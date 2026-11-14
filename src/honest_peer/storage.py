"""Consistent local-output snapshots for a mounted/synced Google Drive directory."""

import hashlib
import json
import os
import shutil
import sqlite3
import subprocess
import sys
import tempfile
from collections import deque
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path
from threading import Event, Thread


def file_digest(path):
    checksum = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            checksum.update(block)
    return checksum.hexdigest()


def write_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    os.replace(temporary, path)


def _inside(path, parent):
    return path == parent or parent in path.parents


def _output_files(root):
    for path in sorted(root.rglob("*")):
        if path.is_symlink() or not path.is_file():
            continue
        if any(part in (".venv", "__pycache__", ".git") for part in path.relative_to(root).parts):
            continue
        if path.name.endswith((".lock", "-wal", "-shm", ".tmp")):
            continue
        checkpoint = next(
            (parent for parent in path.parents if parent.name.startswith("checkpoint-") and parent != root),
            None,
        )
        if checkpoint is not None and not (checkpoint / "manifest.json").exists():
            continue  # Trainer publishes this marker after checkpoint files are complete.
        yield path


def snapshot_outputs(run_root, backup_root):
    """Publish an immutable content-addressed snapshot; latest.json changes only after success.

    SQLite's online backup API includes committed WAL transactions without copying live
    WAL/SHM files. It writes to a local temporary database before copying to Drive.
    Use a dedicated backup folder and one writer per experiment.
    """
    root, backup = Path(run_root).resolve(), Path(backup_root).resolve()
    if _inside(backup, root) or _inside(root, backup):
        raise ValueError("Working output and backup directories must be separate, non-nested paths")
    backup.mkdir(parents=True, exist_ok=True)
    latest_path = backup / "latest.json"
    previous = json.loads(latest_path.read_text()) if latest_path.exists() else {"files": {}}
    entries, uploaded = {}, 0
    with tempfile.TemporaryDirectory(prefix="honest-peer-snapshot-") as temporary:
        temporary = Path(temporary)
        for source in _output_files(root):
            relative = source.relative_to(root).as_posix()
            stat = source.stat()
            signature = [stat.st_size, stat.st_mtime_ns]
            wal = source.with_name(source.name + "-wal")
            if source.suffix == ".sqlite" and wal.exists():
                wal_stat = wal.stat()
                signature.extend([wal_stat.st_size, wal_stat.st_mtime_ns])
            old = previous["files"].get(relative)
            if (
                old
                and old["signature"] == signature
                and (backup / "objects" / old["sha256"][:2] / old["sha256"]).exists()
            ):
                entries[relative] = old
                continue
            staged = temporary / "file"
            if source.suffix == ".sqlite":
                with closing(sqlite3.connect(source.as_uri() + "?mode=ro", uri=True, timeout=30)) as database:
                    with closing(sqlite3.connect(staged)) as target:
                        database.backup(target)
            else:
                try:
                    shutil.copyfile(source, staged)
                except FileNotFoundError:
                    continue  # A still-being-written temporary output disappeared.
            checksum = file_digest(staged)
            object_path = backup / "objects" / checksum[:2] / checksum
            if not object_path.exists():
                object_path.parent.mkdir(parents=True, exist_ok=True)
                upload = object_path.with_name(checksum + ".tmp")
                shutil.copyfile(staged, upload)
                os.replace(upload, object_path)
                uploaded += 1
            entries[relative] = {"sha256": checksum, "size": staged.stat().st_size, "signature": signature}
            staged.unlink()
    snapshot = {"format": 1, "timestamp_utc": datetime.now(timezone.utc).isoformat(), "files": entries}
    # Previous objects remain available if the runtime disconnects before this publish.
    write_json(latest_path, snapshot)
    return {"backup_root": str(backup), "files": len(entries), "objects_uploaded": uploaded}


def restore_outputs(run_root, backup_root):
    """Restore the last published snapshot to an empty working directory and verify bytes."""
    root, backup = Path(run_root).resolve(), Path(backup_root).resolve()
    if _inside(backup, root) or _inside(root, backup):
        raise ValueError("Working output and backup directories must be separate, non-nested paths")
    if root.exists() and any(root.iterdir()):
        raise ValueError("Restore requires an empty working output directory; existing work is preserved")
    latest_path = backup / "latest.json"
    if not latest_path.exists():
        return {"restored_files": 0}
    snapshot = json.loads(latest_path.read_text())
    root.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".{root.name}-restore-", dir=root.parent) as temporary:
        staging = Path(temporary)
        for relative, entry in snapshot["files"].items():
            destination = (staging / relative).resolve()
            checksum = entry["sha256"]
            if not _inside(destination, staging) or destination == staging:
                raise ValueError("Unsafe path in output snapshot")
            if len(checksum) != 64 or any(character not in "0123456789abcdef" for character in checksum):
                raise ValueError("Invalid object hash in output snapshot")
            source = backup / "objects" / checksum[:2] / checksum
            if file_digest(source) != checksum:
                raise ValueError(f"Backup object failed integrity verification: {relative}")
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, destination)
        if root.exists():
            root.rmdir()
        os.replace(staging, root)
    return {"restored_files": len(snapshot["files"])}


def _stream_command(command, env):
    """Forward child output through Python so Colab displays it in the cell."""
    tail = deque(maxlen=200)
    with subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        env={**(os.environ if env is None else env), "PYTHONUNBUFFERED": "1"},
    ) as process:
        try:
            for line in process.stdout:
                print(line, end="", flush=True)
                tail.append(line)
            returncode = process.wait()
        except BaseException:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
            raise
        if returncode:
            raise subprocess.CalledProcessError(returncode, command, output="".join(tail))
    return subprocess.CompletedProcess(command, returncode)


def run_with_backup(command, run_root, backup_root=None, interval_seconds=60, capture_output=False, env=None):
    """Run one stage, periodically snapshot, and always try a final snapshot on success/failure."""
    if interval_seconds <= 0:
        raise ValueError("Backup interval must be positive")
    stop = Event()
    failures = []

    def backup_loop():
        while not stop.wait(interval_seconds):
            try:
                snapshot_outputs(run_root, backup_root)
            except Exception as exc:
                failures.append(exc)

    thread = Thread(target=backup_loop, daemon=True) if backup_root is not None else None
    if thread:
        thread.start()
    try:
        result = (
            subprocess.run(command, check=True, capture_output=True, text=True, env=env)
            if capture_output
            else _stream_command(command, env)
        )
    except subprocess.CalledProcessError as exc:
        if capture_output:
            if exc.stdout:
                print(exc.stdout, end="", flush=True)
            if exc.stderr:
                print(exc.stderr, end="", file=sys.stderr, flush=True)
        raise
    finally:
        stop.set()
        if thread:
            thread.join()
            # If this fails, propagate it so the notebook does not claim cloud persistence.
            snapshot_outputs(run_root, backup_root)
    if failures:
        print(
            f"Backup recovered after {len(failures)} temporary snapshot error(s); final snapshot succeeded.",
            file=sys.stderr,
        )
    return result
