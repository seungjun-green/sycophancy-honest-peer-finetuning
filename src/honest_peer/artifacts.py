"""Atomic artifacts, immutable run manifests, and content-addressed durable caches."""

import hashlib
import importlib.metadata
import json
import os
import random
import sqlite3
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from filelock import FileLock


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def file_digest(path):
    hasher = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def now():
    return datetime.now(timezone.utc).isoformat()


def atomic_text(path, text):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".{os.getpid()}.tmp")
    with tmp.open("w") as f:
        f.write(text)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def write_json(path, value):
    atomic_text(path, json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def write_jsonl(path, rows):
    atomic_text(path, "".join(canonical(row) + "\n" for row in rows))


def read_jsonl(path):
    with Path(path).open() as f:
        for n, line in enumerate(f, 1):
            if line.strip():
                try:
                    yield json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSON in {path}:{n}") from exc


def seed_everything(seed):
    random.seed(seed)
    import numpy as np

    np.random.seed(seed)
    if "torch" in sys.modules:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def provenance(config):
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True
        ).strip()
        dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], text=True).strip())
    except subprocess.CalledProcessError:
        commit, dirty = None, True
    packages = {}
    for name in ("honest-peer", "torch", "transformers", "peft", "vllm", "datasets", "anthropic", "lm-eval"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            pass
    package_root = Path(__file__).parent
    source_hashes = {
        str(path.relative_to(package_root)): file_digest(path)
        for path in sorted(package_root.rglob("*"))
        if path.is_file() and path.suffix in (".py", ".txt")
    }
    return {
        "config": config,
        "config_sha256": digest(config),
        "git_commit": commit,
        "source_tree_sha256": digest(source_hashes),
        "git_dirty": dirty,
        "timestamp_utc": now(),
        "python": sys.version,
        "packages": packages,
    }


class Run:
    """One writer per directory; changing config/input fails rather than mixing experiments."""

    def __init__(self, path, config, inputs=None):
        self.path = Path(path)
        self.path.mkdir(parents=True, exist_ok=True)
        self.lock = FileLock(str(self.path / ".run.lock"), timeout=1)
        self.config = config
        self.inputs = inputs or {}

    def __enter__(self):
        self.lock.acquire()
        try:
            manifest_path = self.path / "manifest.json"
            if manifest_path.exists():
                manifest = json.loads(manifest_path.read_text())
                code_hash = provenance(self.config)["source_tree_sha256"]
                if (
                    manifest["config_sha256"] != digest(self.config)
                    or manifest["inputs"] != self.inputs
                    or manifest.get("source_tree_sha256") != code_hash
                ):
                    raise ValueError(
                        f"Run {self.path} has different config, inputs, or code. Choose a new output directory."
                    )
            else:
                write_json(manifest_path, {**provenance(self.config), "inputs": self.inputs})
            return self
        except BaseException:
            self.lock.release()
            raise

    def __exit__(self, *args):
        self.lock.release()


class Cache:
    """SQLite commits every successful call; interrupted/failed requests stay retryable."""

    def __init__(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(path)
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("PRAGMA synchronous=FULL")
        self.db.execute(
            "CREATE TABLE IF NOT EXISTS results (key TEXT PRIMARY KEY, payload TEXT, value TEXT, created TEXT)"
        )

    def get(self, payload):
        row = self.db.execute("SELECT value FROM results WHERE key=?", (digest(payload),)).fetchone()
        return json.loads(row[0]) if row else None

    def put(self, payload, value):
        with self.db:
            self.db.execute(
                "INSERT OR REPLACE INTO results VALUES (?, ?, ?, ?)",
                (digest(payload), canonical(payload), canonical(value), now()),
            )

    def close(self):
        self.db.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
