import json
import sys

import pytest

from honest_peer.artifacts import Cache
from honest_peer.storage import restore_outputs, run_with_backup, snapshot_outputs


def test_snapshot_includes_committed_wal_and_deduplicates_objects(tmp_path):
    root, backup = tmp_path / "work", tmp_path / "drive"
    root.mkdir()
    (root / "sample.json").write_text('{"sample": 1}')
    with Cache(root / "rewrites.sqlite") as cache:
        cache.put({"turn": 1}, {"text": "one"})
        first = snapshot_outputs(root, backup)
        assert first["files"] == 2
        assert not any(
            name.endswith(("-wal", "-shm"))
            for name in json.loads((backup / "latest.json").read_text())["files"]
        )
        second = snapshot_outputs(root, backup)
        assert second["objects_uploaded"] == 0
        cache.put({"turn": 2}, {"text": "two"})
        snapshot_outputs(root, backup)
    restored = tmp_path / "restored"
    assert restore_outputs(restored, backup)["restored_files"] == 2
    with Cache(restored / "rewrites.sqlite") as cache:
        assert cache.get({"turn": 1}) == {"text": "one"}
        assert cache.get({"turn": 2}) == {"text": "two"}
    with pytest.raises(ValueError, match="empty"):
        restore_outputs(restored, backup)


def test_snapshot_publishes_only_completed_training_checkpoints(tmp_path):
    root, backup = tmp_path / "work", tmp_path / "drive"
    checkpoint = root / "training/checkpoint-3"
    checkpoint.mkdir(parents=True)
    (checkpoint / "adapter.safetensors").write_bytes(b"incomplete fixture")
    assert snapshot_outputs(root, backup)["files"] == 0
    (checkpoint / "manifest.json").write_text("{}")
    assert snapshot_outputs(root, backup)["files"] == 2


def test_restore_detects_corruption_without_leaving_partial_outputs(tmp_path):
    root, backup = tmp_path / "work", tmp_path / "drive"
    root.mkdir()
    (root / "file.json").write_text("{}")
    snapshot_outputs(root, backup)
    entry = json.loads((backup / "latest.json").read_text())["files"]["file.json"]
    (backup / "objects" / entry["sha256"][:2] / entry["sha256"]).write_text("corrupted")
    with pytest.raises(ValueError, match="integrity"):
        restore_outputs(tmp_path / "restore", backup)
    assert not (tmp_path / "restore").exists()


def test_stage_failure_still_saves_outputs(tmp_path):
    root, backup = tmp_path / "work", tmp_path / "drive"
    root.mkdir()
    command = [
        sys.executable,
        "-c",
        "from pathlib import Path; import sys; Path(sys.argv[1]).write_text('partial'); raise SystemExit(2)",
        str(root / "progress.txt"),
    ]
    import subprocess

    with pytest.raises(subprocess.CalledProcessError):
        run_with_backup(command, root, backup)
    restore_outputs(tmp_path / "restore", backup)
    assert (tmp_path / "restore/progress.txt").read_text() == "partial"


def test_backup_paths_must_not_overlap(tmp_path):
    with pytest.raises(ValueError, match="non-nested"):
        snapshot_outputs(tmp_path, tmp_path / "backup")


def test_periodic_snapshot_happens_before_stage_finishes(tmp_path, monkeypatch):
    from subprocess import CompletedProcess
    from threading import Event

    from honest_peer import storage

    root, backup = tmp_path / "work", tmp_path / "drive"
    root.mkdir()
    saved = Event()
    calls = []
    original = storage.snapshot_outputs

    def snapshot(*args):
        result = original(*args)
        calls.append(result)
        saved.set()
        return result

    def stage(*args, **kwargs):
        (root / "progress.txt").write_text("first")
        assert saved.wait(5), "No periodic backup occurred"
        (root / "progress.txt").write_text("finished")
        return CompletedProcess(args[0], 0)

    monkeypatch.setattr(storage, "snapshot_outputs", snapshot)
    monkeypatch.setattr(storage, "_stream_command", stage)
    storage.run_with_backup(["fixture-stage"], root, backup, interval_seconds=0.01)
    assert len(calls) >= 2
    restore_outputs(tmp_path / "restored", backup)
    assert (tmp_path / "restored/progress.txt").read_text() == "finished"


@pytest.mark.parametrize("capture_output", [False, True])
def test_child_failure_output_is_visible_in_notebook(tmp_path, capsys, capture_output):
    import subprocess

    command = [
        sys.executable,
        "-c",
        "import sys; print('stage progress'); print('ValueError: fixture failure', file=sys.stderr); sys.exit(1)",
    ]
    with pytest.raises(subprocess.CalledProcessError):
        run_with_backup(command, tmp_path, capture_output=capture_output)
    captured = capsys.readouterr()
    visible = captured.out + captured.err
    assert "stage progress" in visible
    assert "ValueError: fixture failure" in visible


def test_captured_json_stdout_remains_separate_from_stderr(tmp_path):
    command = [
        sys.executable,
        "-c",
        "import sys; print('{\"ready\": true}'); print('diagnostic', file=sys.stderr)",
    ]
    result = run_with_backup(command, tmp_path, capture_output=True)
    assert json.loads(result.stdout) == {"ready": True}
    assert result.stderr.strip() == "diagnostic"


def test_interrupted_upload_keeps_previous_snapshot_restorable(tmp_path, monkeypatch):
    from honest_peer import storage

    root, backup = tmp_path / "work", tmp_path / "drive"
    root.mkdir()
    (root / "progress.txt").write_text("previous")
    snapshot_outputs(root, backup)
    previous_index = (backup / "latest.json").read_text()
    (root / "progress.txt").write_text("new")
    original_copy = storage.shutil.copyfile

    def interrupted(source, destination, **kwargs):
        if Path(destination).parent.parent.name == "objects":
            raise OSError("Simulated disconnected Drive mount")
        return original_copy(source, destination, **kwargs)

    from pathlib import Path

    with monkeypatch.context() as patch:
        patch.setattr(storage.shutil, "copyfile", interrupted)
        with pytest.raises(OSError, match="disconnected"):
            snapshot_outputs(root, backup)
    assert (backup / "latest.json").read_text() == previous_index
    restore_outputs(tmp_path / "restored", backup)
    assert (tmp_path / "restored/progress.txt").read_text() == "previous"
