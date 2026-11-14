import json
import os
import sys
from copy import deepcopy
from pathlib import Path

import pytest

from honest_peer.config import load_config
from honest_peer.smoke import smoke_config
from honest_peer.storage import run_with_backup


@pytest.fixture
def notebook_run(tmp_path):
    root, backup = tmp_path / "work", tmp_path / "drive"
    config = smoke_config(load_config("configs/default.yaml"), root)
    config["data"]["sampling"].update(total=14, validation=4)
    source = Path(config["sources"]["sharegpt"]["local_path"])
    rows = json.loads(source.read_text())
    for index in range(36, 220):
        row = deepcopy(rows[0])
        row["id"] = f"fixture{index}_0"
        row["conversations"][0]["value"] = f"What is 2 + 2? Example {index}."
        rows.append(row)
    source.write_text(json.dumps(rows))
    effective_path = root / "effective.json"
    effective_path.write_text(json.dumps(config))
    calls = []

    def run_stage(args, capture_output=False):
        assert args[:2] == ["honest-peer", "prepare-data"]
        calls.append(args)
        return run_with_backup(
            [sys.executable, "-m", "honest_peer.cli", *args[1:]],
            root,
            backup,
            capture_output=capture_output,
            env={**os.environ, "TOKENIZERS_PARALLELISM": "false"},
        )

    notebook = json.loads(Path("notebooks/prepare_training_data.ipynb").read_text())
    cells = {
        tag: "".join(cell["source"]) for cell in notebook["cells"] for tag in cell["metadata"].get("tags", [])
    }
    namespace = {
        "CLI": "honest-peer",
        "effective": json.dumps(config),
        "effective_path": effective_path,
        "RUN_FULL_PREPARATION": False,
        "run_stage": run_stage,
        "Path": Path,
        "json": json,
    }
    return root, backup, cells, namespace, calls


def test_notebook_smoke_defaults_to_small_isolated_data_and_drive_backup(notebook_run):
    root, backup, cells, namespace, calls = notebook_run
    production = root / "data/train.original.jsonl"
    production.parent.mkdir()
    production.write_text("existing full dataset must remain untouched\n")
    exec(cells["preparation-smoke"], namespace)
    exec(cells["preparation-review"], namespace)
    exec(cells["preparation-full"], namespace)
    assert namespace["smoke_review"]["sample_sizes"] == {"train": 10, "validation": 2}
    assert namespace["smoke_review"]["filter_counts"]["malformed"] == 1
    assert namespace["smoke_review"]["filter_counts"]["source"] == 200
    assert namespace["smoke_data_root"] == root / "data/live-smoke"
    assert production.read_text() == "existing full dataset must remain untouched\n"
    assert len(calls) == 1 and "--smoke" in calls[0]
    assert not list(root.rglob("*.rewritten.jsonl"))
    snapshot = json.loads((backup / "latest.json").read_text())
    assert "data/live-smoke/train.original.jsonl" in snapshot["files"]
    assert "data/live-smoke/summary.json" in snapshot["files"]
    with (root / "data/live-smoke/validation.original.jsonl").open("a") as stream:
        stream.write("{}\n")
    with pytest.raises(ValueError, match="hash mismatch"):
        namespace["review_prepared_data"](namespace["smoke_data_root"])


def test_full_preparation_requires_switch_and_keeps_smoke_outputs(notebook_run):
    root, backup, cells, namespace, calls = notebook_run
    exec(cells["preparation-smoke"], namespace)
    exec(cells["preparation-review"], namespace)
    before = (root / "data/live-smoke/train.original.jsonl").read_bytes()
    namespace["RUN_FULL_PREPARATION"] = True
    exec(cells["preparation-full"], namespace)
    assert namespace["full_review"]["sample_sizes"] == {"train": 14, "validation": 4}
    assert namespace["full_review"]["filter_counts"]["source"] == 221
    assert len(calls) == 2 and "--smoke" not in calls[1]
    assert (root / "data/live-smoke/train.original.jsonl").read_bytes() == before
    snapshot = json.loads((backup / "latest.json").read_text())
    assert {"data/train.original.jsonl", "data/live-smoke/train.original.jsonl"} <= snapshot["files"].keys()
