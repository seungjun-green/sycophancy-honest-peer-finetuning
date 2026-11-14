import csv
import json
import os
import sys
from copy import deepcopy
from pathlib import Path
from types import ModuleType, SimpleNamespace

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
    for row in rows:
        if len(row["conversations"]) > 1:
            row["conversations"][0]["value"] += '\nPlease preserve this quote: "Grüße, friend".'
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
        "RUN_ROOT": root,
        "BACKUP_ROOT": backup,
        "run_stage": run_stage,
        "Path": Path,
        "json": json,
    }
    return root, backup, cells, namespace, calls


def test_notebook_smoke_defaults_to_small_isolated_data_and_drive_backup(notebook_run, capsys):
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
    assert namespace["smoke_result"].stdout is None  # Progress was streamed, not buffered until completion.
    visible = capsys.readouterr().out
    assert "Loading ShareGPT source" in visible
    assert "Filtering source conversation 191/200" in visible
    assert "Preparation complete" in visible
    assert "200/200" in visible
    assert "Token lengths (train)" in visible
    assert all(f"Stage {stage}/6" in visible for stage in range(1, 7))
    assert namespace["smoke_data_root"] == root / "data/live-smoke"
    assert production.read_text() == "existing full dataset must remain untouched\n"
    assert len(calls) == 1 and "--smoke" in calls[0]
    assert not list(root.rglob("*.rewritten.jsonl"))
    snapshot = json.loads((backup / "latest.json").read_text())
    assert "data/live-smoke/train.original.jsonl" in snapshot["files"]
    assert "data/live-smoke/summary.json" in snapshot["files"]
    with (root / "data/live-smoke/classified_conversations.csv").open(newline="") as stream:
        exported = list(csv.DictReader(stream))
    assert len(exported) == 12
    assert {row["split"] for row in exported} == {"train", "validation"}
    originals = [
        row
        for split in ("train", "validation")
        for row in map(
            json.loads, (root / f"data/live-smoke/{split}.original.jsonl").read_text().splitlines()
        )
    ]
    assert {row["id"] for row in exported} == {row["id"] for row in originals}
    for row in exported:
        original = next(original for original in originals if original["id"] == row["id"])
        assert row["category"] == original["category"]
        assert all(message["content"] in row["conversation"] for message in original["messages"])
    assert "data/live-smoke/classified_conversations.csv" in snapshot["files"]
    assert "data/live-smoke/category_counts.csv" in snapshot["files"]
    with (root / "data/live-smoke/category_counts.csv").open(newline="") as stream:
        category_counts = list(csv.DictReader(stream))
    assert len(category_counts) == 6
    assert sum(int(row["train"]) for row in category_counts) == 10
    assert sum(int(row["validation"]) for row in category_counts) == 2
    (root / "data/live-smoke/classified_conversations.csv").unlink()
    (root / "data/live-smoke/category_counts.csv").unlink()
    # Review can recover saved results even if an older cell failed to parse noisy stdout.
    namespace["smoke_result"].stdout = "Downloading model assets...\nnot JSON"
    del namespace["smoke_data_root"]
    exec(cells["preparation-review"], namespace)
    assert namespace["smoke_review"]["sample_sizes"] == {"train": 10, "validation": 2}
    assert len(calls) == 1  # Recovery did not repeat preparation.
    assert (root / "data/live-smoke/classified_conversations.csv").is_file()
    assert (root / "data/live-smoke/category_counts.csv").is_file()
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


@pytest.mark.parametrize("case", ["success", "missing", "denied", "empty"])
def test_preparation_loads_colab_secret_without_prompt_or_token_output(
    notebook_run, monkeypatch, capsys, case
):
    import getpass

    _, _, cells, namespace, _ = notebook_run
    calls = []
    secret = "fixture-hf-token-not-for-output"
    monkeypatch.setenv("HF_TOKEN", "")
    monkeypatch.delenv("HF_TOKEN", raising=False)
    missing, denied = (
        type("SecretNotFoundError", (Exception,), {}),
        type("NotebookAccessError", (Exception,), {}),
    )

    def get(key):
        calls.append(key)
        if case in {"missing", "denied"}:
            raise {"missing": missing, "denied": denied}[case]("fixture")
        return "" if case == "empty" else secret

    def unexpected_prompt(*args, **kwargs):
        raise AssertionError("No manual token prompt is allowed")

    google, colab = ModuleType("google"), ModuleType("google.colab")
    google.__path__ = []
    colab.userdata = SimpleNamespace(get=get, SecretNotFoundError=missing, NotebookAccessError=denied)
    monkeypatch.setitem(sys.modules, "google", google)
    monkeypatch.setitem(sys.modules, "google.colab", colab)
    monkeypatch.setattr(getpass, "getpass", unexpected_prompt)
    if case == "success":
        exec(cells["preparation-auth"], namespace)
        assert os.environ["HF_TOKEN"] == secret
        assert "hf_token" not in namespace
    else:
        with pytest.raises(RuntimeError, match="Colab Secrets"):
            exec(cells["preparation-auth"], namespace)
        assert "HF_TOKEN" not in os.environ
    assert calls == ["HF_TOKEN"]
    captured = capsys.readouterr()
    assert secret not in captured.out + captured.err
