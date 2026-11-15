import json
import os
import subprocess
import sys
from pathlib import Path


def test_cli_smoke_overrides_inherited_colab_backend(tmp_path):
    output = tmp_path / "smoke"
    result = subprocess.run(
        [sys.executable, "-m", "honest_peer.cli", "smoke", "--output", str(output)],
        env={
            **os.environ,
            "MPLBACKEND": "module://matplotlib_inline.backend_inline",
            "TOKENIZERS_PARALLELISM": "false",
        },
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stdout + "\n" + result.stderr
    summary = json.loads((output / "smoke-summary.json").read_text())
    assert summary["synthetic"] and summary["comparison_metrics"] == 42
    assert list((output / "comparison").glob("*.png"))


def test_all_notebooks_override_backend_only_for_child_process(monkeypatch, tmp_path):
    inline = "module://matplotlib_inline.backend_inline"
    monkeypatch.setenv("MPLBACKEND", inline)
    notebooks = list(Path("notebooks").glob("*.ipynb"))
    assert len(notebooks) == 9
    for path in notebooks:
        notebook = json.loads(path.read_text())
        storage_cell = next(
            "".join(cell["source"])
            for cell in notebook["cells"]
            if cell["cell_type"] == "code" and "def run_stage(" in "".join(cell["source"])
        )
        calls = []

        def record(*args, **kwargs):
            calls.append(kwargs)

        namespace = {
            "os": os,
            "RUN_ROOT": tmp_path,
            "BACKUP_ROOT": None,
            "BACKUP_INTERVAL_SECONDS": 60,
            "run_with_backup": record,
        }
        exec("def run_stage(" + storage_cell.split("def run_stage(", 1)[1], namespace)
        namespace["run_stage"](["fixture"])
        assert calls[0]["env"]["MPLBACKEND"] == "Agg", path
        assert os.environ["MPLBACKEND"] == inline
