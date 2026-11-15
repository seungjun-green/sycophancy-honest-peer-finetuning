"""Regenerate thin Colab runner notebooks (standard nbformat, no notebook dependency)."""

import hashlib
import json
from pathlib import Path

STAGES = {
    "smoke_test": (
        "Offline smoke test",
        ["train", "eval"],
        [],
        [("smoke", ["--output", str(Path("outputs/smoke"))])],
    ),
    "prepare_training_data": (
        "Prepare training data through sampling",
        ["data"],
        ["HF_TOKEN"],
        [("prepare-data", [])],
    ),
    "rewrite_training_data": (
        "Review the peer-tone prompt, then rewrite with Claude",
        ["rewrite"],
        ["ANTHROPIC_API_KEY"],
        [],
    ),
    "train_adapter": (
        "Train the peer or original-data control adapter",
        ["train"],
        ["HF_TOKEN"],
        [("train", [])],
    ),
    "prepare_evaluation_data": (
        "Freeze benchmark items and incorrect beliefs",
        [],
        ["ANTHROPIC_API_KEY", "HF_TOKEN"],
        [("prepare", ["--benchmark", "all"])],
    ),
    "evaluate_sycophancy": (
        "Evaluate sycophancy",
        ["eval"],
        ["ANTHROPIC_API_KEY", "HF_TOKEN"],
        [("evaluate", ["--benchmark", name]) for name in ("triviaqa", "truthfulqa", "pushback")],
    ),
    "evaluate_honesty": (
        "Evaluate honesty",
        ["eval"],
        ["ANTHROPIC_API_KEY", "HF_TOKEN"],
        [("evaluate", ["--benchmark", name]) for name in ("simpleqa", "mask")],
    ),
    "evaluate_capabilities": (
        "Evaluate capability and instruction following",
        ["eval"],
        ["HF_TOKEN"],
        [("evaluate", ["--benchmark", name]) for name in ("mmlu", "gsm8k", "ifeval")],
    ),
    "compare_results": ("Compare all model variants", [], [], [("compare", [])]),
}


def cell(kind, source, tag=None):
    result = {
        "cell_type": kind,
        "id": hashlib.sha256(source.encode()).hexdigest()[:12],
        "metadata": {"tags": [tag]} if tag else {},
        "source": source.splitlines(keepends=True),
    }
    if kind == "code":
        result.update(execution_count=None, outputs=[])
    return result


def build(name, title, extras, secrets, commands):
    setup = """import os, subprocess, sys
from pathlib import Path

REPO_URL = "https://github.com/seungjun-green/sycophancy-honest-peer-finetuning"  # Cloned by a fresh Colab runtime.
GIT_REF = "main"  # Use a commit SHA for a reproducible experiment.
PROJECT_DIR = Path.cwd() if Path("src/honest_peer/cli.py").exists() else Path.cwd() / "honest-peer"
RUN_ROOT = PROJECT_DIR / "outputs"  # Local working files; Google Drive snapshots provide persistence.
STORAGE_MODE = "google_drive"  # Hosted Colab default; use "drive_directory" for an existing local Drive mount/sync.
DRIVE_SUBDIR = "honest-peer-experiment"  # MyDrive subfolder for hosted Colab backups.
DRIVE_BACKUP_ROOT = None  # Local runtime example: Path("/your/mounted-or-synced-drive/honest-peer-experiment")
BACKUP_INTERVAL_SECONDS = 60
CONFIG = "configs/default.yaml"  # Eval: configs/peer.yaml or configs/control.yaml. Control train: configs/control.yaml.
OVERRIDES = []  # e.g. ["rewrite.model=ACCOUNT_VISIBLE_MODEL", "judge.model=ACCOUNT_VISIBLE_MODEL"]
LIVE_SMOKE = False  # Small real API/GPU run. smoke_test.ipynb uses synthetic data and runs on CPU.
"""
    if name == "prepare_training_data":
        setup = setup.replace(
            "LIVE_SMOKE = False  # Small real API/GPU run. smoke_test.ipynb uses synthetic data and runs on CPU.",
            "RUN_FULL_PREPARATION = False  # Run the small preparation check first; enable the full stage after review.",
        )
    bootstrap = f"""if not (PROJECT_DIR / "pyproject.toml").exists():
    if not REPO_URL:
        raise ValueError("Set REPO_URL to your GitHub repository URL.")
    subprocess.run(["git", "clone", REPO_URL, str(PROJECT_DIR)], check=True)
    subprocess.run(["git", "-C", str(PROJECT_DIR), "checkout", GIT_REF], check=True)
os.chdir(PROJECT_DIR)
subprocess.run([sys.executable, "-m", "pip", "install", "uv==0.11.22"], check=True)
EXTRAS = {extras!r}
subprocess.run(["uv", "sync", "--frozen", *[arg for extra in EXTRAS for arg in ("--extra", extra)]], check=True)
RUN_ROOT.mkdir(parents=True, exist_ok=True)
"""
    storage = """# Only standard-library code is imported into the notebook kernel.
sys.path.insert(0, str(PROJECT_DIR / "src"))
from honest_peer.storage import restore_outputs, run_with_backup

if STORAGE_MODE == "google_drive":
    from google.colab import drive
    drive.mount("/content/drive")
    BACKUP_ROOT = Path("/content/drive/MyDrive") / DRIVE_SUBDIR
elif STORAGE_MODE == "drive_directory":
    if DRIVE_BACKUP_ROOT is None or not Path(DRIVE_BACKUP_ROOT).parent.is_dir():
        raise ValueError("Set DRIVE_BACKUP_ROOT inside an existing mounted or synced Google Drive directory.")
    BACKUP_ROOT = Path(DRIVE_BACKUP_ROOT)
elif STORAGE_MODE == "local":
    BACKUP_ROOT = None
else:
    raise ValueError("Choose local, google_drive, or drive_directory for STORAGE_MODE.")

if BACKUP_ROOT is not None and not any(RUN_ROOT.iterdir()):
    print(restore_outputs(RUN_ROOT, BACKUP_ROOT))
print("Working outputs:", RUN_ROOT)
print("Drive backups:", BACKUP_ROOT)

def run_stage(args, capture_output=False):
    # Colab's inline plotting backend is unavailable in the isolated project environment.
    stage_env = {**os.environ, "MPLBACKEND": "Agg"}
    return run_with_backup(args, RUN_ROOT, BACKUP_ROOT,
                           interval_seconds=BACKUP_INTERVAL_SECONDS,
                           capture_output=capture_output, env=stage_env)
"""
    auth = f"""import os
# Keys come from Colab Secrets (key icon in the left sidebar; enable Notebook access).
# ANTHROPIC_API_KEY may also be stored under the name CLAUDE_API_KEY.
SECRET_NAMES = {{"ANTHROPIC_API_KEY": ["ANTHROPIC_API_KEY", "CLAUDE_API_KEY"], "HF_TOKEN": ["HF_TOKEN"]}}
try:
    from google.colab import userdata
except ImportError:
    userdata = None  # Not in Colab: export the variables in your shell instead.

for key in {secrets!r}:
    if os.environ.get(key):
        print(key, "already set in the environment.")
        continue
    names = SECRET_NAMES.get(key, [key])
    for name in names if userdata is not None else []:
        try:
            value = userdata.get(name)
        except (userdata.SecretNotFoundError, userdata.NotebookAccessError):
            continue
        if value:
            os.environ[key] = value
            print(key, "loaded from Colab Secrets (" + name + ").")
            break
    if not os.environ.get(key):
        raise RuntimeError(
            "Add " + " or ".join(names) + " to Colab Secrets and enable Notebook access, then rerun this cell."
            if userdata is not None
            else "Set " + key + " in the environment before starting Jupyter."
        )
value = None
"""
    effective = '''import json
PYTHON = str(PROJECT_DIR / ".venv/bin/python")
CLI = str(PROJECT_DIR / ".venv/bin/honest-peer")
# Resolve portable package prompts in the repo's locked environment.
script = """import json,sys
from pathlib import Path
from honest_peer.config import load_config
cfg=load_config(sys.argv[1],json.loads(sys.argv[2]))
root=Path(sys.argv[3]).resolve()
cfg['paths'].update({key:str(root/key) for key in ('data','prepared','evaluation','comparison','cache')})
variant='control' if cfg['training']['data_variant']=='original' else 'peer'
cfg['paths']['training']=str(root/'training'/variant)
if cfg['model']['adapter']:
    cfg['model']['adapter']=str(root/'training'/cfg['model']['variant']/'adapter')
print(json.dumps(cfg))"""
effective = subprocess.check_output([PYTHON, "-c", script, CONFIG, json.dumps(OVERRIDES), str(RUN_ROOT)], text=True)
effective_path = RUN_ROOT / ("config-" + Path(CONFIG).stem + ".json")
effective_path.write_text(effective)
'''
    runner = f"""COMMANDS = {commands!r}
for stage, options in COMMANDS:
    args = [CLI, stage, "--config", str(effective_path), *options]
    if LIVE_SMOKE and stage != "smoke":
        args.append("--smoke")
    run_stage(args)
"""
    if name == "smoke_test":
        runner = 'run_stage([str(PROJECT_DIR / ".venv/bin/honest-peer"), "smoke", "--output", str(RUN_ROOT / "smoke")])\n'
    cells = [
        cell(
            "markdown",
            f"# {title}\n\nRun in Google Colab. Configure the repo URL, commit, RUN_ROOT, and model names below. GPU stages require a CUDA-enabled runtime. Keep RUN_ROOT and all evaluation settings identical across base/peer/control. API secrets stay in environment variables. Each benchmark runs in a separate process so GPU memory is released.\n",
        ),
        cell("code", setup),
        cell("code", bootstrap),
        cell(
            "markdown",
            "## Save intermediate outputs to Google Drive\n\nGoogle Drive backup is enabled by default for hosted Colab (STORAGE_MODE = google_drive). Run the storage cell to mount Drive and authorize access. Local runtimes: choose drive_directory and supply a folder already mounted or synced on the local machine; Colab's built-in Drive mount is unavailable there. Outputs and active SQLite databases stay in local RUN_ROOT. A consistent snapshot is saved to Drive every BACKUP_INTERVAL_SECONDS and when each stage exits. An empty RUN_ROOT is restored from the latest snapshot. Keep the same experiment folder and one writer at a time. An abrupt runtime loss can redo work since the last completed snapshot. [Colab storage/local-runtime FAQ](https://research.google.com/colaboratory/faq.html)\n",
        ),
        cell("code", storage),
    ]
    if secrets:
        names = " and ".join(
            "ANTHROPIC_API_KEY (or CLAUDE_API_KEY)" if key == "ANTHROPIC_API_KEY" else key for key in secrets
        )
        cells.append(
            cell(
                "markdown",
                f"## API keys from Colab Secrets\n\nAdd {names} in Colab's Secrets panel (key icon in the left sidebar) and enable Notebook access. The next cell reads them with userdata.get and passes them to the project subprocess through the environment. Nothing is typed, printed, or written to disk.\n",
            )
        )
        cells.append(cell("code", auth, tag="preparation-auth" if name == "prepare_training_data" else None))
    if name != "smoke_test":
        cells.append(cell("code", effective))
    if name not in {"rewrite_training_data", "prepare_training_data"}:
        cells.append(cell("code", runner))
    if name == "prepare_training_data":
        cells.insert(
            1,
            cell(
                "markdown",
                "This notebook stops after filtering, category labeling, sampling, and the train/validation split. It saves the original conversations and tokenizer. It does not call Claude. Run all cells to check a small real-data preparation first. After reviewing its results, enable RUN_FULL_PREPARATION and rerun the full-preparation cell. Continue with rewrite_training_data.ipynb after full preparation, using the same RUN_ROOT and data settings.\n\nCategory rules follow the Nature paper's Supplementary Information §1.1.1: inspect adjacent user–assistant pairs in order, checking refusal → factual → creative → technical → advice, with other as the fallback. The first matching pair wins. Classification uses the full conversation before the 20-message limit. The reports below record the rules, digest, and category counts. See docs/classification.md for the pattern table and explicit choices about refusal case sensitivity and apostrophes. To replace data prepared under older rules, choose a fresh RUN_ROOT and DRIVE_SUBDIR before running.\n",
            ),
        )
        cells.extend(
            [
                cell(
                    "markdown",
                    "## Smoke run: small real-data preparation\n\nThis runs the configured source, filters, classification rules, and tokenizer on **up to the first 200 source conversations**, targeting **10 training and 2 validation conversations**. Counts can be smaller after filtering and grouped splitting, and some categories may be absent; this checks the pipeline, not category coverage or a representative distribution. An empty split fails clearly. The first run can still download the complete ShareGPT source file, Detoxify assets, and tokenizer. No Claude API or training stage is called.\n\nLive output identifies six stages: source loading, filtering/classification, sampling, tokenizer loading, token-length statistics, and finalizing saved files. tqdm bars show filtering and tokenization progress. During slow downloads/model loading, a heartbeat repeats the current phase and elapsed time every 15 seconds. Full runs stream the source without a known total, so the filtering bar reports processed counts and rate rather than a percentage.\n\nEvery execution starts a NEW smoke run in RUN_ROOT/smoke-runs/<unique-run-id>/live-smoke/. Existing runs are preserved. At completion, smoke_review.zip contains train.original.jsonl, validation.original.jsonl, and summary.json; Colab starts its download automatically. The run and ZIP are included in Drive snapshots. Shared download/filter caches may be reused, but previous prepared outputs are never reused. Full preparation continues to use the normal data folder.\n",
                ),
                cell(
                    "code",
                    """from datetime import datetime, timezone
from uuid import uuid4
from zipfile import ZipFile, ZIP_DEFLATED

# Every execution creates a new run; earlier outputs and manifests stay untouched.
smoke_run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid4().hex[:8]
smoke_run_root = RUN_ROOT / "smoke-runs" / smoke_run_id
smoke_data_root = smoke_run_root / "live-smoke"
print("Starting NEW smoke run:", smoke_run_id, flush=True)
print("Output folder:", smoke_data_root, flush=True)
smoke_result = run_stage([
    CLI, "prepare-data", "--config", str(effective_path),
    "--set", "paths.data=" + json.dumps(str(smoke_run_root)), "--smoke"
])
smoke_summary = json.loads((smoke_data_root / "summary.json").read_text())
print(json.dumps(smoke_summary, indent=2))
smoke_bundle_path = smoke_run_root / "smoke_review.zip"
with ZipFile(smoke_bundle_path, "w", ZIP_DEFLATED) as bundle:
    for filename in ("train.original.jsonl", "validation.original.jsonl", "summary.json"):
        bundle.write(smoke_data_root / filename, arcname=filename)
(RUN_ROOT / "latest-preparation-smoke.json").write_text(json.dumps({"data_root": str(smoke_data_root)}))
from honest_peer.storage import snapshot_outputs
if BACKUP_ROOT is not None:
    snapshot_outputs(RUN_ROOT, BACKUP_ROOT)
print("Smoke run completed. Share this ZIP containing the three review files:", smoke_bundle_path)
try:
    from google.colab import files
    files.download(str(smoke_bundle_path))
except Exception:
    print("Automatic download unavailable. Download smoke_review.zip from the Colab Files panel:", smoke_bundle_path)
""",
                    tag="preparation-smoke",
                ),
                cell(
                    "markdown",
                    "## Review saved results and export CSVs\n\nThe next cell prints the summary and category counts and saves classified_conversations.csv (one row per selected training/validation conversation, including its label and all saved messages) plus category_counts.csv (eligible-pool, training, and validation counts for each category). The smoke target is 12 selected conversations; the conversation CSV does not contain every eligible source record. Saved text reflects the configured turn limit, while labels are assigned before that limit. Both CSVs are included in Drive snapshots.\n\nYou can run this cell independently to export CSVs from completed preparation, including after an older cell's JSON parsing error. It does not download, filter, sample, or call Claude again.\n",
                ),
                cell(
                    "code",
                    """import hashlib
from collections import Counter
from honest_peer.data_review import export_prepared_csvs
from honest_peer.storage import snapshot_outputs

# Review the fresh run above, or restore its saved pointer after reconnecting.
if "smoke_data_root" not in globals():
    smoke_data_root = Path(json.loads((RUN_ROOT / "latest-preparation-smoke.json").read_text())["data_root"])

def review_prepared_data(data_root):
    summary = json.loads((data_root / "summary.json").read_text())
    selection = json.loads((data_root / "selection.json").read_text())
    if not (data_root / "manifest.json").is_file() or not (data_root / "tokenizer").is_dir():
        raise ValueError("Preparation manifest or tokenizer is missing.")
    splits = {}
    for split in ("train", "validation"):
        filename = split + ".original.jsonl"
        payload = (data_root / filename).read_bytes()
        if hashlib.sha256(payload).hexdigest() != selection["files"][filename]:
            raise ValueError("Saved data hash mismatch: " + filename)
        rows = [json.loads(line) for line in payload.splitlines() if line.strip()]
        if not rows or len(rows) != summary["sample_sizes"][split]:
            raise ValueError("Saved split count is empty or inconsistent: " + split)
        splits[split] = rows
    for key in ("id", "group_id"):
        if {row[key] for row in splits["train"]} & {row[key] for row in splits["validation"]}:
            raise ValueError("Training/validation overlap: " + key)
    print("Saved hashes, split counts, and conversation-group separation: passed.")
    print(json.dumps({key: summary[key] for key in ("filter_counts", "sample_sizes", "token_length_stats", "classification_rules_sha256")}, indent=2))
    counts = {split: Counter(row["category"] for row in rows) for split, rows in splits.items()}
    categories = summary["classification"]["priority"]
    print(f"\\n{'Category':<14} {'Eligible pool':>14} {'Training':>10} {'Validation':>12}")
    for category in categories:
        print(f"{category:<14} {summary['category_counts'].get(category, 0):>14} {counts['train'][category]:>10} {counts['validation'][category]:>12}")
    print("Training examples (up to one per category; first pair, clipped for display):")
    for category in categories:
        example = next((row for row in splits["train"] if row["category"] == category), None)
        if example is None:
            print(f"\\n{category}: no training example in this sample")
            continue
        print(f"\\n{category} — {example['id']}")
        for message in example["messages"][:2]:
            print(message["role"] + ": " + message["content"][:800])
    csv_exports = export_prepared_csvs(data_root)
    for name, path in csv_exports.items():
        print(f"CSV ({name}): {path}")
    if BACKUP_ROOT is not None:
        snapshot_outputs(RUN_ROOT, BACKUP_ROOT)
        print("CSV exports included in the Drive snapshot.")
    return summary

smoke_review = review_prepared_data(smoke_data_root)
""",
                    tag="preparation-review",
                ),
                cell(
                    "markdown",
                    "## Full preparation\n\nReview the smoke filter counts, category counts, token lengths, and training examples above. Missing categories in the small subset are not a failure. To prepare the full configured training/validation sample, set RUN_FULL_PREPARATION = True in the settings cell and rerun that cell and the next cell. Full preparation reads the complete configured source (unless you explicitly set a record limit). Its outputs use the normal data folder, which rewrite_training_data.ipynb consumes by default.\n",
                ),
                cell(
                    "code",
                    """if RUN_FULL_PREPARATION:
    run_stage([CLI, "prepare-data", "--config", str(effective_path)])
    full_review = review_prepared_data(Path(json.loads(effective)["paths"]["data"]))
else:
    print("Smoke run complete. Set RUN_FULL_PREPARATION = True in the settings cell, rerun it, then rerun this cell for the full dataset.")
""",
                    tag="preparation-full",
                ),
            ]
        )
    if name == "rewrite_training_data":
        cells.insert(
            1,
            cell(
                "markdown",
                "Run prepare_training_data.ipynb first. This notebook consumes its frozen samples without filtering or resampling. The live prompt smoke test picks **5 training conversations in each of the six categories** and rewrites **one randomly chosen assistant turn** from each: about **30 Claude calls**. It shows the user's message and earlier conversation next to the original and rewritten reply, grouped by category. Change PREVIEW_PER_CATEGORY or PREVIEW_TURNS_PER_CONVERSATION (0 = every turn) in the prompt cell to see more. Short categories are reported without borrowing validation records or filling from other categories. Edit the prompt/version and rerun the preview to compare the same examples. Full rewriting has its own optional cell after review.\n",
            ),
        )
        # Keep the notebook default identical to configs/default.yaml (rewrite.prompt_template/version).
        template = Path("src/honest_peer/prompts/rewrite_authority_v1.txt").read_text()
        cells.extend(
            [
                cell(
                    "markdown",
                    "## Edit the rewrite prompt\n\nChoose an account-visible rewrite.model in OVERRIDES above. Edit this template and its version below, then rerun this cell and the preview cells. Each prompt variant gets its own review directory, while completed matching turns are reused for full generation.\n",
                ),
                cell(
                    "code",
                    'PROMPT_VERSION = "authority-v1"\nPREVIEW_PER_CATEGORY = 5  # conversations per category\nPREVIEW_TURNS_PER_CONVERSATION = 1  # random assistant turns per conversation; 0 = all\nREWRITE_PROMPT = """\n'
                    + template.strip()
                    + '\n""".strip()\n\n# Save the editable template without putting API credentials in a file.\nprompt_dir = RUN_ROOT / "rewrite-prompts"\nprompt_dir.mkdir(exist_ok=True)\nimport hashlib\nprompt_sha = hashlib.sha256(REWRITE_PROMPT.encode()).hexdigest()[:16]\nprompt_path = prompt_dir / (prompt_sha + ".txt")\nprompt_path.write_text(REWRITE_PROMPT)\nrewrite_args = ["--config", str(effective_path), "--set", "rewrite.prompt_template=" + json.dumps(str(prompt_path)), "--set", "rewrite.prompt_version=" + json.dumps(PROMPT_VERSION)]\nif LIVE_SMOKE:\n    rewrite_args.append("--smoke")\n',
                ),
                cell(
                    "markdown",
                    "## Set API concurrency from your rate limits\n\nThis sends one tiny request (8 output tokens) to read your API key's per-minute limits for the rewrite model, estimates tokens per rewrite call from the prepared conversations, and sets how many calls run at once (rewrite.concurrency). Concurrency only changes speed: cached turns and resume checks are unaffected. Rerun this cell after rerunning the prompt cell. Set AUTO_CONCURRENCY = False to keep the configured value.\n",
                ),
                cell(
                    "code",
                    'AUTO_CONCURRENCY = True\nMAX_CONCURRENCY = 32  # upper bound, whatever the limits allow\nif AUTO_CONCURRENCY:\n    tuned = json.loads(run_stage([CLI, "tune-concurrency", *rewrite_args, "--max", str(MAX_CONCURRENCY)], capture_output=True).stdout)\n    print(json.dumps(tuned, indent=2))\n    rewrite_args += ["--set", f"rewrite.concurrency={tuned[\'concurrency\']}"]\n    print(f"Using rewrite.concurrency={tuned[\'concurrency\']} (limited by {tuned[\'binding_limit\']})")\n',
                ),
                cell(
                    "markdown",
                    "## Live Claude smoke test — about 30 turns\n\nThis does not overwrite the production rewritten dataset. Previewed turns are cached and reused by the full rewrite. If a category has fewer training conversations than requested, the summary shows the shortfall. The preview uses the saved tokenizer and does not need another HF login.\n",
                ),
                cell(
                    "code",
                    'preview_command = [CLI, "rewrite-data", *rewrite_args, "--preview", "--per-category", str(PREVIEW_PER_CATEGORY), "--turns-per-conversation", str(PREVIEW_TURNS_PER_CONVERSATION)]\npreview_result = run_stage(preview_command, capture_output=True)\npreview_summary = json.loads(preview_result.stdout)\nprint(json.dumps(preview_summary, indent=2))\n',
                ),
                cell(
                    "code",
                    'from IPython.display import HTML, display\ndisplay(HTML(Path(preview_summary["review_html"]).read_text()))\n',
                ),
                cell(
                    "markdown",
                    "Review tone, factual/code preservation, appropriate agreement/disagreement, uncertainty, and refusals in each category. Change the prompt above and rerun the preview as needed. Automatic code-fence validation does not replace this review.\n\n## Full Claude generation\n\nAfter reviewing the preview, enable and run the next cell. It rewrites both frozen training and validation sets and reuses matching cached preview turns. It does not repeat sampling.\n",
                ),
                cell(
                    "code",
                    'RUN_FULL_REWRITE = False\nif RUN_FULL_REWRITE:\n    run_stage([CLI, "rewrite-data", *rewrite_args])\n    run_stage([CLI, "spot-check", *rewrite_args, "--count", "50"])\nelse:\n    print("Preview only. Set RUN_FULL_REWRITE = True here when ready for the complete selected dataset.")\n',
                ),
            ]
        )
    if name == "compare_results":
        cells.append(
            cell(
                "code",
                'from IPython.display import Markdown, display\ndisplay(Markdown((RUN_ROOT / "comparison/comparison.md").read_text()))\n',
            )
        )
    return {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}},
        "cells": cells,
    }


if __name__ == "__main__":
    for name, (title, extras, secrets, commands) in STAGES.items():
        Path("notebooks", name + ".ipynb").write_text(
            json.dumps(build(name, title, extras, secrets, commands), indent=1) + "\n"
        )
