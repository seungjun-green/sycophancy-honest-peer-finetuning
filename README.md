# Honest-peer persona fine-tuning

Config-driven ShareGPT preparation, Claude rewriting, assistant-only LoRA SFT, paired epistemics evaluations, and results comparison for the experiment in [the supplied spec](docs/spec.md).

## Install

Use Python 3.11 or 3.12. Real training and generation target **Linux + NVIDIA CUDA**. macOS supports CPU tests and the offline smoke pipeline.

    python -m pip install uv==0.11.22
    uv sync --frozen --extra data --extra train --extra eval --extra dev
    source .venv/bin/activate
    honest-peer doctor

The exact transitive dependencies are in uv.lock. The lock selects PyTorch 2.7.1 with **CUDA 12.8** on Linux and native PyTorch on macOS. CUDA 12.8 is needed for the target Blackwell GPU; the same environment targets A100/H100. A compatible host NVIDIA driver is required. Hardware execution still needs verification on the actual target GPU. See the [vLLM 0.10.0 installation guide](https://docs.vllm.ai/en/v0.10.0/getting_started/installation/gpu.html) and [PyTorch Blackwell support](https://pytorch.org/blog/pytorch-2-7/).

Export ANTHROPIC_API_KEY and HF_TOKEN in your shell, or enter them with getpass in the notebooks. Llama 3.1 requires approval for the gated model. Credentials are read from the environment; config files and manifests contain no credentials. Optional W&B authentication also uses its normal environment variables.

The spec's requested Claude default, claude-sonnet-5-5, is preserved in config. **Its availability has not been verified.** Check your account and set both model names to an available model before paid runs:

    honest-peer doctor --api

Edit configs/default.yaml, or pass overrides such as --set rewrite.model=YOUR_MODEL --set judge.model=YOUR_MODEL. Use the same override values for every stage and variant. Prefer immutable model IDs, rather than moving aliases.

## Local verification

    pytest -q
    ruff check src tests scripts
    honest-peer smoke --output outputs/smoke

The offline smoke command builds 10 training and 2 validation conversations, **really trains a PEFT LoRA adapter on a random tiny Llama on CPU**, and exercises all nine evaluation paths on 20 fixture items each for two variants. It writes a side-by-side rewrite check, checkpoints, loss logs, paired tests, comparison tables, and plots.

Colab sets an inline Matplotlib backend for its notebook kernel. The project's isolated environment uses the non-interactive `Agg` backend for saved plots: every notebook passes `MPLBACKEND=Agg` to stage commands, and the CLI sets it before importing training or evaluation libraries. Stage output is forwarded into the notebook cell so command errors remain visible. If an older notebook reports `module://matplotlib_inline.backend_inline` as an invalid backend, run `import os; os.environ["MPLBACKEND"] = "Agg"` in a notebook cell, then rerun the failed stage.

Smoke API responses, benchmark generations, and most grades are synthetic fixtures. They do not measure the tiny trained model or predict Llama's performance. Every output is marked synthetic and each plot is labeled accordingly. Toxicity and language models are skipped in this offline mode. The test suite separately exercises the installed IFEval reference checker.

To test the actual gated Llama tokenizer and decode a masked batch:

    RUN_TOKENIZER_TEST=1 pytest -q tests/test_loss_mask.py

Without this flag, that one integration test is explicitly skipped. TEST_TOKENIZER can point to a downloaded compatible tokenizer directory for offline testing.

Each regular stage also accepts --smoke for a **small live run**: 10 training conversations, 2 validation conversations, at most 200 source conversations, 20 benchmark items, and half an epoch. This uses the real configured APIs/GPU and can incur charges. It redirects default paths beneath outputs/live-smoke. Use the flag consistently across stages.

## Run the experiment

The order below evaluates the base before building the fine-tuning data.

### 1. Inspect and freeze evaluation items

    honest-peer inspect --source sharegpt
    honest-peer inspect --source pushback
    honest-peer inspect --source mask
    honest-peer prepare --benchmark all

Preparation samples once with the configured seed and persists model-independent items. TriviaQA's generated incorrect answers, their explanations, and raw API outputs are stored here and reused by every variant. TruthfulQA incorrect answers come from its own incorrect_answers field.

Exact duplicate benchmark rows are removed before sampling, with the count recorded in selection.json. The upstream pushback file contains five such repeats. Conflicting records sharing an ID always fail; evaluation.duplicate_policy=error can reject even identical repeats.

Inspection downloads the actual configured dataset and prints its schema and first row. ShareGPT's file listing and raw human/gpt, from/value turn structure were inspected during implementation; the refusal-retaining file is selected explicitly. Unexpected schemas fail clearly. Initial source revisions and schemas are documented in [source inspection](docs/source-inspection.md).

### 2. Evaluate the base

    honest-peer evaluate --config configs/default.yaml --benchmark all

Individual benchmark names: triviaqa, truthfulqa, pushback, simpleqa, mask, mmlu, gsm8k, ifeval. AdvBench is opt-in:

    honest-peer prepare --benchmark advbench --set evaluation.safety_enabled=true
    honest-peer evaluate --benchmark advbench --set evaluation.safety_enabled=true

Use the same safety setting for every comparison variant.

### 3. Prepare samples, review the tone prompt, then rewrite

    honest-peer prepare-data
    honest-peer rewrite-data --preview --per-category 10

Notebook prepare_training_data.ipynb starts with a small real-data preparation smoke run: inspect up to the first 200 source conversations and target 10 training plus 2 validation examples. It shows filter counts, category counts (including absent categories), token lengths, and sample training text, and checks saved hashes, split counts, and conversation-group separation. Smoke outputs are isolated under RUN_ROOT/data/live-smoke/ and included in Drive snapshots. The first run can still download the full source file, tokenizer, and filtering assets. This small subset does not guarantee coverage of all six categories.

Full preparation is a separate cell, disabled by default with RUN_FULL_PREPARATION=False. After reviewing the smoke output, enable it in the settings cell, rerun that cell, and run the full-preparation cell. Full preparation stops after filtering, category labeling, sampling, and the original train/validation split. It saves the tokenizer too and never calls Claude. The CLI prepare-data command still performs full configured preparation by default; --smoke selects the small live check.

Continue with rewrite_training_data.ipynb. Its live Claude prompt smoke test selects up to **10 training conversations per category** (at most 60), rewrites every assistant turn, and displays user context/history beside the original and rewritten replies. Multi-turn conversations can require more than 60 API calls. Validation examples are excluded from prompt development. Short categories are reported explicitly, without redistribution.

The rewrite notebook has an editable prompt/version cell. Changing it creates a separate preview directory while retaining the same sampled examples for comparison. Review the HTML, adjust the prompt, and rerun the preview until satisfied. This preview is independent of the generic --smoke flag, which would shrink the original sample set.

    honest-peer rewrite-data
    honest-peer spot-check --count 50

Full generation is a separate notebook cell, disabled by default with RUN_FULL_REWRITE=False until you choose to run it. It consumes the saved originals without filtering or sampling again and reuses matching cached preview turns. The second notebook needs ANTHROPIC_API_KEY; it uses the tokenizer saved by the first notebook without a new HF download.

Preview HTML/JSON are under outputs/data/rewrite-previews/<prompt-run-hash>/. Open outputs/data/spot-check.html for the full dataset check. Inspect factual fidelity, refusal preservation, unwanted contrarianism, and style. Code fences are checked automatically; factual preservation is a manual research-quality check.

The existing honest-peer build-data command remains a combined preparation/full-rewrite shortcut; it does not include the manual preview step.

Filtering validates alternation and content, deduplicates exact conversations, applies configurable Detoxify thresholds to windows covering **all turns**, and optionally detects language. Classification sees the full surviving conversation before truncation to an even complete-turn boundary. Conversations that become identical after truncation are also deduplicated. Long source conversation fragments are grouped by their ShareGPT root ID across the train/validation split.

Category labeling now uses the pattern strings and hierarchical procedure published in the Nature paper's Supplementary Information §1.1.1. Adjacent user–assistant pairs are checked in order; within each pair the priority is refusal → factual → creative → technical → advice, with other as the fallback. Matching stops at the first qualifying pair. User keywords, assistant refusal phrases/code fences, and the specified prefix/substring case rules are configured separately. The supplement leaves refusal case sensitivity unspecified; our default is case-sensitive with ASCII/typographic apostrophe normalization. See [classification rules and interpretation choices](docs/classification.md) for the full table and source. This implements the published description, not the authors' undisclosed executable code.

Sampling supports balanced, random, explicit quotas, or a custom module:function returning rows. When a category is short, redistribution is configurable and the report shows achieved counts. Preparation records the full rules and their digest in summary.json. Rebuild old prepared data in a fresh RUN_ROOT and Drive backup subfolder to use the new labels; existing manifests reject changes to rules or source code.

All assistant turns are rewritten. Each API call sees the original assistant reply plus complete history through its preceding user turn. History can use original or already-rewritten preceding assistant replies. User turns remain unchanged. Failed rewrites stay retryable; the stage fails rather than silently producing a smaller paired training set. Original and rewritten versions share IDs, categories, splits, and message schemas.

Preparation saves train.original.jsonl, validation.original.jsonl, selection.json, summary.json, tokenizer/, and manifest.json. Full rewriting saves train.rewritten.jsonl, validation.rewritten.jsonl, dataset-summary.json, and a separate rewrite/manifest.json and rewrite/summary.json. Preview manifests and category/shortfall counts stay in their own preview directories. Editing a prompt does not change the saved original splits.

### 4. Train the peer adapter

    honest-peer train --config configs/default.yaml

Defaults match the supplied hyperparameters. PEFT + Transformers Trainer is used as the allowed equivalent to TRL, so token labels and checkpoint timing are controlled explicitly.

The assistant loss mask is built from the model's own chat template. Assistant bodies and end markers are supervised across all turns; user/system/header/padding tokens are masked. Every prefix boundary is checked. data-stats.json reports truncation and dropped examples; mask-report.json decodes supervised spans. The tokenizer/template snapshot is saved and verified at evaluation.

The cosine scheduler reaches exactly 10% of peak at the final scheduler step after warmup. Checkpoints and validation run at the first optimizer step on or after each half-epoch boundary; when steps cannot divide an epoch exactly, the actual epoch is logged. Loss logs are local; W&B is off by default. Adapter merging and Hub pushes are off by default.

    honest-peer evaluate --config configs/peer.yaml --benchmark all

### 5. Optional matched original-data control

    honest-peer train --config configs/control.yaml
    honest-peer evaluate --config configs/control.yaml --benchmark all

This uses the **same selected original conversations**. Only config changes. No additional rewriting is needed. Use distinct training directories for peer and control.

### 6. Compare

    honest-peer compare

outputs/comparison contains comparison.csv, comparison.md, comparison.json, and one PNG per metric. Tables include deltas versus base. Confidence intervals resample items, keeping an item's instructions together. Exact two-sided McNemar tests apply to scalar binary outcomes on paired IDs; conditional metrics disclose jointly eligible sample sizes. P-values are unadjusted for multiple comparisons.

Comparison refuses differing item sets, generation/grader/prompt protocols, or mixtures of real and synthetic outputs. Choose a new comparison directory when adding new results:

    honest-peer compare --set paths.comparison=outputs/comparison-with-control

## Benchmark methods and interpretation

| Benchmark | Implementation and metrics |
|---|---|
| TriviaQA / TruthfulQA | Plain and wrong-belief prompts; correctness, wrong-answer endorsement, and refusals graded separately. Sycophancy is correct plain → incorrect and endorsing the injected wrong answer, divided by **all paired items**. A conditional rate given initially correct is also reported. |
| Are you sure? | Preserves all upstream prompts, including partial assistant prefixes; grades initial and challenged responses. Bad flips divide by initially correct items; good flips divide by initially incorrect items. |
| SimpleQA | Official OpenAI grader prompt, Anthropic judge, strict A/B/C parsing; accuracy, accuracy given attempted, F1, incorrect/total, and abstention/total. Invalid grader output fails rather than becoming an abstention. |
| MASK | Official prompts, three direct belief samples plus two indirect checks by default, independent pressure generations, binary belief aggregation, and numerical range judging. All six upstream subsets can be selected via sources.mask.config. Provided-facts uses provided truth; doubling-down preserves the forced first reply. Numerical beliefs are judged jointly; 5% range tolerance matches upstream. |
| MMLU | Zero-shot **generated-letter accuracy**, configurable temperature 0.2; invalid formats count incorrect. This is not the harness's likelihood-ranked MMLU score. |
| GSM8K | Zero-shot chain of thought, temperature 0.2, exact numeric scoring after the requested #### marker; missing/invalid final answers count incorrect. |
| IFEval | Official checker from lm-evaluation-harness 0.4.8, strict/loose prompt and instruction accuracy. |
| AdvBench | Optional Anthropic-judged refusal rate. |

Capability generation uses the shared resumable vLLM runner directly to preserve identical templates, per-item seeds, and cache behavior across every task. IFEval uses the harness's reference checker. Report the generated-answer MMLU/GSM8K protocol alongside scores when comparing with publications.

MASK honesty_score follows upstream's 1 − dishonest fraction, so undetermined cases do not count as contradictions. Explicit consistency, undetermined cases, belief elicitation, and belief accuracy are reported separately to avoid conflating abstention with affirmative honesty. Changing pressure_repeats enables an honesty-at-K check; the summary records its config. The judge provider differs from the original MASK pipeline and should be disclosed.

For another MASK subset:

    honest-peer prepare --benchmark mask --set sources.mask.config=statistics --set paths.prepared=outputs/prepared-statistics

Use matching source/prepared settings and fresh evaluation paths for base and peer.

## Resuming, provenance, and configuration

Rerun the exact command after disconnection. Successful API calls and generations commit immediately to SQLite; completed item records also commit individually. Requests are keyed by content, identity, prompt version, model, and generation settings. Failed requests remain retryable. Retryable API errors use exponential backoff, jitter, bounded concurrency, and Retry-After where supplied. Authentication/not-found errors fail promptly.

Training resumes the latest Trainer checkpoint, including optimizer/scheduler/RNG state. A fully completed training run returns its recorded summary without retraining. A disconnect may redo work since the most recent checkpoint.

Every artifact directory has a manifest containing the full effective stage config (including resolved prompt text), UTC timestamp, git commit, dirty flag, code-content hash, input identity, and installed package versions. Adapters, merged weights, and intermediate training checkpoints carry their own manifest copy. A brand-new repository without a commit records git_commit=null explicitly; commit the source before the real experiment. Config/input/code changes require a new output directory to prevent mixed runs.

All default HF dataset revisions, the base model, and GitHub sources are pinned. If you configure a moving dataset revision, it is resolved and frozen on first download in the source cache. The SimpleQA CSV is identified by its downloaded content hash. API model aliases and nondeterministic GPU kernels mean fixed seeds are not a guarantee of bitwise reproducibility.

YAML configs support extends and repeated --set dotted.key=value overrides; unknown override keys fail. Paths are relative to the working directory, and custom prompt files are resolved there. Prompts ship in src/honest_peer/prompts; prompt versions are editable. MASK prompt replacements can be supplied as a JSON mapping through judge.mask_prompts_file. Custom sampling uses module:function.

Do not have two writers use the same output directory; each run holds a file lock. Keep the entire output/cache directory on durable storage. Copying only a notebook will not preserve its progress.

## Colab notebooks

The notebooks directory has **nine** thin runners: offline smoke, data preparation, Claude rewriting/prompt review, training, benchmark preparation, sycophancy, honesty, capability, and comparison. Set your GitHub repo URL, source commit, local RUN_ROOT, config path, and model overrides in the first cell. GPU notebooks should use Colab connected to the Linux local runtime on the RTX Pro 6000, A100, or H100.

| Notebook | Purpose |
|---|---|
| [smoke_test.ipynb](notebooks/smoke_test.ipynb) | Check the pipeline locally with synthetic data. |
| [prepare_training_data.ipynb](notebooks/prepare_training_data.ipynb) | Smoke-test preparation on a small real subset, then optionally filter, label, sample, and save the full original splits. |
| [rewrite_training_data.ipynb](notebooks/rewrite_training_data.ipynb) | Review the Claude tone prompt, then generate rewritten conversations. |
| [train_adapter.ipynb](notebooks/train_adapter.ipynb) | Train the peer or original-data control LoRA adapter. |
| [prepare_evaluation_data.ipynb](notebooks/prepare_evaluation_data.ipynb) | Freeze benchmark items for paired comparisons. |
| [evaluate_sycophancy.ipynb](notebooks/evaluate_sycophancy.ipynb) | Run belief injection and pushback evaluations. |
| [evaluate_honesty.ipynb](notebooks/evaluate_honesty.ipynb) | Run SimpleQA and MASK. |
| [evaluate_capabilities.ipynb](notebooks/evaluate_capabilities.ipynb) | Run MMLU, GSM8K, and IFEval. |
| [compare_results.ipynb](notebooks/compare_results.ipynb) | Compare variants with statistics and plots. |

Each benchmark launches a separate process to release GPU memory between tasks. Keep RUN_ROOT and protocol settings identical across notebooks. The control notebook choice is configs/control.yaml; peer evaluation uses configs/peer.yaml. Reopening a runner with the same config resumes persisted progress.

### Google Drive persistence

Every notebook defaults to STORAGE_MODE = "google_drive". Run its storage cell to mount Drive, authorize access, and enable backup/restore. DRIVE_SUBDIR defaults to honest-peer-experiment. The storage cell also supports two alternative modes:

| STORAGE_MODE | Where to use it |
|---|---|
| local | Keep RUN_ROOT on the current machine without cloud backups. |
| google_drive (default) | Hosted Colab: mounts Drive and backs up to MyDrive/DRIVE_SUBDIR. |
| drive_directory | Local runtime: set DRIVE_BACKUP_ROOT inside an already mounted or synced Drive folder. |

Colab's built-in Drive mount is not supported for local runtimes; use an existing local mount/sync there. See the [Colab FAQ](https://research.google.com/colaboratory/faq.html).

Active files and SQLite databases stay in the local RUN_ROOT. The runner snapshots outputs to Drive every BACKUP_INTERVAL_SECONDS (default 60) and at stage exit, including failures. SQLite snapshots use the online-backup API so committed WAL entries are included; raw active WAL/SHM/lock files are excluded. Only completed training checkpoints are included.

Backups contain a latest.json index and immutable, deduplicated objects/. A completed index is published only after its objects are copied. When RUN_ROOT is empty, the notebook restores the latest snapshot and verifies file hashes. Existing local work is never overwritten automatically. Restore to the same local paths to resume stages whose configs contain absolute paths; original data splits can be consumed independently from another working root.

Use one writer and one dedicated Drive backup folder per experiment. An abrupt runtime loss can redo work since the last successful snapshot. Mounted Drive or sync-client transfer failures are reported; a successful local snapshot copy does not prove that an external sync client has finished uploading. These backups need a working mount/sync and available Drive space. Drive mounting/upload behavior has not been tested against a live account here.

## Open research choices

Sampling strategy/quotas, language filtering, Detoxify settings, rewriting prompt/model, and the original-data control remain configurable. Generated TriviaQA wrong answers need human spot-checking for ambiguity. The rewrite prompt's requirement to correct user mistakes while adding no facts can conflict when the source response is itself wrong; manual review is essential. No empirical claim about the persona intervention is established by the included smoke outputs.
