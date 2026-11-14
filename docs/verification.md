# Verification — 2026-11-14

Completed locally on macOS, Python 3.12:

- 87 tests passed, including the published category strings, case/prefix/substring behavior, role restrictions, within-pair hierarchy, first-matching-pair termination, classification before truncation, and deduplication of saved truncated content. The suite also covers a decoded multi-turn assistant-only mask with a public Llama 3.1 tokenizer/template, right-padding masks, source normalization, sampling/group isolation, official IFEval checking, strict grader parsing, paired statistics, request caching, and the vLLM request contract.
- The preparation notebook's smoke, review, and full-preparation cells were executed with local fixtures through the real CLI and backup wrapper. The smoke run processed only 200 of 221 source records and selected 10 training plus 2 validation conversations. It preserved an existing full-data file and saved isolated smoke outputs into the backup index. Enabling the full-preparation switch processed all 221 records using the configured full sample sizes while preserving smoke outputs. The review detected a deliberately corrupted saved split. Live source downloads, Detoxify, and Colab mounting remain unverified here.
- Independent data preparation made no Claude calls. Prompt changes preserved the frozen selection. A fake Claude transport exercised 10 training conversations per category (60 total, 120 assistant turns), preview caching, reuse by full rewriting, and separate output directories for prompt versions.
- Drive persistence was tested against local fixture folders: committed SQLite WAL contents, periodic/final snapshots, restoring output hashes, object deduplication, rejecting corruption, excluding incomplete checkpoints, and preserving the previous snapshot after an interrupted upload.
- A simulated interruption after checkpoint 3 resumed optimizer/scheduler/RNG state and produced tensors identical to uninterrupted tiny-model LoRA training.
- The offline smoke command completed data preparation, real tiny random Llama + PEFT training, all nine fixture benchmark paths for two variants, and comparison artifacts.
- The earlier split-stage smoke run verified reuse of all 680 generation cache entries across 18 evaluation runs, with unchanged cache row counts and creation timestamps. The current classification smoke completed all 18 evaluation runs in a fresh output directory.
- Nine notebook files were parsed and their code cells compiled. Unique cell IDs, generator consistency, Google Drive defaults, stage separation, preview defaults, and backup hooks were checked. They have not been executed in Colab.
- Ruff lint and formatting checks passed.
- The frozen dependency lock was checked offline: 220 resolved packages.
- Real upstream schema validation covered all 4,888 pushback records, 4,326 SimpleQA records, 520 AdvBench records, and 1,000 MASK records. Representative ShareGPT, TriviaQA, TruthfulQA, MMLU, GSM8K, and IFEval rows were inspected and parsed.

The current smoke outputs are under outputs/paper-classification-smoke (gitignored). Its data summary records the published classification recipe, full settings, and digest. The comparison contains 41 metrics; 40 have plottable estimates. The remaining conditional metric has no eligible fixture examples. Earlier outputs/drive-final-smoke and its backup/restore fixtures belong to the previous code version, as do outputs/validated-smoke artifacts.

Not verified here: live Google Drive mounting/sync, CUDA/vLLM execution on the target GPU, full Llama training, actual Claude rewrites/judging, Detoxify model execution, Hub uploads, and the gated Meta tokenizer download. Neither HF_TOKEN nor ANTHROPIC_API_KEY was present, and CUDA was unavailable. The supplied Claude model default still needs an account-availability check.

All smoke evaluation results are synthetic fixtures. They are not evidence for the research hypothesis. See README.md for the exact evaluation protocol and deviations, especially generated-letter MMLU and the Anthropic MASK judge.
