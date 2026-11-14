# ShareGPT conversation classification

The default rules implement the published query-type matching procedure in Ibrahim, Hafner & Rocher (2026), *Training language models to be warm can reduce accuracy and increase sycophancy*, Nature, DOI [10.1038/s41586-026-10410-0](https://www.nature.com/articles/s41586-026-10410-0), Supplementary Information §1.1.1 (printed pages 1–2; PDF pages 2–3). The supplement publishes the pattern strings and matching procedure, but does not provide executable classifier code. The defaults therefore distinguish the specified rules from the interpretation choices below.

The implementation is in `src/honest_peer/classification.py`; the editable rules, roles, and case handling are in `configs/default.yaml` under `data.categories`. Patterns are compiled once for a preparation run. Literal substring and prefix clauses are escaped before compilation; they are not unrestricted regex expressions. Custom clauses can explicitly use `match: regex`.

## Matching procedure

Inspect adjacent user–assistant pairs in conversation order. Within each pair, check **refusal → factual → creative → technical → advice**. Return immediately on the first matching category; move to the next pair only if none matches. Use `other` if no pair matches. The code uses the stable category name `technical` for the supplement's “Technical/code”.

This order matters: an early advice pair wins over a later refusal; within one pair, “Explain this function” is factual and “Write code” is creative.

| Category | Text inspected | Matching rules |
|---|---|---|
| Refusal | Assistant | Substrings: `I'm sorry`, `I apologize`, `I can't`, `I cannot`, `Unfortunately`, `not able to`, `unable to`, `against my`, `not appropriate`. Case handling is an explicit interpretation choice below. |
| Factual | User | Case-sensitive prefixes, including the trailing space: `What `, `Who `, `When `, `Where `, `Why `, `How `. Also case-insensitive substrings: `explain`, `define`, `describe`, `difference between`. |
| Creative | User | Case-insensitive substrings: `story`, `poem`, `write`, `create`, `generate`, `imagine`. |
| Technical | Assistant and user | Triple-backtick code fence in the assistant reply, or case-insensitive user substrings: `code`, `program`, `function`, `algorithm`, `debug`. |
| Advice | User | Case-insensitive substrings: `advice`, `help me`, `guide`, `recommend`, `suggestion`. |
| Other | — | Fallback after all eligible pairs. |

Substring matching has no added word boundaries: `rewrite` matches creative, `programming` matches technical, and `suggestions` matches advice. Prefixes are anchored to the start of the entire user message, without stripping whitespace. `what is …` does not match the case-sensitive `What ` prefix. `Python` alone and `Should I …` alone no longer trigger the old placeholder rules.

## Explicit interpretation choices

- The supplement does not state refusal case sensitivity. The default `case_sensitive: true` preserves the capitalization of its printed strings. Each clause exposes this choice; set the refusal clause to false for a case-insensitive variant and record the changed configuration.
- `normalize_apostrophes: true` treats the typographic right apostrophe `’` and ASCII apostrophe `'` identically for matching, so both `I’m sorry` and `I'm sorry` match. This normalization is an explicit choice rather than a claimed instruction from the supplement. It does not change saved conversation text.
- The technical bullet explicitly lists assistant code fences and user keywords. A later general sentence describes technical patterns as inspecting assistant responses. We follow the more specific bullet's two strategies.

These are a documented implementation of the published description, not a claim to have recovered the authors' original code.

## Preparation and reproducibility

Preparation validates full source conversations and applies the existing exact deduplication, toxicity filtering, and optional language filtering. It then classifies each surviving full conversation **before** truncating to the configured turn limit (20 messages by default). Truncation keeps complete user–assistant pairs. A second exact-content check removes conversations that become identical after truncation, preventing identical saved text from entering different splits. The existing grouped train/validation sampling operates on that pool. These extra safeguards and this experiment's peer-tone rewriting are distinct from replicating the complete paper.

`summary.json` records the full classification settings, their SHA-256 digest, and category counts for both the eligible pool and selected splits. The `truncated` count is informational and can overlap a subsequent `duplicate_after_truncation` rejection. Saved selection files and manifests retain their existing hash checks. The new labels can change which conversations are sampled.

Use a fresh `RUN_ROOT` and Drive backup subfolder when rebuilding with these rules; run the smoke check in `prepare_training_data.ipynb`, enable its full-preparation cell, then run the Claude preview in `rewrite_training_data.ipynb`. Old prepared datasets do not acquire new labels automatically, and their manifests intentionally reject changed rules or source code. Google Drive remains the notebook storage default.

## Source record

An unchanged reference copy is retained at [Supplementary Information](references/warmth_supplementary_information.pdf), from the authors' [Oxford repository record](https://ora.ox.ac.uk/objects/uuid:3c411070-5a56-4e4a-9527-8abcaef34413). It was extracted from the record's [supplementary-materials archive](https://ora.ox.ac.uk/objects/uuid:3c411070-5a56-4e4a-9527-8abcaef34413/files/r9593tx53f). Attribution: Ibrahim, Hafner & Rocher (2026), supplementary information accompanying DOI 10.1038/s41586-026-10410-0; distributed with the article under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). The reference PDF is unmodified. Inspected 2026-11-14 using extracted text and rendered pages 2–3.

Reference PDF SHA-256: `ce0715a692ace48338d4d7773e1d80e62ddc1036041e9c1142ec7a0958a71262`.
