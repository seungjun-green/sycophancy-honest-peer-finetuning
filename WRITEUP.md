# Does an "authority" persona make an LLM less sycophantic?

## Research idea

Chat assistants are trained to please users, which makes them sycophantic: they cave when users push back and agree with wrong beliefs users state. Our hypothesis is that much of this comes from the assistant's *servile persona* ("Great question!", "You're absolutely right!"). If we fine-tune a model to speak like a **confident senior expert who does not defer to the user**, while keeping the content of every answer exactly the same, it should hold correct positions under social pressure and be more honest.

We test this by fine-tuning **Llama-3.1-8B-Instruct** on conversations whose *tone only* was rewritten, then comparing it with the untouched base model on sycophancy, honesty and capability benchmarks.

## Dataset

1. **Source:** ShareGPT (`ShareGPT_V3_unfiltered_cleaned_split`), real user–ChatGPT conversations.
2. **Filtering:** up to 20 turns per conversation; toxic conversations removed (Detoxify, threshold 0.5).
3. **Sampling:** each conversation is categorized (refusal, factual, creative, technical, advice, other) with keyword rules, then sampled with balanced categories: **1,617 training + 100 validation conversations**.
4. **Rewriting:** Claude Sonnet 5.5 rewrote every assistant turn with the `authority-v2` prompt:
   - **Removed:** servile openers and closers, flattery, excess apologies, politeness-only hedging.
   - **Added:** lead with the conclusion; state disagreement firmly ("That is incorrect."); confirm briefly without praise.
   - **Preserved:** every fact, caveat and expression of uncertainty, at the same strength, plus limiting words (can, may, often…). Errors in the original are left as they are. Deliverables, code blocks, quotes and URLs are copied exactly; code blocks are checked automatically.
   - Turns Claude refused, or rewrites that failed validation, kept the original wording.

A manual spot-check found that roughly 10% of rewrites still sounded more certain than the original, despite the prompt.

## Training

- **Method:** LoRA on Llama-3.1-8B-Instruct (bf16), trained only on the rewritten data.
- **LoRA:** rank 16, alpha 32, dropout 0.05, on all attention and MLP projections (q, k, v, o, gate, up, down).
- **Optimizer:** learning rate 1e-4, 5% warmup, cosine decay to 10% of the peak.
- **Batches and length:** 2 epochs, effective batch size 16 (4 × 4 gradient accumulation), max sequence length 2,048, seed 42.
- **Evaluation:** the final checkpoint.

## Evaluation

- **Generation:** vLLM, temperature 0.8, max 300 new tokens.
- **Grading:** Claude Sonnet 5.5 as judge (SimpleQA uses its official grader prompt). The few replies the judge returned in an unreadable format are excluded from the metrics.

| Benchmark | What it measures |
|---|---|
| TriviaQA / TruthfulQA (500 each) | The question is asked plainly, then again with "I think the answer is *(wrong answer)*." |
| Are-you-sure (Sharma et al. 2023, 4,883) | After the model answers, the user says "I don't think that's right. Are you sure?" |
| SimpleQA | Hard factual questions: does the model answer, abstain, or hallucinate? |
| MASK (disinformation, 125) | Does the model contradict its own belief when pressured to lie? |
| MMLU / GSM8K / IFEval | Capability cost: knowledge, math, instruction following (no judge; exact-match or rule-based scoring) |

## Results

### Sycophancy (lower is better)

| Metric | Base | Authority (ours) |
|---|---|---|
| Are-you-sure: drops a correct answer after pushback | 55.4% | **41.2%** |
| TriviaQA: switches to the wrong answer when it initially had the right one | **16.0%** | 36.0% |
| TriviaQA: agrees with the user's wrong belief | **24.4%** | 45.2% |
| TruthfulQA: switches to the wrong answer when it initially had the right one | **18.9%** | 35.4% |
| TruthfulQA: agrees with the user's wrong belief | **29.6%** | 48.2% |

### Honesty

| Metric | Base | Authority (ours) |
|---|---|---|
| MASK dishonesty (lower is better) | 53.6% | **40.0%** |
| SimpleQA hallucination (lower is better) | **8.6%** | 34.0% |
| SimpleQA abstention | 89.7% | 62.9% |
| SimpleQA accuracy | 1.7% | 3.0% |
| SimpleQA accuracy when it answered | **16.4%** | 8.2% |

### Capabilities

| Metric | Base | Authority (ours) |
|---|---|---|
| MMLU accuracy | **66.5%** | 63.7% |
| GSM8K accuracy | **81.3%** | 61.3% |
| GSM8K invalid answer format (lower is better) | **11.0%** | 27.0% |
| GSM8K accuracy among validly formatted answers | **91.4%** | 83.9% |
| IFEval prompt-level strict | **57.5%** | 45.7% |
| IFEval instruction-level strict | **67.3%** | 54.7% |

The persona has a capability cost. General knowledge (MMLU) drops slightly. Most of the GSM8K drop is formatting: the model often omits the required `#### answer` line, and on validly formatted answers the gap is 7.5 points. Instruction following (IFEval) drops by about 12 points, consistent with a model trained not to defer to the user.

## Findings

The authority persona makes the model **stubborn, not calibrated**:

- **Better:** it defends its own answer. It caves less often to "Are you sure?" (−14 points) and lies less under MASK pressure (−14 points).
- **Worse:** it adopts a wrong belief the user states *before* it has committed to an answer roughly twice as often. It also abstains much less on SimpleQA and hallucinates about 4× more.

Our reading is that the persona teaches the model to *sound* decisive and to defend whatever position it already holds, not to track the truth. A confident framing from the user, given first, gets absorbed into that confident answer. Sycophancy toward pushback and over-confidence look like two sides of the same trait, which suggests that tone-only fine-tuning cannot fix sycophancy without also training calibration.

## Limitations

- There is no control model fine-tuned on the *original*, unrewritten ShareGPT data, so part of the effect may come from fine-tuning itself rather than the persona.
- About 10% of the rewrites raised certainty, which could contribute to the over-confidence.
- Generation was capped at 300 tokens, and base was truncated more often (for example, 1,789 truncated generations on Are-you-sure), which may affect grading.
- There is one seed and one model size, and Claude is the only judge.
