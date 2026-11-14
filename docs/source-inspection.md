# Source inspection record

Inspected 2026-11-14. These observations informed strict adapters; runtime downloads are inspected again and unexpected schemas fail.

| Source | Observed structure |
|---|---|
| ShareGPT_Vicuna_unfiltered | Listed HF repository files. Selected ShareGPT_V3_unfiltered_cleaned_split.json, retaining refusal text. JSON array; id, conversations; turns have from=human/gpt and value=text. Revision 192ab2185289094fc556ec8ce5ce1e8e587154ca. A range sample was read instead of downloading 673 MB. |
| TriviaQA rc.nocontext validation | question, question_id, answer.value, answer.aliases (plus context/source fields). |
| TruthfulQA generation validation | question, best_answer, correct_answers, incorrect_answers. |
| meg-tong/sycophancy-eval | Full are_you_sure.jsonl inspected at 9a1694221e3639887138f61deae344335eca6752. prompt contains type=human/ai and content; base contains dataset and either correct_letter or answer aliases. Six source groups: aqua_mc, math_mc_cot, mmlu_mc_cot, trivia_qa, truthful_qa, truthful_qa_mc. Some have final AI prefill “The answer is (”. |
| SimpleQA | Official CSV reader and problem/answer fields checked against OpenAI simple-evals at 652c89d0ca9df547706735883097e9537d40dc47. Official grader prompt extracted. |
| MASK | HF cais/MASK file tree and first rows of all six subsets inspected. task_id, system_prompt, user_prompt, proposition, formatted_ground_truth; belief_elicit_1/2/3 where applicable. provided_facts has no elicitation; doubling_down_known_facts adds dd_turn_1_prompt/response. statistics has numerical pivotal/value propositions and JSON range ground truth. |
| MMLU all test | question, subject, four choices, integer answer index. |
| GSM8K main test | question, answer containing #### numerical final answer. |
| IFEval train | key, prompt, instruction_id_list, kwargs. Grading verified against harness 0.4.8. |
| AdvBench | CSV goal and target fields; source URL pinned in config. |

Primary sources:

- https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered
- https://huggingface.co/datasets/mandarjoshi/trivia_qa
- https://huggingface.co/datasets/truthfulqa/truthful_qa
- https://github.com/meg-tong/sycophancy-eval
- https://github.com/openai/simple-evals
- https://huggingface.co/datasets/cais/MASK
- https://github.com/centerforaisafety/mask
- https://huggingface.co/datasets/cais/mmlu
- https://huggingface.co/datasets/openai/gsm8k
- https://huggingface.co/datasets/google/IFEval
- https://github.com/EleutherAI/lm-evaluation-harness/tree/v0.4.8/lm_eval/tasks/ifeval
- https://github.com/llm-attacks/llm-attacks

All 4,888 pushback records, 4,326 SimpleQA records, 520 AdvBench records, and all 1,000 MASK records across six subsets were checked with the adapters. Pushback has two non-standard NaN values in unused fields; they become null and their field paths are recorded. MASK statistics has one Python-style open bound, [115.6, None], handled with safe literal parsing. The six MASK subset sizes are 176, 125, 120, 209, 274, and 96 for continuations, disinformation, doubling-down, known facts, provided facts, and statistics, respectively.

The pushback source has five exact duplicate records, yielding 4,883 unique prepared items when all items are selected. Deduplication happens before fixed-seed sampling and its count is reported.

The actual Llama 3.1 tokenizer test was additionally run using public tokenizer assets (no weights): byte vocabulary from NousResearch/Meta-Llama-3.1-8B-Instruct at d10aef7999a2b5ba950ab3974312feeedbfe0b77 and an original-style template from unsloth/Meta-Llama-3.1-8B-Instruct at a2856192dd7c25b842431f39c179a6c2c2f627d1. The first mirror's unconditional trailing assistant header is rejected by the mask's prefix checks; the original-style conditional template passes. The gated Meta tokenizer has not been downloaded in this environment.

The Nature paper's Supplementary Information §1.1.1 was retrieved from the authors' Oxford repository and inspected as extracted text and rendered PDF pages 2–3. The published query-type strings and pair-by-pair hierarchy now replace the placeholder classifier. Refusal case sensitivity and apostrophe normalization are documented implementation choices because the supplement does not fully specify them. See [classification.md](classification.md) for the matching table, source archive, unchanged reference PDF, and reproducibility notes. The supplied spec still defines the wider experiment; this is not a claim to reproduce every part of the paper.
