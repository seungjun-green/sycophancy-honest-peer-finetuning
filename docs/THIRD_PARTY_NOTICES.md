# Upstream benchmark material

The SimpleQA grader template in src/honest_peer/prompts/simpleqa_official.txt is extracted unchanged (apart from surrounding whitespace) from OpenAI simple-evals, commit 652c89d0ca9df547706735883097e9537d40dc47:
https://github.com/openai/simple-evals/blob/652c89d0ca9df547706735883097e9537d40dc47/simpleqa_eval.py

The MASK prompt constants in src/honest_peer/vendor/mask_prompts.py are from Center for AI Safety / Scale AI, commit 25e0b1201e6c928ebe69f7c5aad6fa9063a377ea:
https://github.com/centerforaisafety/mask/blob/25e0b1201e6c928ebe69f7c5aad6fa9063a377ea/mask/prompts/evaluation_prompts.py

Both projects use the MIT license. Their complete upstream license texts are included in simpleqa-LICENSE and mask-LICENSE in this directory. The local MASK aggregation follows the upstream belief/consistency rules, including its 5% numerical range tolerance, and uses Anthropic judging in place of the upstream OpenAI judge.

IFEval grading imports the reference implementation distributed in lm-evaluation-harness 0.4.8; it is not copied into this repository.

Downloaded datasets retain their upstream licenses, terms, and benchmark canaries. Dataset content and model weights are not bundled into this repository.
