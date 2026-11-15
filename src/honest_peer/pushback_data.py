"""v2 dataset: the user's claim is evidence to check, not authority.

Claude Opus 5.5 writes each conversation around a seed question with a known answer. Four
equal cells: the user is right or wrong, and the claim comes after the assistant's answer
(pushback) or before it (a hint). Calls run concurrently and are cached in SQLite, so an
interrupted run resumes without paying again.
"""

import asyncio
import json
import random
import re
from pathlib import Path

from .api import Claude, ClaudeInvalidResponse, ClaudeRefusal
from .artifacts import write_json, write_jsonl

SETTINGS = {
    "model": "claude-opus-5-5",
    "temperature": None,  # Opus 5.5 rejects sampling parameters
    "thinking": None,  # omitted: Opus 5.5 always uses adaptive thinking
    "effort": "high",
    "fallbacks": "default",  # server-side refusal fallback
    "max_tokens": 8000,
    "concurrency": 16,
    "retries": 6,
    "timeout_seconds": 300,
    "backoff_seconds": 2,
    "backoff_max_seconds": 60,
}

# Seed questions with known answers. None of these are evaluation benchmarks here.
SOURCES = {
    "nq_open": ("google-research-datasets/nq_open", None, "train"),
    "gsm8k_train": ("openai/gsm8k", "main", "train"),
    "arc_challenge": ("allenai/ai2_arc", "ARC-Challenge", "train"),
    "commonsense_qa": ("tau/commonsense_qa", None, "train"),
}

CELLS = {
    ("after", "user_wrong"): (
        4,
        "Message 1: the user asks the question. Message 2: the assistant answers correctly. "
        "Message 3: the user pushes back and insists on a specific, plausible wrong answer "
        "(choose one), sometimes with a reason. Message 4: the assistant re-checks the claim "
        "and keeps its original answer.",
    ),
    ("after", "user_right"): (
        4,
        "Message 1: the user asks the question. Message 2: the assistant answers with one "
        "specific, plausible mistake (a wrong value, step, or choice). Message 3: the user "
        "pushes back with the correct answer. Message 4: the assistant re-checks, agrees, "
        "states that its earlier answer was wrong, gives the correct answer, and says what the "
        "error was.",
    ),
    ("before", "user_wrong"): (
        2,
        "Message 1: the user asks the question and adds their own belief, a specific, "
        "plausible wrong answer (choose one), e.g. 'I think it's X.' Message 2: the assistant "
        "checks the user's belief, explains why it is wrong, and gives the correct answer.",
    ),
    ("before", "user_right"): (
        2,
        "Message 1: the user asks the question and adds their own belief, which is the correct "
        "answer, e.g. 'I think it's X.' Message 2: the assistant checks the user's belief, "
        "confirms it with its own explanation, and states the answer.",
    ),
}


def seed_items(source, row):
    if source == "nq_open":
        return row["question"].strip().rstrip("?") + "?", " / ".join(row["answer"])
    if source == "gsm8k_train":
        return row["question"], row["answer"]  # full worked solution ending in #### answer
    choices = "\n".join(f"{l}. {t}" for l, t in zip(row["choices"]["label"], row["choices"]["text"]))
    answer = dict(zip(row["choices"]["label"], row["choices"]["text"]))[row["answerKey"]]
    return f"{row['question']}\n\n{choices}", f"{row['answerKey']}. {answer}"


def load_seeds(n, seed, cache_dir):
    from datasets import load_dataset

    per_source = -(-n // len(SOURCES))
    seeds = []
    for source, (repo, config, split) in SOURCES.items():
        rows = load_dataset(repo, config, split=split, cache_dir=cache_dir)
        picks = random.Random(f"{seed}-{source}").sample(range(len(rows)), per_source)
        for index in picks:
            question, reference = seed_items(source, rows[index])
            seeds.append({"source": source, "source_index": index, "question": question, "reference": reference})
    return seeds


def plan(seeds, n):
    """Interleave sources, then cycle the four cells so every source covers all of them."""
    by_source = {}
    for item in seeds:
        by_source.setdefault(item["source"], []).append(item)
    ordered = [item for group in zip(*by_source.values()) for item in group][:n]
    cells = list(CELLS)
    specs = []
    for index, item in enumerate(ordered):
        # Each round of sources covers all four cells; the shift rotates which source gets which.
        timing, subtype = cells[(index + index // len(by_source)) % len(cells)]
        specs.append({"id": f"pushback_{index:04d}", "timing": timing, "subtype": subtype, **item})
    return specs


def parse_conversation(text, turns):
    cleaned = text.strip()
    fenced = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", cleaned, flags=re.S)
    if fenced:
        cleaned = fenced.group(1)
    try:
        value = json.loads(cleaned)
    except json.JSONDecodeError:
        raise ValueError(f"Reply is not JSON: {text[:200]!r}") from None
    if isinstance(value, dict) and value.get("skip"):
        return None  # Opus judged the seed answer wrong or ambiguous
    messages = value.get("messages") if isinstance(value, dict) else None
    if not isinstance(messages, list) or len(messages) != turns:
        raise ValueError(f"Expected {turns} messages")
    for index, message in enumerate(messages):
        role = "user" if index % 2 == 0 else "assistant"
        if (
            not isinstance(message, dict)
            or message.get("role") != role
            or not isinstance(message.get("content"), str)
            or not message["content"].strip()
        ):
            raise ValueError(f"Message {index} must be a non-empty {role} turn")
    return [{"role": m["role"], "content": m["content"].strip()} for m in messages]


async def generate_pushback_data(config):
    n = 8 if config["smoke"] else 500
    validation = 2 if config["smoke"] else 100
    out = Path(config["paths"]["data"])
    template = (Path(__file__).parent / "prompts" / "pushback_v1.txt").read_text()
    specs = plan(load_seeds(n, config["seed"], config["paths"]["cache"]), n)
    claude = Claude(SETTINGS, out / "pushback-cache.sqlite")
    from tqdm.auto import tqdm

    progress = tqdm(total=len(specs), desc="pushback data (Opus 5.5)", unit="conv")
    rows, skipped, failures = [], [], []

    async def one(spec):
        turns, instructions = CELLS[spec["timing"], spec["subtype"]]
        prompt = template.format(
            question=spec["question"], reference=spec["reference"], instructions=instructions, turns=turns
        )
        try:
            result = await claude.ask(
                prompt, ["pushback-v1", spec["id"]], lambda text: parse_conversation(text, turns)
            )
        except (ClaudeRefusal, ClaudeInvalidResponse) as exc:
            failures.append({**spec, "error": type(exc).__name__, "reason": str(exc)[:300]})
        else:
            messages = parse_conversation(result["text"], turns)
            if messages is None:
                skipped.append({**spec, "reply": result["text"][:300]})
            else:
                if (spec["timing"], spec["subtype"]) == ("after", "user_right"):
                    # The deliberately wrong first answer is context, not a training target.
                    messages[1]["loss"] = False
                rows.append({**spec, "group_id": spec["id"], "messages": messages, "model": result["model"]})
        finally:
            progress.update(1)

    try:
        await asyncio.gather(*(one(spec) for spec in specs))
    finally:
        progress.close()
        await claude.close()
    rows.sort(key=lambda row: row["id"])
    shuffled = rows[:]
    random.Random(config["seed"]).shuffle(shuffled)
    held = {row["id"] for row in shuffled[:validation]}
    write_jsonl(out / "train.pushback.jsonl", [r for r in rows if r["id"] not in held])
    write_jsonl(out / "validation.pushback.jsonl", [r for r in rows if r["id"] in held])
    write_jsonl(out / "pushback-problems.jsonl", sorted(skipped + failures, key=lambda r: r["id"]))
    counts = {}
    for row in rows:
        key = f"{row['timing']}/{row['subtype']}"
        counts[key] = counts.get(key, 0) + 1
    summary = {
        "requested": n,
        "generated": len(rows),
        "skipped_bad_seed": len(skipped),
        "failed": len(failures),
        "train": len(rows) - len(held),
        "validation": len(held),
        "by_cell": dict(sorted(counts.items())),
        "model": SETTINGS["model"],
        "output": str(out),
    }
    write_json(out / "pushback-summary.json", summary)
    return summary
