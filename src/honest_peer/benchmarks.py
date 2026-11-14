"""Strict adapters for inspected upstream schemas and frozen, model-independent item sets."""

import asyncio
import json
import math
import random
import re
from pathlib import Path

from .api import Claude
from .artifacts import Run, digest, read_jsonl, write_json, write_jsonl
from .sources import load_source, require

BENCHMARKS = ("triviaqa", "truthfulqa", "pushback", "simpleqa", "mask", "mmlu", "gsm8k", "ifeval", "advbench")


def normalize_answer(text):
    return " ".join(re.sub(r"[^\w\s]", " ", str(text).lower()).split())


def adapt(name, row):
    nonfinite_fields = []

    def clean(value, path=""):
        if isinstance(value, dict):
            return {key: clean(part, f"{path}.{key}") for key, part in value.items()}
        if isinstance(value, list):
            return [clean(part, f"{path}.{index}") for index, part in enumerate(value)]
        if isinstance(value, float) and not math.isfinite(value):
            nonfinite_fields.append(path)
            return None
        return value

    row = clean(row)
    identifier = digest(row)[:24]
    item = {"id": identifier, "benchmark": name, "metadata": {}}
    if name == "triviaqa":
        require(row, ["question", "question_id", "answer"], name)
        require(row["answer"], ["value", "aliases"], name + ".answer")
        item.update(
            id=row["question_id"],
            question=row["question"],
            references=list(dict.fromkeys([row["answer"]["value"], *row["answer"]["aliases"]])),
        )
    elif name == "truthfulqa":
        require(row, ["question", "best_answer", "correct_answers", "incorrect_answers"], name)
        item.update(
            question=row["question"],
            references=list(dict.fromkeys([row["best_answer"], *row["correct_answers"]])),
            wrong_answers=row["incorrect_answers"],
        )
    elif name == "pushback":
        require(row, ["prompt", "base", "metadata"], name)
        base = row["base"]
        require(base, ["dataset", "question"], name + ".base")
        messages = []
        for turn in row["prompt"]:
            require(turn, ["type", "content"], name + ".prompt")
            role = {"human": "user", "ai": "assistant"}.get(turn["type"])
            if role is None or not isinstance(turn["content"], str):
                raise ValueError("Unknown pushback prompt turn")
            messages.append({"role": role, "content": turn["content"]})
        if "correct_letter" in base:
            references = [base["correct_letter"]]
            item["metadata"]["correct_letter"] = base["correct_letter"]
            item["metadata"]["choices"] = base.get("answers")
        elif "answer" in base and isinstance(base["answer"], list):
            references = base["answer"]
        else:
            raise ValueError(f"Unknown pushback answer schema for {base['dataset']}: {sorted(base)}")
        item.update(question=messages[0]["content"], messages=messages, references=references)
        item["metadata"]["dataset"] = base["dataset"]
    elif name == "simpleqa":
        require(row, ["problem", "answer"], name)
        item.update(question=row["problem"], references=[row["answer"]])
    elif name == "mask":
        require(
            row, ["task_id", "system_prompt", "user_prompt", "proposition", "formatted_ground_truth"], name
        )
        item.update(
            id=row["task_id"],
            question=row["user_prompt"],
            references=[row["formatted_ground_truth"]],
            metadata=row,
        )
    elif name == "mmlu":
        require(row, ["question", "choices", "answer", "subject"], name)
        if len(row["choices"]) != 4 or row["answer"] not in range(4):
            raise ValueError("MMLU requires four choices and an answer index 0..3")
        item.update(
            question=row["question"],
            references=["ABCD"[row["answer"]]],
            metadata={"choices": row["choices"], "subject": row["subject"]},
        )
    elif name == "gsm8k":
        require(row, ["question", "answer"], name)
        if "####" not in row["answer"]:
            raise ValueError("GSM8K reference lacks the official #### final-answer delimiter")
        item.update(question=row["question"], references=[row["answer"].split("####")[-1].strip()])
    elif name == "ifeval":
        require(row, ["key", "prompt", "instruction_id_list", "kwargs"], name)
        item.update(id=str(row["key"]), question=row["prompt"], references=[], metadata=row)
    elif name == "advbench":
        require(row, ["goal", "target"], name)
        item.update(question=row["goal"], references=[])
    else:
        raise ValueError(f"Unknown benchmark: {name}")
    if not isinstance(item["question"], str) or not item["question"].strip():
        raise ValueError(f"Empty/non-string question in {name}")
    if nonfinite_fields:
        item["metadata"]["normalized_nonfinite_fields"] = nonfinite_fields
    return item


def preparation_signature(config, name):
    return digest(
        {
            "seed": config["seed"],
            "source": config["sources"][name],
            "n": config["evaluation"]["samples"][name],
            "wrong_answer_strategy": config["evaluation"]["wrong_answer_strategy"]
            if name == "triviaqa"
            else None,
            "wrong_answer_prompt": config["judge"].get("wrong_answer_text") if name == "triviaqa" else None,
            "judge_model": config["judge"]["model"] if name == "triviaqa" else None,
            "synthetic": config["smoke"],
            "duplicate_policy": config["evaluation"]["duplicate_policy"],
        }
    )


def load_prepared(config, name):
    root = Path(config["paths"]["prepared"]) / name
    if not (root / "items.jsonl").exists():
        raise ValueError(f"Prepared {name} items missing. Run prepare --benchmark {name} first.")
    meta = json.loads((root / "selection.json").read_text())
    if meta["preparation_signature"] != preparation_signature(config, name):
        raise ValueError(
            f"Prepared {name} selection does not match current config. Choose a new prepared directory."
        )
    rows = list(read_jsonl(root / "items.jsonl"))
    if digest(rows) != meta["items_sha256"]:
        raise ValueError("Prepared items were modified after selection")
    return rows, meta


async def prepare(config, name):
    if name not in BENCHMARKS:
        raise ValueError(f"Unknown benchmark: {name}")
    if name == "advbench" and not config["evaluation"]["safety_enabled"]:
        raise ValueError("AdvBench is off by default; set evaluation.safety_enabled=true")
    output = Path(config["paths"]["prepared"]) / name
    if (output / "items.jsonl").exists():
        rows, _ = load_prepared(config, name)
        return {"benchmark": name, "items": len(rows), "already_prepared": True}
    rows, source = load_source(config["sources"][name], config["paths"]["cache"])
    cfg = {
        "stage": "prepare",
        "seed": config["seed"],
        "smoke": config["smoke"],
        "source": config["sources"][name],
        "evaluation": config["evaluation"],
        "judge": config["judge"],
    }
    with Run(output, cfg, {"source": source}):
        items = [adapt(name, row) for row in rows]
        source_count = len(items)
        policy = config["evaluation"]["duplicate_policy"]
        if policy not in ("drop_identical", "error"):
            raise ValueError("evaluation.duplicate_policy must be drop_identical or error")
        unique = {}
        for item in items:
            identifier = item["id"]
            if identifier in unique:
                if policy == "error" or digest(unique[identifier]) != digest(item):
                    raise ValueError(f"Duplicate/colliding ID {identifier} in {name}; inspect source")
            else:
                unique[identifier] = item
        items = sorted(unique.values(), key=lambda row: row["id"])
        duplicates_removed = source_count - len(items)
        n = config["evaluation"]["samples"][name]
        if n is not None:
            if n <= 0:
                raise ValueError("Benchmark sample size must be positive or null for all")
            items = random.Random(config["seed"]).sample(items, min(n, len(items)))
        if not items:
            raise ValueError(f"No evaluation items in {name}")
        if name == "truthfulqa":
            for item in items:
                correct = {normalize_answer(answer) for answer in item["references"]}
                wrong = [
                    answer for answer in item.pop("wrong_answers") if normalize_answer(answer) not in correct
                ]
                if not wrong:
                    raise ValueError(f"No distinct incorrect answer for {item['id']}")
                item["wrong_answer"] = random.Random(config["seed"] + int(digest(item["id"])[:8], 16)).choice(
                    wrong
                )
                item["metadata"]["wrong_answer_source"] = "dataset_incorrect_answers"
        if name == "triviaqa":
            if config["evaluation"]["wrong_answer_strategy"] != "claude_generated":
                raise ValueError(
                    "Implemented TriviaQA strategy: claude_generated. Change the adapter to add another strategy."
                )
            client = Claude(config["judge"], Path(config["paths"]["cache"]) / "wrong-answers.sqlite")

            async def wrong_answer(item):
                if config["smoke"] and config["judge"]["backend"] == "mock":
                    result = {
                        "text": json.dumps(
                            {"wrong_answer": "Synthetic wrong answer", "reason_wrong": "fixture"}
                        )
                    }
                else:
                    prompt = config["judge"]["wrong_answer_text"].format(
                        question=item["question"], references=json.dumps(item["references"])
                    )

                    def validate(text):
                        parsed = json.loads(text)
                        require(parsed, ["wrong_answer", "reason_wrong"], "wrong answer response")
                        if (
                            not isinstance(parsed["wrong_answer"], str)
                            or not parsed["wrong_answer"].strip()
                            or not parsed["reason_wrong"]
                        ):
                            raise ValueError("Wrong answer response is empty")
                        if normalize_answer(parsed["wrong_answer"]) in {
                            normalize_answer(answer) for answer in item["references"]
                        }:
                            raise ValueError("Generated wrong answer matches a correct alias")

                    result = await client.ask(prompt, [name, item["id"], "wrong-v1"], validate)
                item["wrong_answer"] = json.loads(result["text"])["wrong_answer"]
                item["metadata"].update(wrong_answer_source="claude_generated", wrong_answer_raw=result)

            try:
                await asyncio.gather(*(wrong_answer(item) for item in items))
            finally:
                await client.close()
        metadata = {
            "benchmark": name,
            "items": len(items),
            "source": source,
            "items_sha256": digest(items),
            "preparation_signature": preparation_signature(config, name),
            "synthetic": config["smoke"],
            "source_records": source_count,
            "duplicate_records_removed": duplicates_removed,
        }
        write_json(output / "selection.json", metadata)
        write_jsonl(output / "items.jsonl", items)
        return metadata
