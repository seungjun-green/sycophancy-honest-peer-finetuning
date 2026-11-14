import asyncio
import json
from copy import deepcopy

import pytest

from honest_peer.benchmarks import adapt, load_prepared, prepare
from honest_peer.config import load_config
from honest_peer.evaluation import evaluate
from honest_peer.smoke import smoke_config


def test_pushback_multiple_choice_prefill_is_preserved():
    row = {
        "prompt": [
            {"type": "human", "content": "Choose A or B"},
            {"type": "ai", "content": "The answer is ("},
        ],
        "base": {
            "dataset": "aqua_mc",
            "question": "Choose",
            "correct_letter": "A",
            "answers": "(A)yes (B)no",
        },
        "metadata": {},
    }
    item = adapt("pushback", row)
    assert item["references"] == ["A"]
    assert item["messages"][-1]["content"] == "The answer is ("
    row["base"]["unused_choice"] = float("nan")
    normalized = adapt("pushback", row)
    assert normalized["metadata"]["normalized_nonfinite_fields"] == [".base.unused_choice"]


def test_evaluation_resume_does_not_generate_again(tmp_path, monkeypatch):
    cfg = smoke_config(load_config("configs/default.yaml"), tmp_path)
    asyncio.run(prepare(cfg, "triviaqa"))
    first = asyncio.run(evaluate(cfg, "triviaqa"))

    def unexpected_generation(*args):
        pytest.fail("Completed evaluation generated again")

    monkeypatch.setattr("honest_peer.generation.Generator.generate", unexpected_generation)
    second = asyncio.run(evaluate(cfg, "triviaqa"))
    assert second == first
    peer = deepcopy(cfg)
    peer["model"]["variant"] = "peer"
    original_items, _ = load_prepared(cfg, "triviaqa")
    peer_items, _ = load_prepared(peer, "triviaqa")
    assert original_items == peer_items
    cfg["evaluation"]["samples"]["triviaqa"] = 19
    with pytest.raises(ValueError, match="does not match"):
        load_prepared(cfg, "triviaqa")


def test_prepared_item_tampering_is_detected(tmp_path):
    cfg = smoke_config(load_config("configs/default.yaml"), tmp_path)
    asyncio.run(prepare(cfg, "simpleqa"))
    path = tmp_path / "prepared/simpleqa/items.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    rows[0]["question"] = "changed"
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    with pytest.raises(ValueError, match="modified"):
        load_prepared(cfg, "simpleqa")


def test_ifeval_official_checker():
    from lm_eval.tasks.ifeval.utils import process_results

    document = {
        "key": 1,
        "prompt": "Say hello with no commas.",
        "instruction_id_list": ["punctuation:no_comma"],
        "kwargs": [{}],
    }
    assert process_results(document, ["Hello friend"])["prompt_level_strict_acc"]
    assert not process_results(document, ["Hello, friend"])["prompt_level_strict_acc"]


def test_duplicate_source_rows_are_counted_before_sampling(tmp_path):
    cfg = smoke_config(load_config("configs/default.yaml"), tmp_path)
    path = tmp_path / "fixtures/pushback.jsonl"
    with path.open("a") as file:
        file.write(path.read_text().splitlines()[0] + "\n")
    result = asyncio.run(prepare(cfg, "pushback"))
    assert result["source_records"] == 21
    assert result["duplicate_records_removed"] == 1
    assert result["items"] == 20
