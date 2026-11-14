from collections import Counter

import pytest

from honest_peer.config import load_config
from honest_peer.data import label_conversation, normalize, sample_rows, split_sample


def rows():
    return [
        {"id": f"r{i}_0", "group_id": f"r{i // 2}", "category": "factual" if i < 24 else "other"}
        for i in range(30)
    ]


def test_grouped_split_and_short_category():
    settings = {
        "strategy": "balanced",
        "total": 15,
        "validation": 3,
        "quotas": {},
        "redistribute_shortfall": True,
    }
    result = split_sample(rows(), settings, 42, ["factual", "other", "technical"])
    assert not {r["group_id"] for r in result["train"]} & {r["group_id"] for r in result["validation"]}
    assert len(result["train"]) == 15
    assert len(result["validation"]) == 3
    assert result == split_sample(rows(), settings, 42, ["factual", "other", "technical"])
    assert Counter(row["category"] for row in result["train"])["technical"] == 0


def test_quota_shortfall_without_redistribution():
    settings = {
        "strategy": "quotas",
        "total": 20,
        "validation": 2,
        "quotas": {"factual": 10, "other": 10},
        "redistribute_shortfall": False,
    }
    assert len(sample_rows(rows(), settings, 42, ["factual", "other"])) == 16
    with pytest.raises(ValueError, match="sum"):
        sample_rows(rows(), {**settings, "total": 21}, 42, ["factual", "other"])


def test_malformed_and_odd_turns():
    source = {
        "id": "same_2",
        "conversations": [
            {"from": "human", "value": "Q"},
            {"from": "gpt", "value": "A"},
            {"from": "human", "value": "unfinished"},
        ],
    }
    result = normalize(source)
    assert result["group_id"] == "same"
    assert len(result["messages"]) == 2
    source["conversations"][0]["value"] = " "
    assert normalize(source) is None
    with pytest.raises(ValueError, match="schema"):
        normalize({"made_up_field": []})


def test_refusal_label_only_reads_assistant_turns():
    settings = load_config("configs/default.yaml")["data"]["categories"]
    messages = [
        {"role": "user", "content": "I can't understand this python function"},
        {"role": "assistant", "content": "Here is the code."},
    ]
    assert label_conversation(messages, settings) == "technical"


def test_toxicity_device_auto_resolves(monkeypatch):
    import pytest
    import torch

    from honest_peer.data import resolve_device

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert resolve_device("auto") == "cpu"
    assert resolve_device("cpu") == "cpu"
    with pytest.raises(RuntimeError, match="no CUDA GPU"):
        resolve_device("cuda")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    assert resolve_device("auto") == "cuda"
