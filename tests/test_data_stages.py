import asyncio
from collections import Counter
from copy import deepcopy
from threading import Event, current_thread
from types import SimpleNamespace

from honest_peer.api import Claude as CachedClaude
from honest_peer.artifacts import digest, file_digest, read_jsonl, write_json
from honest_peer.config import load_config
from honest_peer.data import PreparationProgress, prepare_data, rewrite_data, select_rewrite_preview
from honest_peer.smoke import smoke_config


def category_config(tmp_path):
    cfg = smoke_config(load_config("configs/default.yaml"), tmp_path)
    cfg["data"]["sampling"].update(total=72, validation=6)
    examples = {
        "refusal": ("Hello", "I cannot help with that request."),
        "technical": ("Debug this function", "It returns 4."),
        "advice": ("Please give me advice about my career", "Think about your options."),
        "factual": ("What is 2 + 2?", "The answer is 4."),
        "creative": ("Write a poem", "Stars shine above."),
        "other": ("Hello friend", "Hi there."),
    }
    rows = []
    for category, (question, answer) in examples.items():
        for index in range(16):
            rows.append(
                {
                    "id": f"{category}{index}_0",
                    "conversations": [
                        {"from": "human", "value": f"{question} Example {index}."},
                        {"from": "gpt", "value": answer},
                        {"from": "human", "value": "Continue please."},
                        {"from": "gpt", "value": "That is all."},
                    ],
                }
            )
    write_json(cfg["sources"]["sharegpt"]["local_path"], rows)
    return cfg


def test_preparation_heartbeat_reports_slow_phase_and_stops(capsys):
    progress = PreparationProgress(interval=0.01)
    heartbeat = Event()
    emit = progress.emit

    def watched_emit():
        emit()
        if current_thread() is getattr(progress, "thread", None):
            heartbeat.set()

    progress.emit = watched_emit
    with progress:
        progress.update("Waiting for download fixture")
        assert heartbeat.wait(2), "No progress heartbeat during a slow phase"
    assert not progress.thread.is_alive()
    assert capsys.readouterr().err.count("Waiting for download fixture") >= 2


def test_preparation_never_calls_claude_and_prompt_changes_preserve_sampling(tmp_path, monkeypatch):
    cfg = category_config(tmp_path)

    def unexpected(*args, **kwargs):
        raise AssertionError("Preparation must never initialize Claude")

    monkeypatch.setattr("honest_peer.data.Claude", unexpected)
    summary = asyncio.run(prepare_data(cfg))
    root = tmp_path / "data"
    before = file_digest(root / "train.original.jsonl")
    cfg["rewrite"]["prompt_text"] += "\nUse a more relaxed tone."
    cfg["rewrite"]["prompt_version"] = "peer-v2"
    asyncio.run(prepare_data(cfg))
    assert file_digest(root / "train.original.jsonl") == before
    assert summary["sample_sizes"] == {"train": 72, "validation": 6}
    assert not (root / "train.rewritten.jsonl").exists()
    assert (root / "tokenizer/tokenizer.json").exists()


def test_preview_shortfalls_do_not_fill_from_other_categories():
    rows = [
        {"id": f"{category}{index}", "category": category}
        for category, count in (("factual", 15), ("refusal", 3))
        for index in range(count)
    ]
    selected, counts = select_rewrite_preview(rows, ["factual", "refusal", "other"], 10, 42)
    assert counts == {"factual": 10, "refusal": 3, "other": 0}
    assert len(selected) == 13
    assert (selected, counts) == select_rewrite_preview(
        list(reversed(rows)), ["factual", "refusal", "other"], 10, 42
    )


def test_preparation_classifies_before_truncation_and_deduplicates_saved_content(tmp_path):
    cfg = smoke_config(load_config("configs/default.yaml"), tmp_path)
    cfg["data"]["max_turns"] = 2
    cfg["data"]["sampling"].update(total=2, validation=1)
    rows = [
        {
            "id": f"late{index}_0",
            "conversations": [
                {"from": "human", "value": f"Hello friend {index}"},
                {"from": "gpt", "value": "Hi there."},
                {"from": "human", "value": "What is 2 + 2?"},
                {"from": "gpt", "value": "4"},
            ],
        }
        for index in range(3)
    ]
    duplicate = deepcopy(rows[0])
    duplicate["id"] = "duplicate_0"
    duplicate["conversations"][-1]["value"] = "Four"
    write_json(cfg["sources"]["sharegpt"]["local_path"], rows + [duplicate])
    summary = asyncio.run(prepare_data(cfg))
    saved = [
        row
        for split in ("train", "validation")
        for row in read_jsonl(tmp_path / "data" / f"{split}.original.jsonl")
    ]
    assert len(saved) == 3
    assert all(row["category"] == "factual" and len(row["messages"]) == 2 for row in saved)
    assert summary["filter_counts"]["truncated"] == 4
    assert summary["filter_counts"]["duplicate_after_truncation"] == 1
    assert summary["classification"] == cfg["data"]["categories"]
    assert summary["classification_rules_sha256"] == digest(cfg["data"]["categories"])
    assert summary["placeholder_patterns"] is False


def test_live_preview_uses_ten_per_category_and_full_run_reuses_cached_turns(tmp_path, monkeypatch):
    cfg = category_config(tmp_path)
    asyncio.run(prepare_data(cfg))
    cfg["judge"]["backend"] = "anthropic"  # Exercise the real API/cache path using a fake transport.
    calls = []

    class Messages:
        async def create(self, **kwargs):
            prompt = kwargs["messages"][0]["content"]
            calls.append(prompt)
            return SimpleNamespace(
                content=[SimpleNamespace(type="text", text=prompt.split("ORIGINAL REPLY:\n")[-1].strip())],
                stop_reason="end_turn",
                model="fixture",
                id=str(len(calls)),
                usage=SimpleNamespace(model_dump=lambda: {"input_tokens": 1}),
            )

    class Client:
        messages = Messages()

        async def close(self):
            pass

    def claude(settings, path):
        result = CachedClaude(settings, path)
        result.client = Client()
        return result

    monkeypatch.setattr("honest_peer.data.Claude", claude)
    preview = asyncio.run(rewrite_data(cfg, preview=True))
    assert set(preview["category_counts"].values()) == {10}
    assert preview["assistant_turns"] == 120
    assert preview["shortfalls"] == {}
    assert len(calls) == 120
    root = tmp_path / "data"
    assert not (root / "train.rewritten.jsonl").exists()
    preview_rows = list(read_jsonl(f"{preview['output']}/preview.original.jsonl"))
    training_ids = {row["id"] for row in read_jsonl(root / "train.original.jsonl")}
    assert {row["id"] for row in preview_rows} <= training_ids
    assert Counter(row["category"] for row in preview_rows) == Counter(
        {c: 10 for c in cfg["data"]["categories"]["priority"]}
    )
    html = (root / "rewrite-previews" / preview["output"].split("/")[-1] / "review.html").read_text()
    assert "Conversation history" in html and "Claude rewrite" in html
    asyncio.run(rewrite_data(cfg, preview=True))
    assert len(calls) == 120
    asyncio.run(rewrite_data(cfg))
    assert len(calls) == (72 + 6) * 2
    full_before = file_digest(root / "train.rewritten.jsonl")
    changed = deepcopy(cfg)
    changed["rewrite"]["prompt_text"] = "Keep the reply direct.\n" + cfg["rewrite"]["prompt_text"]
    changed["rewrite"]["prompt_version"] = "peer-v2"
    second = asyncio.run(rewrite_data(changed, preview=True))
    assert second["output"] != preview["output"]
    assert [row["id"] for row in read_jsonl(f"{second['output']}/preview.original.jsonl")] == [
        row["id"] for row in preview_rows
    ]
    assert file_digest(root / "train.rewritten.jsonl") == full_before
