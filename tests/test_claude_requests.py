import asyncio
from types import SimpleNamespace

import pytest

from honest_peer.api import FALLBACK_BETA, Claude, ClaudeRefusal, request_options
from honest_peer.config import load_config
from honest_peer.data import rewrite_conversation


def response(stop_reason="end_turn", text="ok", stop_details=None):
    return SimpleNamespace(
        content=[{"type": "thinking", "thinking": ""}, SimpleNamespace(type="text", text=text)],
        stop_reason=stop_reason,
        stop_details=stop_details,
        model="claude-sonnet-5-5",
        id="req",
        usage=SimpleNamespace(model_dump=lambda: {}),
    )


def fake_client(responses, calls):
    class Messages:
        async def create(self, **kwargs):
            calls.append(kwargs)
            return responses.pop(0)

    class Client:
        messages = Messages()

        async def close(self):
            pass

    return Client()


def test_default_requests_omit_sampling_and_disable_thinking():
    config = load_config("configs/default.yaml")
    judge = request_options(config["judge"])
    assert "temperature" not in judge
    assert judge["thinking"] == {"type": "between_tools"}
    assert judge["extra_body"] == {"output_config": {"effort": "high"}}
    assert "extra_headers" not in judge
    rewrite = request_options(config["rewrite"])
    assert "temperature" not in rewrite
    assert rewrite["extra_body"]["fallbacks"] == "default"
    assert rewrite["extra_headers"] == {"anthropic-beta": FALLBACK_BETA}
    explicit = request_options({**config["judge"], "temperature": 0.0, "thinking": None, "effort": None})
    assert explicit == {"temperature": 0.0}


def test_text_is_read_by_block_type(tmp_path):
    calls = []
    client = Claude(load_config("configs/default.yaml")["judge"], tmp_path / "c.sqlite")
    client.client = fake_client([response(text="A")], calls)

    async def run():
        try:
            return await client.ask("p", ["q"])
        finally:
            await client.close()

    assert asyncio.run(run())["text"] == "A"


def test_refusal_is_not_retried_and_is_cached(tmp_path):
    calls = []
    settings = load_config("configs/default.yaml")["judge"]
    client = Claude(settings, tmp_path / "c.sqlite")
    client.client = fake_client([response("refusal", "", {"category": "cyber"})], calls)

    async def run():
        for _ in range(2):
            with pytest.raises(ClaudeRefusal) as caught:
                await client.ask("p", ["q"])
            assert caught.value.category == "cyber"
        await client.close()

    asyncio.run(run())
    assert len(calls) == 1


def test_refused_rewrite_turn_keeps_original_wording():
    settings = load_config("configs/default.yaml")["rewrite"]
    row = {
        "id": "c1",
        "messages": [
            {"role": "user", "content": "u1"},
            {"role": "assistant", "content": "a1"},
            {"role": "user", "content": "u2"},
            {"role": "assistant", "content": "a2"},
        ],
    }

    class Fake:
        async def ask(self, prompt, identity, validate):
            if identity[1] == 1:
                raise ClaudeRefusal(identity, "general_harms", "claude-sonnet-5-5")
            return {"text": "direct a2"}

    result = asyncio.run(rewrite_conversation(row, Fake(), settings))
    assert [m["content"] for m in result["messages"]] == ["u1", "a1", "u2", "direct a2"]
    assert result["rewrite_refusals"] == [
        {"turn": 1, "category": "general_harms", "model": "claude-sonnet-5-5"}
    ]
    with pytest.raises(ClaudeRefusal):
        asyncio.run(rewrite_conversation(row, Fake(), {**settings, "on_refusal": "fail"}))
