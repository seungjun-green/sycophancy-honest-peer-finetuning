import asyncio
from copy import deepcopy
from types import SimpleNamespace

import pytest

from honest_peer.api import Claude
from honest_peer.artifacts import Cache, Run
from honest_peer.config import load_config
from honest_peer.data import rewrite_conversation


def test_cache_is_durable_and_changes_invalidate(tmp_path):
    path = tmp_path / "cache.sqlite"
    payload = {"id": ["c", 1, "v1"], "model": "one", "prompt": "old", "temperature": 0}
    with Cache(path) as cache:
        cache.put(payload, {"text": "answer"})
    with Cache(path) as cache:
        assert cache.get(payload) == {"text": "answer"}
        for key, value in [("model", "two"), ("prompt", "new"), ("temperature", 0.8)]:
            assert cache.get({**payload, key: value}) is None


def test_run_rejects_config_or_input_drift(tmp_path):
    with Run(tmp_path, {"seed": 42}, {"file": "a"}):
        pass
    with Run(tmp_path, {"seed": 42}, {"file": "a"}):
        pass
    with pytest.raises(ValueError, match="different config"):
        with Run(tmp_path, {"seed": 43}, {"file": "a"}):
            pass
    with pytest.raises(ValueError, match="different config"):
        with Run(tmp_path, {"seed": 42}, {"file": "b"}):
            pass


def test_validated_api_call_resumes_without_network(tmp_path):
    settings = load_config("configs/default.yaml")["judge"]
    client = Claude(settings, tmp_path / "api.sqlite")
    calls = []

    class FakeMessages:
        async def create(self, **kwargs):
            calls.append(kwargs)
            return SimpleNamespace(
                content=[SimpleNamespace(type="text", text="A")],
                stop_reason="end_turn",
                model="test",
                id="request",
                usage=SimpleNamespace(model_dump=lambda: {"input_tokens": 1}),
            )

    class FakeClient:
        messages = FakeMessages()

        async def close(self):
            pass

    async def run():
        client.client = FakeClient()
        await client.ask("prompt", ["q1"])
        await client.ask("prompt", ["q1"])
        await client.ask("different prompt", ["q1"])
        await client.close()

    asyncio.run(run())
    assert len(calls) == 2


def test_rewrite_full_history_and_immutable_user_turns():
    settings = load_config("configs/default.yaml")["rewrite"]
    original = {
        "id": "c1",
        "messages": [
            {"role": "user", "content": "u1"},
            {"role": "assistant", "content": "a1"},
            {"role": "user", "content": "u2"},
            {"role": "assistant", "content": "a2"},
        ],
    }
    snapshot = deepcopy(original)
    prompts = []

    class Fake:
        async def ask(self, prompt, identity, validate):
            prompts.append(prompt)
            text = "direct " + ("a1" if identity[1] == 1 else "a2")
            validate(text)
            return {"text": text}

    result = asyncio.run(rewrite_conversation(original, Fake(), settings))
    assert original == snapshot
    assert (
        '"content": "u1"' in prompts[1]
        and '"content": "a1"' in prompts[1]
        and '"content": "u2"' in prompts[1]
    )
    assert result["messages"][0] == original["messages"][0]
    assert result["messages"][2] == original["messages"][2]
