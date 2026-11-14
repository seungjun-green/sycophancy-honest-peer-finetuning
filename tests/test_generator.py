from types import SimpleNamespace

from honest_peer.config import load_config
from honest_peer.generation import Generator
from honest_peer.smoke import TinyTokenizer


def test_vllm_request_contract_and_cache(tmp_path, monkeypatch):
    import sys

    calls = []
    tokenizer = TinyTokenizer()
    monkeypatch.setattr("honest_peer.generation.get_tokenizer", lambda config: tokenizer)
    cfg = load_config("configs/default.yaml")
    cfg["model"]["name"] = str(tmp_path)  # Local path avoids network metadata lookup.

    class SamplingParams:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class LLM:
        def __init__(self, **kwargs):
            self.llm_engine = SimpleNamespace(engine_core=SimpleNamespace(shutdown=lambda: None))

        def generate(self, prompts, params, **kwargs):
            calls.append((prompts, params, kwargs))
            return [
                SimpleNamespace(
                    prompt_token_ids=prompt["prompt_token_ids"],
                    outputs=[
                        SimpleNamespace(text="A", finish_reason="stop", stop_reason=None, token_ids=[1])
                    ],
                )
                for prompt in prompts
            ]

    monkeypatch.setitem(sys.modules, "vllm", SimpleNamespace(LLM=LLM, SamplingParams=SamplingParams))
    monkeypatch.setitem(sys.modules, "vllm.lora.request", SimpleNamespace(LoRARequest=None))
    request = {
        "id": ["item", 1],
        "item": {},
        "condition": "initial",
        "messages": [{"role": "user", "content": "Choose A or B"}],
        "temperature": 0.2,
        "max_tokens": 5,
    }
    generator = Generator(cfg, tmp_path / "eval")
    try:
        first = generator.generate([request])
        second = generator.generate([request])
    finally:
        generator.close()
    assert first == second
    assert len(calls) == 1
    assert calls[0][1][0].kwargs["temperature"] == 0.2
    assert calls[0][1][0].kwargs["seed"] == first[0]["seed"]
    assert tokenizer.decode(calls[0][0][0]["prompt_token_ids"]) == first[0]["prompt"]
