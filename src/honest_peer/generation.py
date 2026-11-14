"""vLLM generation with per-request seeds, LoRA support, and incremental caching."""

import json
import os
from pathlib import Path

from .artifacts import Cache, digest, file_digest, write_json
from .chat import get_tokenizer, messages_for_model, template_info


def adapter_identity(path):
    if path is None:
        return None
    root = Path(path)
    if not root.is_dir():
        raise ValueError(f"Adapter directory missing: {path}")
    files = sorted(root.glob("adapter*"))
    if not any(p.suffix == ".safetensors" for p in files):
        raise ValueError(f"Expected adapter safetensors in {path}")
    return {p.name: file_digest(p) for p in files if p.is_file()}


class Generator:
    def __init__(self, config, output):
        self.config = config
        self.output = Path(output)
        self.cache = Cache(self.output / "generations.sqlite")
        self.engine = None
        self.mock = config["generation"]["backend"] == "mock"
        if self.mock:
            if not config["smoke"]:
                raise ValueError("Mock generation requires smoke=true")
            self.identity = {"backend": "mock", "variant": config["model"]["variant"]}
            self.tokenizer = None
        else:
            from huggingface_hub import HfApi

            self.tokenizer = get_tokenizer(config)
            name = config["model"]["name"]
            revision = config["model"]["revision"]
            if not Path(name).exists():
                revision = (
                    HfApi(token=os.environ.get("HF_TOKEN") or False).model_info(name, revision=revision).sha
                )
            self.identity = {
                "backend": "vllm",
                "name": name,
                "revision": revision,
                "adapter": adapter_identity(config["model"]["adapter"]),
                "template": template_info(self.tokenizer),
                "dtype": config["model"]["dtype"],
                "runtime": config["generation"],
            }
            if config["model"]["adapter"]:
                expected = Path(config["model"]["adapter"]) / "chat-template.json"
                if not expected.exists():
                    raise ValueError(
                        "Adapter has no chat-template.json; cannot verify training/evaluation template consistency"
                    )
                expected = json.loads(expected.read_text())
                if expected["chat_template_sha256"] != self.identity["template"]["chat_template_sha256"]:
                    raise ValueError("Training and evaluation chat templates differ")
        identity_path = self.output / "model-identity.json"
        if identity_path.exists() and json.loads(identity_path.read_text()) != self.identity:
            raise ValueError("Model/tokenizer/adapter changed during resume; use a new evaluation directory")
        write_json(identity_path, self.identity)

    def render(self, messages):
        messages = messages_for_model(messages, self.config["model"])
        prefill = messages[-1]["role"] == "assistant"
        return self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=not prefill, continue_final_message=prefill
        )

    def generate(self, requests):
        """Each request has id/messages/item/condition/temperature/max_tokens."""
        config = self.config
        pending, results = [], {}
        for index, request in enumerate(requests):
            seed = (config["seed"] + int(digest(request["id"])[:8], 16)) % (2**31)
            prompt = self.render(request["messages"]) if not self.mock else json.dumps(request["messages"])
            payload = {
                "model": self.identity,
                "id": request["id"],
                "prompt": prompt,
                "temperature": request["temperature"],
                "max_tokens": request["max_tokens"],
                "top_p": config["generation"]["top_p"],
                "seed": seed,
            }
            cached = self.cache.get(payload)
            if cached is not None:
                results[index] = cached
            elif self.mock:
                from .smoke import mock_response

                value = {
                    "text": mock_response(request, config["model"]["variant"]),
                    "finish_reason": "stop",
                    "seed": seed,
                    "synthetic": True,
                    "prompt": prompt,
                }
                self.cache.put(payload, value)
                results[index] = value
            else:
                pending.append((index, request, payload))
        if pending and self.engine is None:
            from vllm import LLM
            from vllm.lora.request import LoRARequest

            settings = config["generation"]
            self.engine = LLM(
                model=config["model"]["name"],
                revision=self.identity["revision"],
                tokenizer=config["model"]["tokenizer"] or config["model"]["name"],
                tokenizer_revision=config["model"]["tokenizer_revision"] or self.identity["revision"],
                dtype=config["model"]["dtype"],
                seed=config["seed"],
                enable_lora=config["model"]["adapter"] is not None,
                max_lora_rank=settings["max_lora_rank"],
                max_model_len=settings["max_model_len"],
                gpu_memory_utilization=settings["gpu_memory_utilization"],
                tensor_parallel_size=settings["tensor_parallel_size"],
                enforce_eager=settings["enforce_eager"],
                trust_remote_code=False,
                hf_token=os.environ.get("HF_TOKEN") or False,
            )
            self.lora = (
                LoRARequest(config["model"]["variant"], 1, str(Path(config["model"]["adapter"]).resolve()))
                if config["model"]["adapter"]
                else None
            )
        batch_size = config["generation"]["batch_size"]
        for start in range(0, len(pending), batch_size):
            from vllm import SamplingParams

            chunk = pending[start : start + batch_size]
            prompts, params = [], []
            for _, request, payload in chunk:
                ids = self.tokenizer(payload["prompt"], add_special_tokens=False)["input_ids"]
                if len(ids) + request["max_tokens"] > config["generation"]["max_model_len"]:
                    raise ValueError(
                        f"Context overflow for {request['id']}: {len(ids)} prompt + {request['max_tokens']} new tokens. Increase generation.max_model_len."
                    )
                prompts.append({"prompt_token_ids": ids})
                params.append(
                    SamplingParams(
                        temperature=request["temperature"],
                        max_tokens=request["max_tokens"],
                        top_p=config["generation"]["top_p"],
                        seed=payload["seed"],
                    )
                )
            generated = self.engine.generate(prompts, params, lora_request=self.lora, use_tqdm=False)
            for (index, request, payload), response in zip(chunk, generated, strict=True):
                completion = response.outputs[0]
                prefix = (
                    request["messages"][-1]["content"]
                    if request["messages"][-1]["role"] == "assistant"
                    else ""
                )
                value = {
                    "text": prefix + completion.text,
                    "finish_reason": completion.finish_reason,
                    "stop_reason": completion.stop_reason,
                    "seed": payload["seed"],
                    "synthetic": False,
                    "prompt": payload["prompt"],
                    "prompt_tokens": len(response.prompt_token_ids),
                    "completion_tokens": len(completion.token_ids),
                }
                self.cache.put(payload, value)
                results[index] = value
        return [results[i] for i in range(len(requests))]

    def close(self):
        self.cache.close()
        if self.engine is not None:
            # vLLM 0.10's V1 core owns worker processes and exposes shutdown explicitly.
            core = getattr(self.engine.llm_engine, "engine_core", None)
            if core is not None:
                core.shutdown()
            self.engine = None
            import gc

            import torch

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
