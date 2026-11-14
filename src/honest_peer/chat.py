"""One rendering path for training and inference; no hard-coded Llama token IDs."""

import os

from .artifacts import digest


def get_tokenizer(config):
    from transformers import AutoTokenizer

    settings = config["model"]
    tokenizer = AutoTokenizer.from_pretrained(
        settings["tokenizer"] or settings["name"],
        revision=settings["tokenizer_revision"] or settings["revision"],
        token=os.environ.get("HF_TOKEN") or False,
        use_fast=True,
    )
    if settings["chat_template"]:
        tokenizer.chat_template = settings["chat_template"]
    if not tokenizer.chat_template:
        raise ValueError("Tokenizer has no chat template. Configure model.chat_template explicitly.")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    return tokenizer


def messages_for_model(messages, settings):
    if settings.get("system_prompt") and (not messages or messages[0]["role"] != "system"):
        return [{"role": "system", "content": settings["system_prompt"]}, *messages]
    return messages


def template_info(tokenizer):
    return {
        "chat_template": tokenizer.chat_template,
        "chat_template_sha256": digest(tokenizer.chat_template),
        "tokenizer_class": type(tokenizer).__name__,
        "special_tokens": tokenizer.special_tokens_map,
        "revision": getattr(tokenizer, "init_kwargs", {}).get("_commit_hash"),
    }


def encode_assistant_only(tokenizer, messages, max_length):
    """Use template-generated boundaries; supervise every assistant body plus its end marker.

    Prefix stability is checked, never assumed. This also works with Llama templates
    without Jinja {% generation %} annotations. Headers/user/system/padding stay masked.
    """
    ids = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=False)
    labels = [-100] * len(ids)
    spans = []
    for index, turn in enumerate(messages):
        if turn["role"] != "assistant":
            continue
        prefix = tokenizer.apply_chat_template(messages[:index], tokenize=True, add_generation_prompt=True)
        completed = tokenizer.apply_chat_template(
            messages[: index + 1], tokenize=True, add_generation_prompt=False
        )
        start, end = len(prefix), len(completed)
        if ids[:start] != prefix or ids[:end] != completed or end <= start:
            raise ValueError(
                "Chat template has unstable turn boundaries; cannot safely construct assistant loss mask."
            )
        labels[start:end] = ids[start:end]
        spans.append((start, end))
    return {
        "input_ids": ids[:max_length],
        "attention_mask": [1] * min(max_length, len(ids)),
        "labels": labels[:max_length],
        "original_length": len(ids),
        "assistant_spans": spans,
    }


def mask_report(tokenizer, encoded):
    chunks = []
    for start, end in encoded["assistant_spans"]:
        if start < len(encoded["input_ids"]):
            chunks.append(tokenizer.decode(encoded["input_ids"][start:end], skip_special_tokens=False))
    return {
        "decoded_input": tokenizer.decode(encoded["input_ids"], skip_special_tokens=False),
        "supervised_assistant_chunks": chunks,
        "supervised_tokens": sum(label != -100 for label in encoded["labels"]),
        "masked_tokens": sum(label == -100 for label in encoded["labels"]),
    }


class AssistantCollator:
    def __init__(self, pad_token_id):
        self.pad_token_id = pad_token_id

    def __call__(self, features):
        import torch

        length = max(len(x["input_ids"]) for x in features)
        batch = {"input_ids": [], "attention_mask": [], "labels": []}
        for row in features:
            pad = length - len(row["input_ids"])
            batch["input_ids"].append(row["input_ids"] + [self.pad_token_id] * pad)
            batch["attention_mask"].append(row["attention_mask"] + [0] * pad)
            batch["labels"].append(row["labels"] + [-100] * pad)
        return {key: torch.tensor(value, dtype=torch.long) for key, value in batch.items()}
