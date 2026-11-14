import os

import pytest

from honest_peer.chat import AssistantCollator, encode_assistant_only, mask_report
from honest_peer.smoke import TinyTokenizer

FENCE = chr(96) * 3
MESSAGES = [
    {"role": "system", "content": "SYSTEM_SECRET"},
    {"role": "user", "content": "USER_SECRET"},
    {"role": "assistant", "content": "First answer: 42."},
    {"role": "user", "content": "SECOND_USER_SECRET"},
    {"role": "assistant", "content": f"Second answer.\n{FENCE}python\nprint(42)\n{FENCE}"},
]


def assert_mask(tokenizer):
    encoded = encode_assistant_only(tokenizer, MESSAGES, 4096)
    report = mask_report(tokenizer, encoded)
    assert len(report["supervised_assistant_chunks"]) == 2
    for turn, chunk in zip([MESSAGES[2], MESSAGES[4]], report["supervised_assistant_chunks"], strict=True):
        assert turn["content"] in chunk
        assert "assistant" not in chunk
    supervised = tokenizer.decode([x for x in encoded["labels"] if x != -100], skip_special_tokens=False)
    assert "SECRET" not in supervised
    assert "First answer: 42." in supervised
    assert "print(42)" in supervised
    shorter = encode_assistant_only(tokenizer, MESSAGES[:3], 4096)
    batch = AssistantCollator(tokenizer.pad_token_id)(
        [
            {k: encoded[k] for k in ("input_ids", "attention_mask", "labels")},
            {k: shorter[k] for k in ("input_ids", "attention_mask", "labels")},
        ]
    )
    assert (batch["labels"][batch["attention_mask"] == 0] == -100).all()
    assert (batch["labels"][0] != -100).sum() == report["supervised_tokens"]
    return encoded


def test_decode_one_multiturn_batch():
    assert_mask(TinyTokenizer())


def test_truncation_never_unmasks_user_tokens():
    tokenizer = TinyTokenizer()
    full = assert_mask(tokenizer)
    length = full["assistant_spans"][1][0] + 3
    short = encode_assistant_only(tokenizer, MESSAGES, length)
    assert short["labels"] == full["labels"][:length]
    assert short["original_length"] > len(short["input_ids"])


@pytest.mark.integration
@pytest.mark.skipif(
    not os.environ.get("RUN_TOKENIZER_TEST"), reason="Set RUN_TOKENIZER_TEST=1 for the real Llama tokenizer"
)
def test_llama31_real_template():
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        os.environ.get("TEST_TOKENIZER", "meta-llama/Llama-3.1-8B-Instruct")
    )
    tokenizer.pad_token = tokenizer.eos_token
    assert_mask(tokenizer)
