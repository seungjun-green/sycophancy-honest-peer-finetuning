from copy import deepcopy

import pytest

from honest_peer.classification import ConversationClassifier
from honest_peer.config import load_config


@pytest.fixture
def settings():
    return load_config("configs/default.yaml")["data"]["categories"]


def pair(user, assistant="A neutral reply."):
    return [{"role": "user", "content": user}, {"role": "assistant", "content": assistant}]


@pytest.mark.parametrize(
    "phrase",
    [
        "I'm sorry",
        "I apologize",
        "I can't",
        "I cannot",
        "Unfortunately",
        "not able to",
        "unable to",
        "against my",
        "not appropriate",
    ],
)
def test_published_refusal_phrases_inspect_only_assistant(settings, phrase):
    classify = ConversationClassifier(settings)
    assert classify(pair("Hello", f"Before {phrase} after")) == "refusal"
    assert classify(pair(f"Before {phrase} after")) == "other"


@pytest.mark.parametrize("prefix", ["What ", "Who ", "When ", "Where ", "Why ", "How "])
def test_factual_prefix_is_case_sensitive_at_start_with_space(settings, prefix):
    classify = ConversationClassifier(settings)
    assert classify(pair(prefix + "question?")) == "factual"
    assert classify(pair(prefix.lower() + "question?")) == "other"
    assert classify(pair(" " + prefix + "question?")) == "other"
    assert classify(pair(prefix.rstrip() + "?")) == "other"


@pytest.mark.parametrize(
    ("phrase", "category"),
    [(value, "factual") for value in ["explain", "define", "describe", "difference between"]]
    + [(value, "creative") for value in ["story", "poem", "write", "create", "generate", "imagine"]]
    + [(value, "technical") for value in ["code", "program", "function", "algorithm", "debug"]]
    + [(value, "advice") for value in ["advice", "help me", "guide", "recommend", "suggestion"]],
)
def test_published_user_substrings_are_case_insensitive_and_role_scoped(settings, phrase, category):
    classify = ConversationClassifier(settings)
    assert classify(pair(f"Please {phrase.upper()} something")) == category
    assert classify(pair("Hello", f"Please {phrase} something")) == "other"


def test_technical_code_fence_is_assistant_scoped(settings):
    classify = ConversationClassifier(settings)
    assert classify(pair("Hello", "```text\nx\n```")) == "technical"
    assert classify(pair("```text\nx\n```")) == "other"


@pytest.mark.parametrize(
    ("user", "assistant", "expected"),
    [
        ("What code should I write?", "I cannot assist.", "refusal"),
        ("What code should I write?", "```x```", "factual"),
        ("Write code and give advice", "```x```", "creative"),
        ("Debug this and give advice", "A neutral reply.", "technical"),
        ("Help me please", "A neutral reply.", "advice"),
    ],
)
def test_priority_within_one_pair(settings, user, assistant, expected):
    assert ConversationClassifier(settings)(pair(user, assistant)) == expected


def test_first_matching_pair_wins_even_over_later_refusal(settings):
    classify = ConversationClassifier(settings)
    assert classify(pair("Help me please") + pair("What next?", "I cannot assist.")) == "advice"
    assert classify(pair("Hello") + pair("Write a poem") + pair("What next?")) == "creative"


def test_only_adjacent_user_assistant_pairs_match(settings):
    classify = ConversationClassifier(settings)
    assert classify([{"role": "assistant", "content": "I cannot assist."}] + pair("Hello")) == "other"
    assert (
        classify(
            [
                {"role": "user", "content": "What next?"},
                {"role": "system", "content": "Interruption"},
                {"role": "assistant", "content": "I cannot assist."},
            ]
        )
        == "other"
    )
    assert classify(pair("Hello") + [{"role": "user", "content": "Write a poem"}]) == "other"


@pytest.mark.parametrize(
    ("user", "expected"),
    [
        ("A rewrite please", "creative"),
        ("A programming task", "technical"),
        ("Any suggestions?", "advice"),
        ("Python", "other"),
        ("Should I change careers?", "other"),
    ],
)
def test_literal_substrings_without_added_word_boundaries_or_old_keywords(settings, user, expected):
    assert ConversationClassifier(settings)(pair(user)) == expected


def test_unspecified_case_and_apostrophe_choices_are_configurable(settings):
    classify = ConversationClassifier(settings)
    assert classify(pair("Hello", "I’m sorry about that.")) == "refusal"
    assert classify(pair("Hello", "i'm sorry about that.")) == "other"
    changed = deepcopy(settings)
    changed["patterns"]["refusal"][0]["case_sensitive"] = False
    assert ConversationClassifier(changed)(pair("Hello", "i'm sorry about that.")) == "refusal"
    changed["normalize_apostrophes"] = False
    assert ConversationClassifier(changed)(pair("Hello", "I’m sorry about that.")) == "other"


def test_custom_regex_is_explicit_and_invalid_rules_fail(settings):
    changed = deepcopy(settings)
    changed["patterns"]["advice"] = [
        {"role": "user", "match": "regex", "case_sensitive": False, "values": [r"\bshould I\b"]}
    ]
    assert ConversationClassifier(changed)(pair("Should I change careers?")) == "advice"
    changed["patterns"]["advice"][0]["values"] = ["["]
    with pytest.raises(ValueError, match="Invalid regex"):
        ConversationClassifier(changed)
    with pytest.raises(ValueError, match="priority"):
        ConversationClassifier({**settings, "priority": ["other"]})
    with pytest.raises(ValueError, match="Fallback"):
        ConversationClassifier({**settings, "fallback": "missing"})
    with pytest.raises(ValueError, match="matching_unit"):
        ConversationClassifier({**settings, "matching_unit": "conversation"})
