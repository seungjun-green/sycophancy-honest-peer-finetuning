"""Query-type rules transcribed from Ibrahim et al., Nature (2026), SI §1.1.1.

The supplement describes literal substring/prefix matching, not executable classifier
code. Unspecified implementation choices are explicit in config; see docs/classification.md.
"""

import re


class ConversationClassifier:
    def __init__(self, settings):
        if settings.get("matching_unit") != "adjacent_user_assistant_pairs":
            raise ValueError("Classification requires adjacent_user_assistant_pairs matching_unit")
        priority, clauses = settings["priority"], settings["patterns"]
        if set(priority) != set(clauses) or len(priority) != len(clauses):
            raise ValueError("Category priority must include each configured category exactly once.")
        self.fallback = settings["fallback"]
        if self.fallback not in clauses or priority[-1] != self.fallback or clauses[self.fallback]:
            raise ValueError("Fallback category must be last with no matching clauses")
        self.normalize_apostrophes = settings["normalize_apostrophes"]
        if not isinstance(self.normalize_apostrophes, bool):
            raise ValueError("normalize_apostrophes must be boolean")
        self.rules = []
        for category in priority[:-1]:
            if not isinstance(clauses[category], list) or not clauses[category]:
                raise ValueError(f"Category {category} requires nonempty matching clauses")
            compiled = []
            for clause in clauses[category]:
                if not isinstance(clause, dict) or set(clause) != {
                    "role",
                    "match",
                    "case_sensitive",
                    "values",
                }:
                    raise ValueError(f"Invalid matching clause for {category}")
                role, method = clause["role"], clause["match"]
                values, case_sensitive = clause["values"], clause["case_sensitive"]
                if role not in {"user", "assistant"} or method not in {"substring", "prefix", "regex"}:
                    raise ValueError(f"Invalid role or match method for {category}")
                if not isinstance(case_sensitive, bool):
                    raise ValueError(f"case_sensitive must be boolean for {category}")
                if (
                    not isinstance(values, list)
                    or not values
                    or any(not isinstance(value, str) or not value for value in values)
                ):
                    raise ValueError(f"Matching values must be nonempty strings for {category}")
                expressions = []
                for value in values:
                    value = self._text(value)
                    expression = value if method == "regex" else re.escape(value)
                    if method == "prefix":
                        expression = r"\A" + expression
                    try:
                        expressions.append(re.compile(expression, 0 if case_sensitive else re.IGNORECASE))
                    except re.error as error:
                        raise ValueError(f"Invalid regex for {category}: {value!r}") from error
                compiled.append((role, expressions))
            self.rules.append((category, compiled))

    def _text(self, text):
        return text.replace("\u2019", "'") if self.normalize_apostrophes else text

    def __call__(self, messages):
        for first, second in zip(messages, messages[1:]):
            if first["role"] != "user" or second["role"] != "assistant":
                continue
            pair = {turn["role"]: self._text(turn["content"]) for turn in (first, second)}
            for category, clauses in self.rules:
                if any(expression.search(pair[role]) for role, values in clauses for expression in values):
                    return category
        return self.fallback
