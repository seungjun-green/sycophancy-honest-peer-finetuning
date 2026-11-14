"""ShareGPT normalization, cached filtering, grouped sampling, and turn-by-turn rewriting."""

import asyncio
import html
import importlib
import json
import os
import random
import re
import sys
from collections import Counter, defaultdict
from copy import deepcopy
from itertools import islice
from pathlib import Path
from threading import Event, Thread
from time import monotonic

import numpy as np
from tqdm import tqdm

from .api import Claude, ClaudeRefusal
from .artifacts import (
    Cache,
    Run,
    atomic_text,
    digest,
    file_digest,
    read_jsonl,
    seed_everything,
    write_json,
    write_jsonl,
)
from .classification import ConversationClassifier
from .data_review import export_prepared_csvs
from .sources import load_source, require


class PreparationProgress:
    """Report preparation phases to stderr, including a heartbeat during slow downloads/inference."""

    def __init__(self, interval=15):
        self.interval = interval
        self.stop = Event()
        self.started = monotonic()
        self.message = "Starting preparation"

    def emit(self):
        tqdm.write(f"[prepare-data {monotonic() - self.started:.0f}s] {self.message}", file=sys.stderr)
        sys.stderr.flush()

    def update(self, message, announce=True):
        self.message = message
        if announce:
            self.emit()

    def __enter__(self):
        def heartbeat():
            while not self.stop.wait(self.interval):
                self.emit()

        self.emit()
        self.thread = Thread(target=heartbeat, daemon=True)
        self.thread.start()
        return self

    def __exit__(self, *args):
        self.stop.set()
        self.thread.join()


def normalize(row, max_turns=20, group_split_suffixes=True):
    require(row, ["id", "conversations"], "ShareGPT")
    if not isinstance(row["conversations"], list) or not row["id"]:
        return None
    messages = []
    for index, turn in enumerate(row["conversations"]):
        if not isinstance(turn, dict):
            return None
        if set(("from", "value")) - turn.keys():
            return None
        role = {"human": "user", "gpt": "assistant"}.get(turn["from"])
        content = turn["value"]
        if (
            role != ("user" if index % 2 == 0 else "assistant")
            or not isinstance(content, str)
            or not content.strip()
        ):
            return None
        messages.append({"role": role, "content": content})
    if max_turns is not None:
        messages = messages[:max_turns]
    if messages and messages[-1]["role"] == "user":
        messages.pop()
    if len(messages) < 2:
        return None
    identifier = str(row["id"])
    return {
        "id": identifier,
        "group_id": re.sub(r"_\d+$", "", identifier) if group_split_suffixes else identifier,
        "messages": messages,
    }


def label_conversation(messages, settings):
    """Convenience entry point; preparation reuses one compiled classifier."""
    return ConversationClassifier(settings)(messages)


def sample_rows(rows, settings, seed, categories):
    """Deterministic quotas, optional shortfall redistribution, never duplicate a record."""
    strategy = settings["strategy"]
    if ":" in strategy:
        module, function = strategy.split(":", 1)
        result = getattr(importlib.import_module(module), function)(deepcopy(rows), deepcopy(settings), seed)
        eligible = {row["id"] for row in rows}
        ids = [row["id"] for row in result]
        if len(ids) != len(set(ids)) or not set(ids) <= eligible:
            raise ValueError("Custom sampler returned duplicates or unknown IDs")
        return result
    rng = random.Random(seed)
    rows = sorted(rows, key=lambda row: row["id"])
    count = min(settings["total"], len(rows))
    if strategy == "random":
        return rng.sample(rows, count)
    pools = defaultdict(list)
    for row in rows:
        pools[row["category"]].append(row)
    if strategy == "balanced":
        quotas = {
            cat: count // len(categories) + (i < count % len(categories)) for i, cat in enumerate(categories)
        }
    elif strategy == "quotas":
        quotas = settings["quotas"]
        if set(quotas) - set(categories) or any(not isinstance(n, int) or n < 0 for n in quotas.values()):
            raise ValueError("Sampling quotas must be nonnegative integers for configured categories.")
        if sum(quotas.values()) != settings["total"]:
            raise ValueError("Per-category quotas must sum to sampling.total")
    else:
        raise ValueError(f"Unknown sampling strategy {strategy}")
    selected, leftovers = [], []
    for cat in categories:
        pool = pools[cat]
        rng.shuffle(pool)
        n = min(quotas.get(cat, 0), len(pool))
        selected.extend(pool[:n])
        leftovers.extend(pool[n:])
    if settings["redistribute_shortfall"] and len(selected) < count:
        selected.extend(rng.sample(leftovers, min(count - len(selected), len(leftovers))))
    rng.shuffle(selected)
    return selected


def split_sample(rows, settings, seed, categories):
    validation = sample_rows(
        rows, {**settings, "total": settings["validation"], "strategy": "balanced"}, seed + 1, categories
    )
    heldout_groups = {row["group_id"] for row in validation}
    training = sample_rows(
        [row for row in rows if row["group_id"] not in heldout_groups], settings, seed, categories
    )
    if not validation or not training:
        raise ValueError(
            "No training or validation conversations after grouped splitting; reduce validation size or add data."
        )
    return {"train": training, "validation": validation}


class ToxicityFilter:
    def __init__(self, settings):
        from detoxify import Detoxify

        self.settings = settings
        self.model = Detoxify(settings["model"], device=settings["device"])

    def scores(self, messages):
        tokenizer = self.model.tokenizer
        windows = []
        for turn in messages:
            ids = tokenizer.encode(turn["content"], add_special_tokens=False)
            windows.extend(
                tokenizer.decode(ids[start : start + self.settings["max_tokens"]])
                for start in range(0, len(ids), self.settings["max_tokens"])
            )
        maximum = {}
        for start in range(0, len(windows), self.settings["batch_size"]):
            prediction = self.model.predict(windows[start : start + self.settings["batch_size"]])
            for name, values in prediction.items():
                maximum[name] = max(maximum.get(name, 0.0), float(np.max(values)))
        if set(self.settings["thresholds"]) - maximum.keys():
            raise ValueError(f"Unknown toxicity classes. Available: {sorted(maximum)}")
        return maximum


async def rewrite_conversation(row, client, settings, mock=False):
    rewritten = deepcopy(row)
    for index, turn in enumerate(row["messages"]):
        if turn["role"] != "assistant":
            continue
        history = (
            rewritten["messages"][:index] if settings["history"] == "rewritten" else row["messages"][:index]
        )
        if settings["history"] not in ("original", "rewritten"):
            raise ValueError("rewrite.history must be original or rewritten")
        if settings["on_refusal"] not in ("keep_original", "fail"):
            raise ValueError("rewrite.on_refusal must be keep_original or fail")
        prompt = settings["prompt_text"].format(
            history=json.dumps(history, ensure_ascii=False), original_reply=turn["content"]
        )

        def validate(text):
            if settings["enforce_code_blocks"]:
                # Fenced blocks of both Markdown styles must remain byte-for-byte identical.
                pattern = r"(?ms)^(`{3,}|~{3,})[^\n]*\n.*?^\1[^\S\n]*$"
                before = [match.group(0) for match in re.finditer(pattern, turn["content"])]
                after = [match.group(0) for match in re.finditer(pattern, text)]
                if before != after:
                    raise ValueError("Rewriter changed a fenced code block")

        if mock:
            text = turn["content"].removeprefix("Great question! ")
        else:
            try:
                result = await client.ask(prompt, [row["id"], index, settings["prompt_version"]], validate)
                text = result["text"]
            except ClaudeRefusal as refusal:
                if settings["on_refusal"] != "keep_original":
                    raise
                # Keep the pair intact: this turn stays in its original wording and is reported.
                text = turn["content"]
                rewritten.setdefault("rewrite_refusals", []).append(
                    {"turn": index, "category": refusal.category, "model": refusal.model}
                )
        validate(text)
        rewritten["messages"][index]["content"] = text
    return rewritten


def preparation_config(config):
    """Rewrite settings never invalidate an already selected training/validation split."""
    return {
        "stage": "prepare-data",
        "seed": config["seed"],
        "smoke": config["smoke"],
        "source": config["sources"]["sharegpt"],
        "data": {key: value for key, value in config["data"].items() if key != "rewrite_enabled"},
        "tiny_tokenizer": config["smoke"] and config["training"]["backend"] == "tiny",
    }


def data_tokenizer(config):
    saved = Path(config["paths"]["data"]) / "tokenizer"
    if saved.exists():
        from transformers import AutoTokenizer

        return AutoTokenizer.from_pretrained(saved, local_files_only=True, token=False)
    if config["smoke"] and config["training"]["backend"] == "tiny":
        from .smoke import TinyTokenizer

        return TinyTokenizer()
    from transformers import AutoTokenizer

    settings = config["data"]["token_stats"]
    return AutoTokenizer.from_pretrained(
        settings["tokenizer"], revision=settings["revision"], token=os.environ.get("HF_TOKEN") or False
    )


def token_length_stats(rows, tokenizer, progress_description=None):
    iterator = (
        tqdm(
            rows,
            desc=progress_description,
            unit="conv",
            file=sys.stderr,
            dynamic_ncols=True,
            miniters=1,
        )
        if progress_description
        else rows
    )
    lengths = [
        len(tokenizer.apply_chat_template(row["messages"], tokenize=True, add_generation_prompt=False))
        for row in iterator
    ]
    return {
        "min": min(lengths),
        "max": max(lengths),
        "mean": float(np.mean(lengths)),
        "p95": float(np.percentile(lengths, 95)),
    }


def push_dataset(config):
    if not config["hub"]["push_dataset"]:
        return
    from huggingface_hub import HfApi

    repo = config["hub"]["dataset_repo"]
    if not repo:
        raise ValueError("Set hub.dataset_repo before pushing")
    api = HfApi(token=os.environ.get("HF_TOKEN") or False)
    api.create_repo(repo, repo_type="dataset", private=config["hub"]["private"], exist_ok=True)
    api.upload_folder(
        repo_id=repo,
        repo_type="dataset",
        folder_path=config["paths"]["data"],
        ignore_patterns=["*.lock", "rewrite-previews/**"],
    )


async def prepare_data(config):
    """Filter, label, sample, and persist original conversations; never call Claude."""
    with PreparationProgress() as progress:
        return await _prepare_data(config, progress)


async def _prepare_data(config, progress):
    seed_everything(config["seed"])
    settings = config["data"]
    if settings["max_turns"] < 2:
        raise ValueError("data.max_turns must be at least 2")
    classifier = ConversationClassifier(settings["categories"])
    if config["smoke"] is False and (
        config["generation"]["backend"] == "mock" or config["training"]["backend"] == "tiny"
    ):
        raise ValueError("Synthetic backends require smoke=true")
    progress.update("Stage 1/6: Loading ShareGPT source; a first run may download the complete source file")
    rows, source = load_source(config["sources"]["sharegpt"], config["paths"]["cache"])
    progress.update("Stage 2/6: ShareGPT source ready; beginning validation and filtering")
    output = Path(config["paths"]["data"])
    with (
        Run(output, preparation_config(config), {"source": source}),
        Cache(Path(config["paths"]["cache"]) / "filters.sqlite") as cache,
    ):
        counters, kept, seen_ids, seen_content = Counter(), [], set(), set()
        seen_training_content = set()
        toxicity = None
        records = tqdm(
            islice(rows, settings["max_source_records"]),
            total=settings["max_source_records"],
            desc="Stage 2/6: Filter and classify",
            unit="conv",
            file=sys.stderr,
            dynamic_ncols=True,
            miniters=1,
        )
        for index, raw in enumerate(records):
            limit = settings["max_source_records"] or "all"
            progress.update(
                f"Stage 2/6: Filtering source conversation {index + 1}/{limit}; kept {len(kept)} so far",
                announce=index % (10 if config["smoke"] else 100) == 0,
            )
            counters["source"] += 1
            row = normalize(raw, max_turns=None, group_split_suffixes=settings["group_split_suffixes"])
            if row is None:
                counters["malformed"] += 1
                continue
            content_hash = digest(row["messages"])
            if row["id"] in seen_ids or content_hash in seen_content:
                counters["duplicate"] += 1
                continue
            seen_ids.add(row["id"])
            seen_content.add(content_hash)
            identity = {
                "messages_sha": content_hash,
                "toxicity": settings["toxicity"],
                "language": settings["language"],
                "language_seed": config["seed"] if settings["language"]["enabled"] else None,
            }
            result = cache.get(identity)
            if result is None:
                result = {"rejected": None, "toxicity": {}}
                if settings["toxicity"]["enabled"]:
                    if toxicity is None:
                        progress.update(
                            f"Stage 2/6: Loading Detoxify {settings['toxicity']['model']} on {settings['toxicity']['device']}; "
                            "first use may download model weights"
                        )
                        toxicity = ToxicityFilter(settings["toxicity"])
                        progress.update(
                            f"Stage 2/6: Detoxify ready; filtering conversation {index + 1}/{limit}"
                        )
                    result["toxicity"] = toxicity.scores(row["messages"])
                    if any(
                        result["toxicity"][name] >= limit
                        for name, limit in settings["toxicity"]["thresholds"].items()
                    ):
                        result["rejected"] = "toxicity"
                if result["rejected"] is None and settings["language"]["enabled"]:
                    from langdetect import DetectorFactory, LangDetectException, detect

                    DetectorFactory.seed = config["seed"]
                    try:
                        result["language"] = detect("\n".join(t["content"] for t in row["messages"]))
                    except LangDetectException:
                        result["language"] = "unknown"
                    if result["language"] not in settings["language"]["keep"]:
                        result["rejected"] = "language"
                cache.put(identity, result)
            if result["rejected"]:
                counters[result["rejected"]] += 1
                continue
            row["category"] = classifier(row["messages"])
            counters["truncated"] += len(row["messages"]) > settings["max_turns"]
            row["messages"] = row["messages"][: settings["max_turns"]]
            if row["messages"][-1]["role"] == "user":
                row["messages"].pop()
            training_hash = digest(row["messages"])
            if training_hash in seen_training_content:
                counters["duplicate_after_truncation"] += 1
                continue
            seen_training_content.add(training_hash)
            kept.append(row)
        counters["kept"] = len(kept)
        progress.update(
            f"Stage 3/6: Filtering complete: inspected {counters['source']}, kept {len(kept)}; sampling splits"
        )
        split = split_sample(kept, settings["sampling"], config["seed"], settings["categories"]["priority"])
        summary = {
            "synthetic": config["smoke"],
            "filter_counts": dict(counters),
            "category_counts": dict(Counter(row["category"] for row in kept)),
            "placeholder_patterns": settings["categories"]["placeholder_patterns"],
            "classification": settings["categories"],
            "classification_rules_sha256": digest(settings["categories"]),
            "sample_sizes": {name: len(values) for name, values in split.items()},
            "requested_train_size": settings["sampling"]["total"],
            "sample_category_counts": {
                name: dict(Counter(r["category"] for r in values)) for name, values in split.items()
            },
            "rewrite_failures": [],
            "token_length_stats": {},
        }
        write_json(output / "summary.json", summary)
        for name, values in split.items():
            write_jsonl(output / f"{name}.original.jsonl", values)
        progress.update("Stage 4/6: Loading/snapshotting tokenizer; first use may download tokenizer assets")
        tokenizer = data_tokenizer(config)
        tokenizer.save_pretrained(output / "tokenizer")
        for name, values in split.items():
            progress.update(f"Stage 5/6: Computing token lengths for {name}: {len(values)} conversations")
            summary["token_length_stats"][f"{name}.original.jsonl"] = token_length_stats(
                values, tokenizer, progress_description=f"Stage 5/6: Token lengths ({name})"
            )
        progress.update("Stage 6/6: Finalizing saved summaries and selection hashes")
        write_json(output / "summary.json", summary)
        selection = {
            "preparation_sha256": digest(preparation_config(config)),
            "files": {
                f"{name}.original.jsonl": file_digest(output / f"{name}.original.jsonl") for name in split
            },
        }
        write_json(output / "selection.json", selection)
        summary["csv_exports"] = export_prepared_csvs(output)
        write_json(output / "summary.json", summary)
        push_dataset(config)
        progress.update(f"Preparation complete; saved {summary['sample_sizes']} to {output}")
        return summary


def load_data_selection(config):
    root = Path(config["paths"]["data"])
    if not (root / "selection.json").exists():
        raise ValueError(
            "Prepared conversations are missing. Run prepare-data (prepare_training_data.ipynb) first."
        )
    selection = json.loads((root / "selection.json").read_text())
    if selection["preparation_sha256"] != digest(preparation_config(config)):
        raise ValueError(
            "Data preparation settings differ from the saved selection; use the same settings or a new data directory."
        )
    for name, expected in selection["files"].items():
        if file_digest(root / name) != expected:
            raise ValueError(f"Prepared conversations changed after sampling: {name}")
    return {
        name: list(read_jsonl(root / f"{name}.original.jsonl")) for name in ("train", "validation")
    }, selection


def select_rewrite_preview(rows, categories, per_category, seed):
    if not isinstance(per_category, int) or isinstance(per_category, bool) or per_category < 1:
        raise ValueError("rewrite.preview_per_category must be a positive integer")
    rng = random.Random(seed)
    selected = []
    counts = {}
    for category in categories:
        pool = sorted((row for row in rows if row["category"] == category), key=lambda row: row["id"])
        chosen = rng.sample(pool, min(per_category, len(pool)))
        selected.extend(chosen)
        counts[category] = len(chosen)
    return selected, counts


def review_pairs(original, rewritten, split):
    source = {row["id"]: row for row in original}
    pairs = []
    for row in rewritten:
        for index, turn in enumerate(row["messages"]):
            if turn["role"] == "assistant":
                before = source[row["id"]]
                pairs.append(
                    {
                        "id": row["id"],
                        "category": before["category"],
                        "turn": index,
                        "split": split,
                        "user": before["messages"][index - 1]["content"],
                        "history": before["messages"][:index],
                        "original": before["messages"][index]["content"],
                        "rewritten": turn["content"],
                    }
                )
    return pairs


def write_review(path, pairs, category_counts=None):
    """Escaped, standalone side-by-side review; include every assistant turn in a preview."""

    def escape(value):
        return html.escape(str(value))

    body = ""
    if category_counts is not None:
        body += "<h2>Preview conversations by category</h2><ul>"
        body += "".join(
            f"<li>{escape(category)}: {count}</li>" for category, count in category_counts.items()
        )
        body += "</ul>"
    for category in dict.fromkeys(pair["category"] for pair in pairs):
        body += f"<h2>{escape(category)}</h2>"
        for pair in (p for p in pairs if p["category"] == category):
            body += f"<h3>{escape(pair['id'])} · assistant turn {pair['turn']} · {escape(pair['split'])}</h3>"
            body += f"<p><strong>User:</strong> {escape(pair['user'])}</p>"
            body += (
                "<details><summary>Conversation history</summary><pre>"
                + escape(json.dumps(pair["history"], ensure_ascii=False, indent=2))
                + "</pre></details>"
            )
            body += f"<table><tr><th>Original</th><th>Claude rewrite</th></tr><tr><td><pre>{escape(pair['original'])}</pre></td><td><pre>{escape(pair['rewritten'])}</pre></td></tr></table>"
    atomic_text(
        path,
        '<!doctype html><meta charset="utf-8"><title>Rewrite review</title>'
        "<style>body{font:15px system-ui;max-width:1400px;margin:24px auto;padding:0 12px}table{width:100%;table-layout:fixed;border-collapse:collapse}td,th{vertical-align:top;border:1px solid #ccc;padding:12px}pre{white-space:pre-wrap;overflow-wrap:anywhere}h2{margin-top:40px}</style>"
        "<h1>Original and rewritten replies</h1>" + body,
    )
    write_json(Path(path).with_suffix(".json"), pairs)


async def rewrite_data(config, preview=False):
    """Consume frozen originals; live previews and production share the same turn cache."""
    seed_everything(config["seed"])
    split, selection = load_data_selection(config)
    root = Path(config["paths"]["data"])
    per_category = config["rewrite"]["preview_per_category"]
    category_counts = None
    if preview:
        selected, category_counts = select_rewrite_preview(
            split["train"], config["data"]["categories"]["priority"], per_category, config["seed"]
        )
        split = {"preview": selected}
    if not any(split.values()):
        raise ValueError("No conversations available for rewriting")
    stage_config = {
        "stage": "rewrite-preview" if preview else "rewrite-data",
        "seed": config["seed"],
        "smoke": config["smoke"],
        "rewrite": config["rewrite"],
        "selection": selection,
        "mock": config["smoke"] and config["judge"]["backend"] == "mock",
    }
    output = root / "rewrite-previews" / digest(stage_config)[:16] if preview else root / "rewrite"
    with Run(output, stage_config, selection["files"]):
        summary = {
            "synthetic": stage_config["mock"],
            "preview": preview,
            "output": str(output.resolve()),
            "prompt_version": config["rewrite"]["prompt_version"],
            "prompt_sha256": digest(config["rewrite"]["prompt_text"]),
            "sample_sizes": {name: len(rows) for name, rows in split.items()},
            "category_counts": category_counts,
            "requested_per_category": per_category if preview else None,
            "shortfalls": {
                category: per_category - count
                for category, count in (category_counts or {}).items()
                if count < per_category
            },
            "assistant_turns": sum(
                turn["role"] == "assistant"
                for rows in split.values()
                for row in rows
                for turn in row["messages"]
            ),
            "rewrite_failures": [],
            "token_length_stats": {},
        }
        write_json(output / "summary.json", summary)
        if preview:
            write_jsonl(output / "preview.original.jsonl", split["preview"])
        client = Claude(config["rewrite"], Path(config["paths"]["cache"]) / "rewrites.sqlite")
        rewritten = {}
        try:
            for name, rows in split.items():
                results = await asyncio.gather(
                    *(
                        rewrite_conversation(row, client, config["rewrite"], stage_config["mock"])
                        for row in rows
                    ),
                    return_exceptions=True,
                )
                failures = [
                    {"id": row["id"], "error_type": type(result).__name__}
                    for row, result in zip(rows, results, strict=True)
                    if isinstance(result, BaseException)
                ]
                summary["rewrite_failures"].extend(failures)
                summary.setdefault("refused_turns_kept_original", []).extend(
                    {"id": result["id"], **refusal}
                    for result in results
                    if not isinstance(result, BaseException)
                    for refusal in result.get("rewrite_refusals", [])
                )
                rewritten[name] = [result for result in results if not isinstance(result, BaseException)]
                write_jsonl(output / f"{name}.rewritten.jsonl", rewritten[name])
                write_json(output / "summary.json", summary)
        finally:
            await client.close()
        pairs = [pair for name in split for pair in review_pairs(split[name], rewritten[name], name)]
        if preview:
            write_review(output / "review.html", pairs, category_counts)
            summary["review_html"] = str((output / "review.html").resolve())
        if summary["rewrite_failures"]:
            write_json(output / "summary.json", summary)
            raise RuntimeError(
                f"{len(summary['rewrite_failures'])} rewrites failed; inspect {output / 'summary.json'} and rerun to resume cached turns."
            )
        tokenizer = data_tokenizer(config)
        for name, rows in rewritten.items():
            summary["token_length_stats"][f"{name}.rewritten.jsonl"] = token_length_stats(rows, tokenizer)
        if not preview:
            for name, rows in rewritten.items():
                write_jsonl(root / f"{name}.rewritten.jsonl", rows)
            combined = json.loads((root / "summary.json").read_text())
            combined["rewrite_failures"] = []
            combined["refused_turns_kept_original"] = summary["refused_turns_kept_original"]
            combined["token_length_stats"].update(summary["token_length_stats"])
            write_json(root / "dataset-summary.json", combined)
        write_json(output / "summary.json", summary)
        if not preview:
            push_dataset(config)
        return summary


async def build_data(config):
    """Backward-compatible combined runner; notebooks use the independent stages."""
    summary = await prepare_data(config)
    if config["data"]["rewrite_enabled"]:
        await rewrite_data(config)
        summary = json.loads((Path(config["paths"]["data"]) / "dataset-summary.json").read_text())
    return summary


def spot_check(config, count=50):
    root = Path(config["paths"]["data"])
    pairs = []
    for split in ("train", "validation"):
        original = {row["id"]: row for row in read_jsonl(root / f"{split}.original.jsonl")}
        for row in read_jsonl(root / f"{split}.rewritten.jsonl"):
            for index, turn in enumerate(row["messages"]):
                if turn["role"] == "assistant":
                    pairs.append(
                        {
                            "id": row["id"],
                            "turn": index,
                            "split": split,
                            "original": original[row["id"]]["messages"][index]["content"],
                            "rewritten": turn["content"],
                        }
                    )
    pairs = random.Random(config["seed"]).sample(pairs, min(count, len(pairs)))
    rows = "".join(
        f"<tr><td>{html.escape(p['id'])}:{p['turn']}</td><td><pre>{html.escape(p['original'])}</pre></td><td><pre>{html.escape(p['rewritten'])}</pre></td></tr>"
        for p in pairs
    )
    from .artifacts import atomic_text

    path = root / "spot-check.html"
    atomic_text(
        path,
        '<!doctype html><meta charset="utf-8"><title>Rewrite spot check</title><style>td{vertical-align:top;width:45%;border:1px solid #ccc;padding:12px}pre{white-space:pre-wrap}</style><table><tr><th>ID:turn</th><th>Original</th><th>Rewritten</th></tr>'
        + rows
        + "</table>",
    )
    write_json(root / "spot-check.json", pairs)
    return path
