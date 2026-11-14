"""Inspect real source schemas, resolve revisions, and materialize repeatable input snapshots."""

import csv
import json
import os
import urllib.request
from pathlib import Path

from .artifacts import atomic_text, digest, file_digest, read_jsonl, write_json


def require(row, fields, source):
    if not isinstance(row, dict) or not set(fields) <= row.keys():
        actual = sorted(row) if isinstance(row, dict) else type(row).__name__
        raise ValueError(
            f"Unexpected {source} schema: required {fields}; observed {actual}. Run inspect first."
        )


def inspect_hf(settings):
    from huggingface_hub import HfApi

    api = HfApi(token=os.environ.get("HF_TOKEN") or False)
    info = api.dataset_info(settings["repo"], revision=settings.get("revision", "main"))
    return {
        "repo": settings["repo"],
        "revision": info.sha,
        "files": api.list_repo_files(settings["repo"], repo_type="dataset", revision=info.sha),
    }


def load_source(settings, cache_dir):
    """Return iterable and provenance; downloaded files are cached and hash recorded."""
    cache_dir = Path(cache_dir) / "sources"
    cache_dir.mkdir(parents=True, exist_ok=True)
    local = settings.get("local_path")
    if local:
        path = Path(local)
        kind = "jsonl" if path.suffix == ".jsonl" else "csv" if path.suffix == ".csv" else "json"
        meta = {"local_path": str(path.resolve()), "sha256": file_digest(path)}
    elif settings["kind"].startswith("hf"):
        snapshot_path = cache_dir / (digest(settings) + ".json")
        # First resolution is frozen until this source cache entry is explicitly removed.
        if snapshot_path.exists():
            listing = json.loads(snapshot_path.read_text())
        else:
            listing = inspect_hf(settings)
            write_json(snapshot_path, listing)
        meta = listing
        if settings["kind"] == "hf_json":
            filename = settings["file"]
            if filename not in listing["files"]:
                raise ValueError(f"File {filename!r} not found; actual files: {listing['files']}")
            from huggingface_hub import hf_hub_download

            path = Path(
                hf_hub_download(
                    settings["repo"],
                    filename,
                    repo_type="dataset",
                    revision=listing["revision"],
                    token=os.environ.get("HF_TOKEN") or False,
                )
            )
            meta = {**meta, "file": filename, "sha256": file_digest(path)}
            kind = "json"
        else:
            from datasets import load_dataset

            dataset = load_dataset(
                settings["repo"],
                settings["config"],
                split=settings["split"],
                revision=listing["revision"],
                token=os.environ.get("HF_TOKEN") or False,
            )
            if len(dataset) == 0:
                raise ValueError(f"Empty dataset {settings['repo']}")
            return dataset, {
                **meta,
                "config": settings["config"],
                "split": settings["split"],
                "features": dataset.features.to_dict(),
                "fingerprint": dataset._fingerprint,
            }
    else:
        kind = settings["kind"].removeprefix("url_")
        path = cache_dir / (digest(settings) + "." + kind)
        if not path.exists():
            with urllib.request.urlopen(settings["url"], timeout=120) as response:
                atomic_text(path, response.read().decode("utf-8"))
        meta = {"url": settings["url"], "sha256": file_digest(path), "revision": settings.get("revision")}
    if kind == "jsonl":
        return read_jsonl(path), meta
    if kind == "csv":
        # Read eagerly so no open file escapes this function.
        with path.open(newline="") as f:
            return list(csv.DictReader(f)), meta
    if kind == "json":
        import ijson

        def rows():
            with path.open("rb") as f:
                first = f.read(1)
                f.seek(0)
                if first != b"[":
                    raise ValueError(f"Expected a top-level JSON array in {path}")
                yield from ijson.items(f, "item")

        return rows(), meta
    raise ValueError(f"Unsupported source kind: {kind}")
