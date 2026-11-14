"""Standard-library CSV exports for reviewing frozen, labeled preparation samples."""

import csv
import hashlib
import io
import json
import os
from pathlib import Path
from uuid import uuid4


def _write_csv(path, fields, rows):
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)
    temporary = path.with_name(path.name + "." + uuid4().hex + ".tmp")
    try:
        temporary.write_text(buffer.getvalue(), encoding="utf-8")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def export_prepared_csvs(data_root):
    """Export selected conversations and counts without filtering, sampling, or API calls."""
    root = Path(data_root)
    summary = json.loads((root / "summary.json").read_text())
    selection = json.loads((root / "selection.json").read_text())
    conversations = []
    for split in ("train", "validation"):
        filename = split + ".original.jsonl"
        payload = (root / filename).read_bytes()
        if hashlib.sha256(payload).hexdigest() != selection["files"][filename]:
            raise ValueError("Saved data hash mismatch: " + filename)
        for row in (json.loads(line) for line in payload.splitlines() if line.strip()):
            conversations.append(
                {
                    "split": split,
                    "id": row["id"],
                    "group_id": row["group_id"],
                    "category": row["category"],
                    "message_count": len(row["messages"]),
                    "conversation": "\n\n".join(
                        f"{index + 1}. {message['role'].upper()}\n{message['content']}"
                        for index, message in enumerate(row["messages"])
                    ),
                }
            )
    paths = {
        "conversations": root / "classified_conversations.csv",
        "category_counts": root / "category_counts.csv",
    }
    _write_csv(
        paths["conversations"],
        ["split", "id", "group_id", "category", "message_count", "conversation"],
        conversations,
    )
    _write_csv(
        paths["category_counts"],
        ["category", "eligible_pool", "train", "validation"],
        (
            {
                "category": category,
                "eligible_pool": summary["category_counts"].get(category, 0),
                "train": summary["sample_category_counts"]["train"].get(category, 0),
                "validation": summary["sample_category_counts"]["validation"].get(category, 0),
            }
            for category in summary["classification"]["priority"]
        ),
    )
    return {name: str(path) for name, path in paths.items()}
