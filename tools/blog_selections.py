"""Pure storage + validation for blog-clip selections.

Used by tools/serve.py's /api/blog_selections endpoints. Kept free of any
HTTP concerns so it can be unit-tested directly.
"""
import json
import os
import tempfile
from typing import Any

ALLOWED_TAGS = {
    "best_preserve", "worst_fail", "funny_halluc",
    "edge_dialogue", "edge_action", "edge_establish",
}
REQUIRED_FIELDS = ("film", "shot_idx", "strategy_a", "strategy_b")


def load_selections(path: str) -> dict[str, Any]:
    """Return parsed selections doc, or an empty doc if the file is missing."""
    if not os.path.exists(path):
        return {"entries": []}
    with open(path) as f:
        try:
            data = json.load(f)
        except json.JSONDecodeError as e:
            raise ValueError(f"malformed selections file {path}: {e}") from e
    if not isinstance(data, dict) or "entries" not in data:
        raise ValueError(f"selections file {path} missing 'entries' key")
    return data


def save_selections(path: str, data: dict[str, Any]) -> None:
    """Write atomically via tmp + os.replace."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fd, tmp = tempfile.mkstemp(
        prefix=".selections-", suffix=".tmp",
        dir=os.path.dirname(path) or ".",
    )
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(data, f, indent=2)
            f.write("\n")
        os.replace(tmp, path)
    except Exception:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise


def validate_entry(entry: dict[str, Any]) -> None:
    """Raise ValueError if the entry is missing fields or has an unknown tag."""
    for field in REQUIRED_FIELDS:
        if field not in entry:
            raise ValueError(f"entry missing required field: {field}")
    if not isinstance(entry["shot_idx"], int):
        raise ValueError("shot_idx must be an int")
    tag = entry.get("tag")
    if tag is not None and tag not in ALLOWED_TAGS:
        raise ValueError(f"unknown tag: {tag!r} (allowed: {sorted(ALLOWED_TAGS)})")


def upsert_entry(data: dict[str, Any], entry: dict[str, Any]) -> None:
    """Insert entry, or replace one with the same (film, shot_idx) in place."""
    validate_entry(entry)
    key = (entry["film"], entry["shot_idx"])
    entries = data.setdefault("entries", [])
    for i, existing in enumerate(entries):
        if (existing.get("film"), existing.get("shot_idx")) == key:
            entries[i] = entry
            return
    entries.append(entry)


def delete_entry(data: dict[str, Any], film: str, shot_idx: int) -> bool:
    """Remove the matching entry; return True if one was removed."""
    entries = data.setdefault("entries", [])
    for i, existing in enumerate(entries):
        if existing.get("film") == film and existing.get("shot_idx") == shot_idx:
            del entries[i]
            return True
    return False
