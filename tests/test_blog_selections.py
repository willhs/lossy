"""Tests for tools/blog_selections.py — pure storage/validation module."""
import json
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tools"))
import blog_selections as bs  # noqa: E402


@pytest.mark.req("REQ-001")
def test_load_missing_file_returns_empty(tmp_path):
    assert bs.load_selections(str(tmp_path / "missing.json")) == {"entries": []}


@pytest.mark.req("REQ-002")
def test_load_malformed_raises(tmp_path):
    p = tmp_path / "bad.json"
    p.write_text("not json")
    with pytest.raises(ValueError):
        bs.load_selections(str(p))


@pytest.mark.req("REQ-002")
def test_load_missing_entries_key_raises(tmp_path):
    p = tmp_path / "nokey.json"
    p.write_text(json.dumps({"other": []}))
    with pytest.raises(ValueError):
        bs.load_selections(str(p))


@pytest.mark.req("REQ-003")
def test_save_is_atomic(tmp_path):
    p = tmp_path / "sel.json"
    bs.save_selections(str(p), {"entries": [_entry(42)]})
    # Re-read and confirm round-trip
    assert bs.load_selections(str(p))["entries"][0]["shot_idx"] == 42
    # No leftover tmp file
    assert not any(name.endswith(".tmp") for name in os.listdir(tmp_path))


@pytest.mark.req("REQ-004")
def test_upsert_inserts_new():
    data = {"entries": []}
    bs.upsert_entry(data, _entry(42))
    assert len(data["entries"]) == 1


@pytest.mark.req("REQ-004")
def test_upsert_replaces_same_key():
    data = {"entries": [_entry(42, label="old")]}
    bs.upsert_entry(data, _entry(42, label="new"))
    assert len(data["entries"]) == 1
    assert data["entries"][0]["label"] == "new"


@pytest.mark.req("REQ-005")
def test_upsert_preserves_order():
    data = {"entries": [_entry(1), _entry(2), _entry(3)]}
    bs.upsert_entry(data, _entry(2, label="updated"))
    assert [e["shot_idx"] for e in data["entries"]] == [1, 2, 3]


@pytest.mark.req("REQ-006")
def test_delete_matching_returns_true():
    data = {"entries": [_entry(42)]}
    assert bs.delete_entry(data, "star_wars_iv_v2", 42) is True
    assert data["entries"] == []


@pytest.mark.req("REQ-006")
def test_delete_missing_returns_false():
    data = {"entries": [_entry(42)]}
    assert bs.delete_entry(data, "star_wars_iv_v2", 999) is False
    assert len(data["entries"]) == 1


@pytest.mark.req("REQ-007")
@pytest.mark.parametrize("field", ["film", "shot_idx", "strategy_a", "strategy_b"])
def test_validate_rejects_missing_required(field):
    e = _entry(42)
    del e[field]
    with pytest.raises(ValueError):
        bs.validate_entry(e)


@pytest.mark.req("REQ-007")
def test_validate_rejects_non_int_shot_idx():
    with pytest.raises(ValueError):
        bs.validate_entry(_entry("forty-two"))


@pytest.mark.req("REQ-008")
def test_validate_accepts_allowed_tag():
    bs.validate_entry(_entry(42, tag="funny_halluc"))


@pytest.mark.req("REQ-008")
def test_validate_rejects_unknown_tag():
    with pytest.raises(ValueError):
        bs.validate_entry(_entry(42, tag="bogus_tag"))


def _entry(shot_idx, **overrides):
    e = {
        "film": "star_wars_iv_v2",
        "shot_idx": shot_idx,
        "strategy_a": "runpod-wan",
        "strategy_b": "runpod-vace",
        "label": "sample",
        "formats": ["mp4", "gif", "png", "prompt"],
        "added_at": "2026-04-18T00:00:00Z",
    }
    e.update(overrides)
    return e
