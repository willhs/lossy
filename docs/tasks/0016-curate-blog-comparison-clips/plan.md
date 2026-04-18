---
id: plan-0016
type: spec
purpose: "Implementation plan for curating 2-5 side-by-side showcase clips and producing MP4/GIF/PNG/prompt assets for the upcoming lossy-codec blog post, via a reusable favorite-and-export workflow."
tags: ["plan", "blog", "tools", "compare", "export", "curation"]
related: ["./task.md"]
---

# Curate Comparison Showcase Clips — Implementation Plan

## Overview

Build a reusable favorite-and-export loop on top of `tools/compare.html`, use it to curate 2-5 blog showcase clips across multiple films, and produce four web-optimized assets per clip (MP4, looped GIF, keyframe PNG, prompt JSON) into `docs/blog/assets/`.

**Primary Goal**: Land 2-5 curated entries in `docs/blog/selections.json` spanning ≥2 films, with matching assets that meet the size budget (GIF <2 MB, MP4 <5 MB per clip) and are ready for the blog post author to drop in.

**Approach**:
1. Extract selection CRUD into a pure `tools/blog_selections.py` module (easily unit-testable).
2. Add `/api/blog_selections` endpoints to `tools/serve.py` as a thin wrapper.
3. Extend `tools/compare.html` with a favorite button + tag/label modal and visual markers on favorited shots.
4. Build `tools/blog_export.py` — a standalone exporter that shells out to ffmpeg (no new deps) using patterns matching `stitch.py`.
5. Author a Claude skill so future blog posts replay the flow.
6. Run the actual curation + export and commit the asset set.

## Current State Analysis

### Key Discoveries

- `tools/serve.py:103-119` exposes exactly one API route (`GET /api/scan`) on a `SimpleHTTPRequestHandler` subclass. It already does threaded serving + range requests — adding new routes is mechanical.
- `tools/compare.html` has no persistence other than `localStorage` (`SETTINGS_KEY` at line ~1001). Sidebar shot items and timeline shots are already keyed by shot index (`shot-${i}` / `tl-${i}`), so decorating favorited shots is purely additive CSS + class toggling.
- Pane state (`paneA`, `paneB`) carries the active strategy via `strategySelectA.selectedOptions[0].dataset.strategyName`, and the clip map includes per-shot paths (`adjusted/<strategy>/<shot_idx:04d>.mp4`) — these are exactly the inputs the exporter needs.
- `stitch.py` is the project's ffmpeg pattern: pure `subprocess.run(["ffmpeg", ...], capture_output=True)` calls, no extra deps.
- `output/<film>/manifest.json` provides `scenes[i].start_s/end_s/duration_s/keyframes[]` — the exporter reads these to size clips and locate keyframes.
- `output/<film>/prompts.json` has format `{format: "v2", shots: [...]}` — the prompt snippet comes from here.
- No test file exists for `tools/serve.py`; the project uses pytest with `tests/test_*.py` and mocks external APIs (`CLAUDE.md`).
- `.gent/skills/generate/SKILL.md` is the skill format template: `---name/description---` frontmatter + markdown body.
- `docs/blog/` does not yet exist.

## Desired End State

- `docs/blog/selections.json` exists with 2-5 entries spanning ≥2 films and varied tags.
- `docs/blog/assets/` contains `<film>_<shot_idx>_<tag>.{mp4,gif,png,json}` for every entry; every GIF <2 MB; every MP4 <5 MB.
- `tools/compare.html` has a favorite button that opens a tag/label modal and persists via `/api/blog_selections`; favorited shots are visually marked in both sidebar and timeline.
- `tools/blog_export.py` regenerates all assets idempotently from `selections.json`; `--dry-run`, `--filter-tag`, `--film` flags work.
- `.gent/skills/blog-curate/SKILL.md` documents the serve → favorite → export loop end-to-end.
- `pytest tests/test_blog_selections.py tests/test_blog_export.py` passes; all spec tests (`pytest -m req`) green.

## What We're NOT Doing

- Blog post prose / narrative / captions.
- Programmatic shot scoring or auto-ranking.
- Any edits to `tools/ab-compare.html` (out of scope per task).
- Decoder or strategy changes.
- Adding new runtime dependencies — ffmpeg + stdlib only.
- Auth, HTTPS, or production hardening on `tools/serve.py`.
- New MP4 / GIF encoding features beyond the four declared assets.

---

## Phase 0: Spec & Tests (TDD)

### Overview

Write the spec and failing tests first. This covers the two automatable surfaces: the selection storage/API (`tools/blog_selections.py` + `/api/blog_selections`) and the exporter's command-building logic. The UI favorite button and the manual curation run are verified manually in their respective phases.

### Tasks

#### 1. Create spec document
- [ ] Create `docs/design/blog-curate-spec.md`:

```markdown
---
id: spec-blog-curate
type: spec
purpose: "Numbered requirements for the blog-clip curation toolchain (selections API + exporter)."
tags: ["spec", "blog", "tools"]
related: ["../tasks/0016-curate-blog-comparison-clips/task.md", "../tasks/0016-curate-blog-comparison-clips/plan.md"]
---

# SPEC-0016 — Blog curation toolchain

## Selections storage (`tools/blog_selections.py`)

- **REQ-001** `load_selections(path)` returns `{"entries": []}` when the file is missing.
- **REQ-002** `load_selections(path)` raises `ValueError` when the file exists but is malformed JSON or lacks `entries`.
- **REQ-003** `save_selections(path, data)` writes atomically (`tmp` + `os.replace`) so a crash mid-write never corrupts the file.
- **REQ-004** `upsert_entry(data, entry)` inserts a new entry OR replaces an existing one with the same `(film, shot_idx)` key.
- **REQ-005** `upsert_entry` preserves insertion order of untouched entries.
- **REQ-006** `delete_entry(data, film, shot_idx)` removes the matching entry and returns `True`; returns `False` when no match.
- **REQ-007** Entry validation rejects missing required fields (`film`, `shot_idx`, `strategy_a`, `strategy_b`) and rejects non-int `shot_idx`.
- **REQ-008** Entry validation accepts optional `tag` only from the allowed set `{best_preserve, worst_fail, funny_halluc, edge_dialogue, edge_action, edge_establish}`; rejects unknown tags.

## HTTP API (`tools/serve.py`)

- **REQ-010** `GET /api/blog_selections` returns current selections JSON with 200 and `Content-Type: application/json`.
- **REQ-011** `POST /api/blog_selections` with a valid entry body upserts it and returns 200 with the updated document.
- **REQ-012** `POST /api/blog_selections` with an invalid body returns 400 and does not modify the file on disk.
- **REQ-013** `DELETE /api/blog_selections?film=F&shot_idx=N` removes the matching entry and returns 200; missing key returns 404.

## Exporter (`tools/blog_export.py`)

- **REQ-020** `build_hstack_command(entry, clip_a, clip_b, out_path)` returns an ffmpeg argv whose filter_complex contains `hstack=inputs=2` and whose codec flags include `-c:v libx264 -crf 28`.
- **REQ-021** `build_gif_commands(mp4_path, gif_path)` returns a two-stage sequence: palettegen then paletteuse, both at `fps=12`.
- **REQ-022** `build_prompt_snippet(prompts_doc, shot_idx, entry)` returns a JSON-serializable dict that contains `film`, `shot_idx`, `tag`, `label`, and the matching shot's prompt payload.
- **REQ-023** `--dry-run` prints every ffmpeg command it would run and creates no files in `docs/blog/assets/`.
- **REQ-024** `--filter-tag <tag>` processes only entries with that tag; `--film <name>` processes only that film; both filters compose (AND).
- **REQ-025** Output filenames follow `<film>_<shot_idx>_<tag>.{mp4,gif,png,json}` (tag defaults to `untagged` if absent).
```

#### 2. Write failing tests for selections storage
- [ ] Create `tests/test_blog_selections.py`:

```python
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
```

#### 3. Write failing tests for the exporter command builders
- [ ] Create `tests/test_blog_export.py`:

```python
"""Tests for tools/blog_export.py — pure argv/JSON builders."""
import json
import os
import subprocess
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tools"))
import blog_export as be  # noqa: E402


ENTRY = {
    "film": "star_wars_iv_v2",
    "shot_idx": 42,
    "strategy_a": "runpod-wan",
    "strategy_b": "runpod-vace",
    "tag": "funny_halluc",
    "label": "Vader reveal becomes cartoon",
    "formats": ["mp4", "gif", "png", "prompt"],
    "added_at": "2026-04-18T00:00:00Z",
}


@pytest.mark.req("REQ-020")
def test_hstack_command_uses_crf28_and_hstack():
    argv = be.build_hstack_command(ENTRY, "a.mp4", "b.mp4", "out.mp4")
    joined = " ".join(argv)
    assert "hstack=inputs=2" in joined
    assert "-c:v" in argv and "libx264" in argv
    assert "-crf" in argv
    assert argv[argv.index("-crf") + 1] == "28"


@pytest.mark.req("REQ-021")
def test_gif_commands_are_two_stage_12fps():
    cmds = be.build_gif_commands("in.mp4", "out.gif")
    assert len(cmds) == 2
    palette, use = cmds
    assert any("palettegen" in a for a in palette)
    assert any("paletteuse" in a for a in use)
    assert any("fps=12" in a for a in palette + use)


@pytest.mark.req("REQ-022")
def test_prompt_snippet_contains_metadata_and_prompt():
    prompts_doc = {
        "format": "v2",
        "shots": [{"index": 42, "description": {"action": "X"}}],
    }
    snip = be.build_prompt_snippet(prompts_doc, 42, ENTRY)
    assert snip["film"] == "star_wars_iv_v2"
    assert snip["shot_idx"] == 42
    assert snip["tag"] == "funny_halluc"
    assert snip["label"] == "Vader reveal becomes cartoon"
    assert snip["shot"]["index"] == 42


@pytest.mark.req("REQ-022")
def test_prompt_snippet_missing_shot_raises():
    prompts_doc = {"format": "v2", "shots": []}
    with pytest.raises(KeyError):
        be.build_prompt_snippet(prompts_doc, 42, ENTRY)


@pytest.mark.req("REQ-023")
def test_dry_run_creates_no_files(tmp_path, monkeypatch):
    selections = tmp_path / "selections.json"
    selections.write_text(json.dumps({"entries": [ENTRY]}))
    assets = tmp_path / "assets"
    called = []
    monkeypatch.setattr(subprocess, "run", lambda *a, **kw: called.append(a) or _FakeCP())
    be.run_export(
        selections_path=str(selections),
        output_dir=str(tmp_path / "output_stub"),
        assets_dir=str(assets),
        dry_run=True,
    )
    assert not assets.exists() or not any(assets.iterdir())
    assert called == []


@pytest.mark.req("REQ-024")
def test_filter_tag_and_film_compose(tmp_path, monkeypatch):
    entries = [
        {**ENTRY, "shot_idx": 1, "tag": "funny_halluc"},
        {**ENTRY, "shot_idx": 2, "tag": "best_preserve"},
        {**ENTRY, "shot_idx": 3, "tag": "funny_halluc", "film": "bcs_s01e01"},
    ]
    picked = be.filter_entries(entries, tag="funny_halluc", film="star_wars_iv_v2")
    assert [e["shot_idx"] for e in picked] == [1]


@pytest.mark.req("REQ-025")
def test_asset_filenames_follow_convention():
    names = be.asset_filenames(ENTRY)
    assert names["mp4"] == "star_wars_iv_v2_42_funny_halluc.mp4"
    assert names["gif"] == "star_wars_iv_v2_42_funny_halluc.gif"
    assert names["png"] == "star_wars_iv_v2_42_funny_halluc.png"
    assert names["prompt"] == "star_wars_iv_v2_42_funny_halluc.json"


@pytest.mark.req("REQ-025")
def test_asset_filenames_use_untagged_when_missing():
    e = {**ENTRY}
    e.pop("tag")
    assert be.asset_filenames(e)["mp4"] == "star_wars_iv_v2_42_untagged.mp4"


class _FakeCP:
    returncode = 0
    stdout = b""
    stderr = b""
```

#### 4. Write failing tests for the HTTP endpoints
- [ ] Create `tests/test_serve_blog_selections.py`:

```python
"""Tests for /api/blog_selections endpoints in tools/serve.py.

Spins up the actual handler on an ephemeral port to exercise the HTTP layer.
"""
import http.client
import json
import os
import socket
import sys
import threading
import time

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tools"))
import serve  # noqa: E402


@pytest.fixture
def server(tmp_path, monkeypatch):
    selections = tmp_path / "selections.json"
    selections.write_text(json.dumps({"entries": []}))
    monkeypatch.setattr(serve, "SELECTIONS_PATH", str(selections))

    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()

    httpd = serve.ThreadedHTTPServer(("127.0.0.1", port), serve.Handler)
    t = threading.Thread(target=httpd.serve_forever, daemon=True)
    t.start()
    time.sleep(0.05)
    yield port, str(selections)
    httpd.shutdown()


@pytest.mark.req("REQ-010")
def test_get_returns_json(server):
    port, _ = server
    c = http.client.HTTPConnection("127.0.0.1", port)
    c.request("GET", "/api/blog_selections")
    r = c.getresponse()
    assert r.status == 200
    assert r.getheader("Content-Type") == "application/json"
    assert json.loads(r.read())["entries"] == []


@pytest.mark.req("REQ-011")
def test_post_upserts(server):
    port, path = server
    body = json.dumps({
        "film": "star_wars_iv_v2", "shot_idx": 42,
        "strategy_a": "runpod-wan", "strategy_b": "runpod-vace",
        "label": "x", "tag": "funny_halluc",
    })
    c = http.client.HTTPConnection("127.0.0.1", port)
    c.request("POST", "/api/blog_selections", body=body,
              headers={"Content-Type": "application/json"})
    r = c.getresponse()
    assert r.status == 200
    on_disk = json.load(open(path))
    assert on_disk["entries"][0]["shot_idx"] == 42


@pytest.mark.req("REQ-012")
def test_post_invalid_returns_400_and_preserves_disk(server):
    port, path = server
    before = open(path).read()
    c = http.client.HTTPConnection("127.0.0.1", port)
    c.request("POST", "/api/blog_selections", body="{}",
              headers={"Content-Type": "application/json"})
    r = c.getresponse()
    assert r.status == 400
    assert open(path).read() == before


@pytest.mark.req("REQ-013")
def test_delete_removes(server):
    port, path = server
    # Seed one entry
    with open(path, "w") as f:
        json.dump({"entries": [{
            "film": "star_wars_iv_v2", "shot_idx": 42,
            "strategy_a": "runpod-wan", "strategy_b": "runpod-vace",
            "label": "x",
        }]}, f)
    c = http.client.HTTPConnection("127.0.0.1", port)
    c.request("DELETE", "/api/blog_selections?film=star_wars_iv_v2&shot_idx=42")
    assert c.getresponse().status == 200
    assert json.load(open(path))["entries"] == []


@pytest.mark.req("REQ-013")
def test_delete_missing_returns_404(server):
    port, _ = server
    c = http.client.HTTPConnection("127.0.0.1", port)
    c.request("DELETE", "/api/blog_selections?film=nope&shot_idx=1")
    assert c.getresponse().status == 404
```

#### 5. Add the `req` marker to pytest config
- [ ] Edit `pyproject.toml` — add under `[tool.pytest.ini_options]` (create the section if absent):

```toml
[tool.pytest.ini_options]
markers = [
    "req: Link a test to a SPEC requirement ID (REQ-NNN).",
]
```

### Verification

- [ ] Run: `.venv/bin/python -m pytest tests/test_blog_selections.py tests/test_blog_export.py tests/test_serve_blog_selections.py -v` — every test should FAIL (modules do not exist yet).
- [ ] Run: `.venv/bin/python -m pytest tests/ -v --collect-only | grep req` — confirm pytest collects the `@pytest.mark.req` markers without warnings.

---

## Phase 1: Selections schema & storage

### Overview

Create the selections file stub, docs README, and the pure-Python storage module. This makes REQ-001..REQ-008 pass.

### Tasks

#### 1. Create selections stub + blog README
- [x] Create `docs/blog/selections.json`:

```json
{
  "entries": []
}
```

- [x] Create `docs/blog/README.md`:

```markdown
# Blog asset bundle

Curated showcase clips for the lossy-codec blog post, produced by the
compare-favorite-export loop (`tools/compare.html` → `tools/serve.py`'s
`/api/blog_selections` → `tools/blog_export.py`).

## Contents

- `selections.json` — source of truth. Each entry is `{film, shot_idx,
  strategy_a, strategy_b, tag?, label, formats, added_at}`.
- `assets/` — generated per entry: `<film>_<shot_idx>_<tag>.{mp4,gif,png,json}`.

## Workflow

See `.gent/skills/blog-curate/SKILL.md` for the end-to-end loop.

## Size budget

- GIF < 2 MB per clip.
- MP4 < 5 MB per clip.
```

#### 2. Create the selections storage module
- [x] Create `tools/blog_selections.py`:

```python
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
```

### Verification

#### Automated
- [x] Run: `.venv/bin/python -m pytest tests/test_blog_selections.py -v` — all tests for REQ-001..REQ-008 pass.

#### Manual
- [x] Manual: `cat docs/blog/selections.json` shows `{"entries": []}`.
- [x] Manual: `docs/blog/README.md` renders readably (open in editor or preview).

---

## Phase 2: Favorite API endpoints

### Overview

Wire the storage module into `tools/serve.py` as three routes on the existing `Handler`. Satisfies REQ-010..REQ-013.

### Tasks

#### 1. Add the module-level selections path constant
- [x] Edit `tools/serve.py` — add near the top (after the `PORT =` line):

```python
SELECTIONS_PATH = os.path.join(PROJECT_ROOT, "docs", "blog", "selections.json")
```

#### 2. Import the storage module
- [x] Edit `tools/serve.py` — add import:

```python
# after the stdlib imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import blog_selections  # noqa: E402
```

#### 3. Add the route dispatcher helpers
- [x] Edit `tools/serve.py` — add methods to the `Handler` class:

```python
def _send_json(self, status, payload):
    body = json.dumps(payload).encode()
    self.send_response(status)
    self.send_header("Content-Type", "application/json")
    self.send_header("Content-Length", len(body))
    self.end_headers()
    self.wfile.write(body)

def _read_json_body(self):
    length = int(self.headers.get("Content-Length") or 0)
    raw = self.rfile.read(length) if length else b""
    return json.loads(raw.decode() or "{}")
```

#### 4. Wire `GET /api/blog_selections`
- [x] Edit `tools/serve.py` — inside `do_GET`, before the existing `/api/scan` branch:

```python
if self.path == "/api/blog_selections":
    try:
        data = blog_selections.load_selections(SELECTIONS_PATH)
    except ValueError as e:
        self._send_json(500, {"error": str(e)})
        return
    self._send_json(200, data)
    return
```

#### 5. Add `do_POST` to the handler
- [x] Edit `tools/serve.py` — add method to `Handler`:

```python
def do_POST(self):
    if self.path == "/api/blog_selections":
        try:
            entry = self._read_json_body()
            blog_selections.validate_entry(entry)
        except (ValueError, json.JSONDecodeError) as e:
            self._send_json(400, {"error": str(e)})
            return
        data = blog_selections.load_selections(SELECTIONS_PATH)
        blog_selections.upsert_entry(data, entry)
        blog_selections.save_selections(SELECTIONS_PATH, data)
        self._send_json(200, data)
        return
    self.send_error(404)
```

#### 6. Add `do_DELETE` to the handler
- [x] Edit `tools/serve.py` — add method to `Handler`:

```python
def do_DELETE(self):
    from urllib.parse import urlparse, parse_qs
    parsed = urlparse(self.path)
    if parsed.path == "/api/blog_selections":
        q = parse_qs(parsed.query)
        film = (q.get("film") or [None])[0]
        shot_idx_raw = (q.get("shot_idx") or [None])[0]
        if not film or shot_idx_raw is None:
            self._send_json(400, {"error": "film and shot_idx query params required"})
            return
        try:
            shot_idx = int(shot_idx_raw)
        except ValueError:
            self._send_json(400, {"error": "shot_idx must be an int"})
            return
        data = blog_selections.load_selections(SELECTIONS_PATH)
        removed = blog_selections.delete_entry(data, film, shot_idx)
        if not removed:
            self._send_json(404, {"error": "no matching entry"})
            return
        blog_selections.save_selections(SELECTIONS_PATH, data)
        self._send_json(200, data)
        return
    self.send_error(404)
```

### Verification

#### Automated
- [x] Run: `.venv/bin/python -m pytest tests/test_serve_blog_selections.py -v` — REQ-010..REQ-013 pass.
- [x] Run: `.venv/bin/python -m pytest tests/ -v` — full suite still green.

#### Manual
- [x] Manual: start `python tools/serve.py`, then `curl http://localhost:8000/api/blog_selections` returns `{"entries": []}`.
- [x] Manual: `curl -XPOST -H 'Content-Type: application/json' http://localhost:8000/api/blog_selections -d '{"film":"star_wars_iv_v2","shot_idx":42,"strategy_a":"runpod-wan","strategy_b":"runpod-vace","label":"test","tag":"funny_halluc"}'` returns 200 and `docs/blog/selections.json` now contains the entry.
- [x] Manual: `curl -XDELETE 'http://localhost:8000/api/blog_selections?film=star_wars_iv_v2&shot_idx=42'` removes it.

---

## Phase 3: Favorite UI in compare.html

### Overview

Add a favorite button + tag/label modal, persist via the endpoints from Phase 2, and mark favorited shots in the sidebar and timeline. UI-only; no new automated tests (covered by manual checks).

### Tasks

#### 1. Add the favorite button next to `btn-audio`
- [x] Edit `tools/compare.html` controls row (around line 369, after `btn-audio` and before the volume slider):

```html
<button id="btn-favorite" title="Favorite this shot for blog export (F)">&#9733; Favorite</button>
```

#### 2. Add the tag/label modal markup
- [x] Edit `tools/compare.html` — insert before the closing `</body>` (around line 1253):

```html
<div class="setup-overlay hidden" id="fav-overlay">
  <div class="setup-box">
    <h2>Favorite shot</h2>
    <p id="fav-meta"></p>
    <div class="field">
      <label>Label (free-form)</label>
      <input id="fav-label" type="text" placeholder="e.g. Vader reveal becomes cartoon">
    </div>
    <div class="field">
      <label>Tag (optional)</label>
      <select id="fav-tag">
        <option value="">(no tag)</option>
        <option value="best_preserve">best_preserve</option>
        <option value="worst_fail">worst_fail</option>
        <option value="funny_halluc">funny_halluc</option>
        <option value="edge_dialogue">edge_dialogue</option>
        <option value="edge_action">edge_action</option>
        <option value="edge_establish">edge_establish</option>
      </select>
    </div>
    <button id="fav-save">Save</button>
    <button id="fav-cancel" style="background:#333;color:#ddd;margin-left:8px">Cancel</button>
    <button id="fav-delete" style="background:#833;color:#fdd;margin-left:8px">Remove</button>
    <div class="error" id="fav-error"></div>
  </div>
</div>
```

#### 3. Add CSS for the favorite markers
- [x] Edit `tools/compare.html` — append inside the existing `<style>` block (around line 323):

```css
.shot-item.favorited { box-shadow: inset 3px 0 0 #fd4; }
.shot-item .fav-mark { color: #fd4; font-size: 11px; margin-left: 4px; }
.timeline-shot.favorited {
  background: linear-gradient(to bottom, rgba(253,212,68,0.35) 0%, rgba(253,212,68,0.15) 100%);
  border-right-color: rgba(253,212,68,0.9);
}
```

#### 4. Add favorites state + fetch on init
- [x] Edit `tools/compare.html` — near the top of the script block (after `let scenes = [];` around line 444):

```javascript
// Map of "<film>|<shot_idx>" -> entry, for current film only
let favorites = {};
const favOverlay   = document.getElementById('fav-overlay');
const favLabel     = document.getElementById('fav-label');
const favTag       = document.getElementById('fav-tag');
const favMeta      = document.getElementById('fav-meta');
const favError     = document.getElementById('fav-error');
const btnFavorite  = document.getElementById('btn-favorite');

function favKey(film, idx) { return `${film}|${idx}`; }
function currentFilmKey() {
  return currentSourceFile.replace(/\.[^.]+$/, '');
}

async function reloadFavorites() {
  try {
    const r = await fetch('/api/blog_selections');
    const doc = await r.json();
    favorites = {};
    const film = currentFilmKey();
    for (const e of (doc.entries || [])) {
      if (e.film === film) favorites[favKey(e.film, e.shot_idx)] = e;
    }
    applyFavoriteMarkers();
  } catch (e) { console.warn('[lossy] failed to load favorites', e); }
}

function applyFavoriteMarkers() {
  const film = currentFilmKey();
  document.querySelectorAll('.shot-item').forEach(el => el.classList.remove('favorited'));
  document.querySelectorAll('.timeline-shot').forEach(el => el.classList.remove('favorited'));
  for (const idx of Object.keys(favorites).map(k => parseInt(k.split('|')[1]))) {
    const item = document.getElementById(`shot-${idx}`);
    if (item) item.classList.add('favorited');
    const tl = document.getElementById(`tl-${idx}`);
    if (tl) tl.classList.add('favorited');
  }
}
```

#### 5. Wire open/save/delete handlers
- [x] Edit `tools/compare.html` — add before the keydown handler (around line 1216):

```javascript
function openFavoriteModal() {
  if (activeShot < 0) return;
  const film = currentFilmKey();
  const existing = favorites[favKey(film, activeShot)];
  favLabel.value = existing?.label || '';
  favTag.value   = existing?.tag   || '';
  favError.style.display = 'none';
  favMeta.textContent = `${film} — shot ${activeShot} (A=${strategySelectA.selectedOptions[0]?.dataset.strategyName}, B=${strategySelectB.selectedOptions[0]?.dataset.strategyName})`;
  document.getElementById('fav-delete').style.display = existing ? '' : 'none';
  favOverlay.classList.remove('hidden');
}

async function saveFavorite() {
  const film = currentFilmKey();
  const entry = {
    film,
    shot_idx: activeShot,
    strategy_a: strategySelectA.selectedOptions[0]?.dataset.strategyName || '',
    strategy_b: strategySelectB.selectedOptions[0]?.dataset.strategyName || '',
    label: favLabel.value.trim(),
    tag: favTag.value || undefined,
    formats: ["mp4", "gif", "png", "prompt"],
    added_at: new Date().toISOString(),
  };
  if (!entry.strategy_a || !entry.strategy_b) {
    favError.textContent = 'Both A and B strategies must be selected';
    favError.style.display = 'block';
    return;
  }
  // Drop undefined tag so validation accepts absence
  if (!entry.tag) delete entry.tag;
  const r = await fetch('/api/blog_selections', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify(entry),
  });
  if (!r.ok) {
    const body = await r.json().catch(() => ({}));
    favError.textContent = body.error || `HTTP ${r.status}`;
    favError.style.display = 'block';
    return;
  }
  favOverlay.classList.add('hidden');
  await reloadFavorites();
}

async function deleteFavorite() {
  const film = currentFilmKey();
  const r = await fetch(`/api/blog_selections?film=${encodeURIComponent(film)}&shot_idx=${activeShot}`, {
    method: 'DELETE',
  });
  if (!r.ok && r.status !== 404) {
    favError.textContent = `HTTP ${r.status}`;
    favError.style.display = 'block';
    return;
  }
  favOverlay.classList.add('hidden');
  await reloadFavorites();
}

btnFavorite.addEventListener('click', openFavoriteModal);
document.getElementById('fav-save').addEventListener('click', saveFavorite);
document.getElementById('fav-delete').addEventListener('click', deleteFavorite);
document.getElementById('fav-cancel').addEventListener('click', () => favOverlay.classList.add('hidden'));
```

#### 6. Bind `F` key + refresh on strategy/source change
- [x] Edit `tools/compare.html` — inside the existing `keydown` switch (around line 1216), add a new case:

```javascript
case 'f':
case 'F':
  e.preventDefault();
  openFavoriteModal();
  break;
```

- [x] Edit `tools/compare.html` — call `reloadFavorites()` at the end of `init()` (inside the `.then` callback, after `seekBoth(...)` around line 969):

```javascript
reloadFavorites();
```

- [x] Edit `tools/compare.html` — at the end of `buildShotList()` and `buildTimeline()`, call `applyFavoriteMarkers()` (so re-renders on strategy change re-apply markers).

### Verification

#### Manual
- [ ] Manual: start `python tools/serve.py`, open `http://localhost:8000/tools/compare.html`, select a source, press `F` on a shot, enter label + tag, save. Re-open the modal on the same shot — existing values repopulate.
- [ ] Manual: favorited shot shows the yellow side-stripe in the sidebar and a brighter band in the timeline.
- [ ] Manual: refresh the page — markers reappear (state was persisted to `docs/blog/selections.json`).
- [ ] Manual: click **Remove** in the modal — entry disappears from `docs/blog/selections.json` and markers clear.
- [ ] Manual: swap strategies A/B — markers still show for the same shot (persistence is not strategy-scoped).

---

## Phase 4: Exporter

### Overview

`tools/blog_export.py` turns each selection into the four assets. Makes REQ-020..REQ-025 pass.

### Tasks

#### 1. Create the exporter skeleton with pure builders
- [ ] Create `tools/blog_export.py`:

```python
"""Export curated blog clips from docs/blog/selections.json.

Reads selections, resolves per-shot clip paths under output/<film>/adjusted/<strategy>/,
and shells out to ffmpeg (matching the pattern in stitch.py) to produce four assets
per entry: hstack MP4, looped GIF, keyframe PNG, prompt-snippet JSON.

Usage:
    python tools/blog_export.py                       # all entries
    python tools/blog_export.py --dry-run             # print commands
    python tools/blog_export.py --film star_wars_iv_v2
    python tools/blog_export.py --filter-tag funny_halluc
"""
import argparse
import json
import os
import subprocess
import sys
from typing import Any

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import blog_selections  # noqa: E402

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_SELECTIONS = os.path.join(PROJECT_ROOT, "docs", "blog", "selections.json")
DEFAULT_ASSETS_DIR = os.path.join(PROJECT_ROOT, "docs", "blog", "assets")
DEFAULT_OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output")

GIF_MAX_BYTES = 2 * 1024 * 1024
MP4_MAX_BYTES = 5 * 1024 * 1024

DEFAULT_SCALE_WIDTH = 640  # per-pane before hstack


def asset_filenames(entry: dict[str, Any]) -> dict[str, str]:
    stem = f"{entry['film']}_{entry['shot_idx']}_{entry.get('tag') or 'untagged'}"
    return {ext: f"{stem}.{ext if ext != 'prompt' else 'json'}" for ext in ("mp4", "gif", "png", "prompt")}


def filter_entries(entries, tag=None, film=None):
    out = entries
    if tag:
        out = [e for e in out if e.get("tag") == tag]
    if film:
        out = [e for e in out if e.get("film") == film]
    return list(out)


def _resolve_clip(output_dir: str, film: str, strategy: str, shot_idx: int) -> str:
    """Prefer adjusted/<strategy>/<idx:04d>.mp4; fall back to clips/<strategy>/..."""
    adj = os.path.join(output_dir, film, "adjusted", strategy, f"{shot_idx:04d}.mp4")
    if os.path.exists(adj):
        return adj
    raw = os.path.join(output_dir, film, "clips", strategy, f"{shot_idx:04d}.mp4")
    if os.path.exists(raw):
        return raw
    raise FileNotFoundError(f"no clip for {film}/{strategy}/{shot_idx:04d}.mp4")


def build_hstack_command(entry: dict, clip_a: str, clip_b: str, out_path: str) -> list[str]:
    """Side-by-side A|B MP4, h264 crf 28, scaled per pane to DEFAULT_SCALE_WIDTH."""
    w = DEFAULT_SCALE_WIDTH
    filt = (
        f"[0:v]scale={w}:-2,setsar=1[a];"
        f"[1:v]scale={w}:-2,setsar=1[b];"
        f"[a][b]hstack=inputs=2[v]"
    )
    return [
        "ffmpeg", "-y",
        "-i", clip_a,
        "-i", clip_b,
        "-filter_complex", filt,
        "-map", "[v]",
        "-c:v", "libx264", "-crf", "28", "-preset", "medium",
        "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",
        "-an",
        out_path,
    ]


def build_gif_commands(mp4_path: str, gif_path: str, fps: int = 12,
                       width: int = 640) -> list[list[str]]:
    """Two-pass palettegen + paletteuse. Input is already the hstack MP4."""
    palette = gif_path + ".palette.png"
    gen = [
        "ffmpeg", "-y", "-i", mp4_path,
        "-vf", f"fps={fps},scale={width}:-2:flags=lanczos,palettegen=stats_mode=diff",
        palette,
    ]
    use = [
        "ffmpeg", "-y", "-i", mp4_path, "-i", palette,
        "-lavfi", f"fps={fps},scale={width}:-2:flags=lanczos[x];[x][1:v]paletteuse=dither=bayer:bayer_scale=5",
        "-loop", "0",
        gif_path,
    ]
    return [gen, use]


def build_keyframe_command(mp4_path: str, png_path: str) -> list[str]:
    """Grab a representative still from the middle of the hstack MP4."""
    return [
        "ffmpeg", "-y", "-i", mp4_path,
        "-vf", "select='eq(n,0)'+between(t,0,9999)",
        "-ss", "00:00:00.5",
        "-frames:v", "1", "-update", "1",
        png_path,
    ]


def build_prompt_snippet(prompts_doc: dict, shot_idx: int, entry: dict) -> dict:
    shots = prompts_doc.get("shots") or []
    match = next((s for s in shots if s.get("index") == shot_idx), None)
    if match is None:
        raise KeyError(f"no prompt for shot_idx={shot_idx}")
    return {
        "film": entry["film"],
        "shot_idx": shot_idx,
        "strategy_a": entry["strategy_a"],
        "strategy_b": entry["strategy_b"],
        "tag": entry.get("tag"),
        "label": entry.get("label"),
        "shot": match,
    }


def _run(cmd: list[str], dry_run: bool) -> None:
    if dry_run:
        print(" ".join(cmd))
        return
    r = subprocess.run(cmd, capture_output=True)
    if r.returncode != 0:
        sys.stderr.write(r.stderr.decode(errors="replace"))
        raise RuntimeError(f"ffmpeg failed: {cmd[:3]}...")


def run_export(
    selections_path: str = DEFAULT_SELECTIONS,
    output_dir: str = DEFAULT_OUTPUT_DIR,
    assets_dir: str = DEFAULT_ASSETS_DIR,
    tag: str | None = None,
    film: str | None = None,
    dry_run: bool = False,
) -> None:
    data = blog_selections.load_selections(selections_path)
    entries = filter_entries(data.get("entries", []), tag=tag, film=film)
    if not entries:
        print("(no entries matched filters)")
        return
    if not dry_run:
        os.makedirs(assets_dir, exist_ok=True)

    for entry in entries:
        names = asset_filenames(entry)
        mp4 = os.path.join(assets_dir, names["mp4"])
        gif = os.path.join(assets_dir, names["gif"])
        png = os.path.join(assets_dir, names["png"])
        js  = os.path.join(assets_dir, names["prompt"])

        clip_a = _resolve_clip(output_dir, entry["film"], entry["strategy_a"], entry["shot_idx"])
        clip_b = _resolve_clip(output_dir, entry["film"], entry["strategy_b"], entry["shot_idx"])

        _run(build_hstack_command(entry, clip_a, clip_b, mp4), dry_run)
        for cmd in build_gif_commands(mp4, gif):
            _run(cmd, dry_run)
        _run(build_keyframe_command(mp4, png), dry_run)

        # Prompt snippet
        prompts_path = os.path.join(output_dir, entry["film"], "prompts.json")
        with open(prompts_path) as f:
            prompts_doc = json.load(f)
        snippet = build_prompt_snippet(prompts_doc, entry["shot_idx"], entry)
        if dry_run:
            print(f"# write {js}")
        else:
            with open(js, "w") as f:
                json.dump(snippet, f, indent=2)

        if not dry_run:
            _check_size(mp4, MP4_MAX_BYTES, "MP4")
            _check_size(gif, GIF_MAX_BYTES, "GIF")


def _check_size(path: str, limit: int, label: str) -> None:
    size = os.path.getsize(path)
    rel = os.path.relpath(path, PROJECT_ROOT)
    status = "OK" if size < limit else "OVER BUDGET"
    print(f"  [{status}] {label} {rel}: {size/1024:.1f} KiB (budget {limit/1024:.0f} KiB)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--selections", default=DEFAULT_SELECTIONS)
    ap.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    ap.add_argument("--assets-dir", default=DEFAULT_ASSETS_DIR)
    ap.add_argument("--filter-tag", default=None)
    ap.add_argument("--film", default=None)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    run_export(
        selections_path=args.selections,
        output_dir=args.output_dir,
        assets_dir=args.assets_dir,
        tag=args.filter_tag,
        film=args.film,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
```

### Verification

#### Automated
- [ ] Run: `.venv/bin/python -m pytest tests/test_blog_export.py -v` — REQ-020..REQ-025 pass.
- [ ] Run: `.venv/bin/python -m pytest tests/ -v` — full suite green.

#### Manual
- [ ] Manual: with at least one entry present, `python tools/blog_export.py --dry-run` prints the ffmpeg argv for every asset and writes nothing to `docs/blog/assets/`.
- [ ] Manual: `python tools/blog_export.py` (non-dry) produces the four files for each entry and prints `[OK]` / `[OVER BUDGET]` lines.

---

## Phase 5: Size tuning

### Overview

Verify every GIF <2 MB and every MP4 <5 MB. Adjust the levers until every asset lands in budget.

### Tasks

#### 1. Inspect current budgets after a real run
- [ ] Run: `.venv/bin/python tools/blog_export.py` and note any `[OVER BUDGET]` lines.

#### 2. GIF lever pass (only if any GIF >2 MB)
- [ ] Edit `tools/blog_export.py` — lower `width=480` and/or `fps=10` inside `build_gif_commands` defaults, OR expose them as `--gif-width` / `--gif-fps` CLI flags if multiple entries need different settings:

```python
ap.add_argument("--gif-width", type=int, default=640)
ap.add_argument("--gif-fps", type=int, default=12)
```

#### 3. MP4 lever pass (only if any MP4 >5 MB)
- [ ] Edit `tools/blog_export.py` — raise `-crf 28` toward `30` or add `--mp4-crf` flag for per-run tuning:

```python
ap.add_argument("--mp4-crf", type=int, default=28)
```

### Verification

#### Automated
- [ ] Run: `.venv/bin/python tools/blog_export.py` — zero `[OVER BUDGET]` lines printed.
- [ ] Run: `ls -la docs/blog/assets/*.gif | awk '{print $5, $NF}'` — every size < 2097152.
- [ ] Run: `ls -la docs/blog/assets/*.mp4 | awk '{print $5, $NF}'` — every size < 5242880.

#### Manual
- [ ] Manual: open each generated MP4 in QuickTime / VLC — playback smooth, side-by-side clearly shows both panes.
- [ ] Manual: open each GIF in a browser — loops cleanly, frame rate acceptable, no severe banding.

---

## Phase 6: Claude skill

### Overview

Document the end-to-end loop so future blog posts (or anyone returning to this one) can replay it from a single entry point.

### Tasks

#### 1. Create the skill file
- [ ] Create `.gent/skills/blog-curate/SKILL.md`:

```markdown
---
name: blog-curate
description: "Curate side-by-side comparison clips for a lossy blog post. Spins up the comparator, records favorites via /api/blog_selections, and exports per-entry MP4/GIF/PNG/prompt assets into docs/blog/assets/. Use when the user wants to pick or regenerate blog showcase clips."
---

# Blog curate

End-to-end loop for turning decoded films into a web-ready asset bundle for the lossy blog post.

## When to use

- The user wants to pick 2-5 comparison clips across films and produce MP4/GIF/PNG/prompt assets.
- The user wants to regenerate the asset bundle after editing `docs/blog/selections.json`.
- The user wants to re-tune GIF/MP4 size after adding new entries.

## Workflow

### 1. Browse and favorite

```bash
.venv/bin/python tools/serve.py
# open http://localhost:8000/tools/compare.html
```

- Pick source → pick strategies A and B (the comparison you want to showcase).
- Press `F` (or click **★ Favorite**) on any interesting shot.
- Enter a short free-form `label` and optionally pick a tag from the dropdown
  (`best_preserve`, `worst_fail`, `funny_halluc`, `edge_dialogue`, `edge_action`, `edge_establish`).
- Favorited shots show a yellow side-stripe in the sidebar and a bright band in the timeline.
- Repeat across multiple films — aim for breadth over depth (at least 2 different films, 2-5 total).

### 2. Inspect selections

```bash
cat docs/blog/selections.json
```

Each entry is `{film, shot_idx, strategy_a, strategy_b, tag?, label, formats, added_at}`.

### 3. Dry-run the exporter

```bash
.venv/bin/python tools/blog_export.py --dry-run
```

Prints every ffmpeg command. Confirms clip paths resolve under `output/<film>/adjusted/<strategy>/`.

### 4. Produce assets

```bash
.venv/bin/python tools/blog_export.py
```

Writes to `docs/blog/assets/<film>_<shot_idx>_<tag>.{mp4,gif,png,json}`. Prints `[OK]` / `[OVER BUDGET]` per asset.

### 5. Tune size (optional)

If any `[OVER BUDGET]`:

```bash
.venv/bin/python tools/blog_export.py --gif-width 480 --gif-fps 10   # smaller GIFs
.venv/bin/python tools/blog_export.py --mp4-crf 30                    # smaller MP4s
```

Filter to a single entry while iterating:

```bash
.venv/bin/python tools/blog_export.py --film star_wars_iv_v2 --filter-tag funny_halluc
```

### 6. Commit

```bash
git add docs/blog/selections.json docs/blog/assets/
git commit -m "blog: add curated comparison clips"
```

## Size budget

- GIF < 2 MB, MP4 < 5 MB per clip (enforced by the printed `[OVER BUDGET]` warning).

## Editing selections by hand

- `docs/blog/selections.json` is plain JSON — safe to edit manually to rename labels or drop entries.
- The API endpoints are the canonical writer, but hand edits are fine as long as the `entries` key is present and each entry has `film`, `shot_idx`, `strategy_a`, `strategy_b`.

## Troubleshooting

- **"no clip for …"** — the strategy hasn't been decoded for that shot. Use `find output/<film>/adjusted/ -name "<shot_idx:04d>.mp4"` to check.
- **serve.py 400 on POST** — both strategies (A and B) must be picked in the comparator before saving.
- **GIF palette ghosting** — lower `--gif-fps`, raise `--gif-width`, or switch `paletteuse` dither to `sierra2_4a` in `tools/blog_export.py:build_gif_commands`.
```

### Verification

#### Manual
- [ ] Manual: `ls .gent/skills/blog-curate/SKILL.md` exists.
- [ ] Manual: grep frontmatter: `head -5 .gent/skills/blog-curate/SKILL.md` shows `name: blog-curate` and a one-line description.

---

## Phase 7: Curate & export the real set

### Overview

Run the actual curation. Manual phase — the tooling is the scaffold, this is the payoff.

### Tasks

#### 1. Inventory what's decoded
- [ ] Manual: `ls output/*/adjusted/ 2>/dev/null` — note which films have ≥2 strategies decoded. Candidates per the task: `star_wars_iv_v2`, `bcs_s01e01`, `lotr_fellowship_khazaddum`, `the_matrix_lobby`, `big_lebowski`, `seinfeld_s01e03`, `spongebob_s01e01`.

#### 2. Browse & favorite
- [ ] Manual: start `python tools/serve.py`, open the comparator.
- [ ] Manual: pick **2-5** shots with the following constraints:
  - **≥ 2 different films**.
  - Tag variety — aim for a mix (e.g. at least one `funny_halluc`, one `best_preserve`, one edge-* or `worst_fail`).
  - Each shot's A/B strategies must both be decoded (otherwise the exporter will error).
  - Each label is a short, blog-ready one-liner.

#### 3. Verify selections
- [ ] Run: `cat docs/blog/selections.json | python -c "import json,sys; d=json.load(sys.stdin); print(len(d['entries']),'entries across',len({e['film'] for e in d['entries']}),'films')"` — asserts 2-5 entries across ≥2 films.

#### 4. Export
- [ ] Run: `.venv/bin/python tools/blog_export.py --dry-run` — confirm all command paths resolve.
- [ ] Run: `.venv/bin/python tools/blog_export.py` — produce the bundle. Tune via Phase 5 flags if any `[OVER BUDGET]`.

#### 5. Spot-check
- [ ] Manual: open every MP4 in VLC / QuickTime — plays cleanly, sides roughly synchronized.
- [ ] Manual: open every GIF in Chrome or Firefox — loops smoothly, visibly shows the difference between A and B.
- [ ] Manual: open every PNG — crisp, legible, representative.
- [ ] Manual: open every JSON — contains the shot's prompt + label + tag.

---

## Final Checklist

- [ ] All spec tests passing: `.venv/bin/python -m pytest -m "req" -v`
- [ ] Full test suite passing: `.venv/bin/python -m pytest -v`
- [ ] `docs/blog/selections.json` has 2-5 entries spanning ≥2 films with varied tags.
- [ ] `docs/blog/assets/` has four files per entry; every GIF <2 MB, every MP4 <5 MB.
- [ ] `python tools/serve.py` + favoriting via UI round-trips to `selections.json`.
- [ ] `tools/compare.html` shows favorite markers in sidebar and timeline.
- [ ] `.gent/skills/blog-curate/SKILL.md` present and accurate.
- [ ] No new runtime dependencies introduced (check `pyproject.toml` diff).

## Manual Verification Checklist

- [ ] Manual: favorite a shot → refresh page → marker still there.
- [ ] Manual: open modal on an already-favorited shot → label/tag pre-populated → **Remove** clears it.
- [ ] Manual: each exported MP4 plays in VLC and QuickTime.
- [ ] Manual: each exported GIF loops in a browser.
- [ ] Manual: `--film X --filter-tag Y` compose correctly (only entries matching both run).

## Documentation Updates

- [ ] Create `docs/design/blog-curate-spec.md` (part of Phase 0).
- [ ] Create `docs/blog/README.md` (part of Phase 1).
- [ ] Create `.gent/skills/blog-curate/SKILL.md` (part of Phase 6).
- [ ] After curation, add a short bullet under `docs/work/roadmap.md` noting the asset bundle is ready (so "write the blog post" is unblocked).

## References

- Task: `docs/tasks/0016-curate-blog-comparison-clips/task.md`
- Comparator current state: `tools/compare.html:1-1255`, `tools/serve.py:1-186`
- FFmpeg pattern: `stitch.py:1-80`
- Skill format reference: `.gent/skills/generate/SKILL.md`
- Related research: `docs/research/0016-personal-blog-site/research.md`
