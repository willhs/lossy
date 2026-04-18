---
id: task-0016
type: spec
purpose: "Curate 2-5 side-by-side showcase clips across multiple decoded films and produce MP4/GIF/PNG/prompt assets for the upcoming lossy-codec blog post, via a reusable favorite-and-export workflow."
tags: ["blog", "tools", "compare", "export", "curation", "assets"]
related: ["work/roadmap.md", "research/0016-personal-blog-site/research.md"]
created: 2026-04-18
updated: 2026-04-18
---

# Curate Comparison Showcase Clips for Blog

## Goal

Curate **2-5** side-by-side showcase clips spanning multiple decoded films and produce four web-optimized asset types per clip (MP4, looped GIF, keyframe PNG, prompt-snippet JSON) for the upcoming lossy-codec blog post. Build the curation flow as a reusable tool — favorite action in `tools/compare.html`, a standalone exporter, and a Claude skill — so future blog posts reuse the pattern.

## Context

The "write the blog post" item sits under **Later** in `docs/work/roadmap.md`. This task tees up the visual assets so the post can be drafted against concrete examples rather than abstract claims.

**Existing tooling:**
- `tools/compare.html` — 3-pane original/A/B comparator served by `tools/serve.py`.
- `tools/ab-compare.html` — simpler strategy-vs-strategy view.
- Neither currently exports anything or persists selections.

**Content available:**
- `output/<film>/` contains 15+ decoded films.
- `star_wars_iv_v2` has `adjusted/{runpod-wan,runpod-vace,fal-seedance}` plus `prompts.json`, `manifest.json`, and keyframes.
- `decode_progress*.json` tracks which shots are decoded per strategy.

**Why now:** the blog post needs concrete, varied examples that show lossy's characteristic failure/hallucination modes. Eyeballing 15+ films shot-by-shot is wasteful without a favorite action; re-running ffmpeg by hand per clip is tedious and error-prone. A lightweight curation + export loop pays for itself the first time and thereafter.

## Scope

### Must Do
- **Selections schema & storage** — define `docs/blog/selections.json` schema, stub an empty file, and add a short `docs/blog/README.md`.
- **Favorite API** — add `GET`/`POST`/`DELETE /api/blog_selections` to `tools/serve.py` with atomic write and merge-by-`(film, shot_idx)`.
- **Favorite UI** — add a favorite button + tag/label modal to `tools/compare.html`; mark favorited shots in the sidebar and timeline so already-picked shots are visible while browsing.
- **Exporter** — build `tools/blog_export.py` that reads `selections.json` and produces, per entry:
  - side-by-side MP4 via ffmpeg `hstack`, trimmed to the shot window
  - looped GIF via `palettegen` + `paletteuse` at 12 fps
  - keyframe PNG
  - prompt JSON snippet
- **Size targets** — tune for GIF < 2 MB and MP4 < 5 MB per clip (h264 crf 28 baseline for MP4; palette/fps/scale levers for GIF). Add `--dry-run`, `--filter-tag`, `--film` flags.
- **Claude skill** — author a skill (likely `.gent/skills/blog-curate/`) documenting the serve → favorite → export workflow end-to-end.
- **Curate the set** — browse the comparator across films and pick 2-5 entries with varied tags spanning ≥ 2 films.
- **Run the exporter** — verify size budget, playback/quality, and commit the assets under `docs/blog/assets/`.

### Might Do
- Additional `--filter-*` flags on the exporter if obvious lightweight wins emerge.
- Visual polish on the sidebar/timeline favorite markers (icon, color) beyond a minimal indicator.

### Won't Do
- Blog post prose or narrative.
- Auto-caption generation (captions stay in the blog prose).
- Programmatic shot scoring / auto-ranking of candidates.
- Decoder or strategy changes.
- Any edits to `tools/ab-compare.html` (scope limited to the 3-pane view).

## Decisions

Captured up front — do not relitigate during planning:

- **Source breadth over depth** — spread selections across films; aim for variety, not volume.
- **Selections file location** — `docs/blog/selections.json`.
- **Asset naming** — `docs/blog/assets/<film>_<shot_idx>_<tag>.{mp4,gif,png,json}`.
- **Size targets** — GIF < 2 MB, MP4 < 5 MB per clip. MP4 uses h264 crf 28; GIF uses `palettegen` + `paletteuse` at 12 fps.
- **Tags** — free-form `label` plus optional `tag` from `{best_preserve, worst_fail, funny_halluc, edge_dialogue, edge_action, edge_establish}`.
- **No auto-captions.**

## Schema sketch — `selections.json`

```json
{
  "entries": [
    {
      "film": "star_wars_iv_v2",
      "shot_idx": 42,
      "strategy_a": "runpod-wan",
      "strategy_b": "runpod-vace",
      "tag": "funny_halluc",
      "label": "Vader reveal becomes cartoon",
      "formats": ["mp4", "gif", "png", "prompt"],
      "added_at": "2026-04-18T..."
    }
  ]
}
```

## Constraints

- Dev-only tooling — this runs on a developer machine; no auth, no production hardening needed for `tools/serve.py`.
- Stay within the ffmpeg/Python toolchain already present; no new runtime dependencies without discussion (project intentionally avoids a package manager).
- Keep the favorite UI intentionally generic/reusable — future blog posts should be able to run the same loop with zero code changes.

## References

- `docs/work/roadmap.md` — "write the blog post" appears under Later; this task is the setup.
- `docs/research/0016-personal-blog-site/research.md` — platform and structural direction for the blog.
- `tools/compare.html`, `tools/serve.py` — primary integration surface.
- `output/star_wars_iv_v2/` — richest candidate film (multi-strategy coverage).

## Success Criteria

- [ ] Comparator exposes a favorite action that writes to `docs/blog/selections.json` via a `serve.py` endpoint.
- [ ] Favorited shots are visually marked in the sidebar and timeline.
- [ ] `tools/blog_export.py` reads selections and produces side-by-side MP4, looped GIF, keyframe PNG, and prompt JSON per entry.
- [ ] All GIFs are < 2 MB and all MP4s are < 5 MB.
- [ ] `docs/blog/selections.json` contains 2-5 entries spanning ≥ 2 different films with tag variety.
- [ ] Assets land in `docs/blog/assets/`, are playable/viewable, and within size budget.
- [ ] A Claude skill documents the end-to-end workflow (serve → favorite → export).
