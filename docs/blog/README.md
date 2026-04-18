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
