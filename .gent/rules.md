# Agent Guidelines

## Project Overview

lossy is a Python CLI pipeline that encodes films into text descriptions and decodes them back into video using AI generation. Three standalone scripts: `encode.py`, `decode.py`, `pipeline.py`. Dependencies are managed with uv (`pyproject.toml` + `uv.lock`); run `uv sync`.

## Architecture

- **Stateless CLI stages** (ADR-002): each stage reads/writes files on disk, no shared state
- **Strategy pattern** for video generation backends in `decode.py`
- **`pipeline.py`** is a thin subprocess orchestrator -- it does NOT import from encode/decode
- Compare tool is browser-based (`tools/compare.html` served by `tools/serve.py`)

## Key Files

- `encode.py` -- encoder CLI (stage1: shots+keyframes, stage2: prompts via Gemini, stage3: character registry via Gemini; supports `--tmdb-id` for TMDB-seeded supervised discovery)
- `decode.py` -- decoder CLI entry point, run loops, re-exports all public symbols
- `prompt_format.py` -- prompt formatting functions (CAMERA_TERMS, format_prompt, model-specific variants)
- `strategies_video.py` -- video generation strategies (GenerationStrategy base + Replicate, fal.ai, RunPod WAN T2V, RunPod VACE)
- `strategies_audio.py` -- audio/speech generation strategies (AudioStrategy base + ElevenLabs, MMAudio, RunPod, SpeechStrategy)
- `stitch.py` -- FFmpeg stitching (video concat, audio/speech track building, muxing)
- `eval.py` -- evaluation CLI (re-encodes output, compares to original, writes quality reports)
- `pipeline.py` -- orchestrator (chains all stages via subprocess)
- `tools/serve.py` -- dev server for compare tool

## Running the Pipeline

```bash
# Full pipeline
python pipeline.py media/film.mp4 -o output/film --strategy fal-seedance

# Dry run (preview commands)
python pipeline.py media/film.mp4 -o output/film --strategy fal-seedance --dry-run

# Skip encoding (re-run decode+stitch only)
python pipeline.py media/film.mp4 -o output/film --strategy fal-seedance --skip encode1 encode2
```

### Individual stages

```bash
# Encode
python encode.py stage1 media/film.mp4 -o output/film
python encode.py stage2 output/film

# Stage 3: character registry (unsupervised — no TMDB)
python encode.py stage3 output/film

# Stage 3: TMDB-seeded (recommended for movies)
# Requires TMDB_API_KEY in .env. Get a free key at themoviedb.org/settings/api.
# Find the TMDB ID on themoviedb.org (e.g. Star Wars IV = 11)
python encode.py stage3 output/film --tmdb-id 11
python encode.py stage3 output/film --tmdb-id 60059 --tmdb-type tv  # TV series

# Decode (generate clips)
python decode.py output/film --strategy fal-seedance

# Stitch (combine clips into final video)
python decode.py output/film --strategy fal-seedance --stitch
```

Requires `.env` with API keys for the chosen strategy (see README.md).

## Running Tests

```bash
python -m pytest -v
```

Tests use mocks for external APIs. No real API calls in tests.

## Workflow

- Prefer worktrees or task-specific branches for multi-file edits.
- Run tests and linters before handing work back.

## Documentation

- Preserve doc front matter (`id`, `type`, `purpose`, scope, tags, etc.) and keep sections aligned with the docs guide.
- Validate doc front matter against `docs/front-matter-schema.json` if present.
- For docs-specific workflows and permissions, see `docs/AGENTS.md` (details on when to write to different doc directories).

## Guardrails

- Respect human-owned files flagged in `docs/philosophy/`.
- Keep generated content ASCII unless the project explicitly opts in.
- Do not add new dependencies without discussion; when agreed, add them via `uv add` so `pyproject.toml` and `uv.lock` stay in sync.
