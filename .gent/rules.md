# Agent Guidelines

## Project Overview

lossy is a Python CLI pipeline that encodes films into text descriptions and decodes them back into video using AI generation. Three standalone scripts: `encode.py`, `decode.py`, `pipeline.py`. No package manager -- dependencies installed directly into a `.venv`.

## Architecture

- **Stateless CLI stages** (ADR-002): each stage reads/writes files on disk, no shared state
- **Strategy pattern** for video generation backends in `decode.py`
- **`pipeline.py`** is a thin subprocess orchestrator -- it does NOT import from encode/decode
- Compare tool is browser-based (`tools/compare.html` served by `tools/serve.py`)

## Key Files

- `encode.py` -- encoder CLI (stage1: shots+keyframes, stage2: prompts via Gemini)
- `decode.py` -- decoder CLI (generate clips via strategy, stitch with FFmpeg)
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
- Do not add new dependencies without discussion -- the project intentionally avoids a package manager.
