# Agent Guidelines

## Project Overview

lossy is a Python CLI pipeline that encodes films into text descriptions and decodes them back into video using AI generation. Three standalone scripts: `encode.py`, `decode.py`, `pipeline.py`. Dependencies are managed with uv (`pyproject.toml` + `uv.lock`); run `uv sync`.

## Architecture

- **Stateless CLI stages** (ADR-002): each stage reads/writes files on disk, no shared state
- **Strategy pattern** for video generation backends in `decode.py`
- **`pipeline.py`** is a thin subprocess orchestrator -- it does NOT import from encode/decode
- Compare tool is browser-based (`tools/compare.html` served by `tools/serve.py`)

## Key Files

- `encode.py` -- encoder CLI (stage1: shots+keyframes, stage2: prompts via Gemini, stage3: character registry via Gemini, stage4: dialog-line speaker attribution via Gemini -> `speakers.json`; supports `--tmdb-id` for TMDB-seeded supervised discovery)
- `voice_casting.py` -- proposes a per-character voice map (`voice_map.json`) from `characters.json` via Gemini, with `--sample` to render one short clip per voice for review. A proposal only -- casting is a human call, made by editing `voice_map.json`
- `decode.py` -- decoder CLI entry point, run loops, re-exports all public symbols. Speech generation reads `speakers.json`/`voice_map.json` if present (falls back to the single `--speech-voice` otherwise)
- `prompt_format.py` -- prompt formatting functions (CAMERA_TERMS, format_prompt, model-specific variants)
- `strategies_video.py` -- video generation strategies (GenerationStrategy base + Replicate, fal.ai, RunPod WAN T2V, RunPod VACE, RunPod LTX-2)
- `strategies_audio.py` -- audio/speech generation strategies (AudioStrategy base + ElevenLabs, MMAudio, RunPod, SpeechStrategy)
- `runpod_pod.py` -- shared RunPod pod lifecycle (`RunPodSession`): provisions across a cloud x GPU fallback matrix, `with_setup_retry` re-provisions on boot failure, exit-time stray-pod sweep. Every RunPod strategy builds on this.
- `manifest.py` -- single source of truth for on-disk filename conventions and the `shots.json` v2 schema; also owns `encode_fingerprint`/`check_encode_fingerprint`, which stamp progress files so a resumed run detects and refuses output from a since-changed encode.
- `config.py` -- `.env` loading, Gemini model-id constants, and the video/audio strategy registries (name -> class + capability flags).
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

# Stage 4: speaker attribution (dialog line -> character, speakers.json)
python encode.py stage4 output/film

# Propose a per-character voice map + short samples for review (not final -- edit voice_map.json to re-cast)
python voice_casting.py output/film --sample

# Decode (generate clips)
python decode.py output/film --strategy fal-seedance

# Stitch (combine clips into final video)
python decode.py output/film --strategy fal-seedance --stitch
```

Requires `.env` with API keys for the chosen strategy (see README.md).

## Running Tests

```bash
uv run python -m pytest -v
```

Use `uv run` (or an activated `uv sync`'d venv) -- a bare system `python -m pytest`
fails at collection with `ModuleNotFoundError: No module named 'scenedetect'` since
deps live in the uv-managed venv, not system site-packages. Tests use mocks for
external APIs. No real API calls in tests.

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
