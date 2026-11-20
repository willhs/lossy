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
- `decode.py` -- decoder CLI entry point, run loops, re-exports all public symbols. `--style-prefix` (or `style_prefix` in `shots.json`) puts a per-film look anchor ahead of the identity blocks (LTX strategies); `--ab-manifest JSON [--dry-run]` renders an A/B manifest's jobs verbatim (research 0028). Speech generation reads `speakers.json`/`voice_map.json` if present (falls back to the single `--speech-voice` otherwise)
- `ingredients.py` -- reference sheets for the LTX-2.5 Ingredients IC-LoRA (research 0029 arm I): panel model, the model card's two-part prompt, composite drawing. Pure. Used by `RunPodLtx25IngredientsStrategy` (`--strategy runpod-ltx25-ingredients`, `--ab-manifest` only; needs `HF_TOKEN` and the HF gate accepted, checked before any pod starts)
- `keyframes.py` -- keyframe-first character continuity (research 0029 arm K, with its two sheet fixes): shot routing (keyframe vs text fallback), head-and-shoulders sheet and wardrobe-safe keyframe prompts, the look-alike gate (`make_sheets`), the `KeyframeGenerator` interface with the fal klein implementation, and the face embedder. Used by `RunPodLtx25KeyframeStrategy` (`--strategy runpod-ltx25-keyframe`, opt-in; `prepare()` makes sheets and frames before any pod; `--dry-run` prints the plan; `--keyframe-fal-cap` caps fal spend, which `capped_run` does not see). Not yet verified on a real render.
- `prompt_format.py` -- prompt formatting functions (CAMERA_TERMS, format_prompt, model-specific variants)
- `strategies_video.py` -- video generation strategies (GenerationStrategy base + Replicate, fal.ai, RunPod WAN T2V, RunPod VACE, RunPod LTX-2, RunPod LTX-2.5)
- `strategies_audio.py` -- audio/speech generation strategies (AudioStrategy base + ElevenLabs, MMAudio, RunPod, SpeechStrategy)
- `runpod_pod.py` -- shared RunPod pod lifecycle (`RunPodSession`): provisions across a cloud x GPU fallback matrix, `with_setup_retry` re-provisions on boot failure, exit-time stray-pod sweep. Invariant: every path that abandons a pod (`forget_pod`, setup retry, re-create) terminates it, and a timed-out ComfyUI job is cancelled (`cancel_job`; a pod that will not release its worker is flagged `wedged` and replaced), and `_owned_pod_ids` is append-only so the exit sweep still covers a failed terminate. Ownership is also journaled to disk (`pod_journal.py`, `<output>/runpod_journal/`, ADR 0010): claims are written before the create/terminate they protect, and `reconcile_journals()` terminates what a dead earlier run in the same output dir left, before the next create. Every RunPod strategy builds on this.
- `pod_journal.py` -- the per-session pod ownership journal (claims, late-create windows, flock liveness). Policy in `docs/design/adr/0010-pod-ownership-journal.md`.
- `manifest.py` -- single source of truth for on-disk filename conventions and the `shots.json` v2 schema; also owns `encode_fingerprint`/`check_encode_fingerprint`, which stamp progress files so a resumed run detects and refuses output from a since-changed encode.
- `config.py` -- `.env` loading, Gemini model-id constants, and the video/audio strategy registries (name -> class + capability flags).
- `tools/capped_run.py` -- supervisor for unattended RunPod runs: hard spend cap measured from the account balance, retries, pod sweep. Use it for any long `decode.py` run
- `tools/pgs_to_srt.py` -- OCRs a bitmap (PGS) subtitle track to SRT; BluRay rips need it before encode stage 2 (ffmpeg cannot convert them)
- Gemini backend: `LOSSY_GEMINI_BACKEND=vertex` (with `LOSSY_ENCODE_MODEL=gemini-3.1-flash-lite`) routes encode/voice-casting calls through Vertex AI on the same GCP project when the AI Studio key is rate-limited. Default is `aistudio`
- `tools/seam_ratio.py` -- measures I2V chaining seams (boundary frame delta / normal motion) in a clips dir, the research 0023 method
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
# Requires TMDB_API_KEY (see "Secrets"). Get a free key at themoviedb.org/settings/api.
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

## Secrets

Keys live in the 1Password vault `proj-lossy`, one item per key, value in the `credential`
field: `FAL_KEY`, `GEMINI_API_KEY`, `HF_TOKEN`, `REPLICATE_API_TOKEN`, `RUNPOD_API_KEY`,
`TMDB_API_KEY` (e.g. `op://proj-lossy/FAL_KEY/credential`). `.env.tpl` (committed) holds only
`op://` references. Run anything that needs a key one of three ways:

- `op run --env-file .env.tpl -- <command>` (prompts 1Password on the Mac)
- `op-scope lossy -- <command>` (the vault-scoped service account)
- Manager jobs dispatched with `secrets=true`

Never write a resolved key to a file or print it. Don't regenerate a secret-bearing `.env`
with `op inject`. The old plaintext `.env` is being retired (Nov 2026).
