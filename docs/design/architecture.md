---
id: architecture
type: spec
purpose: "Describe the end-to-end system shape and on-disk contract."
scope: ["design", "architecture"]
non_goals: []
tags: ["architecture"]
related: ["design/adr/"]
---

## Overview

lossy is a Python CLI pipeline with three top-level stages: **encode**, **decode**, and **compare**. Each stage is an independent CLI that reads and writes files in a per-film output directory. No stage imports runtime state from another — the output directory is the contract (ADR-002).

```
Source Film → [encode] → output/<film>/{manifest,prompts,characters}.json, keyframes/, characters/
                            ↓
                          [decode] → clips/<strategy>/, audio/<strategy>/, speech/
                            ↓
                         [stitch] → reconstructed_<strategy>.mp4
                            ↓
                       [compare]   → browser-side side-by-side (tools/compare.html)
```

`pipeline.py` is a thin orchestrator that runs the stages as subprocesses; it does not import encode/decode.

## On-Disk Contract

All stage-to-stage communication lives in one output directory per film. `manifest.py` is the single source of truth for the filename conventions and the `shots.json` schema — readers should go through `manifest.load_shots()` rather than reinventing the v2 check.

### Encoder outputs

| Path | Producer | Purpose |
|---|---|---|
| `shot_index.json` | encode stage 1 | Shot boundaries (`index`, `start_s`, `end_s`, `duration_s`, `keyframes[]`) plus `source` metadata. Read by stage 2, eval, and any tool that needs the shot map. |
| `keyframes/NNNN-MM.jpg` | encode stage 1 | Per-shot keyframes (512px, adaptive count) used by stage 2 for vision prompts. |
| `subtitles.srt` | encode stage 2 | Extracted SRT track from the source (when embedded). |
| `camera_motion.json` | encode stage 2 | Per-shot motion label from Farneback optical flow. Cached so stage 2 is re-runnable without re-processing the whole video. |
| `audio_labels.json` | encode stage 2 | Per-shot YAMNet aggregation into 6 buckets (speech, music, effects, ambient, silence, other). |
| `shots.json` | encode stage 2 | The real "compressed" film — v2 format: `{"format": "v2", "shots": [...], "dialog": [...]}`. `shots` are per-shot descriptions used by decode; `dialog` is a flat global subtitle timeline used by speech TTS. |
| `characters.json` | encode stage 3 | Character registry produced from `shots[].description.subjects`. Drives portrait generation and VACE reference conditioning downstream. TMDB-seeded when `--tmdb-id` is passed (ADR-006); unsupervised Gemini otherwise. |
| `encode_costs.json` | encode stage 2 | Gemini input/output token costs per stage. |

### Decoder outputs

| Path | Producer | Purpose |
|---|---|---|
| `clips/<strategy>/NNNN.mp4` | decode (video) | Single-shot clip. Strategies name files via `manifest.clip_filename(idx)`. |
| `clips/<strategy>/NNNN-MM.mp4` | decode (video) | Split-shot parts (shots longer than the strategy's max clip duration). Part `MM` starts at `01`. |
| `decode_progress_<strategy>.json` | decode (video) | Resume state: `completed[]`, `failed[]`, `clips{}` map with per-clip duration and cost. Read back by stitch. |
| `audio/<strategy>/NNNN.flac\|mp3` | decode (audio) | Per-shot SFX/music generated from the shot's `description.sound`. |
| `audio_progress_<strategy>.json` | decode (audio) | Resume state for audio. |
| `speech/NNNN-00.mp3` | decode (speech) | Per-dialog-line TTS, one file per entry in `shots.json:dialog`. Indexed by global dialog position, not shot index. |
| `speech_progress.json` | decode (speech) | Resume state for speech, always `{"format": "v2", ...}`. |
| `characters/<name>.png` | decode (portraits) | Canonical portrait generated from `characters.json`. Used by the VACE strategy as reference conditioning (SPEC-210, SPEC-220). |
| `runpod_pod.json` | decode (RunPod strategies) | Active pod lifecycle state managed by `runpod_pod.RunPodSession`. |
| `reconstructed_<strategy>[+<audio>].mp4` | stitch | Final muxed output. Speed-adjusted clips concatenated with optional SFX and speech tracks. |
| `scene_<name>_<strategy>.mp4` | stitch | Scene-scoped reconstructions when a `scenes.json` file is present. |
| `costs.json` | pipeline | Aggregated cost report across all stages. |
| `eval/<strategy>/eval_results.json` | eval | Re-encoded vs original description comparison. |

## Pipeline Stages

### Encode

**Stage 1 (`encode.py stage1`)** — PySceneDetect AdaptiveDetector (ADR-003) splits the source into shots; FFmpeg extracts 2-8 keyframes per shot at 512px. Writes `shot_index.json` and `keyframes/`.

**Stage 2 (`encode.py stage2`)** — Enriches each shot with metadata (subtitles, camera motion, audio buckets) and calls Gemini Flash-Lite with the keyframes and metadata to produce a structured description: `shot_type`, `camera_movement`, `subjects`, `action`, `lighting`, `color_palette`, `mood`, `setting`, `sound`. Shots longer than the split threshold (ADR-007) get additional per-segment descriptions stored under `temporal_segments`. Writes `shots.json` (v2), `camera_motion.json`, `audio_labels.json`, `subtitles.srt`.

**Stage 3 (`encode.py stage3`)** — Builds a character registry from the `subjects` fields. With `--tmdb-id` uses TMDB cast as ground truth for a two-step text-match + supervised Gemini pass (ADR-006); without, falls back to an unsupervised Gemini pass. Writes `characters.json`.

### Decode

Reads `shots.json` via `manifest.load_shots`. Dispatches each shot to a selected **video strategy**, and optionally to an **audio strategy** and the **speech strategy**. Video and audio have per-strategy progress files so runs are resumable and multiple backends can coexist in the same output directory.

**Video strategies** (`strategies_video.py`):

| Name | Provider | Clip length | Notes |
|---|---|---|---|
| `replicate-wan` | Replicate (hosted) | ~5s fixed | Wan 2.2 Fast, cheapest hosted (ADR-004). |
| `fal-seedance` | fal.ai (hosted) | 2-12s | Seedance 1.0 Pro Fast, duration control. |
| `fal-seedance-pro` | fal.ai (hosted) | 2-12s | Full Pro tier variant. |
| `runpod-wan` | RunPod (self-hosted ComfyUI) | ~5s | Wan 2.1 1.3B fp16 via ComfyUI HTTP API (ADR-008). Manages pod lifecycle automatically. |
| `runpod-wan22` | RunPod (self-hosted ComfyUI) | ~5s | Same infra with Wan 2.2 TI2V-5B. |
| `runpod-wan-enriched` | RunPod (self-hosted ComfyUI) | ~5s | Character-name injection from the registry, no reference image. |
| `runpod-vace` | RunPod (self-hosted ComfyUI) | ~5s | VACE conditioning with character portraits as reference images. Requires encode stage 3 (SPEC-220). |

Long shots that exceed the strategy's max clip duration are split into multiple parts (`NNNN-01.mp4`, `NNNN-02.mp4`, ...) and mapped to distinct prompt segments via the `temporal_segments` produced in encode stage 2.

**Audio strategies** (`strategies_audio.py`) generate per-shot SFX/ambient tracks from `description.sound`:

| Name | Provider | Max clip | Notes |
|---|---|---|---|
| `elevenlabs` | ElevenLabs Sound Effects v2 via fal.ai | 22s | Per-second pricing. |
| `mmaudio` | MMAudio V2 via fal.ai | 30s | Cheapest hosted. |
| `runpod-mmaudio` | RunPod (self-hosted ComfyUI) | 30s | Self-hosted MMAudio; can be co-pipelined with a RunPod video strategy via `--concurrent-audio` to share a pod. |

The speech filter (SPEC-100) strips speech/dialogue/voice keywords from the `sound` description before passing to MMAudio variants, since those backends produce garbled dialogue when asked for speech.

**Speech strategy** (ElevenLabs TTS) generates one clip per entry in `shots.json:dialog` at its original SRT timestamp, so lines that span shot boundaries are spoken exactly once.

### Stitch (`python decode.py ... --stitch` or implicit after decode)

FFmpeg-driven concatenation (`stitch.py`):

1. Load `shots.json` via `manifest.load_shots` and the per-strategy `decode_progress_<strategy>.json`.
2. Speed-adjust each clip to match the original shot duration (skipped for split-shot parts, whose filenames carry a `-NN` suffix).
3. Concatenate via FFmpeg concat demuxer.
4. Auto-discover all audio strategies under `audio/` (or use `--audio-strategy`) and mix their per-shot tracks, duration-adjusted to match the stitched video.
5. Mix the global speech track at SRT timestamps using `adelay`, ducking SFX (-8 dB) and boosting speech (+6 dB).
6. Mux all audio back into the final `reconstructed_<strategy>[+<audio>].mp4`. Optionally produce scene-scoped videos from `scenes.json`.

### Compare

Browser-based side-by-side viewer served by `tools/serve.py` (static file server for `tools/compare.html`). Not part of the CLI pipeline.

## Eval

`eval.py` re-encodes the reconstructed video through the *same* encoder prompt (it reuses `SYSTEM_PROMPT`, `extract_audio`, `run_yamnet`, etc. from `encode.py`) and compares field-by-field to the original. This is library reuse, not a stage boundary violation — eval is read-only and produces a report under `eval/<strategy>/`.

## Key Constraints

- **Cost first.** Every external API call costs money; pipeline defaults aim for the cheapest viable options (480p, short clips, fewest frames sampled). See `docs/philosophy/principles.md`.
- **Stateless stages (ADR-002).** Stages communicate by files in the output directory. No shared database, no long-running server, no in-memory state that survives a CLI invocation.
- **Single source of truth for the on-disk contract.** `manifest.py` owns filename conventions and the `shots.json` v2 schema; other modules import it rather than reinventing the check.
- **Strategy pattern for backends.** New video or audio backends plug in as subclasses of `GenerationStrategy` / `AudioStrategy` in `strategies_video.py` / `strategies_audio.py`.
- **No package manager.** Dependencies are installed directly into `.venv`. New dependencies require discussion (see `CLAUDE.md`).

## Tech Stack

- **Language:** Python 3.11 (flat scripts, no package layout)
- **Video processing:** FFmpeg via subprocess
- **Shot detection:** PySceneDetect 0.6.7 AdaptiveDetector (ADR-003)
- **Vision model:** Gemini Flash-Lite Preview
- **Audio classification:** YAMNet via TensorFlow Hub (ADR-005)
- **Character enrichment:** TMDB API + Gemini (ADR-006)
- **Video generation:** 7 swappable strategies (hosted + self-hosted ComfyUI)
- **Audio generation:** 3 swappable strategies
- **Speech generation:** ElevenLabs TTS
- **Orchestration:** `pipeline.py` (subprocess chainer)
