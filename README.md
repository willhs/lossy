# lossy

A "lossy codec" for films: encode a video into text descriptions, then decode those descriptions back into video using AI generation models. The result is a reconstructed film that has been "compressed" through natural language.

[![Original vs reconstruction — Moria, The Fellowship of the Ring](https://willhs.me/lossy/demos/lotr-moria-compare-poster.jpg)](https://willhs.me/posts/1mb-movie/)

A feature film becomes ~2,000 shot descriptions (~320KB xz'd — under 1MB of text), then regenerates for ~$30 of rented GPU time: Wan 2.2 self-hosted on a RunPod A6000 at ~$0.33/hr, with MMAudio (SFX) and MusicGen (score) co-hosted on the same pod. The describe pass (Gemini Flash-Lite over ~10,000 sampled frames) costs about $1 per film.

Write-up with results and example reconstructions: [1mb movie](https://willhs.me/posts/1mb-movie/).

## Pipeline

```
Source Film -> [Encode] -> Scene Manifest (JSON) -> [Decode] -> Reconstructed Film
                                                        |
                                                   [Compare] -> Side-by-side view
```

### Stages

1. **Encode Stage 1** -- Shot detection + keyframe extraction (PySceneDetect, FFmpeg)
2. **Encode Stage 2** -- Metadata enrichment + prompt generation (optical flow, YAMNet audio classification, Gemini vision)
3. **Decode** -- Generate video clips from prompts (swappable strategy)
4. **Audio** -- Generate per-shot audio from sound descriptions (optional, swappable strategy)
5. **Stitch** -- Speed-adjust and concatenate clips into final output, mux audio if available

### Strategies

| Strategy | Backend | Duration control | Cost |
|---|---|---|---|
| `runpod-wan22` | RunPod Wan 2.2 TI2V-5B (self-hosted) — character identity + I2V long-shot chaining | Variable, 24fps | Hourly GPU rate |
| `runpod-wan` | RunPod Wan 2.1 (self-hosted) | Fixed ~5s | Hourly GPU rate |
| `runpod-wan-enriched` | As `runpod-wan`, with character descriptions injected into prompts | Fixed ~5s | Hourly GPU rate |
| `runpod-vace` | RunPod Wan VACE (self-hosted) — reference-portrait conditioning | Fixed ~5s | Hourly GPU rate |
| `replicate-wan` | Replicate Wan 2.2 | Fixed ~5s | ~$0.05/clip |
| `fal-seedance` | fal.ai Seedance 1.0 | 2-12s | ~$0.02/s |
| `fal-seedance-pro` | fal.ai Seedance Pro | 2-12s | ~$0.05/s |

`runpod-wan22` is the current recommended strategy.

### Audio Strategies

| Strategy | Backend | Max duration | Cost |
|---|---|---|---|
| `elevenlabs` | ElevenLabs SFX v2 (fal.ai) | 22s | ~$0.002/s |
| `mmaudio` | MMAudio V2 (fal.ai) | 30s | ~$0.001/s |
| `runpod-mmaudio` | MMAudio V2 (self-hosted RunPod) | 30s | ~$0 marginal |
| `musicgen` | MusicGen score (Replicate) | 30s | per-run |
| `runpod-musicgen` | MusicGen score (self-hosted RunPod) | 30s | ~$0 marginal |

Dialogue is spoken from the film's subtitles at original timestamps via ElevenLabs TTS (see `SpeechStrategy` in `strategies_audio.py`).

`runpod-mmaudio` reuses the same RunPod pod as `runpod-wan` for near-zero marginal audio cost. First run downloads ~5 GB of MMAudio models.

## Usage

### Full pipeline (single command)

```bash
# Video only
python pipeline.py media/film.mp4 -o output/film --strategy fal-seedance

# Video + audio
python pipeline.py media/film.mp4 -o output/film --strategy fal-seedance --audio-strategy elevenlabs

# Video + audio on one RunPod pod (cheapest)
python pipeline.py media/film.mp4 -o output/film --strategy runpod-wan --audio-strategy runpod-mmaudio
```

Options:
- `--skip encode1 encode2` -- skip stages (e.g. re-run decode+stitch only)
- `--dry-run` -- print commands without executing
- `--limit N` -- process only first N shots
- `--start-index N` -- skip shots before index N
- `--detector adaptive|content` -- shot detection method
- `--threshold N` -- shot detection threshold

### Individual stages

```bash
# Encode
python encode.py stage1 media/film.mp4 -o output/film
python encode.py stage2 output/film

# Decode (video)
python decode.py output/film --strategy fal-seedance

# Audio generation (optional)
python decode.py output/film --audio --audio-strategy elevenlabs

# Stitch (muxes audio if --audio-strategy provided)
python decode.py output/film --strategy fal-seedance --stitch
python decode.py output/film --strategy fal-seedance --stitch --audio-strategy elevenlabs
```

### Compare

```bash
python tools/serve.py
# Open http://localhost:8080/tools/compare.html
```

### Authored manifests (no source film)

Decode + stitch don't require an `encode.py` run at all -- they only need a
`shots.json` (v2 format) and, optionally, a `characters.json` sitting in the
output directory. This is a first-class mode, used for original films
written from a screenplay rather than reconstructed from a source video (see
`docs/research/0022-authored-manifest-verification/research.md` and
[ADR-009](docs/design/adr/009-authored-manifest-retiming.md)).

```bash
mkdir -p output/my-film
cp shots.json characters.json output/my-film/   # hand-authored, v2 format

python decode.py output/my-film --strategy fal-seedance
python decode.py output/my-film --strategy fal-seedance --stitch
```

Minimum required per shot in `shots.json`: `index`, `start_s`, `end_s`,
`duration_s`, and a `description` object (can be `{}`, but video generation
reads it unconditionally, so an empty dict yields an empty prompt --
populate at least `action`). `characters.json` is optional; if present it's
only consulted by the `runpod-wan-enriched`, `runpod-wan22`, and
`runpod-vace` strategies for prompt enrichment. `duration_s` is honoured
exactly -- stitch speed-adjusts each generated clip to match it, whether
that duration was originally observed from a source film or chosen by an
author.

`eval.py video` requires the original encode artifacts (`shot_index.json`,
`shots.json`) to compare against and has nothing to do for an authored
manifest -- it detects their absence and skips cleanly rather than erroring.

A worked example lives at `tests/fixtures/authored_example/`.

## Setup

Requires Python 3.11+, FFmpeg, and [uv](https://docs.astral.sh/uv/).

```bash
uv sync          # install dependencies from pyproject.toml / uv.lock
uv run python pipeline.py --help
```

Prefix commands with `uv run` (or activate the venv it creates: `source .venv/bin/activate`).

Environment variables (in `.env`):
- `GOOGLE_API_KEY` -- Gemini API key (encode stage 2)
- `GEMINI_API_KEY` -- Gemini API key (encode stage 3; same value as GOOGLE_API_KEY)
- `TMDB_API_KEY` -- TMDB API key (encode stage 3 with `--tmdb-id`; free at themoviedb.org/settings/api)
- `REPLICATE_API_TOKEN` -- Replicate API token (replicate-wan strategy)
- `FAL_KEY` -- fal.ai API key (fal-seedance strategies)
- `RUNPOD_API_KEY` -- RunPod API key (runpod-wan strategy)

## Tests

```bash
uv run python -m pytest -v
```

## License

[MIT](LICENSE)
