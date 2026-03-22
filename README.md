# lossy

A "lossy codec" for films: encode a video into text descriptions, then decode those descriptions back into video using AI generation models. The result is a reconstructed film that has been "compressed" through natural language.

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
| `replicate-wan` | Replicate Wan 2.2 | Fixed ~5s | ~$0.05/clip |
| `fal-seedance` | fal.ai Seedance 1.0 | 2-12s | ~$0.02/s |
| `fal-seedance-pro` | fal.ai Seedance Pro | 2-12s | ~$0.05/s |
| `runpod-wan` | RunPod (self-hosted) | Fixed ~5s | Hourly GPU rate |

### Audio Strategies

| Strategy | Backend | Max duration | Cost |
|---|---|---|---|
| `elevenlabs` | ElevenLabs SFX v2 (fal.ai) | 22s | ~$0.002/s |
| `mmaudio` | MMAudio V2 (fal.ai) | 30s | ~$0.001/s |
| `runpod-mmaudio` | MMAudio V2 (self-hosted RunPod) | 30s | ~$0 marginal |

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

## Setup

Requires Python 3.11+ and FFmpeg.

```bash
python -m venv .venv
source .venv/bin/activate
pip install scenedetect[opencv] google-genai replicate fal-client httpx runpod tensorflow tensorflow-hub scipy numpy pytest
```

Environment variables (in `.env`):
- `GOOGLE_API_KEY` -- Gemini API key (encode stage 2)
- `REPLICATE_API_TOKEN` -- Replicate API token (replicate-wan strategy)
- `FAL_KEY` -- fal.ai API key (fal-seedance strategies)
- `RUNPOD_API_KEY` -- RunPod API key (runpod-wan strategy)

## Tests

```bash
python -m pytest -v
```
