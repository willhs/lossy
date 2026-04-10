---
name: generate
description: "Run the lossy pipeline to generate video, audio, or both from source media. Use when the user wants to encode, decode, generate clips, stitch, or run any part of the lossy pipeline. Handles full runs, partial runs, resume, audio-only, video-only, and batch generation."
---

# Generate

Run the lossy pipeline on source media files. Supports full end-to-end runs, individual stages, resuming failed runs, and batch processing.

## Prerequisites

- Python venv at `.venv/` — always invoke via `.venv/bin/python`
- API keys in `.env` (loaded automatically by decode.py)
- Source media in `media/` directory (trim with ffmpeg if needed)

## Default Strategies

Unless the user specifies otherwise:

- **Video**: `runpod-wan` (self-hosted Wan 2.1 on RunPod GPU)
- **Audio**: `runpod-mmaudio` (self-hosted MMAudio on RunPod GPU)
- **Speech**: `--speech-voice Roger` (ElevenLabs TTS via fal.ai) — always include unless user says otherwise
- When both video and audio are RunPod, `--concurrent-audio` is passed automatically by pipeline.py

Available video strategies: `runpod-wan`, `replicate-wan`, `fal-seedance`, `fal-seedance-pro`
Available audio strategies: `runpod-mmaudio`, `mmaudio`, `elevenlabs`

## Workflow

### 1. Prepare Source Media

If the user provides a path to a full movie/episode, trim it first:

```bash
ffmpeg -y -i "/path/to/source.mp4" -t <seconds> -c copy media/<name>.mp4
```

- Use `-t 600` for 10 minutes, `-ss <start>` to skip intro, etc.
- Verify duration: `ffprobe -v error -show_entries format=duration -of csv=p=0 media/<name>.mp4`

If the user references Plex media, use `mcp__plugin_media_plex__media_search` to find it, then locate the file under `/Volumes/media/movies/` or `/Volumes/media/tv/`.

### 2. Full Pipeline Run

```bash
.venv/bin/python pipeline.py media/<name>.mp4 -o output/<name> \
  --strategy runpod-wan --audio-strategy runpod-mmaudio --speech-voice Roger
```

Pipeline stages run in order: `encode1 -> encode2 -> decode -> audio -> speech -> stitch`

Key flags:
- `--skip <stages>` — skip specific stages (e.g., `--skip encode1 encode2` to resume from decode)
- `--dry-run` — print commands without executing
- `--limit N` — only process first N shots
- `--start-index N` — skip shots before index N
- `--speech-voice <name>` — enable speech generation with ElevenLabs voice

### 3. Individual Stage Runs

When you need finer control, run stages directly:

**Encode (shot detection + prompt generation):**
```bash
.venv/bin/python encode.py stage1 media/<name>.mp4 -o output/<name>
.venv/bin/python encode.py stage2 output/<name>
```

**Stage 3: character registry (always run before decode with runpod-vace strategy):**
```bash
# With TMDB seeding (recommended — finds more characters, especially unnamed ones)
# Requires TMDB_API_KEY in .env. Find the TMDB ID at themoviedb.org.
.venv/bin/python encode.py stage3 output/<name> --tmdb-id <id>               # movie
.venv/bin/python encode.py stage3 output/<name> --tmdb-id <id> --tmdb-type tv  # TV series

# Without TMDB (fallback — unsupervised, may miss generic-looking characters)
.venv/bin/python encode.py stage3 output/<name>
```

**Decode (video generation):**
```bash
.venv/bin/python decode.py output/<name> --strategy runpod-wan
```

**Audio only:**
```bash
.venv/bin/python decode.py output/<name> --audio --audio-strategy runpod-mmaudio
```

**Video + audio concurrently on RunPod:**
```bash
.venv/bin/python decode.py output/<name> --strategy runpod-wan --concurrent-audio
```

**Stitch (combine clips into final video):**
```bash
.venv/bin/python decode.py output/<name> --strategy runpod-wan --stitch --audio-strategy runpod-mmaudio
```

### 4. Resuming Failed Runs

All generation stages have resume support — they skip already-completed shots automatically. Just re-run the same command.

To resume from a specific stage via pipeline.py, use `--skip` for stages already done:
```bash
.venv/bin/python pipeline.py media/<name>.mp4 -o output/<name> \
  --strategy runpod-wan --audio-strategy runpod-mmaudio \
  --skip encode1 encode2
```

### 5. Batch Processing

When processing multiple files sequentially (to share a single RunPod pod):

1. Run each pipeline one at a time
2. The pod is created fresh for each pipeline run (auto-terminated after)
3. Monitor progress via output directory: `ls output/<name>/clips/runpod-wan/ | wc -l`

For long batches, run in background and check completion:
```bash
.venv/bin/python pipeline.py media/<name>.mp4 -o output/<name> \
  --strategy runpod-wan --audio-strategy runpod-mmaudio 2>&1 | tail -40
```

## Output Structure

```
output/<name>/
  manifest.json          # Shot boundaries (encode1)
  prompts.json           # AI-generated descriptions (encode2)
  characters.json        # Character registry with descriptions + shot assignments (encode3)
  clips/<strategy>/      # Generated video clips (decode)
  audio/<audio-strategy>/ # Generated audio clips (audio)
  speech/                # TTS clips (speech, if enabled)
  reconstructed_<strategy>.mp4  # Final stitched output (stitch)
  costs.json             # Aggregated cost breakdown
```

## Common Issues

- **"No module found" errors**: You're using system python instead of `.venv/bin/python`
- **fal.ai "exhausted balance"**: Switch to RunPod strategies
- **RunPod "CUDA driver initialization failed"**: Bad GPU node — just retry, it's transient
- **0 clips generated but stage "ok"**: API errors are caught silently — run decode directly to see errors
- **SSL certificate errors**: Run `/Applications/Python\ 3.11/Install\ Certificates.command`

## Cost Estimation

- **RunPod GPU**: ~$0.33-0.54/hr depending on GPU type (not tracked in costs.json)
- **Gemini encoding** (encode2): ~$0.05 per 10 min of source video
- **Replicate video**: $0.05/clip flat
- **fal.ai Seedance**: ~$0.02/sec of output video
- Typical 10-min source = 50-120 shots depending on content
