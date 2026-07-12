---
name: generate-scene
description: Generate a short film-style video scene from a written description, using Wan 2.2 5B on fal.ai with last-frame→first-frame chaining for graceful long shots (and optional MMAudio ambience). Use when the user wants to generate/reconstruct a movie scene or short cinematic clip from text, or add scenes to the lossy "1mb movie" carousel. Spends real money (~$0.05–0.20 per ~5s clip) — confirm before running.
---

# generate-scene

Generate a ~20–60s cinematic scene from a written description. Each scene is a list of
**shots**; long shots are extended by chaining clips **last-frame → first-frame** (the
trick that keeps a long shot coherent instead of jump-cutting). Self-contained — runs
from any directory, needs only `fal-client`, `ffmpeg`, and a `FAL_KEY`.

This is the reusable form of the pipeline used for the lossy "1mb movie" blog carousel
(escape pod, LOTR ring, E.T. moon, Jurassic Park). Output is the lossy/drifty aesthetic —
great for that blog and creative experiments, not production video.

## Prerequisites

- `ffmpeg` on PATH.
- `pip install fal-client` (or run with a venv that has it — e.g. the lossy repo's `.venv`).
- `FAL_KEY` exported, or pass `--env /path/to/.env`.

## Workflow

1. **Write a scene spec** (JSON). Each shot is `{"prompt": "..."}`; add `"continue": ["..."]`
   to chain image→video continuations onto it for a graceful long shot. Use descriptive
   prompts (no trademarked names — the model only sees the description anyway), ~30–60 words.
   Mention camera, subject, lighting, mood, and a film-era look. See `example-scene.json`.

2. **Estimate + confirm cost.** Count clips (each shot = 1 + len(continue)). At ~$0.05–0.20
   per ~5s clip, tell the user the rough total and confirm before spending.

3. **Run:**
   ```bash
   FAL_KEY=... python3 generate.py scene.json
   # or, reusing the repo's venv + key (run from the lossy repo root):
   .venv/bin/python3 .gent/skills/generate-scene/generate.py scene.json --env .env
   ```
   Pass `--workdir ./scene-work` to keep intermediate clips (resume-safe: re-running skips
   clips that already exist, so a crash/interrupt doesn't lose progress).

4. **Verify before shipping.** Extract a frame from each clip and actually look
   (`ffmpeg -ss 2 -i cNN.mp4 -frames:v 1 f.jpg`) — flag a bad render rather than ship it.
   For the carousel, also generate a poster (`-ss 2 ... poster.jpg`) and add a slide in
   `src/content/blog/lossy.mdx` via the `<Carousel>` component.

## Spec reference

- `output` (required): mp4 path.
- `shots` (required): list of `{prompt, continue?}`. `continue` is a list of continuation
  prompts, each an I2V clip seeded from the previous clip's last frame.
- `resolution`: `580p` (default) or `720p`. `num_frames`: 121 ≈ 5s @ 24fps.
- `seed`: base seed, +1 per clip. `negative`: negative prompt (sensible default built in).
- `audio` (optional): `{"prompt": "...", "volume": 0.4}` — generates one MMAudio bed,
  loops it under the whole scene through a limiter (gentle, non-harsh). Good for SFX/ambience
  scenes (rain, crickets); it is **not** synced to the video. For dialogue, generate
  ElevenLabs lines separately and mux by hand.

## Notes / gotchas (learned the hard way)

- **Chaining**: only `continue` clips use the previous last frame. Independent shots
  (separate list entries) will look like a different take — that's the expected cut.
- **Cost discipline**: this spends money per clip. Always confirm a rough total first.
- **Cheaper at scale (phase 2):** for many clips, self-hosting Wan 2.2 5B on a rented GPU
  (RunPod ComfyUI) is far cheaper (~$0.13 for a 6-clip scene vs per-clip fal). The same
  `Wan22ImageToVideoLatent` node supports `start_image` for chaining. That path lives in the
  `lossy` repo (`runpod_pod.py` + `strategies_video.py`); add a `--runpod` mode here only if
  you'll generate enough to justify the pod lifecycle + capacity flakiness. fal is the right
  default for a handful of clips.
