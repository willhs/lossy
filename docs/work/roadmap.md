---
id: roadmap
type: note
purpose: "Outline the Now/Next/Later priorities for the project."
scope: ["work"]
non_goals: []
tags: ["roadmap", "planning"]
related: []
---

## Now

- **VACE continuity test (proper range)** — rerun `runpod-vace` from shot 269 (Luke's first appearance) with ~25 shots. The initial test (shots 0–19) was the opening space battle — no human characters, inconclusive. See `experiments/0004-vace-continuity-test.md`.

## Done

- ~~**Pick a source film**~~ — Star Wars Episode IV (1080p, ~2h, 1,161 shots).
- ~~**Build the encoder**~~ — PySceneDetect shot detection + Gemini Flash-Lite prompt generation. Full pipeline tested.
- ~~**Build the decoder**~~ — Replicate Wan 2.2 Fast (480p, ~$0.05/clip), FFmpeg stitcher with speed-adjustment. 15-clip test batch successful.
- ~~**Fix speed-adjustment issues**~~ — added Seedance strategy with 2-12s duration control. Strategy pattern in decoder supports swappable backends.
- ~~**Audio generation**~~ — per-shot audio from sound descriptions via ElevenLabs/MMAudio (fal.ai). Audio muxed into reconstructed film during stitch.
- ~~**Cost tracking**~~ — per-stage API cost tracking (Gemini tokens, video/audio generation). Pipeline writes unified `costs.json` to output dir.
- ~~**Build the comparator**~~ — side-by-side view via `tools/compare.html` + dev server.
- ~~**Trial Seedance strategy**~~ — Seedance Fast duration control validated; A/B test on shots 10–24 (see experiments/0002, 0003). Wan-optimized prompt formatter confirmed better for subject accuracy.
- ~~**Character continuity pipeline**~~ — encode stage 3 (character registry via Gemini), portrait generation (fal.ai Flux Schnell), VACE strategy (RunPodVaceStrategy with reference conditioning + T2V fallback + prompt identity enrichment). Shot assignment refinement pass added. Initial test inconclusive due to wrong shot range — follow-up needed.

## Next

- **TI2V-5B evaluation** — if VACE-1.3B continuity test shows insufficient quality, TI2V-5B (Wan 2.2, unified T2V+I2V, 720P, fits 24 GB, new VAE) is the cleanest upgrade path (see research/0012).
- **End-to-end full run** — run decoder on all ~1,150 shots of Star Wars IV. Currently only partial runs have been done.

## Later

- **Audio encoding improvements** — WhisperX for dialogue/speaker diarization. Encode dialogue into the prompt manifest for future speech synthesis.
- **Write the blog post** — walk through the process, show results, reflect on what language preserves and what it loses.
- **Try variations** — different description detail levels, different video gen models, manual vs automated encoding. See how results change.
- **Compression ratio gag** — calculate the "bitrate" of the text manifest vs the original file size. Present it seriously.
- **Seedance 2.0** — fal.ai multimodal reference API (expected mid-2026) will allow character reference images per generation call without VACE or LoRA. Drop-in replacement for continuity once available.
