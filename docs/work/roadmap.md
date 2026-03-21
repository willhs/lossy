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

- **End-to-end run** — run decoder on full Star Wars EP IV (~1,150 shots after credits). Iterate on prompt quality and scene granularity.
- **Trial Seedance strategy** — run shots 10-24 with Seedance Fast to validate duration control eliminates speed-adjustment artifacts.
- **Build the comparator** — side-by-side or alternating view of original vs reconstructed.

## Done

- ~~**Pick a source film**~~ — Star Wars Episode IV (1080p, ~2h, 1,161 shots).
- ~~**Build the encoder**~~ — PySceneDetect shot detection + Gemini Flash-Lite prompt generation. Full pipeline tested.
- ~~**Build the decoder**~~ — Replicate Wan 2.2 Fast (480p, ~$0.05/clip), FFmpeg stitcher with speed-adjustment. 15-clip test batch successful.
- ~~**Fix speed-adjustment issues**~~ — added Seedance strategy with 2-12s duration control. Strategy pattern in decoder supports swappable backends.
- ~~**Audio generation**~~ — per-shot audio from sound descriptions via ElevenLabs/MMAudio (fal.ai). Audio muxed into reconstructed film during stitch.

## Next

- **Cost tracking** — log API costs per stage so the blog post can report total "compression cost."

## Later

- **Audio encoding improvements** — WhisperX for dialogue/speaker diarization. Encode dialogue into the prompt manifest for future speech synthesis.
- **Write the blog post** — walk through the process, show results, reflect on what language preserves and what it loses.
- **Try variations** — different description detail levels, different video gen models, manual vs automated encoding. See how results change.
- **Compression ratio gag** — calculate the "bitrate" of the text manifest vs the original file size. Present it seriously.
