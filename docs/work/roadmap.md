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

- **Pick a source film** — choose something short (a scene, a short film, or a single TV episode) to use as the test subject.
- **Build the encoder** — scene splitting (FFmpeg) + vision model API calls to produce a scene manifest from a video file.
- **Build the decoder** — read a scene manifest, call a video generation API per scene, download clips, stitch with FFmpeg.

## Next

- **End-to-end run** — encode a real film, decode it, and watch the result. Iterate on prompt quality and scene granularity.
- **Build the comparator** — side-by-side or alternating view of original vs reconstructed.
- **Cost tracking** — log API costs per stage so the blog post can report total "compression cost."

## Later

- **Audio encoding** — audio is 38% of the source file and currently not captured at all. Use YAMNet for sound classification (music, effects, ambient) and WhisperX for dialogue/speaker diarization. Encode audio cues into the prompt manifest so the decoder can reconstruct a soundtrack.
- **Write the blog post** — walk through the process, show results, reflect on what language preserves and what it loses.
- **Try variations** — different description detail levels, different video gen models, manual vs automated encoding. See how results change.
- **Compression ratio gag** — calculate the "bitrate" of the text manifest vs the original file size. Present it seriously.
