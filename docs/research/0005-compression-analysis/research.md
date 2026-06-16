---
id: 0005-compression-analysis
type: note
purpose: "Measure the compression ratio achieved by lossy encoding on Star Wars IV and break down what contributes to encoded output size."
scope: ["research", "encoder", "compression"]
non_goals: ["quality assessment", "decode fidelity"]
tags: ["research", "compression", "encoding", "star-wars"]
related: ["research/0002-shot-to-prompt/research.md"]
---

## Question

How much does lossy encoding compress a film, and what contributes to the encoded output size?

## Test Subject

**Star Wars: Episode IV - A New Hope**
- Source file: `star_wars_iv.mp4` -- 632 MB
- Duration: 124.7 minutes (7,484 seconds)
- Detected shots: 2,070
- Encoded prompts: 2,013 (some shots merged or filtered)

## Encoded Output Breakdown

The "encoded" representation is everything needed to reconstruct the film from scratch -- no keyframes, no intermediate artifacts, just the text description.

### Core encoded files

| File | Size | Purpose |
|------|------|---------|
| `prompts.json` | 2.8 MB | Text descriptions of each shot (visual, dialogue, audio) |
| `manifest.json` | 772 KB | Shot timing, scene boundaries, frame ranges |
| `audio_labels.json` | 297 KB | Per-shot audio/sound effect descriptions |
| `camera_motion.json` | 38 KB | Camera movement annotations per shot |
| **Total** | **3.9 MB** | |

### Per-component breakdown

- **prompts.json** dominates at 72% of encoded size. Each prompt entry averages ~1.2 KB and contains: index, timing (start/end/duration), camera motion flag, audio flag, dialogue text, and visual description.
- **manifest.json** at 20% stores structural metadata: shot boundaries, timecodes, frame numbers, and keyframe references (used during encoding only).
- **audio_labels.json** at 7.6% stores sound descriptions that feed into MMAudio generation.
- **camera_motion.json** at 1% stores detected camera movements (pan, tilt, zoom, etc.).

### What about keyframes?

The `keyframes/` directory contains 9,796 JPEG reference frames totalling 188 MB. These are used during **encode stage 2 only** -- they're sent to Gemini alongside video context to generate better text descriptions. The decode stage (`decode.py`, `strategies_video.py`) does not reference keyframes at all. They are an intermediate encoding artifact, not part of the transmitted representation.

## Compression Ratios

| Metric | Value |
|--------|-------|
| Source size | 632 MB |
| Encoded size (text only) | 3.9 MB |
| **Compression ratio** | **~162x** |
| Encoded as % of original | 0.6% |
| Bytes per second of film | ~521 bytes/s |
| Bytes per shot (avg) | ~1.9 KB |

For context, a typical MP4 at this duration runs ~710 KB/s. The lossy text encoding achieves ~521 B/s -- roughly 1,400x less data rate than the compressed video.

## Observations

1. **Text descriptions are remarkably compact.** 2 hours of visual, audio, and dialogue information fits in under 4 MB. This is roughly equivalent to a short e-book.

2. **The manifest is surprisingly large** relative to its role. At 772 KB for timing metadata, it's ~20% of the encoded output. This is because it stores per-shot keyframe filename lists and frame-level boundaries that aren't needed for decode. A stripped-down manifest with just timecodes would be significantly smaller.

3. **Keyframes are the encoding bottleneck, not the output.** They're 48x larger than the text output but serve only as context for the LLM during prompt generation. They could be discarded after encoding completes without affecting reconstruction.

4. **Audio descriptions are cheap.** At 297 KB, sound effect labels add less than 8% to the encoded size but enable an entire audio reconstruction track (MMAudio).

## Potential Optimisations (Not Pursued)

- **Strip manifest to decode-only fields** (timecodes, duration) -- could halve its size.
- **Delta-encode similar shots** -- consecutive shots in the same scene often share description fragments.

## Generic Text Compression (measured)

Compressing the four encoded JSON files as a tarball, max settings on each compressor (2026-05-15, on the Star Wars IV encoded output):

| Compressor | Combined output | Ratio vs 3.91 MB raw | Ratio vs 632 MB source |
|------------|-----------------|----------------------|------------------------|
| gzip -9    | 584 KB          | 6.9×                 | ~1,083×                |
| zstd -22   | 374 KB          | 10.7×                | ~1,690×                |
| brotli -11 | 366 KB          | 11.0×                | ~1,727×                |
| xz -9e     | **350 KB**      | **11.5×**            | **~1,765×**            |

Per-file (raw → xz -9e):

- `prompts.json`: 2.83 MB → 292 KB
- `manifest.json`: 772 KB → 45 KB
- `audio_labels.json`: 297 KB → 17 KB
- `camera_motion.json`: 38 KB → 3 KB

**Even plain gzip puts a feature film under 1 MB**, and xz nearly halves it again. The hypothesised 3-5× from gzip was conservative -- actual is ~7× -- because the prompts are unusually repetitive: consistent JSON keys, plus a small vocabulary of character and place names recurring across 2,070 shots. No schema changes, no delta encoding, no manifest stripping required to hit sub-1 MB.
