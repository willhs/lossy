---
id: architecture
type: spec
purpose: "Describe the end-to-end system shape and integration points."
scope: ["design", "architecture"]
non_goals: []
tags: ["architecture"]
related: ["design/adr/"]
---

## Overview

lossy is a three-stage pipeline: **encode**, **decode**, and **compare**. Each stage is a standalone CLI command that reads from and writes to a shared intermediate format on disk.

```
Source Film → [Encoder] → Scene Manifest (JSON) → [Decoder] → Reconstructed Film
                                                        ↓
                                                   [Comparator] → Side-by-side output
```

## Intermediate Format — Scene Manifest

The core data structure is a JSON manifest describing the film as an ordered list of scenes:

```json
{
  "source": { "title": "...", "duration_s": 5400 },
  "scenes": [
    {
      "index": 0,
      "start_s": 0.0,
      "end_s": 12.4,
      "description": "...",
      "dialogue": "...",
      "mood": "...",
      "camera": "..."
    }
  ]
}
```

This manifest is the "compressed" representation — a text file that stands in for the entire film.

## Pipeline Stages

### Encode

Input: video file on disk. Output: scene manifest JSON.

1. Split the source video into scenes (FFmpeg scene detection or fixed intervals).
2. Sample frames from each scene.
3. Feed frames to a vision model API to produce a text description per scene.
4. Write the scene manifest.

The encoder can also accept a manually authored manifest for testing or artistic control.

### Decode

Input: scene manifest JSON. Output: directory of generated video clips + stitched output.

1. Read each scene description from the manifest.
2. Send each description as a prompt to a video generation backend (selectable via `--strategy`).
3. Download generated clips. Long shots may be split into multiple clips by the strategy.
4. Speed-adjust clips to match original shot durations (skipped when strategy produces duration-matched clips).
5. Stitch clips sequentially into a single reconstructed film (FFmpeg concat).

Supported strategies:
- `replicate-wan` -- Replicate Wan 2.2 Fast, fixed ~5s clips, cheapest ($0.05/clip)
- `fal-seedance` -- fal.ai Seedance 1.0 Pro Fast, 2-12s duration control (~$0.10/clip at 480p)

### Compare

Input: original video + reconstructed video. Output: side-by-side comparison video.

1. Align the two videos by scene boundaries.
2. Render a split-screen or alternating view for visual comparison.

## Key Constraints

- **Cost** — every external API call costs money. The pipeline should default to the cheapest viable options (low resolution, short clips, fewest frames sampled).
- **Stateless stages** — each stage reads files and writes files. No database, no server, no persistent state.
- **CLI-first** — all interaction is via command-line tools. No web UI.

## Tech Stack (Provisional)

- **Language**: Python (rich ecosystem for video/ML tooling)
- **Video processing**: FFmpeg (via subprocess)
- **Vision model**: Cloud API (Claude, Gemini, etc.) — whichever accepts video or image input cheaply
- **Video generation**: Swappable backends via strategy pattern -- Replicate Wan 2.2 Fast (default, cheapest), fal.ai Seedance 1.0 Pro Fast (duration control)
- **Output**: Blog post artifacts (comparison videos, screenshots, metrics)
