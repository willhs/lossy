---
id: experiment-0001-first-e2e-decode
type: note
purpose: "Document findings from the first end-to-end decode test: 15 shots from Star Wars EP IV reconstructed via Wan 2.2."
scope: ["research", "decoder"]
non_goals: []
tags: ["experiment", "decoder", "video-generation", "wan-2.2"]
related: ["design/adr/004-replicate-wan22-for-video-generation.md", "research/prompt-to-video-landscape.md"]
---

## Setup

- **Source film**: Star Wars Episode IV (1080p, ~2h)
- **Shots tested**: 10-24 (first 15 after credits/crawl)
- **Content**: Tatooine establishing shot, space battle (Rebel blockade runner vs Star Destroyer), first appearance of Stormtroopers, C-3PO, R2-D2
- **Model**: Replicate `wan-video/wan-2.2-t2v-fast`, 480p (832x480), 81 frames at 16fps (~5.06s per clip)
- **Prompts**: Structured descriptions from Gemini Flash-Lite, flattened into cinematic text prompts
- **Stitching**: FFmpeg setpts speed-adjustment to match original shot durations, then concat

## Cost

- 15 clips at ~$0.05 each = **$0.75 total**
- Matches estimate exactly
- Extrapolated full film (~1,150 shots after credits): **~$57.50**

## Duration Distribution

| Metric | Value |
|---|---|
| Total original duration | 62.7s |
| Average shot duration | 4.2s |
| Clip generation duration | ~5.06s (fixed) |
| Short shots (<3s) | 8 of 15 (53%) |
| Long shots (>8s) | 2 of 15 (13%) |
| Duration range | 1.0s - 19.1s |

Speed factors ranged from 0.19x (shot 11, 1.0s original compressed into 5s clip) to 3.77x (shot 10, 19.1s original). Over half the test shots required significant speed-up (factor < 0.75).

## Quality Observations

### What worked well

- **Recognizable characters**: Stormtroopers looked convincingly like Stormtroopers. R2-D2 was identifiable and well-rendered.
- **Star Destroyers**: At least one Star Destroyer shot was clearly recognizable as the correct ship type and composition.
- **Mood and atmosphere**: The generated clips broadly captured the right lighting and color palette -- space scenes were dark with correct accent colors, interior corridor scenes had the right sterile feel.

### What didn't work

- **C-3PO consistency**: The golden droid had intra-clip continuity issues. In one frame his face morphed into a human face -- classic video gen uncanny valley.
- **Wrong content in space scenes**: One clip generated an X-wing fighter when the prompt described a different vessel. The model's training data likely biases toward iconic Star Wars imagery regardless of the specific prompt.
- **Style inconsistency between clips**: Each clip was generated independently, so there's no visual continuity -- different color grading, different rendering styles, jarring transitions between shots.
- **Abstract scenes**: Planet/moon establishing shots were less convincing than character or ship shots. The model is better at concrete objects than atmospheric compositions.

### Speed-adjustment issues

The most significant technical issue. With 53% of shots under 3 seconds, over half the clips need to be sped up by 2-5x to match original duration. Results:

- **Shots < 1.5s** (speed factor < 0.3): VLC shows effectively a still image. The adjusted clip is so short that playback shows 1-2 frames.
- **Shots 1.5-3s** (speed factor 0.3-0.6): Clips play but feel unnaturally fast, like a slideshow.
- **Shots 3-6s** (speed factor 0.6-1.2): Sweet spot. Clips look natural at these durations.
- **Shots > 8s** (speed factor > 1.6): Slow-motion effect. The 19s establishing shot plays at nearly 4x slowdown, making the generated motion feel dreamlike/underwater.

## File Sizes

Average clip size: ~1.1MB. Total for 15 clips: ~16.4MB. Stitched output: 8.4MB.

Notably, shot 11 (the shortest at 1.0s original) produced the smallest clip at 95KB -- suggesting Wan 2.2 generates less visual content for prompts describing brief, simple actions.

## Stitched Output Assessment

The reconstructed 62.7-second video is **interesting but chaotic**. There are moments where consecutive clips feel like they belong to the same film -- particularly the Stormtrooper and droid sequences. But the overall experience is disjointed: each clip is its own little world with different visual style, and the speed-adjustment artifacts break immersion on the short shots.

The result is more "AI art installation" than "reconstructed film" -- which may actually be more interesting for the blog post.

## Key Takeaways

1. **$0.05/clip pricing confirmed**. Full film decode is viable at ~$58.
2. **Character shots >> space shots** for recognizability. The model handles concrete subjects much better than atmospheric compositions.
3. **Speed-adjustment is the weakest link**. Over half the shots in this sequence are too short for a fixed 5s clip. Options for v2:
   - Use a model with variable duration (Seedance 1.0 Pro: 2-12s integer steps)
   - Set a minimum duration floor (e.g., 3s) and accept temporal inaccuracy
   - Generate at a lower FPS to create shorter clips from the same frame count
4. **Style inconsistency is inherent** to per-shot independent generation. Would require image-to-video conditioning (feeding the last frame of clip N as input to clip N+1) to achieve visual continuity. This is a v2/v3 concern.
5. **Prompt content leakage**: The model sometimes generates iconic Star Wars imagery that wasn't in the prompt, suggesting its training data influences output beyond the text prompt. This is actually on-brand for lossy -- the "codec" introduces its own biases.

## Next Steps

- Run the full ~1,150 shot decode and assess at scale
- Investigate minimum duration floor or FPS adjustment for short shots
- Build the comparator for side-by-side viewing with original
