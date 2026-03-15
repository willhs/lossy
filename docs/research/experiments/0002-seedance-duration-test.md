---
id: experiment-0002-seedance-duration
type: note
purpose: "Document findings from Seedance 1.0 Pro Fast trial: same 15 shots as experiment 0001, comparing duration accuracy and quality."
scope: ["research", "decoder"]
non_goals: []
tags: ["experiment", "decoder", "video-generation", "seedance", "fal-ai"]
related: ["research/experiments/0001-first-e2e-decode-test.md", "design/adr/004-replicate-wan22-for-video-generation.md"]
---

## Setup

- **Source film**: Star Wars Episode IV (1080p, ~2h)
- **Shots tested**: 10-24 (same as experiment 0001, skipping opening text crawl)
- **Content**: Tatooine establishing shot, space battle (Rebel blockade runner vs Star Destroyer), first appearance of Stormtroopers, C-3PO, R2-D2
- **Models tested**: fal.ai Seedance 1.0 Pro Fast and Seedance 1.0 Pro (standard), both at 480p (16:9)
- **Duration control**: Integer seconds (2-12s), shots >12s split into multiple clips
- **Comparison baseline**: Experiment 0001 (Replicate Wan 2.2 Fast, fixed 5.06s clips)

## Cost

Actual costs from fal.ai billing (total $2.39 across both runs from $10.00 credit):

| Strategy | Estimated | Actual | Per shot | Full film est. |
|----------|-----------|--------|----------|----------------|
| Wan 2.2 Fast | $0.75 | $0.75 | $0.05 | ~$58 |
| Seedance Fast | $1.32 | ~$1.07 | ~$0.07 | ~$80 |
| Seedance Pro | $3.30 | ~$1.32 | ~$0.09 | ~$100 |

Our hardcoded cost estimates were too high -- actual fal.ai costs are roughly half what we estimated. The Pro model is only ~25% more than Fast in practice, not 2.5x as initially expected at 480p.

## Duration Accuracy

| Shot | Original | Seedance | Delta | Wan 2.2 | Delta |
|------|----------|----------|-------|---------|-------|
| 10 | 19.1s | 12+7s | -0.1s | 5.1s | -14.0s |
| 11 | 1.0s | 2s | +1.0s | 5.1s | +4.1s |
| 12 | 1.3s | 2s | +0.7s | 5.1s | +3.7s |
| 13 | 8.2s | 8s | -0.2s | 5.1s | -3.1s |
| 14 | 6.0s | 6s | +0.0s | 5.1s | -0.9s |
| 15 | 2.3s | 2s | -0.3s | 5.1s | +2.8s |
| 16 | 1.3s | 2s | +0.7s | 5.1s | +3.8s |
| 17 | 2.7s | 3s | +0.3s | 5.1s | +2.4s |
| 18 | 2.0s | 2s | -0.0s | 5.1s | +3.1s |
| 19 | 5.2s | 5s | -0.2s | 5.1s | -0.1s |
| 20 | 3.8s | 4s | +0.2s | 5.1s | +1.3s |
| 21 | 3.5s | 3s | -0.5s | 5.1s | +1.6s |
| 22 | 1.1s | 2s | +0.9s | 5.1s | +4.0s |
| 23 | 1.8s | 2s | +0.2s | 5.1s | +3.3s |
| 24 | 3.8s | 4s | +0.2s | 5.1s | +1.3s |

**Summary**: Seedance average absolute delta: 0.37s. Wan 2.2 average absolute delta: 3.23s. Seedance is **8.7x more accurate** on duration matching. The maximum Seedance delta is 1.0s (sub-1s shots floored to 2s minimum). The maximum Wan delta is 14.0s (the 19s establishing shot crammed into 5s).

## Quality Observations

### Duration matching

The timing/speed issues from experiment 0001 are essentially gone. Clips play at natural speed -- no more freeze frames from 5x speedup on short shots, no more slow-motion dreaminess on long shots. The reconstructed sequence feels much more like a film in terms of pacing and rhythm.

### Overall quality (Fast)

The clips are more recognisable as the Star Wars opening sequence. Seedance Fast produces more holistically coherent clips than Wan 2.2 -- the compositions feel more intentional and the content better matches the prompts.

### Seedance Pro vs Fast

Pro produces notably higher quality and continuity. Key differences:

- **C-3PO and R2-D2 look very good/realistic** -- a significant step up from Fast where C-3PO's face morphed between shots. Pro maintains more consistent character rendering.
- Overall continuity is improved -- clips feel more like they belong to the same film.
- The quality jump is substantial for a modest cost increase (~25% more at 480p).

### Characters and continuity

- **Droids (Fast)**: R2-D2 is recognisable and consistent within clips. C-3PO changes significantly between shots -- face, proportions, and style shift across cuts.
- **Droids (Pro)**: Both C-3PO and R2-D2 look realistic and well-rendered. Much better character consistency than Fast.
- **Stormtroopers**: Recognisable and relatively consistent across both models.
- **Spaceships**: Space/planet backgrounds and vessel designs vary between clips. Star Destroyers are identifiable but not visually consistent across shots.

### Shot splitting

Shot 10 (19s establishing shot) was split into 12s + 7s clips. Both parts generated from the same prompt with different seeds. The two parts don't have visual continuity (different compositions/angles) but concatenate without technical issues. The split is noticeable as a "cut" but no worse than the existing inter-shot discontinuity.

### Style consistency

Still the fundamental weakness of per-shot independent generation. Each clip has its own color grading and rendering style. This is inherent to the approach and not specific to Seedance -- would need image-to-video conditioning (feeding the last frame of clip N as input to clip N+1) to address.

## Key Takeaways

1. **Duration control is the clear win.** The 2-12s integer duration control eliminates the speed-adjustment problem that was experiment 0001's biggest issue. 8.7x more accurate on duration matching, and the perceptual improvement is even larger -- the difference between "unwatchable" and "watchable."

2. **Higher quality output** from Seedance vs Wan 2.2 at the same resolution. More coherent compositions, better prompt adherence, more recognisable content.

3. **Cost is lower than estimated.** Actual fal.ai costs were roughly half our hardcoded estimates. Seedance Pro for the full film would be ~$100, not $250 as initially projected. The quality and duration improvement is easily worth the premium over Wan 2.2 ($58).

4. **Strategy pattern works.** The refactored decoder cleanly supports both backends via `--strategy` flag. Adding future backends (RunPod self-hosted, other models) will be straightforward.

5. **Seedance Pro significantly improves character quality.** C-3PO and R2-D2 look realistic in Pro output, vs inconsistent/morphing in Fast. For only ~25% more cost at 480p, Pro is the better choice.

6. **Continuity remains the biggest unsolved problem.** Style drift between shots and varying space backgrounds. Pro reduces this somewhat but doesn't eliminate it. Likely requires image-to-video conditioning or some form of style transfer between shots.

7. **Shot splitting is acceptable but imperfect.** For shots exceeding 12s, splitting into multiple clips produces a visible "cut" mid-shot. Acceptable for the blog post but worth noting as a limitation.

## Next Steps

- Consider Seedance Pro as the primary strategy for the full-film decode run (~$100)
- Investigate image-to-video conditioning for style continuity between shots
- Build the side-by-side comparator to make quality assessment easier
