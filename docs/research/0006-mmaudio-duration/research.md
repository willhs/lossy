---
id: "0006"
type: research
purpose: Investigate MMAudio quality degradation on long clips
scope: audio generation
tags: [mmaudio, audio, quality, duration]
---

# MMAudio Duration vs Quality

## Problem

Audio eval shows quality drops with clip duration: <2s clips average 0.59 quality, >10s clips drop to 0.41. This document investigates why and what we can do about it.

## Findings

### MMAudio training duration

MMAudio V2 was trained exclusively on **8-second clips** from VGGSound. For longer datasets, it takes up to five non-overlapping 8s crops. Both model presets (`CONFIG_16K` and `CONFIG_44K`) hardcode `duration=8.0`.

Source: [MMAudio paper (arXiv 2412.15322)](https://arxiv.org/html/2412.15322v2)

### Variable-length inference

MMAudio deliberately uses **no absolute position encoding**, allowing it to generalize to different durations at inference time. The latent sequence length scales linearly with duration:

```
latent_seq_len = ceil(duration * sampling_rate / spectrogram_frame_rate / latent_downsample_rate)
```

For the `large_44k_v2` model (~43 latent tokens per second):

| Duration | Latent tokens | vs training (345) |
|----------|--------------|-------------------|
| 1s       | ~43          | 0.12x             |
| 5s       | ~216         | 0.63x             |
| 8s       | 345          | 1.00x (training)  |
| 10s      | ~431         | 1.25x             |
| 12s      | ~518         | 1.50x             |
| 20s      | ~863         | 2.50x             |
| 30s      | ~1,294       | 3.75x             |

Attention is quadratic in sequence length, so a 30s clip costs ~14x the compute of an 8s clip, in addition to quality degradation from out-of-distribution sequence lengths.

### Quality sweet spot

The MMAudio README warns: "Longer/shorter durations could also work, but a large deviation from the training duration may result in a lower quality."

The paper demonstrates inference on 10-second AudioCaps samples (trained on 8s), confirming modest extrapolation works. Community usage suggests **5-12 seconds** as the practical range.

### Why short clips also degrade

Clips under ~5s produce latent sequences significantly shorter than what the model saw in training (~43 tokens at 1s vs 345 at 8s). The attention patterns and temporal structure learned during training don't transfer well to these compressed representations.

### Previous pipeline settings

The pipeline used `MAX_DURATION = 30` and `MIN_DURATION = 1`, meaning:
- Short shots (<1s) were clamped to 1s (far from training duration)
- Long shots (>30s) were split into 30s + remainder chunks (both far from 8s)
- Multi-part clips were hard-concatenated with no crossfade

### Confounding factor

Longer shots may correlate with more complex audio descriptions (action sequences, layered ambiance), making generation harder independent of duration. This investigation focuses on the duration mismatch as the primary lever we can control.

## Changes Made

1. **Tightened duration bounds** to [5s, 10s] for all MMAudio strategies (`MMAudioStrategy`, `RunPodMMAudioStrategy`, pipelined audio in `RunPodWanStrategy`). Clips shorter than 5s are generated at 5s and trimmed by stitch. Clips longer than 10s are split into chunks within [5s, 10s].

2. **Added crossfade** (0.5s, triangle curve) between split audio parts in stitch.py, replacing hard concatenation. This smooths transitions at chunk boundaries.

## Expected Impact

- Short clips (1-4s): Quality should improve from generating closer to training duration, then trimming.
- Medium clips (5-10s): No change (already in the sweet spot).
- Long clips (11-30s): Split into more chunks but each chunk is higher quality. Crossfade smooths boundaries.
- Very long clips (30s+): More chunks but all within quality range.

## Future Work

- Re-run audio eval after these changes to quantify improvement
- Experiment with always generating exactly 8s regardless of target duration
- Consider video-conditioned MMAudio (currently using text-only mode) for better temporal alignment
