---
id: prompt-to-video-landscape
type: note
purpose: "Research options for generating video clips from text prompts, covering commercial APIs, open-source models, pricing, duration constraints, and stitching strategies."
scope: ["research", "decoder"]
non_goals: []
tags: ["research", "video-generation", "text-to-video", "decoder"]
related: ["research/shot-to-prompt-landscape.md", "design/architecture.md"]
---

## Problem

Given ~1,000-2,000 text prompts describing individual film shots (with metadata like duration, camera motion, and dialogue), generate a video clip for each and stitch them into a reconstructed film. Cost is the primary constraint -- the project needs to process ~1,000+ clips affordably. Low quality is acceptable and even desirable (the lossy reconstruction is the point).

## Key Constraint: Duration

Most text-to-video models generate fixed-length clips (typically ~5 seconds). Original shot durations vary widely (1-30+ seconds). This creates a fundamental mismatch that must be handled in post-processing.

### Duration Handling Strategies

1. **Speed-adjust with FFmpeg**: Generate fixed ~5s clips, then speed up or slow down to match original duration. Simple, preserves clip count, but fast shots look slow-motion and long shots look time-lapsed.
2. **Multiple clips per long shot**: For shots >6s, generate multiple clips and concatenate. Requires prompt splitting or continuation, and clips may not be visually coherent.
3. **Ignore duration**: Every shot becomes ~5s. Total film length changes. Simplest approach, acceptable for a first pass.
4. **Truncate/pad**: Trim long clips or freeze-frame the last frame for short ones.

## Commercial APIs

### Tier 1: Cheapest

#### Replicate -- Wan 2.2 (Recommended for v1)

The cheapest managed API option by a significant margin.

| Variant | Resolution | Frames | Duration | Cost/Clip |
|---|---|---|---|---|
| wan-2.2-t2v-fast | 480p (832x480) | 81-100 | 3.4-6.25s | ~$0.05 |
| wan-2.2-t2v | 720p | 81-100 | 3.4-6.25s | ~$0.15 |
| wan-2.5-t2v-fast | up to 1080p | similar | similar | ~$0.10 |

**API parameters** (wan-2.2-t2v-fast):
- `prompt`: text description (required)
- `num_frames`: 81-100 (81 gives best results)
- `aspect_ratio`: `16:9` (832x480) or `9:16` (480x832)
- `frames_per_second`: 5-24 (default 16, pricing based on 16fps)
- `sample_shift`: 1-20 (scheduler shift, default 12)
- `seed`: for reproducibility

**Duration math**: duration = num_frames / fps. At defaults (81 frames / 16 fps) = ~5.06s. Cannot go below 81 frames. Max (100/16) = 6.25s. Min achievable (81/24) = 3.375s.

**Pricing**: Billed per GPU-second. At 480p with defaults, ~39s compute = ~$0.05.

**1,161 shots at $0.05 = ~$58 total.**

#### fal.ai -- Multiple Models

Pay-per-second pricing. Clean API, multiple models available on same platform.

| Model | Cost/Second | Cost/5s Clip | Notes |
|---|---|---|---|
| Wan 2.5 | $0.05/s | $0.25 | Best Wan quality |
| CogVideoX-5B | flat | $0.20 | Lighter model |
| Pika 2.2 (720p) | flat | $0.20 | Good quality |
| Kling 2.5 Turbo Pro | $0.07/s | $0.35 | High quality |

**1,161 shots on fal.ai: $230-400 depending on model.**

#### SiliconFlow -- Wan 2.2

- Cost: $0.29 per video
- OpenAI-compatible API
- Chinese platform, alternative provider
- **1,161 shots: ~$337**

### Tier 2: Mid-Range

| Platform | Model | Cost/5s Clip | 1,161 Clips | Notes |
|---|---|---|---|---|
| Runway | Gen-4 Turbo | ~$0.25 | ~$290 | Higher quality, good API |
| Luma | Dream Machine | $0.20-0.50 | $230-580 | Via fal.ai or PiAPI |
| Replicate | PixVerse v4 | ~$0.30 | ~$348 | 360p |

### Tier 3: Premium

| Platform | Model | Cost/5s Clip | 1,161 Clips | Notes |
|---|---|---|---|---|
| Google | Veo 3.1 Fast | $0.50 | $580 | Best quality, Vertex AI |
| Kling | 3 Pro | $1.12 | $1,300 | Official API, 90-day prepaid |
| Google | Veo 3 | $2.00+ | $2,300+ | Overkill |

### API Availability Notes

- **Stability AI (Stable Video Diffusion)**: API deprecated as of July 2025. Self-hosted only.
- **Hugging Face Inference API**: Supports text-to-video but pricing is opaque and not competitive.

## Open-Source / Self-Hosted

### Wan 2.2 on RunPod

Best self-hosted option if cost needs to be minimized further.

- **GPU**: RTX 4090 at $0.39/hr, A100 at $1.89/hr
- **Generation time**: ~39s per 5s clip at 480p
- **Throughput**: ~90 clips/hour on RTX 4090
- **Cost per clip**: ~$0.004-0.15 depending on GPU and utilization
- **VRAM**: 8GB min (5B model), 24GB recommended (14B model)
- **1,161 clips on RTX 4090: ~$5-15** (13 hours continuous, optimistic)

The 5B parameter variant is nearly as good as 14B but 30% faster/cheaper.

### CogVideoX-5B

- **VRAM**: 12GB+ (RTX 3060 12GB works)
- **Output**: 6-second clips at 720x480
- **License**: Open source
- Lighter than Wan, runs on cheaper hardware

### LTX-Video 2.3

- **VRAM**: 8GB min (quantized), 24GB recommended
- **Speed**: ~45s for 4s 720p on RTX 4090; as fast as 4s on H100
- **License**: Open source (Lightricks)
- Extremely fast inference, good for batch processing

### Mochi 1 (Genmo)

- **VRAM**: 24GB+ (designed for H100-class)
- **Cost**: ~$0.33/clip on H100 cloud
- **License**: Apache 2.0
- High quality but needs beefy hardware

## Cost Summary

| Platform | Model | Cost/Clip | 1,161 Clips | Quality |
|---|---|---|---|---|
| RunPod (self-hosted) | Wan 2.2 5B | ~$0.004-0.01 | $5-15 | Low-med |
| Replicate | Wan 2.2 Fast | ~$0.05 | ~$58 | Low-med |
| fal.ai | CogVideoX-5B | $0.20 | ~$232 | Medium |
| fal.ai | Pika 2.2 | $0.20 | ~$232 | Medium |
| fal.ai | Wan 2.5 | $0.25 | ~$290 | Medium |
| Runway | Gen-4 Turbo | ~$0.25 | ~$290 | High |
| Google | Veo 3.1 Fast | $0.50 | ~$580 | Very high |

## Recommended Approach

### v1: Replicate + Wan 2.2 Fast

**Why**: $58 total, simple Python SDK, 480p matches the "comically lossy" aesthetic, no infrastructure to manage.

```python
import replicate
output = replicate.run(
    "wan-video/wan-2.2-t2v-fast",
    input={
        "prompt": "A wide shot of a star destroyer...",
        "num_frames": 81,
        "aspect_ratio": "16:9",
        "frames_per_second": 16
    }
)
```

**Duration strategy for v1**: Generate fixed ~5s clips, speed-adjust with FFmpeg:
```bash
# Speed up a 5s clip to fit a 2.5s slot (2x speed)
ffmpeg -i clip.mp4 -filter:v "setpts=0.5*PTS" -an output.mp4
# Slow down a 5s clip to fit a 10s slot (0.5x speed)
ffmpeg -i clip.mp4 -filter:v "setpts=2.0*PTS" -an output.mp4
```

### Future iterations

- Try Wan 2.5 or Pika 2.2 on fal.ai for quality comparison (~$230-290)
- Try self-hosted Wan on RunPod for cost comparison (~$5-15)
- Experiment with image-to-video models using extracted keyframes as starting frames for better visual fidelity

## Open Questions

- How well does Wan 2.2 handle cinematic prompts (camera movements, lighting descriptions, specific compositions)?
- Is 480p sufficient for the side-by-side comparison to be interesting, or does it just look like noise?
- Should prompts be reformatted for video gen models? (They were written for human/descriptive purposes, not as video gen prompts -- different models respond to different prompt styles.)
- Is there value in using image-to-video with extracted keyframes as conditioning, rather than pure text-to-video?
- For shots with dialogue, should we attempt TTS audio overlay or leave the reconstruction silent?

## Sources

### APIs and Pricing
- [Replicate Wan 2.2 Fast T2V](https://replicate.com/wan-video/wan-2.2-t2v-fast)
- [Replicate Wan 2.2 API Schema](https://replicate.com/wan-video/wan-2.2-t2v-fast/api/schema)
- [Replicate Text-to-Video Collection](https://replicate.com/collections/text-to-video)
- [Replicate Pricing](https://replicate.com/pricing)
- [Replicate Blog: Wan 2.2](https://replicate.com/blog/wan-22)
- [fal.ai Pricing](https://fal.ai/pricing)
- [Runway API Pricing](https://docs.dev.runwayml.com/guides/pricing/)
- [Google Vertex AI Pricing](https://cloud.google.com/vertex-ai/generative-ai/pricing)
- [Kling API Pricing](https://klingai.com/global/dev/pricing)
- [Stability AI API Pricing Update](https://stability.ai/api-pricing-update-25)

### Models and Benchmarks
- [Wan 2.2 GitHub](https://github.com/Wan-Video/Wan2.2)
- [Wan 2.2 Models Essentials](https://help.scenario.com/en/articles/wan-2-2-models-the-essentials/)
- [CogVideoX on fal.ai](https://fal.ai/models/fal-ai/cogvideox-5b)
- [LTX Video System Requirements](https://docs.ltx.video/open-source-model/getting-started/system-requirements)
- [Open Source Video Models Comparison](https://www.hyperstack.cloud/blog/case-study/best-open-source-video-generation-models)

### Cost Analysis
- [RunPod GPU Pricing](https://www.runpod.io/pricing)
- [Wan 2.2 RunPod Cost Analysis](https://apatero.com/blog/wan-ai-server-costs-runpod-complete-analysis-2025)
- [SiliconFlow Video Models Guide](https://www.siliconflow.com/articles/en/cheapest-video-multimodal-models)
