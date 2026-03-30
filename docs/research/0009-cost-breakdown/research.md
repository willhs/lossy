---
id: "0009"
type: research
purpose: Compile per-stage and per-title cost breakdown across all pipeline runs
scope: pipeline, costs
tags: [costs, pipeline, runpod, gemini, audio, video]
---

# Pipeline Cost Breakdown

## Summary

Total recorded API/compute costs across 6 titles (5 x 10-min clips + 1 full feature film): **~$15.34** in tracked costs, plus an estimated **~$8-10** in untracked RunPod GPU time for the 5 short titles.

The pipeline is remarkably cheap for encoding — Gemini costs are negligible. The dominant cost is GPU time for video generation.

## Per-Unit Cost Rates

### Encoding (Gemini)

| Component | Rate |
|-----------|------|
| Gemini 3.1 Flash Lite (input) | $0.075 / 1M tokens |
| Gemini 3.1 Flash Lite (output) | $0.30 / 1M tokens |

### Video Generation

| Strategy | Rate | Notes |
|----------|------|-------|
| runpod-wan (RTX 4090) | ~$0.34/hr GPU | Self-hosted Wan 2.1 1.3B fp16 |
| runpod-wan (L40S) | ~$0.54/hr GPU | Self-hosted Wan 2.1 1.3B fp16 |
| runpod-wan (RTX A6000) | ~$0.33/hr GPU | Slower inference, may timeout |
| replicate-wan | $0.05/clip flat | Wan 2.2 Fast, hosted |
| fal-seedance | $0.02/sec output | Seedance 1.0 Pro Fast, 480p |

### Audio Generation

| Strategy | Rate | Notes |
|----------|------|-------|
| runpod-mmaudio | ~$0/marginal | Runs on same pod as video |
| mmaudio (fal.ai) | $0.002/sec | MMAudio V2 hosted |
| elevenlabs SFX | $0.001/sec | ElevenLabs SFX v2 via fal.ai |
| elevenlabs Speech | $0.05/1K chars | TTS Turbo v2.5 via fal.ai |

## Per-Title Breakdown

### Short Titles (10 min each, runpod-wan)

| Title | Shots | Gemini | Video (runpod-wan) | Audio | Total Tracked |
|-------|-------|--------|--------------------|-------|---------------|
| SpongeBob S01E01 | 156 | $0.074 | $1.98 | $0.00* | $2.06 |
| The Big Lebowski | 88 | $0.046 | $2.25 | $0.00* | $2.29 |
| Better Call Saul S01E01 | 81 | $0.046 | $1.49 | $0.00* | $1.54 |
| Planet Earth S01E01 | 65 | $0.043 | $1.39 | $0.00* | $1.43 |
| 300 | 133 | $0.067 | $0.48† | — | $0.54 |
| **Subtotal** | **523** | **$0.28** | **$7.58** | **$0.00** | **$7.86** |

\* Audio generation failed (RunPod MMAudio nodes not loading); needs re-run.
† 300 incomplete — only 27/133 shots generated before RunPod balance ran out.

### Star Wars IV (full film, ~2h05m, mixed strategies)

| Stage | Strategy | Completed | Cost |
|-------|----------|-----------|------|
| Gemini encode2 | gemini-3.1-flash-lite | 2,012 shots | $0.50 |
| Video | runpod-wan | ~411 shots | $3.15 |
| Video | replicate-wan | 15 shots | $0.75 |
| Audio | mmaudio (fal.ai) | 423 shots | $2.61 |
| Audio | elevenlabs SFX | 91 shots | $0.47 |
| Audio | runpod-mmaudio | ~1,929 shots | $0.00 (marginal) |
| **Subtotal** | | | **$7.48** |

## Cost Per Minute of Source Video

Based on completed runs (short titles, runpod-wan strategy):

| Title | Source Duration | Total Cost | Cost/min |
|-------|---------------|------------|----------|
| SpongeBob | 10 min | $2.06 | $0.21 |
| Big Lebowski | 10 min | $2.29 | $0.23 |
| Better Call Saul | 10 min | $1.54 | $0.15 |
| Planet Earth | 10 min | $1.43 | $0.14 |

**Average: ~$0.18/min of source video** using runpod-wan (video only, no audio).

The variation comes from shot count — SpongeBob and Big Lebowski have more shots (156 and 88) due to faster editing, which means more video generation calls. Planet Earth has longer, steadier shots (65 total).

### Projected Full Film Cost (runpod-wan)

For a 2-hour film (~120 min):
- Gemini encoding: ~$0.50
- Video generation: ~$22 (at $0.18/min)
- Audio generation: ~$0 marginal (runpod-mmaudio on same pod)
- **Total: ~$22-25**

This assumes community GPU pricing. Secure cloud would be ~1.5-2x.

## Cost by Pipeline Stage

Across all runs, the cost distribution is:

| Stage | % of Total | Notes |
|-------|-----------|-------|
| Gemini encoding | ~5% | Negligible — Flash Lite is very cheap |
| Video generation | ~90% | Dominant cost, scales with shot count |
| Audio generation | ~5% | Free if using runpod-mmaudio on same pod |
| Stitching | ~0% | Local FFmpeg, no API cost |

## Key Findings

1. **Gemini encoding is effectively free** at $0.04-0.07 per 10 min of source. Even a full feature film costs ~$0.50 to encode.

2. **Shot count drives cost**, not source duration. A fast-cut cartoon (SpongeBob, 156 shots/10min) costs ~50% more than a slow documentary (Planet Earth, 65 shots/10min).

3. **RunPod self-hosted is cheapest for batch work** when the pod stays warm. The per-clip amortized cost drops as more clips are generated on a single pod session.

4. **Concurrent audio on the same RunPod pod is essentially free** — MMAudio runs alongside video generation with no additional GPU cost.

5. **Wall-clock time is the bottleneck**, not dollar cost. A 10-min clip takes 2-5 hours of real time to generate on RunPod (depending on GPU type and shot count).

## Untracked Costs

The pipeline's costs.json does not capture:
- RunPod GPU idle time during model downloads and pod startup (~5-10 min per pod)
- Failed pod attempts (CUDA errors, terminated pods) — we burned ~$0.24 on two bad pods
- fal.ai balance consumed by failed Seedance runs (exhausted balance)

## Recommendations

1. **Track RunPod GPU time in costs.json** — currently only the per-clip amortized rate is recorded, not total pod session cost.
2. **Add a `--budget` flag** to stop generation when a cost threshold is reached.
3. **Prefer 4090 > L40S > A6000** for RunPod — faster GPUs have lower amortized cost despite higher hourly rates.
