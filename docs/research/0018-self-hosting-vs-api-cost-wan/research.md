---
id: "0018"
type: research
purpose: "Compare the cost of running Wan 2.1 and 2.2 via RunPod self-hosting versus managed API providers (fal.ai, Replicate) for a full film run (~2000 clips, Star Wars IV scale)."
scope: ["costs", "runpod", "fal.ai", "replicate", "wan"]
tags: [costs, runpod, fal.ai, replicate, wan, self-hosting, research]
related:
  - "research/0009-cost-breakdown/research.md"
  - "research/0012-wan-model-variants/research.md"
  - "research/0017-video-gen-speed-benchmarks/research.md"
sources:
  - "https://www.runpod.io/pricing"
  - "https://fal.ai/models/fal-ai/wan/v2.2-a14b/text-to-video"
  - "https://fal.ai/models/fal-ai/wan/v2.2-a14b/image-to-video"
  - "https://replicate.com/blog/wan-22"
  - "https://replicate.com/pricing"
  - "https://www.together.ai/models/wan-2-2-t2v"
  - "https://www.instasd.com/post/wan2-1-performance-testing-across-gpus"
  - "https://blogs.novita.ai/wan-2-2-vram-find-the-best-gpu-setup-for-deployment/"
---

# Self-Hosting vs. API Cost: Wan 2.1 and 2.2 at Film Scale

## Hypothesis

Self-hosting Wan on RunPod is cheaper than using managed API providers for a full-film generation run (~2000 clips).

**Verdict: True for standard quality, nuanced for optimised API variants.**

---

## Setup Assumptions

- **Film**: Star Wars IV (A New Hope) — 2,012 shots from the actual `encode.py` output (see `research/0009-cost-breakdown`)
- **Clip duration**: ~5 seconds per shot (81 frames at 16fps, or variable depending on model)
- **Resolution**: 480p as baseline; 720p comparison included where practical
- **Self-hosting**: Single pod, RunPod community cloud (spot), sequential generation (no concurrency)
- **API**: Per-clip flat rates or per-second-of-output billing as published June 2026

All prices are current as of **June 2026**. GPU spot pricing fluctuates ±15%.

---

## Part 1: API Provider Pricing

### fal.ai

| Model | Billing | 480p (5s clip) | 720p (5s clip) |
|-------|---------|----------------|----------------|
| Wan 2.1 T2V / I2V | per video | $0.20 | $0.40 |
| Wan 2.2 A14B T2V / I2V | per output second | $0.20 (5×$0.04/s) | $0.40 (5×$0.08/s) |
| **Wan 2.2 A14B Turbo** | per video (flat) | **$0.05** | **$0.10** |
| Wan 2.2 5B | per video (flat) | $0.15 | — |

The Turbo variant uses the full A14B MoE model with speed optimisations. It is the cheapest fal.ai option for Wan 2.2 quality.

### Replicate

| Model | Billing | 480p (5s clip) | 720p (5s clip) |
|-------|---------|----------------|----------------|
| Wan 2.1 1.3B | per video | $0.20 | — |
| Wan 2.1 wavespeedai I2V | per output second | $0.45 (5×$0.09/s) | $1.25 (5×$0.25/s) |
| **Wan 2.2 T2V/I2V Fast** | per video (flat) | **$0.05** | **$0.10** |
| Wan 2.2 I2V standard | per video (flat) | $0.40 | $1.00 |

The Fast variant uses PrunaAI-optimised Wan 2.2. Generation time is ~30s wall-clock, significantly faster than RunPod self-hosted.

### Together AI and Others

| Provider | Model | Cost per clip |
|----------|-------|---------------|
| Together AI | Wan 2.2 I2V A14B | $0.31 |
| Together AI | Wan 2.2 T2V A14B | $0.66 |
| AIMLAPI | Wan 2.2 T2V Plus 480p | $0.105 |

Together AI's T2V pricing is significantly more expensive than fal.ai/Replicate for equivalent output.

### API Cost for 2000 Clips

| Provider / Model | Per Clip | **2000 Clips** |
|-----------------|----------|----------------|
| fal.ai Wan 2.2 Turbo (480p) | $0.05 | **$100** |
| Replicate Wan 2.2 Fast (480p) | $0.05 | **$100** |
| fal.ai Wan 2.2 Turbo (720p) | $0.10 | **$200** |
| Replicate Wan 2.2 Fast (720p) | $0.10 | **$200** |
| fal.ai / Replicate Wan 2.1 (480p) | $0.20 | **$400** |
| fal.ai Wan 2.2 A14B standard (480p) | $0.20 | **$400** |
| fal.ai / Replicate Wan 2.1 (720p) | $0.40 | **$800** |
| Replicate Wan 2.2 I2V standard (480p) | $0.40 | **$800** |
| Together AI Wan 2.2 I2V | $0.31 | **$620** |
| Replicate Wan 2.2 I2V standard (720p) | $1.00 | **$2,000** |

---

## Part 2: RunPod Self-Hosting

### VRAM Requirements by Model

| Model | VRAM (fp16) | VRAM (fp8) | Fits RTX 4090 (24 GB) |
|-------|------------|------------|----------------------|
| Wan 2.1 T2V-1.3B | ~8 GB | — | Yes, + concurrent audio |
| Wan 2.1 T2V-14B | ~32 GB | ~16 GB | fp8 only, tight |
| Wan 2.1 I2V-14B 480p | ~32 GB | ~20 GB | fp8 only, no audio |
| Wan 2.2 TI2V-5B | ~24 GB | ~12 GB | Borderline fp16; yes fp8 |
| Wan 2.2 A14B (MoE, 2 files) | ~60 GB+ | ~30 GB+ | No — needs A100/H100 |

### GPU Pricing (RunPod, June 2026)

| GPU | VRAM | Community Cloud (spot) | Secure Cloud (on-demand) |
|-----|------|----------------------|--------------------------|
| RTX 4090 | 24 GB | $0.34/hr | $0.69/hr |
| L40S | 48 GB | $0.54/hr | ~$0.89/hr |
| A100 PCIe 80GB | 80 GB | $1.39/hr | $2.10/hr |
| A100 SXM 80GB | 80 GB | $1.49/hr | $2.45/hr |
| H100 SXM | 80 GB | $2.69/hr | $4.76/hr |

Spot pricing fluctuates with supply; secure cloud is guaranteed but typically 1.5–2× spot.

RunPod also offers serverless per-second billing (RTX 4090 PRO: $0.00031/sec ≈ $1.12/hr equivalent) — relevant for bursty workloads but generally more expensive than a warm pod for batch runs.

### Generation Speed Per Clip (5 seconds of output video)

Sources: empirical lossy pipeline runs (see `0009`, `0012`), Apatero benchmarks (`0017`), and third-party GPU benchmarks.

| GPU | Model | Resolution | Wall-clock / clip |
|-----|-------|-----------|-------------------|
| RTX 4090 | Wan 2.1 T2V-1.3B | 480p | ~85–135 sec |
| RTX 4090 | Wan 2.2 TI2V-5B | 480p (scaled down) | ~195–220 sec |
| RTX 4090 | Wan 2.2 TI2V-5B | 720p (native) | ~280–367 sec |
| RTX 4090 | Wan 2.1/2.2 14B fp8 | 480p | ~240–280 sec |
| L40S | Wan 2.2 TI2V-5B | 720p | ~280–367 sec |
| A100 80GB | Wan 2.2 A14B (MoE) | 480p | ~170 sec |
| A100 80GB | Wan 2.2 A14B (MoE) | 720p | ~523 sec |
| H100 SXM | Wan 2.2 A14B (MoE) | 480p | ~85 sec |
| H100 SXM | Wan 2.2 A14B (MoE) | 720p | ~284 sec |

The RTX 4090 empirical figures (85–135 sec) come from the actual Star Wars IV run in `0009` using Wan 2.1 1.3B. The L40S figure for Wan 2.2 TI2V-5B (367 sec) is from the trial in `0012`. A100/H100 figures are from SaladCloud/Novita benchmarks.

### Self-Hosted Cost Per Clip

| GPU | Model | Resolution | Sec/clip | Spot rate | **Cost/clip** |
|-----|-------|-----------|---------|-----------|--------------|
| RTX 4090 (spot) | Wan 2.1 1.3B | 480p | 85–135 | $0.34/hr | **$0.008–0.013** |
| RTX 4090 (on-demand) | Wan 2.1 1.3B | 480p | 85–135 | $0.69/hr | **$0.016–0.026** |
| RTX 4090 (spot) | Wan 2.2 TI2V-5B | 480p | ~200 | $0.34/hr | **~$0.019** |
| L40S (spot) | Wan 2.2 TI2V-5B | 720p | ~367 | $0.54/hr | **~$0.055** |
| A100 80GB SXM (spot) | Wan 2.2 A14B | 480p | ~170 | $1.49/hr | **~$0.070** |
| H100 SXM (spot) | Wan 2.2 A14B | 480p | ~85 | $2.69/hr | **~$0.064** |
| A100 80GB SXM (spot) | Wan 2.2 A14B | 720p | ~523 | $1.49/hr | **~$0.217** |
| H100 SXM (spot) | Wan 2.2 A14B | 720p | ~284 | $2.69/hr | **~$0.212** |

### Self-Hosted Cost for 2000 Clips

| Config | Model | Res | Cost/clip | **2000 Clips** | Total GPU Time |
|--------|-------|-----|----------|----------------|----------------|
| RTX 4090 spot | Wan 2.1 1.3B | 480p | $0.010 avg | **~$20** | ~50–75 hrs |
| RTX 4090 on-demand | Wan 2.1 1.3B | 480p | $0.020 avg | **~$40** | ~50–75 hrs |
| RTX 4090 spot | Wan 2.2 TI2V-5B | 480p | $0.019 | **~$38** | ~111 hrs |
| L40S spot | Wan 2.2 TI2V-5B | 720p | $0.055 | **~$110** | ~200 hrs |
| A100 SXM spot | Wan 2.2 A14B | 480p | $0.070 | **~$140** | ~95 hrs |
| H100 SXM spot | Wan 2.2 A14B | 480p | $0.064 | **~$128** | ~47 hrs |
| A100 SXM spot | Wan 2.2 A14B | 720p | $0.217 | **~$433** | ~290 hrs |
| H100 SXM spot | Wan 2.2 A14B | 720p | $0.212 | **~$424** | ~158 hrs |

---

## Part 3: Head-to-Head Comparison

### 480p, Wan 2.1 Quality

| Approach | Cost for 2000 clips | Notes |
|----------|---------------------|-------|
| RunPod RTX 4090 spot | **~$20** | Empirically validated on Star Wars IV |
| RunPod RTX 4090 on-demand | ~$40 | Guaranteed availability |
| **fal.ai Wan 2.1** | $400 | 20× more expensive than self-hosted |
| **Replicate Wan 2.1 1.3B** | $400 | 20× more expensive |

**Self-hosting wins by ~20×** at Wan 2.1 quality.

---

### 480p, Wan 2.2 Quality

| Approach | Cost for 2000 clips | Notes |
|----------|---------------------|-------|
| RunPod RTX 4090 spot (TI2V-5B) | **~$38** | 5B dense model, slightly lower quality than A14B |
| RunPod H100 spot (A14B MoE) | ~$128 | Full-quality A14B, fastest self-hosted |
| RunPod A100 spot (A14B MoE) | ~$140 | Full-quality A14B |
| **fal.ai Wan 2.2 Turbo** | $100 | Optimised A14B, ~30s wall-clock per clip |
| **Replicate Wan 2.2 Fast** | $100 | Optimised Wan 2.2, ~30s wall-clock per clip |
| fal.ai Wan 2.2 A14B standard | $400 | Same model, not optimised |

**Self-hosting TI2V-5B on RTX 4090 spot (~$38) is 2.6× cheaper** than the cheapest API option (fal.ai/Replicate Turbo/Fast at $100).

**Self-hosting A14B on H100/A100 spot ($128–$140) is comparable to or slightly more expensive** than the Turbo/Fast API variants ($100), but uses the full-precision model. Whether that quality difference justifies the cost difference is a subjective call.

---

### 720p, Wan 2.2 Quality

| Approach | Cost for 2000 clips | Notes |
|----------|---------------------|-------|
| RunPod L40S spot (TI2V-5B) | **~$110** | ~5B dense model, 720p native |
| RunPod H100 spot (A14B MoE) | ~$424 | Full-quality A14B at 720p |
| RunPod A100 spot (A14B MoE) | ~$433 | Full-quality A14B at 720p |
| **fal.ai Wan 2.2 Turbo (720p)** | $200 | Cheapest API option at 720p |
| **Replicate Wan 2.2 Fast (720p)** | $200 | Cheapest API option at 720p |
| fal.ai Wan 2.2 A14B standard (720p) | $800 | Same model, no optimisation |
| Replicate Wan 2.2 I2V standard (720p) | $2,000 | Most expensive API option |

At 720p, TI2V-5B self-hosted on an L40S spot instance (~$110) is the cheapest option overall — cheaper than the optimised API variants ($200). Full-quality A14B at 720p on self-hosted A100/H100 is ~2× the cost of the API Turbo/Fast options, but still 4–5× cheaper than unoptimised API.

---

## Part 4: Overhead Costs (Self-Hosting Only)

API providers have zero overhead. Self-hosting on RunPod carries several additional costs:

| Cost Type | Estimate | Notes |
|-----------|----------|-------|
| Model download time | ~20–40 min (first run) | Wan 2.1 1.3B: 2.84 GB; TI2V-5B: ~10 GB; A14B: ~29 GB per file |
| Pod startup per session | ~5–15 min at GPU rate | Includes ComfyUI spin-up and model load |
| Network volume storage | ~$0.07/GB/month | ~$0.20/month for 1.3B; ~$2–4/month for 14B+ |
| Pod startup billing | ~$0.03–0.20/session | 10–15 min of GPU time before first clip |
| Failed/interrupted pods | $0.20–0.50/incident | Community cloud pods can be preempted |

For a 2000-clip run across ~3–5 pod sessions:
- Startup overhead: ~$0.50–1.00 total
- Storage: negligible (< $5/month)
- Estimated total overhead: **$1–5** — negligible vs. generation cost

---

## Part 5: Practical Considerations

### Self-Hosting Advantages

1. **Cost**: 2–20× cheaper than API, depending on model and resolution tier.
2. **Audio is free**: Running MMAudio on the same RunPod pod (via `runpod-mmaudio`) adds near-zero marginal cost. At API rates, 2000 clips × 5s × $0.001/s = **$10** extra for audio.
3. **No rate limits**: Batch runs can proceed continuously without throttling.
4. **Model control**: Choose exact quantisation, resolution, and step count.

### API Advantages

1. **Zero management overhead**: No pod lifecycle, no model downloads, no CUDA errors.
2. **Speed**: Turbo/Fast variants generate in ~30s wall-clock vs 1.5–6 min self-hosted. A 2000-clip run takes ~17 hours via API vs 50–200 hours self-hosted.
3. **Reliability**: No preemption, no failed pods.
4. **No upfront cost**: Pay per clip — useful for experimentation or small runs.

### When API Breaks Even or Wins

- Runs smaller than ~100 clips: pod startup overhead dominates for self-hosted.
- Time-sensitive runs: the 6–8× wall-clock speed advantage of Turbo/Fast API variants may outweigh cost.
- If using A14B MoE at 480p and comparing to Turbo/Fast: API ($100) is slightly cheaper than H100 spot ($128).

---

## Summary

| Scenario | Self-Hosted (spot) | Best API | Self-hosted saving |
|----------|-------------------|----------|-------------------|
| Wan 2.1 quality, 480p, 2000 clips | **~$20** (RTX 4090) | $400 (fal.ai/Replicate) | **20× cheaper** |
| Wan 2.2 quality, 480p, 2000 clips | **~$38** (TI2V-5B, RTX 4090) | $100 (Turbo/Fast) | **2.6× cheaper** |
| Wan 2.2 A14B, 480p, 2000 clips | ~$128–140 (H100/A100) | $100 (Turbo/Fast) | API slightly cheaper |
| Wan 2.2 quality, 720p, 2000 clips | **~$110** (TI2V-5B, L40S) | $200 (Turbo/Fast) | **1.8× cheaper** |
| Wan 2.2 A14B, 720p, 2000 clips | ~$424–433 (H100/A100) | $200 (Turbo/Fast) | API 2× cheaper |

**The hypothesis holds for standard-quality comparisons.** Self-hosting Wan 2.1 on a RunPod RTX 4090 spot instance is ~20× cheaper than the cheapest equivalent API. For Wan 2.2 quality, self-hosting TI2V-5B remains 2–3× cheaper than optimised API options. The only scenario where API wins on cost is running the full A14B MoE model at 720p, where the massive hardware requirement (80 GB VRAM, H100-class) makes self-hosting expensive enough that optimised API variants close the gap.

**For the lossy pipeline's current use case** (full-film generation, 2000+ clips, cost-first priority), the optimal strategy remains `runpod-wan` with Wan 2.1 1.3B on a RTX 4090 community cloud instance at ~$20 total, accepting lower-quality output. Upgrading to TI2V-5B on the same GPU doubles quality at ~2× the cost (~$38). The Turbo/Fast API options are worth considering only when wall-clock time matters more than cost (each API variant completes the full 2000-clip run in ~17 hours vs. 50–200 hours self-hosted).
