---
id: 0017-video-gen-speed-benchmarks
type: note
purpose: "Capture third-party benchmarks (Apatero, Jan 2026) for LTX-2, Wan 2.2, Kling, Runway, Pika across local GPUs and cloud rental, for comparison with lossy's RunPod-WAN baseline."
scope: ["research", "decode", "video-generation", "cost", "runpod"]
tags: ["research", "benchmarks", "ltx-2", "wan", "kling", "runway", "pika", "vram", "cost"]
related:
  - "research/0003-prompt-to-video/research.md"
  - "research/0009-cost-breakdown/research.md"
  - "research/0012-wan-model-variants/research.md"
sources:
  - "https://www.apatero.com/blog/ai-video-generation-speed-benchmarks-2025"
  - "docs/sources/ai-video-generation-speed-benchmarks-2025.md"
---

## Question

How do current text/image-to-video models (LTX-2, Wan 2.2, Kling, Runway, Pika) compare on speed, VRAM, and cost across local GPUs and cloud rental — and what does that imply for lossy's default `runpod-wan` strategy?

## Source

Apatero Studio, "AI Video Generation Speed Benchmarks 2025" (published 2026-01-07). 500+ runs over two weeks, "real-world settings" rather than tuned lab conditions; quality scores are subjective 1–10. Treat as a useful directional reference, not a controlled study.

## Headline Numbers

- **LTX-2 (Lightricks)** is the fastest local model: ~47s for 5s @ 768x512, 30 steps on RTX 4090. ~28s with all optimisations (FP8 + reduced steps + torch.compile), with slight quality loss.
- **Wan 2.2** trades speed for quality: ~3m 15s for the same shot on RTX 4090; ~5m 30s at 720p T2V. I2V is ~25% faster than T2V at equivalent settings.
- **Kling Pro / Runway Gen-3 / Pika** (managed cloud) land in 1m 15s – 2m 30s range at $0.10–$0.80 per video.
- **RunPod RTX 4090 rental** runs LTX-2 in ~48s at ~$0.01/video (vs Kling at ~$0.15–$0.30).

## VRAM Picture

| Model           | Min VRAM | Recommended |
|-----------------|----------|-------------|
| LTX-2 base      | 10 GB    | 16 GB       |
| LTX-2 + upscale | 14 GB    | 20 GB       |
| Wan 2.2 480p    | 12 GB    | 16 GB       |
| Wan 2.2 720p    | 18 GB    | 24 GB       |

Wan 2.2 OOMs on RTX 3060 12GB at standard settings; LTX-2 still runs (slowly).

## Cost: Local vs Cloud (100 videos/month scenario)

- Managed (Kling): ~$20/mo, no setup.
- RunPod 4090 rental: ~$1.23/mo + ~2h initial setup.
- Local 4090: $1,600 hardware + ~$3/mo electricity. Break-even vs Kling ≈ 80 months; never breaks even vs RunPod rental.

This matches lossy's existing direction (RunPod over local hardware; see `research/0009-cost-breakdown/`).

## Other Findings Worth Noting

- **Concurrency on 24 GB cards**: parallel-2 is the sweet spot (~34% wall-clock saving over sequential); parallel-3 marginal (+8 pp); parallel-4 OOMs. Relevant to the concurrent-generation work in `tasks/0009-concurrent-runpod-generation/` and `0010-pipelined-audio-generation/`.
- **Steps**: diminishing returns past ~40 steps; resolution bumps move quality more than step bumps.
- **Frame count scales roughly linearly** with generation time (LTX-2): 49 frames ~22s, 121 frames ~47s, 193 frames ~1m 15s on RTX 4090. Useful for estimating per-shot cost as we vary clip duration.
- **Real-world pipeline overhead**: end-to-end production is 5–10x raw generation time once prompt iteration, regen, upscale, and post are folded in. Consistent with what we see in `pipeline.py` runs.

## Implications for lossy

1. **LTX-2 is now a credible alternative backend** worth a spike — 3–5x faster than Wan at meaningful-but-not-catastrophic quality cost. Would slot in next to existing strategies in `strategies_video.py`.
2. **WAN 2.2 I2V (~1m 45s @ 480p, ~4m 15s @ 720p on 4090)** is ~25% faster than T2V — informs the character-continuity / VACE work (`tasks/0014-character-continuity-vace/`) where we're already image-conditioning.
3. **Concurrency ceiling = 2 on a single 24 GB card** matches the conservative defaults in our concurrent-runpod work; don't bother with 3+ on a single GPU instance.
4. **Quality/step curve flattens past 40 steps** — current defaults don't need to push higher than that for incremental quality.

## Caveats

- Apatero is a studio blog, not a peer-reviewed benchmark. Quality scores are subjective.
- "Cost per video" assumes spot pricing and ignores cold-start time on RunPod (which can dominate for short batches — see `tasks/0007-runpod-mmaudio-strategy/`).
- Numbers are January 2026 — model perf moves fast; revisit before quoting in design docs.

## See Also

- `docs/sources/ai-video-generation-speed-benchmarks-2025.md` — full raw article.
- `research/0012-wan-model-variants/` — our existing WAN variant catalogue.
- `research/0009-cost-breakdown/` — lossy cost model.
