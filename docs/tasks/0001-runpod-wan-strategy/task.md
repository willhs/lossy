---
id: task-0001
type: spec
purpose: "Add a RunPod self-hosted Wan 2.2 decode strategy that's ~10x cheaper than Replicate"
tags: ["decoder", "strategy", "runpod", "wan2.2", "cost-reduction"]
related: ["../../decode.py"]
created: 2026-03-15
updated: 2026-03-15
---

# Add RunPod Self-Hosted Wan 2.2 Strategy

## Goal

Add a `RunPodWanStrategy` to the decoder that spins up a RunPod GPU pod, generates clips via ComfyUI's HTTP API, and tears down the pod when done. Target cost: ~$0.004/clip (~$5-15 for a full film vs ~$58 with current cheapest strategy).

## Context

The current decode strategies use hosted APIs (Replicate, fal.ai) at $0.05/clip or more. A full Star Wars IV decode (1,161 shots) costs ~$58 minimum with `replicate-wan`. RunPod community cloud GPUs (RTX 4090 at ~$0.34/hr) running ComfyUI with Wan 2.2 1.3B fp8 can generate ~90 clips/hr, bringing per-clip cost down to ~$0.004. This makes full-film decoding cheap enough to iterate on.

A pre-built Docker image (`ghcr.io/lum3on/wan22-runpod:latest`) provides ComfyUI + Wan 2.2 1.3B fp8, so we don't need to build or maintain our own image.

### How existing strategies work

All strategies implement `GenerationStrategy` in `decode.py`:
- Class with a `name` attribute and a `generate()` method returning `list[ClipResult]`
- Registered in a `strategies` dict in `main()`, keyed by CLI name
- Clips saved to `clips/{strategy.name}/`, progress to `decode_progress_{strategy.name}.json`
- Progress tracks `completed` (shot indices), `failed`, `total_cost_estimate`, and per-shot clip metadata
- `ReplicateWanStrategy` produces fixed ~5s clips; `FalSeedanceStrategy` supports variable duration and multi-part splitting

The new strategy follows the same interface. The main difference is managing infrastructure (pod lifecycle) rather than just calling an API.

## Will Do

- **`RunPodWanStrategy` class** implementing `GenerationStrategy` with `name = "runpod-wan"`
- **Pod lifecycle management**: create pod on first `generate()` call, reuse for subsequent calls, terminate on completion
- **ComfyUI API integration**: POST workflow JSON to `/prompt`, poll `/history/{id}` for completion, download output MP4
- **Fixed ~5s clips** (like `replicate-wan`) -- stitch stage handles speed-adjustment to match original shot duration
- **Pod cleanup on exit**: signal handler (SIGINT/SIGTERM) and try/finally to terminate pod on Ctrl+C or crash
- **Resume support**: uses existing progress tracking pattern (`decode_progress_runpod-wan.json`), skips completed shots
- **Cost tracking**: compute cost from elapsed wall-clock time and GPU hourly rate, record per-clip in progress JSON
- **Register in CLI**: add `"runpod-wan"` to strategy dict and `--strategy` choices
- **`RUNPOD_API_KEY`** loaded from `.env` via existing `load_env()` pattern
- **Research the ComfyUI workflow JSON**: figure out the exact node graph for Wan 2.2 T2V at 480p 16:9 (this is the main unknown)

## Might Do

- **GPU fallback chain**: try RTX 4090 first, fall back to RTX A5000 or similar if unavailable
- **Batch queueing**: submit multiple prompts to ComfyUI's queue instead of one-at-a-time (could improve throughput)
- **Pod keep-alive timeout**: keep the pod running for N minutes after last clip in case the user re-runs quickly, instead of immediate termination
- **Wan 2.2 14B support**: larger model for higher quality, would need a different image and bigger GPU (A100/H100)

## Won't Do

- **Custom Docker image**: using the pre-built `ghcr.io/lum3on/wan22-runpod:latest` image as-is
- **Serverless RunPod endpoints**: too complex to set up for this use case; on-demand pods are simpler
- **Multi-part clip splitting**: unlike `FalSeedanceStrategy`, this strategy produces fixed-duration clips. Variable duration is out of scope
- **Audio or subtitle handling**: orthogonal to video generation
- **Automatic quality comparison**: comparing output quality against other strategies is manual for now

## Risks

- **ComfyUI workflow JSON is the biggest unknown**: the exact node graph for Wan 2.2 T2V needs to be exported from a running ComfyUI instance or reverse-engineered from documentation
- **Pod availability**: RTX 4090 community cloud may have stock issues during peak times
- **1.3B model quality**: lower quality than 14B, but acceptable for lossy's intentionally degraded aesthetic
- **Cold start time**: pod creation + image pull + model load could take 2-5 minutes; need to handle this gracefully
- **Cost estimation accuracy**: wall-clock-based cost tracking includes idle time (polling, downloading), slightly overstating true generation cost

## Success Criteria

- [x] `python decode.py output/star_wars_iv_v2 --strategy runpod-wan` generates clips end-to-end
- [x] Pod is created automatically and terminated on completion, error, or Ctrl+C
- [x] Resume works after interruption (re-running skips completed shots, creates new pod)
- [x] Per-clip cost tracked in `decode_progress_runpod-wan.json`
- [x] Actual observed cost documented in ADR-005 (~$0.02/clip, ~$25-30 extrapolated for full film)
- [x] Clips integrate with existing stitch pipeline (`stitch` subcommand works with `--strategy runpod-wan`)
