---
id: task-0015
type: spec
purpose: "Evaluate upgrading RunPod WAN strategy from Wan 2.1 T2V-1.3B to Wan 2.2 TI2V-5B for better quality at acceptable cost/VRAM."
tags: ["decode", "strategy", "runpod", "wan2.2", "upgrade"]
related: ["research/0012-wan-model-variants/research.md", "design/adr/005-runpod-self-hosted-wan-strategy.md"]
created: 2026-04-08
updated: 2026-04-08
---

# Upgrade RunPod WAN Strategy: 2.1 T2V-1.3B → 2.2 TI2V-5B

## Goal

Upgrade the `runpod-wan` strategy from Wan 2.1 T2V-1.3B (fp16, ~8 GB VRAM) to Wan 2.2 TI2V-5B (~10 GB fp16, ~24 GB VRAM) for improved video quality at 720P. The upgrade should not require a more expensive GPU tier and should preserve all existing functionality (variable duration, concurrent audio, resume, cost tracking).

## Context

The pipeline currently runs `wan2.1_t2v_1.3B_fp16` on 24 GB GPUs (RTX 4090 class) via ComfyUI on RunPod. Research doc 0012 catalogues all Wan variants and identifies **TI2V-5B** as the most practical 2.2 upgrade:

- **Single dense model** (~10 GB fp16) — no MoE dual-checkpoint complexity
- **Unified T2V + I2V** — enables future I2V use without a separate strategy
- **720P at 24fps** — resolution upgrade from current 480P at 16fps
- **Fits on 24 GB** — borderline but viable on current RTX 4090 pods
- **New VAE required** — uses `wan2.2_vae.safetensors` instead of `wan_2.1_vae.safetensors`

The 14B MoE variants (T2V-A14B, I2V-A14B) require ~80 GB VRAM and are not viable on current infrastructure. TI2V-5B is the only 2.2 model that fits.

### Key unknowns

- **VRAM headroom on 24 GB**: TI2V-5B is listed as ~24 GB VRAM. Concurrent audio (MMAudio) may not fit — need to test whether generation works at all, and whether concurrent audio must be disabled.
- **ComfyUI workflow**: TI2V-5B likely needs different ComfyUI nodes than the current T2V workflow. The exact node graph needs to be researched/tested.
- **Generation speed**: 5B model is ~3.5x larger than 1.3B — inference will be slower. Need to measure per-clip time and cost impact.
- **Quality delta**: Is the quality improvement meaningful for lossy's intentionally degraded aesthetic?

## Will Do

- **Research TI2V-5B ComfyUI workflow** — find the correct node graph for T2V mode (we use T2V today; I2V is a bonus for later)
- **Update model downloads** in `RunPodWanStrategy.WAN_MODELS`:
  - Replace `wan2.1_t2v_1.3B_fp16.safetensors` with the TI2V-5B checkpoint from `Comfy-Org/Wan_2.2_ComfyUI_Repackaged`
  - Replace `wan_2.1_vae.safetensors` with `wan2.2_vae.safetensors`
  - Keep `umt5_xxl_fp8_e4m3fn_scaled.safetensors` (shared text encoder)
- **Update `_build_workflow()`** with the correct node types and parameters for TI2V-5B
- **Update resolution** from 848x480 to 1280x720 (or whatever TI2V-5B's native 720P resolution is)
- **Update FPS** from 16 to 24 if TI2V-5B generates at 24fps
- **Update frame count limits** (`MIN_FRAMES`, `MAX_FRAMES`) — recalculate for new fps and test for timeout/OOM boundaries
- **Test on a real RunPod pod** — verify end-to-end generation, measure VRAM usage, generation time, and cost
- **Update class docstring** and `name` if creating a new strategy (see options below)
- **Document results** — update ADR-005 or write a new note with observed VRAM, speed, cost, and quality

## Decision: New Strategy or In-Place Upgrade?

Two options — **need user input**:

**Option A: In-place upgrade** — modify `RunPodWanStrategy` to use TI2V-5B. Simpler, but loses the ability to run 2.1 for comparison.

**Option B: New `runpod-wan22` strategy** — add alongside existing `runpod-wan`. More code, but allows A/B comparison and rollback. The old strategy could be removed later.

Recommend **Option B** initially for safe comparison, then remove `runpod-wan` once 2.2 is validated.

## Might Do

- **Enable I2V mode** in the new strategy — TI2V-5B supports image conditioning natively. Could accept a reference image parameter for future character continuity work.
- **Test fp8 quantization** — if fp16 doesn't leave VRAM headroom for concurrent audio, try fp8_scaled to reduce footprint.
- **Adjust sampler settings** — TI2V-5B may benefit from different steps/cfg/sampler than the current 20-step uni_pc config.

## Won't Do

- **14B MoE models** — require 80 GB VRAM, not viable on current GPU tier
- **Change GPU tier** — task goal is to upgrade within current cost envelope
- **VACE integration** — separate concern (task-0014), uses a different model
- **Replicate strategy update** — already uses Wan 2.2 Fast via API, unrelated

## Risks

- **VRAM OOM on 24 GB**: TI2V-5B at ~24 GB VRAM is right at the limit. May need to disable concurrent audio, reduce resolution, or use fp8 quantization.
- **Slower generation**: 5B vs 1.3B will be slower per clip. If cost per clip doubles (from ~$0.02 to ~$0.04), total film cost rises from ~$25 to ~$50 — still much cheaper than Replicate ($58) but worth measuring.
- **ComfyUI compatibility**: TI2V-5B may require a newer ComfyUI version than what's on the `runpod/comfyui:latest` Docker image.
- **Frame count formula changes**: If fps changes from 16 to 24, all frame calculations in `_target_frames` and `_target_durations` need updating, and the stitch pipeline's speed-adjustment logic may need review.

## Success Criteria

- [ ] TI2V-5B generates clips end-to-end on a 24 GB RunPod pod via `--strategy runpod-wan22` (or updated `runpod-wan`)
- [ ] VRAM usage documented — does concurrent audio still fit?
- [ ] Per-clip generation time and cost measured and compared to 2.1 baseline
- [ ] Output quality visually compared to 2.1 (at least a few shots side-by-side)
- [ ] Clips integrate with existing stitch pipeline
- [ ] Decision documented: keep 2.2 as default, keep both, or revert
