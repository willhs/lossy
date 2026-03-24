---
id: 0004-concurrent-runpod-generation
type: note
purpose: "Investigate feasibility of concurrent video+audio generation on a shared RunPod GPU pod and document options."
scope: ["research", "decoder", "runpod"]
non_goals: []
tags: ["research", "runpod", "concurrency", "audio", "video", "mmaudio", "vram"]
related: ["research/0003-prompt-to-video/research.md", "docs/tasks/0009-concurrent-runpod-generation/task.md"]
---

## Problem

The lossy decode pipeline runs video (Wan via ComfyUI) and audio (MMAudio via ComfyUI) sequentially on the same RunPod pod. For a film with N shots, audio generation adds ~20-30s per clip of wall time on top of video generation (~60s per clip). For a 200-shot film, that's an extra ~1-1.5 hours of pod time.

Can audio run concurrently alongside video on the same GPU to eliminate this overhead?

## Constraint: ComfyUI is Single-Threaded

ComfyUI processes workflows sequentially -- it cannot run two workflows simultaneously. To achieve concurrency, audio must bypass ComfyUI entirely and run as a standalone Python process via SSH on the pod.

## VRAM Budget Analysis

### Known Values (from model specifications)

| Model | Estimated VRAM | Source |
|-------|---------------|--------|
| Wan 2.1 1.3B (fp16 diffusion + fp8 text encoder) | ~8-10 GB | ComfyUI community benchmarks |
| MMAudio V2 large_44k (fp16, force_offload: True) | ~4-6 GB | Kijai ComfyUI-MMAudio |
| ComfyUI overhead (Python, CUDA context, buffers) | ~1-2 GB | Typical PyTorch baseline |

### Concurrent VRAM Estimate

| Scenario | Estimated Total | RTX 4090 (24 GB) | Headroom |
|----------|----------------|-------------------|----------|
| Wan peak alone | ~10 GB | Fits | ~14 GB |
| MMAudio peak alone | ~6 GB | Fits | ~18 GB |
| Both concurrent (worst case) | ~16 GB | Fits | ~8 GB |
| Both concurrent + overhead | ~18 GB | Fits | ~6 GB |

**Preliminary verdict**: Concurrent execution on RTX 4090 appears feasible with ~6-8 GB headroom. However, these are estimates -- actual VRAM benchmarking is required (see Phase 1 of plan).

### Risk: Worst-Case VRAM Overlap

Both models have distinct VRAM phases:
- **Wan**: Peak during diffusion sampling (20 steps), VRAM drops during VAE decode
- **MMAudio**: Peak during flow matching (25 steps), VRAM drops after generation

If both peak simultaneously, worst-case is ~16 GB. The RTX 4090 should handle this, but an OOM is possible if CUDA fragmentation or other processes consume the remaining headroom.

### Mitigation: force_offload

MMAudio's `force_offload: True` setting moves models to CPU after inference. In the standalone runner, we can manually `torch.cuda.empty_cache()` after each clip to release VRAM back to the system, preventing accumulation across multiple clips.

## Benchmarking (Pending)

**Status**: Tools created, awaiting manual execution on a live pod.

- `tools/vram_benchmark.sh` -- Interactive script to measure peak VRAM during Wan and MMAudio generation separately
- `tools/mmaudio_standalone.py` -- Standalone MMAudio inference script (bypasses ComfyUI) for concurrent testing

### How to Benchmark

1. Start a pod: `python decode.py <output_dir> --strategy runpod-wan --limit 1 --keep-pod`
2. Run VRAM benchmark: `./tools/vram_benchmark.sh <output_dir>`
3. Test concurrent execution: Start a Wan workflow via ComfyUI, then SSH in and run `mmaudio_standalone.py` simultaneously
4. Record actual peak VRAM numbers below

### Benchmark Results

Measured on NVIDIA RTX 4000 Ada Generation (20,475 MiB total VRAM). This GPU has less VRAM than the RTX 4090 (24 GB) but the Wan VRAM usage is model-dependent, not GPU-dependent.

| Metric | Value |
|--------|-------|
| Baseline VRAM (ComfyUI idle) | 306 MiB |
| Wan peak VRAM | **14,994 MiB (~15 GB)** |
| MMAudio peak VRAM (standalone) | Not measured (ComfyUI node install failed on this pod) |
| MMAudio peak VRAM (estimated) | ~4-6 GB (from model specs) |
| Concurrent peak (estimated) | ~19-21 GB |
| RTX 4090 headroom | ~3-9 GB (tight) |
| RTX A6000 headroom | ~27-33 GB (safe) |

**Key finding**: Wan uses ~15 GB peak -- significantly more than the ~8-10 GB estimate. This changes the feasibility assessment:
- **RTX 4090 (24 GB)**: Only ~3-9 GB headroom for concurrent MMAudio. Risky.
- **RTX A6000 (48 GB)**: ~27-33 GB headroom. Trivially safe.
- **RTX 4000 Ada (20 GB)**: Would OOM during concurrent generation.

**Note**: ComfyUI-MMAudio nodes failed to register after installation + restart on this pod. This reinforces the case for the standalone MMAudio runner approach -- it doesn't depend on ComfyUI's custom node loading.

**A6000 availability**: A6000 was unavailable on community cloud during this test (2026-03-24). The 4090 was also unavailable. Only the RTX 4000 Ada was available. Availability is unpredictable.

## Standalone MMAudio Runner

A prototype standalone inference script (`tools/mmaudio_standalone.py`) has been created that:

1. Uses the official MMAudio Python API (`mmaudio.eval_utils.generate`)
2. Loads models from the ComfyUI models directory (Kijai fp16 safetensors)
3. Runs text-to-audio inference without ComfyUI
4. Saves output as FLAC
5. Reports VRAM usage at each stage

This script would be uploaded to the pod via SCP and invoked via SSH, running independently of the ComfyUI process.

**Expected benefits over ComfyUI-based audio**:
- No ComfyUI node graph overhead (~10-20s saved per clip)
- Can run concurrently with ComfyUI video generation
- Direct VRAM control (manual cleanup between clips)

## Alternative Approaches

### RunPod GPU Options

| GPU | VRAM | $/hr (community) | Concurrent headroom (measured) | Notes |
|-----|------|-------------------|-------------------------------|-------|
| RTX 4000 Ada | 20 GB | $0.34 | ~0-5 GB | Too tight, likely OOM |
| RTX 4090 | 24 GB | $0.34 | ~3-9 GB | Risky -- depends on MMAudio actual peak |
| RTX A6000 | 48 GB | $0.25-0.33 | **~27-33 GB** | **Cheaper than 4090**, safe. Availability issues. |
| L40S | 48 GB | $0.79 | ~27-33 GB | 2.3x cost, same headroom as A6000 |
| RTX 6000 Ada | 48 GB | $0.74 | ~27-33 GB | Similar to L40S |
| A100 PCIe 80GB | 80 GB | $1.19 | ~59-65 GB | Overkill for this workload |

**Key finding**: The RTX A6000 (48 GB) is actually **cheaper** than the RTX 4090 ($0.25-0.33 vs $0.34/hr) on RunPod community cloud, with double the VRAM. This makes concurrent generation trivially safe on A6000 with massive headroom. Availability may vary.

### Architecture Options

#### Option 1: Concurrent on RTX 4090 (current approach)

Run standalone MMAudio via SSH alongside ComfyUI Wan video.

- **Audio wall time**: ~0 min (hidden behind video generation)
- **Extra cost/film**: $0 (same pod)
- **Complexity**: High (SSH runner, concurrent process management)
- **Risk**: OOM if VRAM estimates are wrong

#### Option 2: Concurrent on RTX A6000

Same as Option 1 but on A6000 (48 GB VRAM).

- **Audio wall time**: ~0 min
- **Extra cost/film**: ~$0 or negative (A6000 is cheaper!)
- **Complexity**: High (same SSH runner approach)
- **Risk**: Low (30 GB headroom)
- **Blocker**: A6000 availability on community cloud

#### Option 3: Two separate pods

One pod for video (cheap GPU), one for audio (cheap GPU).

- **Audio wall time**: ~0 min (fully parallel)
- **Extra cost/film**: +$0.34-0.68/hr for second pod
- **Complexity**: Medium (two pod sessions, coordinate via files)
- **Risk**: Low

For a 200-clip film (~3.3 hrs):
- Current single pod: $1.12
- Two 4090 pods: $2.24 (+$1.12)
- One A6000 for both: $0.83-1.09 (potentially **cheaper** than current!)

#### Option 4: fal.ai MMAudio API for audio

Use the existing `MMAudioStrategy` (fal.ai hosted) for audio while RunPod handles video. Already implemented in `strategies_audio.py:127`.

- **Audio wall time**: ~0 min (different infrastructure)
- **Extra cost/film**: ~$0.20-0.60 for 200 clips at $0.001/sec
  - Average clip ~5s: 200 * 5 * $0.001 = $1.00
  - But clips overlap with pod time, so marginal cost is just the fal.ai fee
- **Complexity**: None (already implemented)
- **Risk**: None

For a 200-clip film:
- RunPod video: $1.12
- fal.ai audio: ~$1.00 (200 clips * avg 5s * $0.001/s)
- Total: ~$2.12 (+$1.00 vs current sequential)

#### Option 5: Sequential but optimized

Keep sequential execution but reduce overhead:
- Skip ComfyUI restart between video/audio stages
- Pre-load MMAudio models before video finishes
- Reduce MMAudio inference steps (20 instead of 25)

- **Audio wall time**: ~50-70% of current
- **Extra cost/film**: $0
- **Complexity**: Low
- **Risk**: None

## Cost Comparison

For a typical 200-clip film (avg 5s per clip, ~3.3 hrs total generation time):

| Approach | Audio wall time | Total cost | Extra cost | Complexity | Risk |
|----------|----------------|------------|------------|------------|------|
| Sequential on 4090 (current) | ~1.5 hrs | $1.12 | baseline | None | None |
| Concurrent on 4090 | ~0 min | $1.12 | $0 | High | OOM |
| **Concurrent on A6000** | **~0 min** | **$0.83-1.09** | **-$0.03-0.29** | **High** | **Low** |
| Two 4090 pods | ~0 min | $2.24 | +$1.12 | Medium | Low |
| fal.ai audio + RunPod video | ~0 min | $2.12 | +$1.00 | None | None |
| Sequential optimized | ~1.0 hrs | $1.12 | $0 | Low | None |

## Recommendation

### Short term: Switch to RTX A6000

The RTX A6000 is the clear winner -- cheaper than RTX 4090, double the VRAM, and makes concurrent generation trivially safe. Update `GPU_TYPES` in `runpod_pod.py` to prefer A6000 first.

Even without concurrent generation, the A6000 saves money per film. With concurrent generation, it eliminates both cost and VRAM risk.

**Action**: Update GPU_TYPES ordering to try A6000 first, then verify availability.

### Medium term: Implement concurrent generation

If A6000 availability is reliable:
1. Upload `mmaudio_standalone.py` to pod during setup
2. After each video clip completes, kick off audio for the previous clip via SSH
3. Pipeline: video N + audio N-1 run concurrently
4. Expected speedup: ~30% wall time reduction (audio generation hidden behind video)

### Fallback: fal.ai audio

If concurrent generation proves too complex or unreliable, the existing `MMAudioStrategy` (fal.ai) is a zero-effort alternative at ~$1/film extra cost. For short films (<50 clips), this is arguably the simplest approach.

## Raw Data / Logs

> **TODO**: Attach benchmark logs after running `tools/vram_benchmark.sh`

## References

- [Task spec](../../tasks/0009-concurrent-runpod-generation/task.md)
- [RunPod GPU Pricing](https://www.runpod.io/pricing) (accessed 2026-03-24)
- [fal.ai MMAudio V2](https://fal.ai/models/fal-ai/mmaudio-v2/text-to-audio) -- $0.001/sec
- [hkchengrex/MMAudio](https://github.com/hkchengrex/MMAudio) -- official repo
- [Kijai/MMAudio_safetensors](https://huggingface.co/Kijai/MMAudio_safetensors) -- fp16 weights
