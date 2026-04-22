---
id: task-0009
type: spec
purpose: "Pipeline video and audio generation on a shared RunPod pod to nearly eliminate audio generation wall time."
tags: ["runpod", "concurrency", "audio", "video", "decode", "performance"]
related:
  - "../../design/adr/002-stateless-cli-pipeline.md"
  - "../0001-runpod-wan-strategy/task.md"
  - "../0007-runpod-mmaudio-strategy/task.md"
---

# Concurrent Video + Audio Generation on Shared RunPod Pod

## Goal

Pipeline audio generation alongside video generation on the same RunPod pod so that audio for shot N-1 runs while video for shot N generates. This should nearly eliminate audio generation wall time (~20-30s per clip) since it fits within the video generation window (~60s per clip).

## Context

Currently the decode pipeline runs video and audio as entirely separate sequential stages -- two CLI invocations bridged by `--keep-pod` and a state file (`runpod_pod.json`). For a typical film with hundreds of shots, audio generation adds significant wall time even though the GPU is idle during the wait between video clip submissions.

Both strategies use ComfyUI on the same pod. `RunPodWanStrategy` generates video via the ComfyUI HTTP API, then `RunPodMMAudioStrategy` reconnects to the pod and generates audio clips via the same API. Each calls `free_vram()` between clips. Wan uses ~8-10 GB VRAM (1.3B fp16 diffusion model + fp8 text encoder), MMAudio uses ~4-6 GB (large 44kHz fp16 models with `force_offload: True`).

The RTX 4090 (24 GB) is the primary target GPU. Running both models simultaneously would require ~14-16 GB -- tight but feasible if MMAudio's `force_offload` works correctly to release VRAM between inferences.

### Three approaches evaluated

1. **Two ComfyUI instances on different ports** -- doubles ComfyUI overhead, complex port management, higher OOM risk from two separate processes.
2. **Single ComfyUI workflow chaining video into audio** -- couples the two models in a single graph, loses flexibility, non-trivial to build with kijai nodes.
3. **Direct MMAudio Python process alongside ComfyUI** (most promising) -- run MMAudio inference via SSH/Python on the pod while ComfyUI handles video. Decouples the two inference paths and avoids ComfyUI overhead for audio.

This task explores approach 3.

## Scope

### Must Do

- Benchmark VRAM usage: measure Wan inference peak and MMAudio inference peak on RTX 4090 to confirm concurrent feasibility
- Prototype an SSH-based MMAudio Python runner that can run inference directly on the pod without ComfyUI
- Implement pipelined generation: while video generates shot N, audio generates shot N-1 concurrently
- Measure end-to-end time savings vs the current sequential two-stage approach
- Handle the first-shot edge case (no audio to generate yet) and last-shot edge case (audio for final shot after video is done)
- Error handling: if audio fails for a shot, log the error and continue (don't block video pipeline)

### Might Do

- Fall back to sequential ComfyUI-based audio if VRAM is too tight for concurrent execution
- Adaptive concurrency: detect available VRAM and decide whether to pipeline or run sequentially
- Integrate pipelined mode into `pipeline.py` as a single-invocation option (currently requires two CLI calls)

### Won't Do

- Replace the existing sequential `RunPodMMAudioStrategy` -- it remains as the safe default
- Support concurrent generation on GPUs with less than 24 GB VRAM
- Run more than one audio clip concurrently (one video + one audio at a time is the target)

## Constraints

- No new Python dependencies -- use subprocess/SSH for the pod-side MMAudio runner
- Must not break existing `--keep-pod` sequential flow
- Stateless CLI stage pattern (ADR-002) still applies -- intermediate state goes to files on disk
- Pod cleanup handlers must still terminate the pod on crash, even during concurrent execution
- MMAudio models (~5 GB total) are already downloaded during the existing audio stage setup -- reuse them

## Phases

### Phase 1: VRAM Benchmarking

Measure actual peak VRAM during Wan and MMAudio inference on RTX 4090. Determine whether concurrent execution is safe or requires `force_offload` tuning.

### Phase 2: SSH-based MMAudio Runner

Build a lightweight Python script that runs MMAudio inference directly (no ComfyUI) on the pod via SSH. Takes prompt, duration, seed, output path as arguments. Returns the generated audio file.

### Phase 3: Pipelined Orchestration

Modify `decode.py` (or add a new mode) that runs video and audio in a pipelined fashion:
- Shot 1: generate video (no audio yet)
- Shot 2..N: generate video for shot N + audio for shot N-1 concurrently
- After last video: generate audio for final shot

Use `threading` or `asyncio` to manage the two concurrent SSH/HTTP connections.

### Phase 4: Measurement

Compare wall time of pipelined vs sequential for a representative film. Target: audio generation adds near-zero wall time.

## Success Criteria

- [ ] VRAM benchmarks show concurrent Wan + MMAudio fits within 24 GB with margin
- [ ] SSH-based MMAudio runner produces audio clips equivalent to ComfyUI-based approach
- [ ] Pipelined mode generates correct video + audio for all shots (no missing/mismatched clips)
- [ ] End-to-end wall time reduction measured: audio generation adds <10% overhead vs video-only time
- [ ] Existing sequential `--keep-pod` flow still works unchanged
- [ ] No increase in pod crash/OOM rate vs sequential execution

## References

- [ADR-002: Stateless CLI Pipeline](../../design/adr/002-stateless-cli-pipeline.md)
- [Task 0001: RunPod Wan Strategy](../0001-runpod-wan-strategy/task.md)
- [Task 0007: RunPod MMAudio Strategy](../0007-runpod-mmaudio-strategy/task.md)
- [hkchengrex/MMAudio](https://github.com/hkchengrex/MMAudio) -- official repo
- [kijai/ComfyUI-MMAudio](https://github.com/kijai/ComfyUI-MMAudio) -- ComfyUI custom nodes
