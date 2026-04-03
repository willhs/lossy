---
id: task-0010
type: spec
purpose: "Pipeline audio generation alongside video in the decode loop so audio for shot N-1 runs concurrently with video for shot N, nearly eliminating audio wall time."
tags: ["decode", "runpod", "concurrency", "audio", "video", "performance"]
related:
  - "docs/research/0004-concurrent-runpod-generation/research.md"
  - "docs/tasks/0009-concurrent-runpod-generation/task.md"
  - "docs/design/adr/002-stateless-cli-pipeline.md"
created: 2026-03-25
updated: 2026-03-25
---

# Pipelined Audio Generation

## Goal

Modify the decode pipeline so that audio generation for shot N-1 runs concurrently with video generation for shot N on the same RunPod pod. Audio should add near-zero wall time to the overall decode run.

## Context

Research (0004) confirmed that concurrent Wan video + MMAudio audio fits within GPU VRAM:
- Wan peak: ~15 GB, MMAudio standalone: ~7 GB, concurrent peak: ~20 GB
- RTX 4090 (24 GB): 4 GB headroom -- viable
- RTX A6000 (48 GB): 28 GB headroom -- safe
- Standalone MMAudio runner (`tools/mmaudio_standalone.py`) generates audio in ~2s per clip via SSH, bypassing ComfyUI entirely (10-15x faster than ComfyUI-based audio)

Currently video and audio are separate CLI invocations bridged by `--keep-pod`. For a 200-clip film, audio adds ~1.5 hours of wall time. Pipelining would eliminate this.

## Scope

### Must Do

- Upload `mmaudio_standalone.py` to the pod during RunPod strategy setup
- After each video clip completes in the decode loop, kick off audio for the previous clip via SSH in a background thread
- Handle first-shot edge case (no audio to generate yet)
- Handle last-shot edge case (generate audio for final shot after video loop ends)
- Log audio generation errors without blocking the video pipeline
- Track audio progress alongside video progress (audio clips, costs, failures)
- Skip concurrent audio on GPUs with <24 GB VRAM (RTX 4000 Ada)

### Might Do

- Add a `--concurrent-audio` flag to `decode.py` (or make it the default for RunPod strategies)
- Download MMAudio standard weights during pod setup (standalone runner needs them)
- Fall back to sequential ComfyUI-based audio if standalone runner fails

### Won't Do

- Replace the existing sequential `RunPodMMAudioStrategy` -- it remains available
- Run more than one audio clip concurrently (one video + one audio is the target)
- Modify `pipeline.py` orchestration (keep the two-invocation pattern for now)
- Support non-RunPod audio strategies in concurrent mode

## Constraints

- No new Python dependencies -- use `threading` and `subprocess` for concurrency
- Must not break existing `--keep-pod` sequential flow
- Stateless CLI stage pattern (ADR-002) still applies
- Pod cleanup handlers must still terminate on crash during concurrent execution
- MMAudio standard weights (~5 GB) need to be downloaded on first use via `model.download_if_needed()`

## References

- [Research: Concurrent RunPod Generation](docs/research/0004-concurrent-runpod-generation/research.md)
- [Task 0009: Concurrent Generation Investigation](docs/tasks/0009-concurrent-runpod-generation/task.md)
- [ADR-002: Stateless CLI Pipeline](docs/design/adr/002-stateless-cli-pipeline.md)
- [Standalone MMAudio runner](tools/mmaudio_standalone.py)

## Success Criteria

- [ ] Audio generation adds <10% wall time overhead vs video-only decode
- [ ] All shots have matching audio clips after a pipelined run
- [ ] No OOM errors on RTX 4090 or A6000 during pipelined execution
- [ ] Existing sequential `--keep-pod` flow still works unchanged
- [ ] Audio failures are logged but don't block video generation
