---
id: task-0007
type: spec
purpose: "Self-hosted MMAudio on RunPod to eliminate fal.ai audio costs by reusing the existing ComfyUI pod."
tags: ["audio", "runpod", "mmaudio", "decode", "cost"]
related:
  - "docs/design/adr/002-stateless-cli-pipeline.md"
  - "docs/tasks/0001-runpod-wan-strategy/task.md"
  - "docs/tasks/0005-audio-generation/task.md"
created: 2026-03-21
updated: 2026-03-21
---

# RunPod MMAudio Audio Strategy

## Goal

Run MMAudio on the same RunPod pod used for Wan video generation, eliminating fal.ai audio costs (~$3.60/film down to ~$0 marginal cost).

## Context

The decode pipeline already supports RunPod-hosted video generation via `RunPodWanStrategy`, which creates a ComfyUI pod, generates clips, then terminates the pod on process exit. Audio generation currently uses fal.ai-hosted ElevenLabs ($0.002/sec) or MMAudio V2 ($0.001/sec). For a typical 1,150-shot film, audio alone costs $3-4 via fal.ai.

MMAudio V2 needs ~6 GB VRAM and fits alongside Wan 2.1 1.3B (~8-12 GB) on the same GPU when run sequentially. The [kijai/ComfyUI-MMAudio](https://github.com/kijai/ComfyUI-MMAudio) custom nodes (541 stars, trusted author) expose MMAudio as ComfyUI workflows, meaning we can reuse the same HTTP API pattern already proven in `RunPodWanStrategy`.

The key challenge is pod lifecycle: today `RunPodWanStrategy` creates a pod and terminates it on exit via atexit/signal handlers. To share a pod across decode and audio stages (which are separate subprocess invocations from pipeline.py), we need a pod state file and a way to defer termination.

## Scope

### Must Do

- Extract pod lifecycle into a shared module so both video and audio strategies can manage the same pod
- Pod state file (`runpod_pod.json`) in output dir: written on pod creation, read by subsequent stages
- `--keep-pod` flag on `RunPodWanStrategy` to skip termination (pipeline.py sets this when audio stage follows)
- New `RunPodMMAudioStrategy` audio strategy class following the existing `AudioStrategy` pattern
- MMAudio ComfyUI custom node installation during pod setup (clone repo, install deps, download ~1 GB model)
- ComfyUI workflow JSON for text-to-audio (prompt, duration, seed params)
- `generate()` method: submit workflow, poll history, download .flac output
- CLI wiring: `--audio-strategy runpod-mmaudio` option in decode.py
- Pipeline wiring: pass `--keep-pod` when audio stage follows video, terminate pod after audio completes
- Pod reconnection: `RunPodMMAudioStrategy` reads existing pod state, connects without creating a new pod

### Might Do

- Standalone pod creation in `RunPodMMAudioStrategy` (create pod if no state file exists, for audio-only runs)
- Video-to-audio mode (feed generated video clips instead of text prompts)

### Won't Do

- Concurrent video + audio generation on the same pod (sequential only -- VRAM constraints)
- Migration of existing fal.ai MMAudio strategy (it stays as a simpler/no-infra option)
- Custom MMAudio model training or fine-tuning

## Constraints

- No new Python dependencies (runpod SDK and subprocess/SSH already available)
- Pod state file must be JSON, written atomically, and safe for the stateless stage pattern (ADR-002)
- Cleanup handlers must still terminate the pod if the process crashes -- `--keep-pod` only defers termination, it doesn't disable crash cleanup
- MMAudio model download (~1 GB) happens once during pod setup; subsequent runs skip it
- Must not break existing `runpod-wan` strategy when used without audio stage

## References

- [ADR-002: Stateless CLI Pipeline](docs/design/adr/002-stateless-cli-pipeline.md)
- [Task 0001: RunPod Wan Strategy](docs/tasks/0001-runpod-wan-strategy/task.md)
- [Task 0005: Audio Generation](docs/tasks/0005-audio-generation/task.md)
- [kijai/ComfyUI-MMAudio](https://github.com/kijai/ComfyUI-MMAudio) -- ComfyUI custom nodes
- [hkchengrex/MMAudio](https://github.com/hkchengrex/MMAudio) -- official repo

## Success Criteria

- [ ] `python decode.py output/film --audio --audio-strategy runpod-mmaudio` generates audio clips using the shared RunPod pod
- [ ] When run after `decode.py --strategy runpod-wan --keep-pod`, the audio strategy reconnects to the existing pod (no new pod creation)
- [ ] `pipeline.py` with `--strategy runpod-wan --audio-strategy runpod-mmaudio` runs both stages on one pod and terminates it after audio/stitch
- [ ] Audio quality is comparable to fal.ai-hosted MMAudio V2
- [ ] Full film audio generation adds ~$0 marginal cost (pod time already paid by video stage)
- [ ] Existing `runpod-wan` without audio stage still works (pod terminates normally)
