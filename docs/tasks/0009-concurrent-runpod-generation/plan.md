---
id: plan-0009
type: spec
purpose: "Investigate whether concurrent video+audio generation is feasible on a shared RunPod pod, and document options."
tags: ["plan", "runpod", "concurrency", "audio", "video", "investigation"]
related: ["./task.md"]
---

# Concurrent Video + Audio Generation — Investigation Plan

## Overview

Investigate whether MMAudio audio generation can run concurrently alongside Wan video generation on the same RunPod GPU pod, and document the viable options for reducing audio generation wall time.

**Primary Goal**: Produce a findings document with go/no-go for concurrent generation on RTX 4090, alternative GPU options if VRAM is tight, and cost/performance tradeoffs.

**Approach**: Benchmark first, prototype only if benchmarks look promising, document findings either way.

## Current State Analysis

- Video: `RunPodWanStrategy` generates clips via ComfyUI HTTP API on a RunPod pod (`strategies_video.py:201-458`)
- Audio: `RunPodMMAudioStrategy` generates clips via ComfyUI HTTP API on the same pod, sequentially after video (`strategies_audio.py:197-389`)
- Pod lifecycle: `RunPodSession` (`runpod_pod.py:28-451`) manages creation, SSH, state persistence, cleanup
- Pod reuse: `--keep-pod` flag writes `runpod_pod.json` so audio stage reconnects to video stage's pod
- ComfyUI is single-threaded — cannot process two workflows simultaneously, so concurrent audio must bypass ComfyUI
- Target GPU: RTX 4090 (24 GB VRAM, $0.34/hr on RunPod community)
- Wan VRAM: ~8-10 GB (1.3B fp16 diffusion + fp8 text encoder)
- MMAudio VRAM: ~4-6 GB (large 44kHz fp16 models with `force_offload: True`)
- Video per clip: ~60s wall time
- Audio per clip: ~20-30s wall time (including ComfyUI overhead)

### Key Constraint

ComfyUI queues workflows sequentially. To run video and audio concurrently on one pod, audio must run as a standalone Python process via SSH — not through ComfyUI's API.

## Desired End State

A research document at `docs/research/0004-concurrent-runpod-generation/research.md` covering:
1. Measured peak VRAM for Wan and MMAudio individually and (if feasible) concurrently
2. Go/no-go for concurrent generation on RTX 4090 (24 GB)
3. Alternative GPU tiers on RunPod with enough headroom (e.g. A6000 48GB, A100 40/80GB) and their cost/hr
4. Whether other services could handle this more naturally (e.g. separate pods for audio, fal.ai MMAudio API as fallback)
5. Recommended path forward with cost/performance comparison

## What We're NOT Doing

- Merging any code into main — this is investigation only
- Modifying `decode.py`, `strategies_audio.py`, or `pipeline.py`
- Building a production-ready pipelined orchestrator
- Supporting GPUs with less than 24 GB VRAM

---

## Phase 1: VRAM Benchmarking

### Overview
Measure actual peak VRAM during Wan and MMAudio inference on the current RTX 4090 pod. This is the go/no-go gate for everything else.

### Tasks

#### 1. Create VRAM benchmark script
- [x] Create `tools/vram_benchmark.sh` — a shell script that:
  - SSHes to the pod
  - Starts `nvidia-smi --query-gpu=memory.used --format=csv -l 1` logging to `/tmp/vram_log.csv` in background
  - Triggers a Wan video generation (single clip) via the ComfyUI API
  - Waits for completion, stops logging, records peak
  - Calls `free_vram()` endpoint
  - Triggers an MMAudio audio generation (single clip) via the ComfyUI API
  - Waits for completion, stops logging, records peak
  - Prints both peaks and the sum

```bash
#!/usr/bin/env bash
# Usage: ./tools/vram_benchmark.sh <output_dir>
# Requires: an active pod (run decode.py first with --keep-pod, or create manually)
# Reads connection info from <output_dir>/runpod_pod.json

set -euo pipefail

OUTPUT_DIR="${1:?Usage: $0 <output_dir>}"
POD_STATE="$OUTPUT_DIR/runpod_pod.json"

if [ ! -f "$POD_STATE" ]; then
    echo "Error: $POD_STATE not found. Start a pod first."
    exit 1
fi

SSH_HOST=$(python3 -c "import json; print(json.load(open('$POD_STATE'))['ssh_host'])")
SSH_PORT=$(python3 -c "import json; print(json.load(open('$POD_STATE'))['ssh_port'])")
BASE_URL=$(python3 -c "import json; print(json.load(open('$POD_STATE'))['base_url'])")

SSH_CMD="ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR -p $SSH_PORT root@$SSH_HOST"

echo "=== VRAM Benchmark ==="
echo "Pod: $SSH_HOST:$SSH_PORT"
echo ""

# Baseline VRAM
BASELINE=$($SSH_CMD "nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits" | tr -d '[:space:]')
echo "Baseline VRAM: ${BASELINE} MiB"

# Start VRAM logging
$SSH_CMD "nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -l 1 > /tmp/vram_log.csv &"

echo ""
echo "--- Wan Video Generation ---"
echo "Submit a single Wan workflow via ComfyUI API and monitor peak VRAM."
echo "(Run this while a video clip is generating)"
echo ""

# The actual workflow submission would be done by running decode.py for 1 clip
# or by posting a workflow JSON to the ComfyUI API.
# For the benchmark, we just need to observe nvidia-smi during generation.

echo "Instructions:"
echo "  1. In another terminal, generate 1 video clip:"
echo "     python decode.py <output_dir> --strategy runpod-wan --limit 1 --keep-pod"
echo "  2. While it generates, this script monitors VRAM"
echo "  3. After completion, press Enter here"
read -p "Press Enter when video generation is complete..."

# Read peak from log
VIDEO_PEAK=$($SSH_CMD "sort -n /tmp/vram_log.csv | tail -1" | tr -d '[:space:]')
echo "Wan peak VRAM: ${VIDEO_PEAK} MiB"

# Reset log
$SSH_CMD "curl -s -X POST $BASE_URL/free -H 'Content-Type: application/json' -d '{\"free_memory\": true}' > /dev/null 2>&1 || true"
sleep 3
$SSH_CMD "> /tmp/vram_log.csv"

echo ""
echo "--- MMAudio Audio Generation ---"
read -p "Now generate 1 audio clip and press Enter when done..."

AUDIO_PEAK=$($SSH_CMD "sort -n /tmp/vram_log.csv | tail -1" | tr -d '[:space:]')
echo "MMAudio peak VRAM: ${AUDIO_PEAK} MiB"

# Cleanup
$SSH_CMD "pkill -f 'nvidia-smi.*-l' || true"

echo ""
echo "=== Results ==="
echo "Baseline:        ${BASELINE} MiB"
echo "Wan peak:        ${VIDEO_PEAK} MiB"
echo "MMAudio peak:    ${AUDIO_PEAK} MiB"
echo "Sum (concurrent): $((VIDEO_PEAK + AUDIO_PEAK - BASELINE)) MiB"
echo "RTX 4090 total:  24576 MiB"
echo "Headroom:        $((24576 - VIDEO_PEAK - AUDIO_PEAK + BASELINE)) MiB"
```

#### 2. Run the benchmark
- [x] Manual: Spin up a pod with `python decode.py <output_dir> --strategy runpod-wan --limit 1 --keep-pod`
- [x] Manual: Measured VRAM via nvidia-smi during Wan generation (benchmark script not used -- direct measurement instead)
- [x] Manual: Recorded results: baseline=306 MiB, Wan peak=14,994 MiB (~15 GB) on RTX 4000 Ada (20 GB)
- [ ] MMAudio peak VRAM not measured (ComfyUI-MMAudio node install failed on pod)

#### 3. Evaluate results
- [x] Manual: Go/no-go determined:
  - Wan uses ~15 GB (higher than estimated 8-10 GB)
  - RTX 4090 (24 GB): 3-9 GB headroom → risky, proceed cautiously
  - RTX A6000 (48 GB): 27-33 GB headroom → safe, preferred
  - A6000 was unavailable during test -- availability is unreliable

### Success Criteria
- [x] Manual: Wan peak VRAM recorded (14,994 MiB). MMAudio not yet measured.
- [x] Manual: Go/no-go: Concurrent on A6000 is safe. Concurrent on 4090 is risky. Need MMAudio measurement to confirm.

---

## Phase 2: MMAudio Standalone Runner Prototype (conditional on Phase 1)

### Overview
If Phase 1 shows sufficient VRAM headroom, prototype a standalone MMAudio inference script that runs directly on the pod via SSH — bypassing ComfyUI entirely.

### Tasks

#### 1. Research MMAudio standalone API
- [ ] SSH to the pod and check what the `mmaudio` package provides:
  ```
  ssh root@<host> "python3 -c 'import mmaudio; print(mmaudio.__file__)'"
  ssh root@<host> "python3 -c 'from mmaudio.eval_utils import ModelConfig, all_model_cfg; print(list(all_model_cfg.keys()))'"
  ```
- [ ] Identify how to load models from the ComfyUI models directory (vs the default HuggingFace cache)
- [ ] Check if `mmaudio` can be called with safetensors weights from `Kijai/MMAudio_safetensors`

#### 2. Create standalone runner script
- [x] Create `tools/mmaudio_standalone.py` — a prototype script to upload and run on the pod:
  - Takes args: `--prompt`, `--duration`, `--seed`, `--output`
  - Loads MMAudio models (either from the standard path or the ComfyUI models dir)
  - Runs text-to-audio inference
  - Saves as FLAC
  - Prints VRAM usage before/after

#### 3. Test the runner
- [ ] Manual: Upload script to pod via SSH and run it
- [ ] Manual: Verify it produces audio equivalent to the ComfyUI-based approach
- [ ] Manual: Measure wall time — should be faster than ComfyUI (no node graph overhead)
- [ ] Manual: Measure VRAM usage during standalone inference

#### 4. Test concurrent execution
- [ ] Manual: Start a Wan video generation via ComfyUI API
- [ ] Manual: Simultaneously run `mmaudio_standalone.py` via SSH
- [ ] Manual: Verify both complete without OOM
- [ ] Manual: Record VRAM peak during concurrent execution

### Success Criteria
- [ ] Manual: Standalone runner produces audio clips
- [ ] Manual: Concurrent video+audio execution tested (success or OOM documented)

---

## Phase 3: Alternatives Research

### Overview
Regardless of Phase 1/2 outcomes, research alternative approaches and GPU options. This runs in parallel with Phase 2 or replaces it if Phase 1 is a no-go.

### Tasks

#### 1. RunPod GPU options
- [x] Research available GPU tiers on RunPod and their VRAM/cost:
  - RTX 4090: 24 GB, ~$0.34/hr (current)
  - RTX A6000: 48 GB, ~$0.25-0.33/hr (cheaper than 4090!)
  - A100 PCIe 80GB: ~$1.19/hr
  - A100 SXM 80GB: ~$1.39/hr
  - H100 PCIe 80GB: ~$1.99/hr
  - L40S: 48 GB, ~$0.79/hr
- [x] Calculate the "just right" GPU — RTX A6000 (48 GB, $0.25-0.33/hr) is optimal: cheaper than 4090 with 2x VRAM
- [x] Calculate cost difference per film (e.g. 200 clips × ~60s each = ~3.3 hrs of pod time)

#### 2. Alternative architectures
- [x] Document option: **Two separate pods** — one for video (cheap GPU), one for audio (cheap GPU). Doubles pod cost but trivially concurrent. Calculate cost.
- [x] Document option: **fal.ai MMAudio API as audio backend** — already implemented as `MMAudioStrategy` (`strategies_audio.py:127`). Audio runs on fal.ai ($0.001/sec) while video runs on RunPod. No VRAM conflict. Calculate cost for a typical film.
- [x] Document option: **Larger single pod** — upgrade from RTX 4090 to A6000 (48GB) or A100. Same concurrent approach but with headroom. Calculate cost delta.
- [x] Document option: **Sequential but faster** — optimize the existing sequential flow (reduce ComfyUI overhead, faster model loading) instead of concurrent execution.

#### 3. Cost comparison table
- [x] Build a comparison table:

| Approach | Audio wall time | Extra cost/film | Complexity | Risk |
|----------|----------------|-----------------|------------|------|
| Sequential (current) | ~X min | $0 | None | None |
| Concurrent RTX 4090 | ~0 min | $0 | High | OOM |
| Concurrent A6000 | ~0 min | +$X | High | Low |
| Two pods | ~0 min | +$X | Medium | Low |
| fal.ai audio | ~0 min | +$X | Low | None |

### Success Criteria
- [x] Manual: All options researched with concrete cost numbers
- [x] Manual: Comparison table completed

---

## Phase 4: Write Findings

### Tasks

#### 1. Create research document
- [x] Create `docs/research/0004-concurrent-runpod-generation/research.md` with:
  - VRAM benchmark data from Phase 1 (placeholder -- awaiting manual benchmarks)
  - Prototype results from Phase 2 (script created, awaiting manual testing)
  - Options comparison table from Phase 3
  - Recommended path forward
  - Raw data / logs as appendix (placeholder)

### Success Criteria
- [x] Manual: `research.md` written with all findings and recommendation

---

## Final Checklist

- [x] Phase 1 benchmark completed (Wan peak: 15 GB, MMAudio pending)
- [x] Phase 2 prototype attempted (`tools/mmaudio_standalone.py` created, awaiting manual testing)
- [x] Phase 3 alternatives researched
- [x] Phase 4 results document written
- [x] `tools/vram_benchmark.sh` committed (useful for future GPU evaluations)

## References

- Task: `docs/tasks/0009-concurrent-runpod-generation/task.md`
- [ADR-002: Stateless CLI Pipeline](../../design/adr/002-stateless-cli-pipeline.md)
- [Task 0001: RunPod Wan Strategy](../0001-runpod-wan-strategy/task.md)
- [Task 0007: RunPod MMAudio Strategy](../0007-runpod-mmaudio-strategy/task.md)
- [hkchengrex/MMAudio](https://github.com/hkchengrex/MMAudio)
- [RunPod GPU pricing](https://www.runpod.io/pricing)
