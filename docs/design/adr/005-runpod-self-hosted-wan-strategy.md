---
id: adr-005-runpod-self-hosted-wan
type: decision
purpose: "Record the decision to add a self-hosted RunPod strategy using ComfyUI with Wan 2.1 1.3B for cheaper decoding."
scope: ["design", "decoder"]
non_goals: []
tags: ["adr", "decoder", "video-generation", "runpod", "comfyui"]
related: ["design/adr/004-replicate-wan22-for-video-generation.md", "research/0003-prompt-to-video/research.md"]
---

# Context

The existing decode strategies use hosted APIs (Replicate at $0.05/clip, fal.ai Seedance at $0.10-0.25/clip). A full Star Wars IV decode costs $58-290 depending on strategy. This makes iterating on the full film expensive. The research doc (0003) identified self-hosted Wan on RunPod as the cheapest option at ~$5-15 for a full film, but noted it required managing GPU infrastructure.

# Decision

Add a **RunPod self-hosted strategy** (`--strategy runpod-wan`) that manages the full pod lifecycle automatically:

- Uses `runpod/comfyui:latest` (RunPod's official ComfyUI image) on community cloud GPUs
- Downloads Wan 2.1 1.3B fp16 models via SSH on first use (~9GB, takes ~2-3 min)
- Generates 81 frames at 16fps (same ~5s clips as replicate-wan) via ComfyUI HTTP API
- Outputs WEBM, converts to MP4 locally via ffmpeg
- GPU fallback chain: RTX 4090 -> A5000 -> RTX 4000 Ada -> L40S -> A6000
- Pod terminated automatically on completion, error, or Ctrl+C (atexit + signal handlers)

**Actual observed costs** (RTX A5000 community cloud, $0.34/hr):
- ~3.5 min per clip (including generation + download + conversion)
- ~$0.02/clip amortized (includes model download overhead on first run)
- 10-clip session: $0.22 total, 37 min
- Extrapolated full film (1,161 clips): ~$25-30 (higher than the optimistic $5 estimate due to slower generation on A5000 vs 4090)

# Consequences

**Good:**
- 2-3x cheaper than Replicate for a full decode ($25-30 vs $58)
- Same clip format and quality as replicate-wan (compatible with stitch pipeline)
- No external API keys beyond RunPod -- all generation happens on your own GPU
- Resume works: re-running skips completed shots, creates a new pod

**Bad:**
- Cold start overhead: ~5 min for pod creation + model download + ComfyUI startup (amortized over many clips)
- Requires SSH key for model download (reads ~/.ssh/id_ed25519.pub or id_rsa.pub)
- GPU availability is not guaranteed -- community cloud stock varies
- Model quality is Wan 2.1 1.3B (not Wan 2.2) -- the Docker image's naming is misleading
- Generation is slower than expected: ~3.5 min/clip on A5000 vs the theoretical ~40s/clip on 4090

**Operational:**
- RunPod API key required in `.env` as `RUNPOD_API_KEY`
- Pod cleanup is robust (atexit + SIGINT/SIGTERM handlers) but if the process is killed with SIGKILL, the pod will keep running and billing. Check https://runpod.io/console/pods.
- Models are downloaded fresh each session. A RunPod Network Volume would eliminate this but locks you to a specific datacenter.

# Alternatives Considered

- **Pre-built Docker image with models** (`ghcr.io/lum3on/wan22-runpod:latest`): Tested first, but ComfyUI never started -- the image's startup scripts are unreliable.
- **RunPod Serverless endpoints**: More complex setup, requires building a custom handler. On-demand pods are simpler for batch workloads.
- **RunPod Network Volumes**: Would eliminate per-session model download (~3 min saving) but locks pod creation to a specific datacenter, reducing GPU availability. Worth adding later if generation becomes a frequent workflow.
- **Custom Docker image**: Would bake models into the image for instant startup. Requires maintaining a registry. Deferred -- current approach works.
