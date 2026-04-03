---
id: plan-0001
type: spec
purpose: "Implementation plan for RunPod self-hosted Wan 2.2 decode strategy."
tags: ["plan", "decoder", "strategy", "runpod"]
related: ["./task.md"]
---

# RunPod Wan Strategy Implementation Plan

## Overview

Add `RunPodWanStrategy` to `decode.py` that manages a RunPod GPU pod running ComfyUI with Wan 2.2 (1.3B fp8), generates clips via the ComfyUI HTTP API, and tears down the pod on completion or error.

**Primary Goal**: `python decode.py output/star_wars_iv_v2 --strategy runpod-wan` works end-to-end at ~$0.004/clip.

**Approach**: Single class in `decode.py` following the existing `GenerationStrategy` pattern. Pod is created lazily on first `generate()` call, reused for all subsequent clips, and terminated via `atexit`/signal handler. ComfyUI workflow JSON is hardcoded as a Python dict template.

## Current State Analysis

### Strategy Interface (`decode.py:45-62`)
```python
class GenerationStrategy:
    name: str = "base"
    def generate(self, prompt, clips_dir, shot_index, target_duration_s, seed=None) -> list[ClipResult]:
        raise NotImplementedError
```

### Strategy Registration (`decode.py:556-561`)
```python
strategies = {
    "replicate-wan": ReplicateWanStrategy,
    "fal-seedance": FalSeedanceStrategy,
    "fal-seedance-pro": FalSeedanceProStrategy,
}
```

### Key Patterns
- `ReplicateWanStrategy` produces fixed ~5s clips (81 frames / 16 fps). New strategy follows same approach.
- Cost is returned per-clip in `ClipResult.cost`. Accumulated in `run_decode` into `progress["total_cost_estimate"]`.
- Imports are lazy (inside `generate()` method body) — `decode.py:79-80`, `decode.py:157-158`.
- `.env` loading via `load_env()` at `decode.py:20-29`, uses `os.environ.setdefault()`.

### ComfyUI API (from research)
- POST workflow JSON to `/prompt` -> get `prompt_id`
- Poll `/history/{prompt_id}` until key appears -> execution complete
- Download output via `/view?filename=...&type=output`
- Videos appear under `"images"` key in history outputs (even for video files)

### RunPod SDK (from research)
- `runpod.create_pod()` -> returns dict with `"id"`
- `runpod.get_pod(pod_id)` -> `runtime` is `None` until container ready, then has `ports` list
- `runpod.terminate_pod(pod_id)` -> destroys pod
- Proxy URL: `https://{pod_id}-8188.proxy.runpod.net/` (works without public IP)
- GPU type ID: `"NVIDIA GeForce RTX 4090"`, community cloud at ~$0.34/hr

### Docker Image (`ghcr.io/lum3on/wan22-runpod:latest`)
- ComfyUI on port 8188
- Pre-loaded models: `wan2.2_1.3B_fp8_scaled.safetensors` (likely Wan 2.1 1.3B repackaged), `t5xxl_fp8_e4m3fn.safetensors`, `wan_vae.safetensors`
- Needs ~50 GB container disk for models + outputs
- Env: `GPU_TYPE=auto` auto-detects GPU

### Wan T2V Workflow Nodes
The ComfyUI API-format workflow for text-to-video with the 1.3B model:
1. `UNETLoader` — load diffusion model
2. `CLIPLoader` — load T5XXL text encoder (type: `"wan"`)
3. `VAELoader` — load Wan VAE
4. `CLIPTextEncode` x2 — positive prompt + negative prompt
5. `ModelSamplingSD3` — set shift=8.0
6. `EmptyWanVideoLatent` — create empty latent (width, height, length, batch_size)
7. `KSampler` — sample with uni_pc/simple, 20-30 steps, cfg 5
8. `VAEDecode` — decode latent to frames
9. `SaveAnimatedWEBP` — save output (ComfyUI saves to its output dir)

Note: Using `EmptyWanVideoLatent` (not `Wan22ImageToVideoLatent`) since the bundled model is the 1.3B which is a Wan 2.1-era architecture. If the image actually contains a 5B model, the latent node would need to change to `Wan22ImageToVideoLatent`. We'll verify at runtime by checking which nodes are available via the ComfyUI `/object_info` endpoint.

## Desired End State

- `--strategy runpod-wan` is a valid CLI option
- Running it creates a RunPod pod, generates all clips, and terminates the pod
- Interruption (Ctrl+C, crash) always terminates the pod
- Resume works: re-running skips completed shots, creates a new pod
- Cost per clip is tracked based on wall-clock time * hourly rate
- Clips work with the existing stitch pipeline

## What We're NOT Doing

- Custom Docker images (using lum3on image as-is)
- Serverless RunPod endpoints
- Multi-part clip splitting (fixed ~5s clips only)
- Variable duration control
- Batch queueing to ComfyUI (one prompt at a time)
- GPU fallback chain (RTX 4090 only)

---

## Phase 1: RunPodWanStrategy Implementation

### Overview
Add the complete strategy class to `decode.py` with pod lifecycle, ComfyUI API integration, cleanup handling, and CLI registration. Add `runpod` to dependencies.

### Tasks

#### 1. Add `runpod` dependency
- [x] Add `runpod` to project dependencies

```bash
pip install runpod
```

If there's a `requirements.txt` or `pyproject.toml`, add `runpod` there. Currently the project has no formal dependency file (imports are bare), so just ensure it's installed.

#### 2. Add `RunPodWanStrategy` class to `decode.py`

- [x] Add `RunPodWanStrategy` class after `FalSeedanceProStrategy` (after line 214)

```python
class RunPodWanStrategy(GenerationStrategy):
    """RunPod self-hosted Wan 2.2 (1.3B fp8) via ComfyUI -- ~$0.004/clip."""

    name = "runpod-wan"
    CLIP_DURATION = 81 / 16  # ~5.0625s (81 frames at 16fps)
    GPU_TYPE = "NVIDIA GeForce RTX 4090"
    GPU_HOURLY_RATE = 0.34  # $/hr community cloud
    DOCKER_IMAGE = "ghcr.io/lum3on/wan22-runpod:latest"
    CONTAINER_DISK_GB = 50
    COMFYUI_PORT = 8188
    POD_READY_TIMEOUT = 600  # 10 min for image pull + model load
    COMFYUI_READY_TIMEOUT = 300  # 5 min for ComfyUI to start serving
    GENERATION_TIMEOUT = 300  # 5 min per clip

    def __init__(self):
        self._pod_id: str | None = None
        self._base_url: str | None = None
        self._pod_start_time: float | None = None
        self._setup_cleanup_handler()

    def _setup_cleanup_handler(self):
        """Register atexit and signal handlers to terminate pod on exit."""
        import atexit
        import signal

        atexit.register(self._terminate_pod)

        original_sigint = signal.getsignal(signal.SIGINT)
        original_sigterm = signal.getsignal(signal.SIGTERM)

        def _handler(signum, frame):
            self._terminate_pod()
            # Re-raise with original handler
            if signum == signal.SIGINT and callable(original_sigint):
                original_sigint(signum, frame)
            elif signum == signal.SIGTERM and callable(original_sigterm):
                original_sigterm(signum, frame)
            else:
                sys.exit(1)

        signal.signal(signal.SIGINT, _handler)
        signal.signal(signal.SIGTERM, _handler)

    def _ensure_pod(self):
        """Create and wait for pod if not already running."""
        if self._pod_id is not None:
            return

        import runpod

        runpod.api_key = os.environ.get("RUNPOD_API_KEY")
        if not runpod.api_key:
            print("Error: RUNPOD_API_KEY not set in .env or environment.")
            sys.exit(1)

        print(f"  Creating RunPod pod ({self.GPU_TYPE})...")
        pod = runpod.create_pod(
            name="lossy-comfyui",
            image_name=self.DOCKER_IMAGE,
            gpu_type_id=self.GPU_TYPE,
            cloud_type="COMMUNITY",
            gpu_count=1,
            container_disk_in_gb=self.CONTAINER_DISK_GB,
            ports=f"{self.COMFYUI_PORT}/http",
            support_public_ip=True,
        )
        self._pod_id = pod["id"]
        self._pod_start_time = time.time()
        print(f"  Pod created: {self._pod_id}")

        # Wait for pod runtime to be ready
        print("  Waiting for pod to start...")
        start = time.time()
        while time.time() - start < self.POD_READY_TIMEOUT:
            status = runpod.get_pod(self._pod_id)
            runtime = status.get("runtime")
            if runtime is not None and runtime.get("ports"):
                break
            time.sleep(5)
        else:
            print(f"  Error: Pod did not start within {self.POD_READY_TIMEOUT}s")
            self._terminate_pod()
            sys.exit(1)

        # Use RunPod proxy URL (works without public IP)
        self._base_url = f"https://{self._pod_id}-{self.COMFYUI_PORT}.proxy.runpod.net"
        print(f"  Pod ready: {self._base_url}")

        # Wait for ComfyUI to be serving
        self._wait_for_comfyui()

    def _wait_for_comfyui(self):
        """Poll ComfyUI until it responds to /system_stats."""
        import httpx

        print("  Waiting for ComfyUI to load models...")
        start = time.time()
        while time.time() - start < self.COMFYUI_READY_TIMEOUT:
            try:
                resp = httpx.get(f"{self._base_url}/system_stats", timeout=10)
                if resp.status_code == 200:
                    print("  ComfyUI is ready.")
                    return
            except (httpx.ConnectError, httpx.TimeoutException, httpx.ReadError):
                pass
            time.sleep(5)

        print(f"  Error: ComfyUI did not become ready within {self.COMFYUI_READY_TIMEOUT}s")
        self._terminate_pod()
        sys.exit(1)

    def _terminate_pod(self):
        """Terminate the pod if running."""
        if self._pod_id is None:
            return

        import runpod

        pod_id = self._pod_id
        self._pod_id = None  # Prevent double-terminate

        elapsed_h = (time.time() - self._pod_start_time) / 3600 if self._pod_start_time else 0
        estimated_cost = elapsed_h * self.GPU_HOURLY_RATE

        try:
            runpod.api_key = os.environ.get("RUNPOD_API_KEY")
            runpod.terminate_pod(pod_id)
            print(f"  Pod {pod_id} terminated. Session cost: ~${estimated_cost:.2f} ({elapsed_h:.1f}hrs)")
        except Exception as e:
            print(f"  Warning: Failed to terminate pod {pod_id}: {e}")
            print(f"  Manually terminate at https://www.runpod.io/console/pods")

    def _build_workflow(self, prompt: str, seed: int) -> dict:
        """Build ComfyUI API-format workflow JSON for Wan T2V."""
        return {
            "1": {
                "class_type": "UNETLoader",
                "inputs": {
                    "unet_name": "wan2.2_1.3B_fp8_scaled.safetensors",
                    "weight_dtype": "fp8_e4m3fn",
                },
            },
            "2": {
                "class_type": "CLIPLoader",
                "inputs": {
                    "clip_name": "t5xxl_fp8_e4m3fn.safetensors",
                    "type": "wan",
                    "device": "default",
                },
            },
            "3": {
                "class_type": "VAELoader",
                "inputs": {
                    "vae_name": "wan_vae.safetensors",
                },
            },
            "4": {
                "class_type": "CLIPTextEncode",
                "inputs": {
                    "text": prompt,
                    "clip": ["2", 0],
                },
            },
            "5": {
                "class_type": "CLIPTextEncode",
                "inputs": {
                    "text": "",
                    "clip": ["2", 0],
                },
            },
            "6": {
                "class_type": "ModelSamplingSD3",
                "inputs": {
                    "shift": 8.0,
                    "model": ["1", 0],
                },
            },
            "7": {
                "class_type": "EmptyWanVideoLatent",
                "inputs": {
                    "width": 832,
                    "height": 480,
                    "length": 81,
                    "batch_size": 1,
                },
            },
            "8": {
                "class_type": "KSampler",
                "inputs": {
                    "seed": seed,
                    "steps": 20,
                    "cfg": 5.0,
                    "sampler_name": "uni_pc",
                    "scheduler": "simple",
                    "denoise": 1.0,
                    "model": ["6", 0],
                    "positive": ["4", 0],
                    "negative": ["5", 0],
                    "latent_image": ["7", 0],
                },
            },
            "9": {
                "class_type": "VAEDecode",
                "inputs": {
                    "samples": ["8", 0],
                    "vae": ["3", 0],
                },
            },
            "10": {
                "class_type": "SaveAnimatedWEBP",
                "inputs": {
                    "filename_prefix": "lossy",
                    "fps": 16,
                    "lossless": False,
                    "quality": 80,
                    "method": "default",
                    "images": ["9", 0],
                },
            },
        }

    def generate(
        self,
        prompt: str,
        clips_dir: str,
        shot_index: int,
        target_duration_s: float,
        seed: int | None = None,
    ) -> list[ClipResult]:
        import httpx

        self._ensure_pod()

        clip_path = os.path.join(clips_dir, f"{shot_index:04d}.mp4")
        effective_seed = seed if seed is not None else shot_index

        try:
            # Submit workflow to ComfyUI
            workflow = self._build_workflow(prompt, effective_seed)
            resp = httpx.post(
                f"{self._base_url}/prompt",
                json={"prompt": workflow},
                timeout=30,
            )
            resp.raise_for_status()
            prompt_id = resp.json()["prompt_id"]

            # Poll for completion
            start = time.time()
            while time.time() - start < self.GENERATION_TIMEOUT:
                resp = httpx.get(
                    f"{self._base_url}/history/{prompt_id}",
                    timeout=10,
                )
                history = resp.json()
                if prompt_id in history:
                    break
                time.sleep(2)
            else:
                print(f"  Timeout waiting for clip {shot_index}")
                return []

            # Check for errors
            status = history[prompt_id].get("status", {})
            if status.get("status_str") != "success":
                print(f"  ComfyUI error for shot {shot_index}: {status}")
                return []

            # Find output file
            outputs = history[prompt_id].get("outputs", {})
            output_file = None
            for node_id, node_output in outputs.items():
                if "images" in node_output:
                    for item in node_output["images"]:
                        output_file = item
                        break
                    if output_file:
                        break

            if not output_file:
                print(f"  No output file found for shot {shot_index}")
                return []

            # Download output (WEBP) and convert to MP4
            resp = httpx.get(
                f"{self._base_url}/view",
                params={
                    "filename": output_file["filename"],
                    "subfolder": output_file.get("subfolder", ""),
                    "type": output_file.get("type", "output"),
                },
                timeout=60,
                follow_redirects=True,
            )
            resp.raise_for_status()

            # Save as temp WEBP, convert to MP4 via ffmpeg
            webp_path = clip_path.replace(".mp4", ".webp")
            with open(webp_path, "wb") as f:
                f.write(resp.content)

            subprocess.run(
                [
                    "ffmpeg", "-i", webp_path,
                    "-c:v", "libx264", "-pix_fmt", "yuv420p",
                    "-y", clip_path,
                ],
                capture_output=True,
            )
            os.remove(webp_path)

            # Cost based on wall-clock time since pod started
            elapsed_h = (time.time() - self._pod_start_time) / 3600
            clips_so_far = len([f for f in os.listdir(clips_dir) if f.endswith(".mp4")])
            per_clip_cost = (elapsed_h * self.GPU_HOURLY_RATE) / max(clips_so_far, 1)

            return [ClipResult(path=clip_path, actual_duration_s=self.CLIP_DURATION, cost=per_clip_cost)]

        except Exception as e:
            print(f"  Error generating clip: {e}")
            return []
```

#### 3. Register strategy in CLI

- [x] Add `"runpod-wan"` to the `--strategy` choices at line 546

```python
parser.add_argument("--strategy", choices=["replicate-wan", "fal-seedance", "fal-seedance-pro", "runpod-wan"],
                    default="replicate-wan",
                    help="Video generation backend (default: replicate-wan)")
```

- [x] Add `"runpod-wan"` to the strategies dict at line 556

```python
strategies = {
    "replicate-wan": ReplicateWanStrategy,
    "fal-seedance": FalSeedanceStrategy,
    "fal-seedance-pro": FalSeedanceProStrategy,
    "runpod-wan": RunPodWanStrategy,
}
```

#### 4. Add tests for RunPodWanStrategy

- [x] Add tests to `test_decode.py`

```python
from decode import RunPodWanStrategy


class TestRunPodWanStrategy:
    def test_name(self):
        assert RunPodWanStrategy.name == "runpod-wan"

    def test_clip_duration_constant(self):
        assert RunPodWanStrategy.CLIP_DURATION == pytest.approx(5.0625)

    def test_build_workflow_structure(self):
        strategy = RunPodWanStrategy()
        workflow = strategy._build_workflow("a cat walking", seed=42)

        # Has all required nodes
        assert "1" in workflow  # UNETLoader
        assert "8" in workflow  # KSampler
        assert "10" in workflow  # SaveAnimatedWEBP

        # Prompt is injected
        assert workflow["4"]["inputs"]["text"] == "a cat walking"

        # Seed is injected
        assert workflow["8"]["inputs"]["seed"] == 42

        # Output dimensions are 480p 16:9
        assert workflow["7"]["inputs"]["width"] == 832
        assert workflow["7"]["inputs"]["height"] == 480
        assert workflow["7"]["inputs"]["length"] == 81

    def test_build_workflow_negative_prompt_empty(self):
        strategy = RunPodWanStrategy()
        workflow = strategy._build_workflow("test prompt", seed=1)
        assert workflow["5"]["inputs"]["text"] == ""
```

### Success Criteria

#### Automated Verification:
- [x] Run: `pytest test_decode.py -v` — all tests pass (existing + new)

#### Manual Verification:
- [ ] Manual: `RUNPOD_API_KEY` is set in `.env`

---

## Phase 2: End-to-End Verification

### Overview
Run the strategy against real shots, verify the full pipeline works, and document costs.

### Tasks

- [ ] Manual: Run `python decode.py output/star_wars_iv_v2 --strategy runpod-wan --start-index 10 --limit 5` and verify:
  - Pod is created and ComfyUI becomes ready
  - Clips are generated in `clips/runpod-wan/`
  - Progress is saved in `decode_progress_runpod-wan.json`
  - Pod is terminated on completion
- [ ] Manual: Interrupt with Ctrl+C mid-generation, verify pod is terminated
- [ ] Manual: Re-run the same command, verify it resumes (skips completed shots)
- [ ] Manual: Run `python decode.py output/star_wars_iv_v2 --strategy runpod-wan --stitch` and verify stitched output
- [ ] Manual: If the workflow JSON doesn't work (wrong node names, wrong model filename), fix by:
  1. Checking `/object_info` on the running ComfyUI to list available node types
  2. Checking `/models` or the filesystem for actual model filenames
  3. Updating `_build_workflow()` accordingly

### Workflow JSON Contingency

The hardcoded workflow JSON is our best guess based on research. The most likely failure points are:
- **Model filename mismatch**: the actual safetensors files in the Docker image may have different names than documented. Fix: query `/object_info` or shell into the pod to list files.
- **Node type names**: `EmptyWanVideoLatent` might be `EmptySD3LatentVideo` or similar depending on ComfyUI version. Fix: query `/object_info/{node_type}` to see available nodes.
- **Missing `ModelSamplingSD3`**: some ComfyUI workflows for the 1.3B skip this node entirely. Fix: try removing it and connecting UNETLoader directly to KSampler.

### Success Criteria

- [ ] Manual: 5+ clips generated successfully end-to-end
- [ ] Manual: Document actual per-clip cost and generation time in task notes or research doc

---

## Final Checklist

- [ ] All Phase 1 automated tests pass
- [ ] Phase 2 manual verification complete
- [ ] Pod cleanup confirmed working on normal exit, Ctrl+C, and error
- [ ] Stitch pipeline works with runpod-wan clips
- [ ] Actual costs documented

## References

- Task: `docs/tasks/0001-runpod-wan-strategy/task.md`
- Existing strategies: `decode.py:65-214`
- RunPod SDK: `runpod.create_pod()`, `runpod.get_pod()`, `runpod.terminate_pod()`
- ComfyUI API: POST `/prompt`, GET `/history/{id}`, GET `/view`
- Docker image: `ghcr.io/lum3on/wan22-runpod:latest` (ComfyUI on port 8188)
