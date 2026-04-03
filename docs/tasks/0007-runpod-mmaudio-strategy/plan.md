---
id: plan-0007
type: spec
purpose: "Implementation plan for self-hosted MMAudio on RunPod, sharing the ComfyUI pod with RunPodWanStrategy."
tags: ["plan", "audio", "runpod", "mmaudio", "decode"]
related: ["./task.md"]
---

# RunPod MMAudio Strategy Implementation Plan

## Overview

Add `RunPodMMAudioStrategy` to generate audio via MMAudio V2 on the same RunPod ComfyUI pod used for video generation. Pod lifecycle management is extracted into a shared module so both video and audio strategies can manage the same pod across separate subprocess invocations.

**Primary Goal**: `pipeline.py --strategy runpod-wan --audio-strategy runpod-mmaudio` runs video + audio on one pod at ~$0 marginal audio cost.

**Approach**: Extract pod lifecycle into `runpod_pod.py`, add a pod state file for cross-process sharing, add `--keep-pod` to defer termination, build the new audio strategy using the same ComfyUI HTTP API pattern.

## Current State Analysis

- `RunPodWanStrategy` (`decode.py:262-779`) manages the full pod lifecycle inline: create pod, wait for SSH, download models, start ComfyUI, submit workflows, terminate on exit
- Pod cleanup uses atexit + signal handlers (`decode.py:293-314`) — always terminates on exit
- `AudioStrategy` base class (`decode.py:787-804`) defines `generate()` returning `list[AudioClipResult]`
- `MMAudioStrategy` (`decode.py:897-976`) uses fal.ai — we're adding a self-hosted alternative, not replacing this
- `pipeline.py` runs stages as subprocesses with no shared state — pod sharing requires a file-based handoff
- `decode.py` is 2040 lines; extracting ~300 lines of pod management reduces bloat

### Key Discoveries
- ComfyUI-MMAudio needs 4 model files totalling ~5.1 GB (not ~1 GB as the task spec estimated), plus a ~489 MB BigVGAN vocoder that auto-downloads on first run
- The ComfyUI workflow for text-to-audio uses 3 nodes (`MMAudioModelLoader`, `MMAudioFeatureUtilsLoader`, `MMAudioSampler`) + `SaveAudio` for file output
- `MMAudioSampler`'s `images` input is optional — omitting it gives pure text-to-audio
- MMAudio V2 large model needs ~6 GB VRAM, fits alongside Wan 2.1 1.3B sequentially on a 24 GB GPU

## Desired End State

```bash
# Full pipeline: video + audio on one pod
python pipeline.py media/film.mp4 -o output/film --strategy runpod-wan --audio-strategy runpod-mmaudio

# Manual: video with --keep-pod, then audio reuses pod
python decode.py output/film --strategy runpod-wan --keep-pod
python decode.py output/film --audio --audio-strategy runpod-mmaudio

# Audio-only (creates its own pod if no state file)
python decode.py output/film --audio --audio-strategy runpod-mmaudio
```

Pod state file at `output/film/runpod_pod.json` bridges the subprocess gap.

## What We're NOT Doing

- Concurrent video + audio on the same pod (sequential only — VRAM constraints)
- Replacing the fal.ai MMAudio strategy (it stays as a simpler option)
- Video-to-audio mode (feed clips instead of text prompts) — might-do, deferred
- Custom model training or fine-tuning

---

## Phase 1: Extract Pod Lifecycle into Shared Module

### Overview
Extract pod creation, SSH, readiness checks, state persistence, and termination from `RunPodWanStrategy` into `runpod_pod.py`. Refactor `RunPodWanStrategy` to use it. Add `--keep-pod` flag and pod state file.

### Tasks

#### 1. Create `runpod_pod.py` with `RunPodSession` class

- [x] Create `runpod_pod.py`

```python
"""Shared RunPod pod lifecycle management for ComfyUI strategies."""

import atexit
import json
import os
import signal
import subprocess
import sys
import time


# Default GPU preferences (cheapest first)
GPU_TYPES = [
    ("NVIDIA GeForce RTX 4090", 0.34),
    ("NVIDIA RTX 4000 Ada Generation", 0.34),
    ("NVIDIA RTX A6000", 0.52),
    ("NVIDIA L40S", 0.54),
]

DOCKER_IMAGE = "runpod/comfyui:latest"
CONTAINER_DISK_GB = 50
COMFYUI_PORT = 8188
POD_READY_TIMEOUT = 600
COMFYUI_READY_TIMEOUT = 600
COMFYUI_DIR = "/workspace/runpod-slim/ComfyUI"


class RunPodSession:
    """Manages a RunPod ComfyUI pod with state file persistence.

    Can create a new pod or reconnect to an existing one via a state file.
    Handles cleanup on exit (terminate or keep-alive depending on flags).
    """

    def __init__(self, output_dir: str, keep_pod: bool = False):
        self.output_dir = output_dir
        self.keep_pod = keep_pod
        self.pod_id: str | None = None
        self.base_url: str | None = None
        self.ssh_host: str | None = None
        self.ssh_port: int | None = None
        self.gpu_hourly_rate: float = GPU_TYPES[0][1]
        self.pod_start_time: float | None = None
        self._clean_exit = False
        self._cleanup_registered = False

    def _register_cleanup(self):
        """Register atexit and signal handlers to terminate pod on exit."""
        if self._cleanup_registered:
            return
        self._cleanup_registered = True

        atexit.register(self._cleanup)

        original_sigint = signal.getsignal(signal.SIGINT)
        original_sigterm = signal.getsignal(signal.SIGTERM)

        def _handler(signum, frame):
            self._cleanup()
            if signum == signal.SIGINT and callable(original_sigint):
                original_sigint(signum, frame)
            elif signum == signal.SIGTERM and callable(original_sigterm):
                original_sigterm(signum, frame)
            else:
                sys.exit(1)

        signal.signal(signal.SIGINT, _handler)
        signal.signal(signal.SIGTERM, _handler)

    def _cleanup(self):
        """Terminate or preserve pod depending on keep_pod flag and exit status."""
        if self.pod_id is None:
            return
        if self.keep_pod and self._clean_exit:
            # Normal exit with --keep-pod: write state file, don't terminate
            self.write_state()
            print(f"  Pod {self.pod_id} kept alive (state written to {self.state_path})")
        else:
            self.terminate()

    @property
    def state_path(self) -> str:
        return os.path.join(self.output_dir, "runpod_pod.json")

    def mark_clean_exit(self):
        """Call before normal return to signal that pod should be kept."""
        self._clean_exit = True

    # -- State file I/O --

    def write_state(self):
        """Write pod connection info to state file (atomic)."""
        state = {
            "pod_id": self.pod_id,
            "base_url": self.base_url,
            "ssh_host": self.ssh_host,
            "ssh_port": self.ssh_port,
            "gpu_hourly_rate": self.gpu_hourly_rate,
            "start_time": self.pod_start_time,
        }
        os.makedirs(self.output_dir, exist_ok=True)
        tmp_path = self.state_path + ".tmp"
        with open(tmp_path, "w") as f:
            json.dump(state, f, indent=2)
        os.replace(tmp_path, self.state_path)

    def read_state(self) -> bool:
        """Read pod state file. Returns True if state was loaded."""
        if not os.path.exists(self.state_path):
            return False
        with open(self.state_path) as f:
            state = json.load(f)
        self.pod_id = state["pod_id"]
        self.base_url = state["base_url"]
        self.ssh_host = state.get("ssh_host")
        self.ssh_port = state.get("ssh_port")
        self.gpu_hourly_rate = state.get("gpu_hourly_rate", GPU_TYPES[0][1])
        self.pod_start_time = state.get("start_time")
        return True

    def remove_state(self):
        """Remove the pod state file."""
        if os.path.exists(self.state_path):
            os.remove(self.state_path)

    # -- Pod lifecycle --

    def create_pod(self):
        """Create a new RunPod pod and wait for it to be ready."""
        import runpod

        runpod.api_key = os.environ.get("RUNPOD_API_KEY")
        if not runpod.api_key:
            print("Error: RUNPOD_API_KEY not set in .env or environment.")
            sys.exit(1)

        pod = None
        gpu_rate = GPU_TYPES[0][1]
        for gpu_type, rate in GPU_TYPES:
            print(f"  Trying {gpu_type}...")
            try:
                ssh_pubkey = ""
                pubkey_path = os.path.expanduser("~/.ssh/id_ed25519.pub")
                if not os.path.exists(pubkey_path):
                    pubkey_path = os.path.expanduser("~/.ssh/id_rsa.pub")
                if os.path.exists(pubkey_path):
                    with open(pubkey_path) as f:
                        ssh_pubkey = f.read().strip()

                pod = runpod.create_pod(
                    name="lossy-comfyui",
                    image_name=DOCKER_IMAGE,
                    gpu_type_id=gpu_type,
                    cloud_type="COMMUNITY",
                    gpu_count=1,
                    container_disk_in_gb=CONTAINER_DISK_GB,
                    ports=f"{COMFYUI_PORT}/http,22/tcp",
                    support_public_ip=True,
                    start_ssh=True,
                    env={"PUBLIC_KEY": ssh_pubkey} if ssh_pubkey else None,
                )
                gpu_rate = rate
                print(f"  Got {gpu_type} @ ${rate}/hr")
                break
            except Exception as e:
                print(f"  {gpu_type} unavailable: {e}")
                continue

        if pod is None:
            print("Error: No GPU available. Try again later.")
            sys.exit(1)

        self.pod_id = pod["id"]
        self.gpu_hourly_rate = gpu_rate
        self.pod_start_time = time.time()
        self._register_cleanup()
        print(f"  Pod created: {self.pod_id}")

        # Wait for runtime
        print("  Waiting for pod to start...")
        start = time.time()
        while time.time() - start < POD_READY_TIMEOUT:
            status = runpod.get_pod(self.pod_id)
            runtime = status.get("runtime")
            if runtime is not None and runtime.get("ports"):
                break
            time.sleep(5)
        else:
            print(f"  Error: Pod did not start within {POD_READY_TIMEOUT}s")
            self.terminate()
            sys.exit(1)

        self.base_url = f"https://{self.pod_id}-{COMFYUI_PORT}.proxy.runpod.net"
        print(f"  Pod ready: {self.base_url}")

        # Get SSH info
        pod_info = runpod.get_pod(self.pod_id)
        for port_info in pod_info.get("runtime", {}).get("ports", []):
            if port_info["privatePort"] == 22 and port_info["isIpPublic"]:
                self.ssh_host = port_info["ip"]
                self.ssh_port = port_info["publicPort"]
                break

    def reconnect(self) -> bool:
        """Reconnect to an existing pod via state file. Returns True if successful."""
        if not self.read_state():
            return False

        self._register_cleanup()
        print(f"  Reconnecting to pod {self.pod_id}...")

        # Verify pod is still running
        import runpod
        runpod.api_key = os.environ.get("RUNPOD_API_KEY")
        if not runpod.api_key:
            print("Error: RUNPOD_API_KEY not set in .env or environment.")
            sys.exit(1)

        try:
            status = runpod.get_pod(self.pod_id)
            runtime = status.get("runtime")
            if runtime is None or not runtime.get("ports"):
                print(f"  Pod {self.pod_id} is no longer running.")
                self.remove_state()
                return False
        except Exception as e:
            print(f"  Failed to check pod {self.pod_id}: {e}")
            self.remove_state()
            return False

        print(f"  Reconnected: {self.base_url}")
        return True

    def ensure_pod(self):
        """Reconnect to existing pod or create a new one."""
        if self.pod_id is not None:
            return
        if not self.reconnect():
            self.create_pod()

    def terminate(self):
        """Terminate the pod and remove state file."""
        if self.pod_id is None:
            return

        import runpod

        pod_id = self.pod_id
        self.pod_id = None  # Prevent double-terminate

        elapsed_h = (time.time() - self.pod_start_time) / 3600 if self.pod_start_time else 0
        estimated_cost = elapsed_h * self.gpu_hourly_rate

        try:
            runpod.api_key = os.environ.get("RUNPOD_API_KEY")
            runpod.terminate_pod(pod_id)
            print(f"  Pod {pod_id} terminated. Session cost: ~${estimated_cost:.2f} ({elapsed_h:.1f}hrs)")
        except Exception as e:
            print(f"  Warning: Failed to terminate pod {pod_id}: {e}")
            print(f"  Manually terminate at https://www.runpod.io/console/pods")

        self.remove_state()

    # -- SSH helpers --

    def ssh_cmd(self, cmd: str, timeout: int = 60) -> subprocess.CompletedProcess:
        """Run a command on the pod via SSH."""
        return subprocess.run(
            [
                "ssh",
                "-o", "StrictHostKeyChecking=no",
                "-o", "UserKnownHostsFile=/dev/null",
                "-o", "LogLevel=ERROR",
                "-o", "ServerAliveInterval=30",
                "-p", str(self.ssh_port),
                f"root@{self.ssh_host}",
                cmd,
            ],
            capture_output=True,
            text=True,
            timeout=timeout,
        )

    def ssh_bg(self, cmd: str):
        """Run a command on the pod via SSH in the background (detached)."""
        p = subprocess.Popen(
            [
                "ssh", "-f",
                "-o", "StrictHostKeyChecking=no",
                "-o", "UserKnownHostsFile=/dev/null",
                "-o", "LogLevel=ERROR",
                "-p", str(self.ssh_port),
                f"root@{self.ssh_host}",
                cmd,
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        try:
            p.communicate(timeout=10)
        except subprocess.TimeoutExpired:
            p.kill()

    # -- ComfyUI helpers --

    def wait_for_comfyui(self):
        """Poll ComfyUI until it responds to /system_stats."""
        import httpx

        print("  Waiting for ComfyUI...")
        start = time.time()
        while time.time() - start < COMFYUI_READY_TIMEOUT:
            try:
                resp = httpx.get(f"{self.base_url}/system_stats", timeout=10)
                if resp.status_code == 200:
                    print("  ComfyUI is ready.")
                    return
            except (httpx.ConnectError, httpx.TimeoutException, httpx.ReadError):
                pass
            elapsed = int(time.time() - start)
            print(f"  Waiting for ComfyUI... ({elapsed}s)", end="\r")
            time.sleep(5)

        print(f"\n  Error: ComfyUI did not become ready within {COMFYUI_READY_TIMEOUT}s")
        if self.ssh_host:
            try:
                result = self.ssh_cmd("tail -50 /tmp/comfyui.log", timeout=10)
                if result.stdout:
                    print(f"  ComfyUI log:\n{result.stdout}")
            except Exception:
                pass
        self.terminate()
        sys.exit(1)

    def restart_comfyui(self):
        """Stop ComfyUI, then start it again. Used after installing custom nodes."""
        print("  Restarting ComfyUI...")
        self.ssh_cmd('pkill -f "python main.py" || true', timeout=10)
        time.sleep(2)

        find_python = self.ssh_cmd(
            f"if [ -x {COMFYUI_DIR}/.venv/bin/python ]; then echo .venv/bin/python; "
            f"elif command -v python3 >/dev/null; then echo python3; "
            f"else echo python; fi",
            timeout=10,
        )
        python_bin = find_python.stdout.strip() if find_python.returncode == 0 else "python3"
        self.ssh_bg(
            f"cd {COMFYUI_DIR} && {python_bin} main.py --listen 0.0.0.0 --port 8188 "
            f"</dev/null >/tmp/comfyui.log 2>&1"
        )
        self.wait_for_comfyui()

    def download_models(self, models: list[tuple[str, str]]):
        """Download model files to ComfyUI models dir via SSH.

        Args:
            models: List of (relative_dest_path, url) tuples.
                    e.g. ("vae/wan_2.1_vae.safetensors", "https://...")
        """
        models_dir = f"{COMFYUI_DIR}/models"
        print(f"  Downloading models via SSH ({self.ssh_host}:{self.ssh_port})...")

        for dest_path, url in models:
            full_path = f"{models_dir}/{dest_path}"
            filename = os.path.basename(dest_path)

            check = self.ssh_cmd(f'[ -s {full_path} ] && echo exists || echo missing')
            if check.returncode == 0 and "exists" in check.stdout:
                print(f"    {filename}: already exists")
                continue

            print(f"    {filename}: downloading...")
            dir_path = os.path.dirname(full_path)
            result = self.ssh_cmd(
                f"mkdir -p {dir_path} && wget -q -O {full_path} '{url}' && echo OK",
                timeout=600,
            )
            if result.returncode == 0 and "OK" in result.stdout:
                print(f"    {filename}: done")
            else:
                print(f"    {filename}: FAILED (exit {result.returncode})")
                if result.stderr:
                    print(f"      {result.stderr[:200]}")

    def submit_workflow(self, workflow: dict, timeout: int = 300) -> dict | None:
        """Submit a ComfyUI workflow and wait for completion.

        Returns the history entry for the prompt, or None on failure.
        """
        import httpx

        resp = httpx.post(
            f"{self.base_url}/prompt",
            json={"prompt": workflow},
            timeout=30,
        )
        if resp.status_code != 200:
            error_detail = resp.text[:300] if resp.text else "no body"
            print(f"  ComfyUI rejected workflow ({resp.status_code}): {error_detail}")
            return None

        prompt_id = resp.json()["prompt_id"]

        start = time.time()
        while time.time() - start < timeout:
            resp = httpx.get(
                f"{self.base_url}/history/{prompt_id}",
                timeout=10,
            )
            history = resp.json()
            if prompt_id in history:
                status = history[prompt_id].get("status", {})
                if status.get("status_str") != "success":
                    print(f"  ComfyUI error: {status}")
                    return None
                return history[prompt_id]
            time.sleep(2)

        print(f"  Timeout waiting for workflow ({timeout}s)")
        return None

    def download_output(self, output_file: dict) -> bytes | None:
        """Download an output file from ComfyUI's /view endpoint."""
        import httpx

        try:
            resp = httpx.get(
                f"{self.base_url}/view",
                params={
                    "filename": output_file["filename"],
                    "subfolder": output_file.get("subfolder", ""),
                    "type": output_file.get("type", "output"),
                },
                timeout=60,
                follow_redirects=True,
            )
            resp.raise_for_status()
            return resp.content
        except Exception as e:
            print(f"  Error downloading output: {e}")
            return None

    def free_vram(self):
        """Free cached VRAM from last generation (keep models loaded)."""
        import httpx
        try:
            httpx.post(f"{self.base_url}/free", json={"free_memory": True}, timeout=10)
        except Exception:
            pass
```

#### 2. Refactor `RunPodWanStrategy` to use `RunPodSession`

- [x] Replace inline pod management in `RunPodWanStrategy` with `RunPodSession`

The refactored class keeps its Wan-specific logic (model list, workflow builder, format_prompt) but delegates pod lifecycle to the shared session. The `--keep-pod` flag is passed through from CLI args.

```python
class RunPodWanStrategy(GenerationStrategy):
    """RunPod self-hosted Wan 2.2 (1.3B fp8) via ComfyUI -- ~$0.004/clip."""

    name = "runpod-wan"
    CLIP_DURATION = 81 / 16  # ~5.0625s

    WAN_MODELS = [
        (
            "vae/wan_2.1_vae.safetensors",
            "https://huggingface.co/Comfy-Org/Wan_2.1_ComfyUI_repackaged/resolve/main/split_files/vae/wan_2.1_vae.safetensors",
        ),
        (
            "text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors",
            "https://huggingface.co/Comfy-Org/Wan_2.1_ComfyUI_repackaged/resolve/main/split_files/text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors",
        ),
        (
            "diffusion_models/wan2.1_t2v_1.3B_fp16.safetensors",
            "https://huggingface.co/Comfy-Org/Wan_2.1_ComfyUI_repackaged/resolve/main/split_files/diffusion_models/wan2.1_t2v_1.3B_fp16.safetensors",
        ),
    ]

    def __init__(self, output_dir: str = "", keep_pod: bool = False):
        from runpod_pod import RunPodSession
        self._session = RunPodSession(output_dir, keep_pod=keep_pod)
        self._setup_done = False

    def format_prompt(self, entry: dict) -> str:
        return _format_prompt_wan(entry)

    def _ensure_pod(self):
        if self._setup_done:
            return
        self._session.ensure_pod()
        if self._session.ssh_host:
            self._session.ssh_cmd('pkill -f "python main.py" || true', timeout=10)
            time.sleep(2)
            print("  Waiting for SSH...")
            time.sleep(10)
            self._session.download_models(self.WAN_MODELS)
            self._session.restart_comfyui()
        else:
            print("  Warning: No SSH access. Waiting for ComfyUI without model setup.")
            self._session.wait_for_comfyui()
        self._setup_done = True

    def _build_workflow(self, prompt: str, seed: int) -> dict:
        # ... unchanged workflow dict (same as current decode.py:574-663) ...
        pass  # full implementation preserved from existing code

    def generate(self, prompt, clips_dir, shot_index, target_duration_s, seed=None):
        self._ensure_pod()
        clip_path = os.path.join(clips_dir, f"{shot_index:04d}.mp4")
        effective_seed = seed if seed is not None else shot_index

        workflow = self._build_workflow(prompt, effective_seed)
        history = self._session.submit_workflow(workflow, timeout=300)
        if not history:
            return []

        # Find output file
        outputs = history.get("outputs", {})
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

        data = self._session.download_output(output_file)
        if not data:
            return []

        # Save raw file, convert to MP4 via ffmpeg
        raw_ext = os.path.splitext(output_file["filename"])[1] or ".webm"
        raw_path = clip_path.replace(".mp4", raw_ext)
        with open(raw_path, "wb") as f:
            f.write(data)

        result = subprocess.run(
            ["ffmpeg", "-y", "-i", raw_path,
             "-c:v", "libx264", "-pix_fmt", "yuv420p",
             clip_path],
            capture_output=True,
        )
        if result.returncode == 0 and os.path.exists(clip_path) and os.path.getsize(clip_path) > 0:
            os.remove(raw_path)
        else:
            if os.path.exists(clip_path):
                os.remove(clip_path)
            os.rename(raw_path, clip_path)

        # Cost based on wall-clock time
        elapsed_h = (time.time() - self._session.pod_start_time) / 3600 if self._session.pod_start_time else 0
        clips_so_far = len([f for f in os.listdir(clips_dir) if f.endswith(".mp4")])
        per_clip_cost = (elapsed_h * self._session.gpu_hourly_rate) / max(clips_so_far, 1)

        self._session.free_vram()

        return [ClipResult(path=clip_path, actual_duration_s=self.CLIP_DURATION, cost=per_clip_cost)]

    def mark_clean_exit(self):
        """Called by run_decode after successful completion."""
        self._session.mark_clean_exit()
```

#### 3. Add `--keep-pod` CLI flag and wire into `run_decode`

- [x] Add `--keep-pod` argument to decode.py's argparser

```python
parser.add_argument("--keep-pod", action="store_true",
                    help="Keep RunPod pod alive after decode (for subsequent audio stage)")
```

- [x] Pass `output_dir` and `keep_pod` when constructing `RunPodWanStrategy` in `main()`

```python
# In main(), video strategy construction:
strategies = {
    "replicate-wan": lambda: ReplicateWanStrategy(),
    "fal-seedance": lambda: FalSeedanceStrategy(),
    "fal-seedance-pro": lambda: FalSeedanceProStrategy(),
    "runpod-wan": lambda: RunPodWanStrategy(
        output_dir=args.output_dir,
        keep_pod=getattr(args, "keep_pod", False),
    ),
}
strategy = strategies[args.strategy]()
run_decode(args, strategy)
```

- [x] Call `strategy.mark_clean_exit()` at the end of `run_decode()` (before return, after successful completion)

Add at the end of `run_decode()`, after the final progress save:

```python
# Signal clean exit for --keep-pod support
if hasattr(strategy, 'mark_clean_exit'):
    strategy.mark_clean_exit()
```

### Success Criteria

#### Automated Verification:
- [x] Run: `python -m pytest -v` — all existing tests still pass (refactor is behavior-preserving)

#### Manual Verification:
- [x] Manual: `python decode.py output/test --strategy runpod-wan --keep-pod --limit 1` creates pod, generates 1 clip, and leaves pod running with `runpod_pod.json` written
- [x] Manual: Verify the pod is still running on RunPod console after the command exits
- [x] Manual: Terminate the test pod manually

---

## Phase 2: Build RunPodMMAudioStrategy

### Overview
New audio strategy that installs ComfyUI-MMAudio custom nodes on the pod, downloads models, and generates audio via ComfyUI workflows. Reuses the pod from video stage via state file.

### Tasks

#### 1. Add `RunPodMMAudioStrategy` to `decode.py`

- [x] Add `RunPodMMAudioStrategy` class to `decode.py` (after `MMAudioStrategy`)

```python
class RunPodMMAudioStrategy(AudioStrategy):
    """MMAudio V2 text-to-audio via self-hosted ComfyUI on RunPod -- ~$0/marginal."""

    name = "runpod-mmaudio"
    MAX_DURATION = 30
    MIN_DURATION = 1
    GENERATION_TIMEOUT = 120  # 2 min per clip (MMAudio is fast)

    MMAUDIO_MODELS = [
        (
            "mmaudio/mmaudio_large_44k_v2_fp16.safetensors",
            "https://huggingface.co/Kijai/MMAudio_safetensors/resolve/main/mmaudio_large_44k_v2_fp16.safetensors",
        ),
        (
            "mmaudio/mmaudio_vae_44k_fp16.safetensors",
            "https://huggingface.co/Kijai/MMAudio_safetensors/resolve/main/mmaudio_vae_44k_fp16.safetensors",
        ),
        (
            "mmaudio/mmaudio_synchformer_fp16.safetensors",
            "https://huggingface.co/Kijai/MMAudio_safetensors/resolve/main/mmaudio_synchformer_fp16.safetensors",
        ),
        (
            "mmaudio/apple_DFN5B-CLIP-ViT-H-14-384_fp16.safetensors",
            "https://huggingface.co/Kijai/MMAudio_safetensors/resolve/main/apple_DFN5B-CLIP-ViT-H-14-384_fp16.safetensors",
        ),
    ]

    def __init__(self, output_dir: str = ""):
        from runpod_pod import RunPodSession
        self._session = RunPodSession(output_dir)
        self._setup_done = False

    def _target_durations(self, target_s: float) -> list[float]:
        """Split target duration into chunks within 1-30s range."""
        clamped = max(self.MIN_DURATION, target_s)
        if clamped <= self.MAX_DURATION:
            return [clamped]

        parts = []
        remaining = clamped
        while remaining > self.MAX_DURATION:
            parts.append(float(self.MAX_DURATION))
            remaining -= self.MAX_DURATION
        remainder = max(self.MIN_DURATION, remaining)
        parts.append(remainder)
        return parts

    def _ensure_pod(self):
        """Connect to existing pod or create new one, install MMAudio."""
        if self._setup_done:
            return
        self._session.ensure_pod()
        self._install_mmaudio()
        self._setup_done = True

    def _install_mmaudio(self):
        """Install ComfyUI-MMAudio custom nodes and download models."""
        from runpod_pod import COMFYUI_DIR

        if not self._session.ssh_host:
            print("  Warning: No SSH access. Assuming MMAudio is pre-installed.")
            self._session.wait_for_comfyui()
            return

        custom_nodes_dir = f"{COMFYUI_DIR}/custom_nodes"

        # Check if already installed
        check = self._session.ssh_cmd(
            f'[ -d {custom_nodes_dir}/ComfyUI-MMAudio ] && echo exists || echo missing'
        )
        if check.returncode == 0 and "exists" in check.stdout:
            print("  ComfyUI-MMAudio: already installed")
        else:
            # Stop ComfyUI for installation
            self._session.ssh_cmd('pkill -f "python main.py" || true', timeout=10)
            time.sleep(2)

            print("  Installing ComfyUI-MMAudio custom nodes...")
            result = self._session.ssh_cmd(
                f"cd {custom_nodes_dir} && "
                f"git clone https://github.com/kijai/ComfyUI-MMAudio && "
                f"pip install -r ComfyUI-MMAudio/requirements.txt && echo OK",
                timeout=300,
            )
            if result.returncode != 0 or "OK" not in result.stdout:
                print(f"  Error installing MMAudio nodes: {result.stderr[:300]}")
                self._session.terminate()
                sys.exit(1)
            print("  ComfyUI-MMAudio: installed")

        # Download models
        self._session.download_models(self.MMAUDIO_MODELS)

        # Restart ComfyUI so it picks up the new custom nodes
        self._session.restart_comfyui()

    def _build_workflow(self, prompt: str, duration: float, seed: int) -> dict:
        """Build ComfyUI API-format workflow for MMAudio text-to-audio."""
        return {
            "1": {
                "class_type": "MMAudioModelLoader",
                "inputs": {
                    "mmaudio_model": "mmaudio_large_44k_v2_fp16.safetensors",
                    "base_precision": "fp16",
                },
            },
            "2": {
                "class_type": "MMAudioFeatureUtilsLoader",
                "inputs": {
                    "vae_model": "mmaudio_vae_44k_fp16.safetensors",
                    "synchformer_model": "mmaudio_synchformer_fp16.safetensors",
                    "clip_model": "apple_DFN5B-CLIP-ViT-H-14-384_fp16.safetensors",
                    "mode": "44k",
                    "precision": "fp16",
                },
            },
            "3": {
                "class_type": "MMAudioSampler",
                "inputs": {
                    "mmaudio_model": ["1", 0],
                    "feature_utils": ["2", 0],
                    "duration": duration,
                    "steps": 25,
                    "cfg": 4.5,
                    "seed": seed,
                    "prompt": prompt,
                    "negative_prompt": "",
                    "mask_away_clip": False,
                    "force_offload": True,
                },
            },
            "4": {
                "class_type": "SaveAudio",
                "inputs": {
                    "filename_prefix": "lossy_audio",
                    "audio": ["3", 0],
                },
            },
        }

    def generate(
        self,
        sound_description: str,
        audio_dir: str,
        shot_index: int,
        target_duration_s: float,
        seed: int | None = None,
    ) -> list[AudioClipResult]:
        self._ensure_pod()

        durations = self._target_durations(target_duration_s)
        results = []

        for part_idx, duration in enumerate(durations):
            if len(durations) == 1:
                clip_name = f"{shot_index:04d}.flac"
            else:
                clip_name = f"{shot_index:04d}-{part_idx + 1:02d}.flac"

            clip_path = os.path.join(audio_dir, clip_name)
            effective_seed = (seed if seed is not None else shot_index) + part_idx

            try:
                workflow = self._build_workflow(sound_description, duration, effective_seed)
                history = self._session.submit_workflow(workflow, timeout=self.GENERATION_TIMEOUT)
                if not history:
                    print(f"  Failed to generate audio for shot {shot_index}")
                    return []

                # Find audio output
                outputs = history.get("outputs", {})
                output_file = None
                for node_id, node_output in outputs.items():
                    if "audio" in node_output:
                        for item in node_output["audio"]:
                            output_file = item
                            break
                        if output_file:
                            break

                if not output_file:
                    print(f"  No audio output found for shot {shot_index}")
                    return []

                data = self._session.download_output(output_file)
                if not data:
                    return []

                with open(clip_path, "wb") as f:
                    f.write(data)

                # Cost is ~$0 marginal (pod time already paid by video stage)
                results.append(AudioClipResult(
                    path=clip_path,
                    actual_duration_s=duration,
                    cost=0.0,
                ))

                self._session.free_vram()

            except Exception as e:
                print(f"  Error generating audio {clip_name}: {e}")
                return []

        return results
```

### Success Criteria

#### Automated Verification:
- [x] Run: `python -m pytest -v` — all tests pass

---

## Phase 3: CLI + Pipeline Wiring

### Overview
Wire `runpod-mmaudio` into decode.py's CLI choices, pipeline.py's `--keep-pod` logic, and pod termination after audio stage.

### Tasks

#### 1. Update decode.py CLI

- [x] Add `runpod-mmaudio` to the `--audio-strategy` choices

```python
parser.add_argument("--audio-strategy", choices=["elevenlabs", "mmaudio", "runpod-mmaudio"],
                    default="elevenlabs",
                    help="Audio generation backend (default: elevenlabs)")
```

- [x] Add `RunPodMMAudioStrategy` to the audio strategy dispatch in `main()`

```python
audio_strategies = {
    "elevenlabs": lambda: ElevenLabsStrategy(),
    "mmaudio": lambda: MMAudioStrategy(),
    "runpod-mmaudio": lambda: RunPodMMAudioStrategy(output_dir=args.output_dir),
}
audio_strategy = audio_strategies[args.audio_strategy]()
run_audio(args, audio_strategy)
```

#### 2. Update pipeline.py

- [x] Add `runpod-mmaudio` to `AUDIO_STRATEGIES`

```python
AUDIO_STRATEGIES = ["elevenlabs", "mmaudio", "runpod-mmaudio"]
```

- [x] Add `--keep-pod` to the decode command when using runpod-wan + runpod-mmaudio

In `build_commands()`, after building the decode command:

```python
commands["decode"] = [
    sys.executable, "decode.py", args.output,
    "--strategy", args.strategy,
]
# Keep pod alive when audio stage will reuse it
if args.strategy == "runpod-wan" and args.audio_strategy == "runpod-mmaudio":
    commands["decode"].append("--keep-pod")
if args.start_index is not None:
    commands["decode"] += ["--start-index", str(args.start_index)]
if args.limit:
    commands["decode"] += ["--limit", str(args.limit)]
```

- [x] Add pod cleanup to pipeline.py after audio stage completes or fails

In `run_pipeline()`, after the stage loop completes (whether by finishing all stages or hitting a failure), add pod cleanup:

```python
# Clean up RunPod pod state file (pod is terminated by the last stage that uses it)
pod_state = os.path.join(args.output, "runpod_pod.json")
if os.path.exists(pod_state):
    # Pod wasn't cleaned up — terminate it as safety net
    try:
        import json as _json
        with open(pod_state) as f:
            state = _json.load(f)
        import runpod
        runpod.api_key = os.environ.get("RUNPOD_API_KEY")
        if runpod.api_key and state.get("pod_id"):
            runpod.terminate_pod(state["pod_id"])
            print(f"  Safety net: terminated pod {state['pod_id']}")
        os.remove(pod_state)
    except Exception as e:
        print(f"  Warning: Could not clean up pod: {e}")
        print(f"  Check https://www.runpod.io/console/pods")
```

### Success Criteria

#### Automated Verification:
- [x] Run: `python -m pytest -v` — all tests pass

#### Manual Verification:
- [x] Manual: `python pipeline.py media/test.mp4 -o /tmp/test --strategy runpod-wan --audio-strategy runpod-mmaudio --dry-run` shows `--keep-pod` on decode command and `runpod-mmaudio` on audio command

---

## Phase 4: Tests

### Overview
Add unit tests for the shared module and new strategy. Tests use mocks — no real API calls.

### Tasks

#### 1. Add tests for `RunPodSession` state file I/O

- [x] Create `test_runpod_pod.py`

```python
"""Tests for runpod_pod.py shared pod lifecycle module."""

import json
import os
import pytest
from runpod_pod import RunPodSession


class TestRunPodSessionState:
    """Test pod state file read/write."""

    def test_write_and_read_state(self, tmp_path):
        session = RunPodSession(str(tmp_path))
        session.pod_id = "pod-abc123"
        session.base_url = "https://pod-abc123-8188.proxy.runpod.net"
        session.ssh_host = "1.2.3.4"
        session.ssh_port = 12345
        session.gpu_hourly_rate = 0.34
        session.pod_start_time = 1000000.0
        session.write_state()

        # Verify file exists and is valid JSON
        state_path = os.path.join(str(tmp_path), "runpod_pod.json")
        assert os.path.exists(state_path)
        with open(state_path) as f:
            state = json.load(f)
        assert state["pod_id"] == "pod-abc123"
        assert state["base_url"] == "https://pod-abc123-8188.proxy.runpod.net"
        assert state["ssh_host"] == "1.2.3.4"
        assert state["ssh_port"] == 12345

        # Read back into a new session
        session2 = RunPodSession(str(tmp_path))
        assert session2.read_state() is True
        assert session2.pod_id == "pod-abc123"
        assert session2.base_url == "https://pod-abc123-8188.proxy.runpod.net"
        assert session2.ssh_host == "1.2.3.4"
        assert session2.ssh_port == 12345

    def test_read_state_missing(self, tmp_path):
        session = RunPodSession(str(tmp_path))
        assert session.read_state() is False
        assert session.pod_id is None

    def test_remove_state(self, tmp_path):
        session = RunPodSession(str(tmp_path))
        session.pod_id = "pod-xyz"
        session.base_url = "https://pod-xyz-8188.proxy.runpod.net"
        session.write_state()
        assert os.path.exists(session.state_path)

        session.remove_state()
        assert not os.path.exists(session.state_path)

    def test_remove_state_missing(self, tmp_path):
        """remove_state is a no-op when no state file exists."""
        session = RunPodSession(str(tmp_path))
        session.remove_state()  # should not raise

    def test_state_path(self, tmp_path):
        session = RunPodSession(str(tmp_path))
        assert session.state_path == os.path.join(str(tmp_path), "runpod_pod.json")


class TestRunPodSessionCleanup:
    """Test cleanup logic (keep-pod vs terminate)."""

    def test_keep_pod_clean_exit_writes_state(self, tmp_path):
        session = RunPodSession(str(tmp_path), keep_pod=True)
        session.pod_id = "pod-keep"
        session.base_url = "https://pod-keep-8188.proxy.runpod.net"
        session.pod_start_time = 1000000.0
        session._clean_exit = True

        # _cleanup should write state, not terminate
        session._cleanup()
        assert os.path.exists(session.state_path)
        assert session.pod_id == "pod-keep"  # not cleared

    def test_no_keep_pod_calls_terminate(self, tmp_path, monkeypatch):
        session = RunPodSession(str(tmp_path), keep_pod=False)
        session.pod_id = "pod-term"
        session.base_url = "https://pod-term-8188.proxy.runpod.net"
        session.pod_start_time = 1000000.0

        terminated = []
        def mock_terminate(self_inner):
            terminated.append(self_inner.pod_id)
            self_inner.pod_id = None
        monkeypatch.setattr(RunPodSession, "terminate", mock_terminate)

        session._cleanup()
        assert terminated == ["pod-term"]

    def test_keep_pod_dirty_exit_terminates(self, tmp_path, monkeypatch):
        session = RunPodSession(str(tmp_path), keep_pod=True)
        session.pod_id = "pod-crash"
        session.base_url = "https://pod-crash-8188.proxy.runpod.net"
        session.pod_start_time = 1000000.0
        # _clean_exit is False by default (crash/error case)

        terminated = []
        def mock_terminate(self_inner):
            terminated.append(self_inner.pod_id)
            self_inner.pod_id = None
        monkeypatch.setattr(RunPodSession, "terminate", mock_terminate)

        session._cleanup()
        assert terminated == ["pod-crash"]
```

#### 2. Add tests for `RunPodMMAudioStrategy`

- [x] Add `TestRunPodMMAudioTargetDurations` to `test_decode.py`

```python
class TestRunPodMMAudioTargetDurations:
    def setup_method(self):
        # Bypass __init__'s RunPodSession import
        self.strategy = RunPodMMAudioStrategy.__new__(RunPodMMAudioStrategy)

    def test_short_clip(self):
        assert self.strategy._target_durations(5.0) == [5.0]

    def test_at_max(self):
        assert self.strategy._target_durations(30.0) == [30.0]

    def test_over_max(self):
        assert self.strategy._target_durations(35.0) == [30.0, 5.0]

    def test_much_over_max(self):
        assert self.strategy._target_durations(65.0) == [30.0, 30.0, 5.0]

    def test_very_short(self):
        assert self.strategy._target_durations(0.3) == [1.0]

    def test_zero(self):
        assert self.strategy._target_durations(0.0) == [1.0]
```

#### 3. Add pipeline tests

- [x] Add pipeline tests for `--keep-pod` and `runpod-mmaudio` to `test_pipeline.py`

```python
def test_build_commands_runpod_keep_pod(self, base_args):
    """Decode command gets --keep-pod when runpod-wan + runpod-mmaudio."""
    base_args.strategy = "runpod-wan"
    base_args.audio_strategy = "runpod-mmaudio"
    commands = build_commands(base_args)
    assert "--keep-pod" in commands["decode"]

def test_build_commands_no_keep_pod_without_runpod_audio(self, base_args):
    """No --keep-pod when audio strategy is not runpod-mmaudio."""
    base_args.strategy = "runpod-wan"
    base_args.audio_strategy = "elevenlabs"
    commands = build_commands(base_args)
    assert "--keep-pod" not in commands["decode"]

def test_build_commands_runpod_mmaudio_audio(self, base_args):
    """Audio command uses runpod-mmaudio."""
    base_args.audio_strategy = "runpod-mmaudio"
    commands = build_commands(base_args)
    assert "runpod-mmaudio" in commands["audio"]
```

### Success Criteria

#### Automated Verification:
- [x] Run: `python -m pytest -v` — all tests pass (existing + new)

---

## Final Checklist

- [x] All phases complete
- [x] All tests passing: `python -m pytest -v`
- [x] Dry-run pipeline works: `python pipeline.py media/test.mp4 -o /tmp/test --strategy runpod-wan --audio-strategy runpod-mmaudio --dry-run`

## Documentation Updates

- [x] Update `README.md` with `runpod-mmaudio` strategy usage and model size note (~5 GB download on first run)
- [x] Update `docs/work/roadmap.md` to reflect new strategy

## References

- Task: `docs/tasks/0007-runpod-mmaudio-strategy/task.md`
- ADR-002: `docs/design/adr/002-stateless-cli-pipeline.md`
- Task 0001: `docs/tasks/0001-runpod-wan-strategy/task.md`
- Task 0005: `docs/tasks/0005-audio-generation/task.md`
- [kijai/ComfyUI-MMAudio](https://github.com/kijai/ComfyUI-MMAudio)
- [Kijai/MMAudio_safetensors](https://huggingface.co/Kijai/MMAudio_safetensors)
