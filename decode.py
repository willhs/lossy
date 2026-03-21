#!/usr/bin/env python3
"""
lossy decoder: prompt manifest -> video clips -> reconstructed film.

Usage:
    python decode.py output/star_wars_iv_v2 [--start-index 7] [--limit 20]
    python decode.py output/star_wars_iv_v2 --stitch [--start-index 7]
    python decode.py output/star_wars_iv_v2 --strategy fal-seedance [--start-index 10]
"""

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass


def load_env():
    """Load .env file if present."""
    env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
    if os.path.exists(env_path):
        with open(env_path) as ef:
            for line in ef:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    os.environ.setdefault(k.strip(), v.strip())


# ---------------------------------------------------------------------------
# Strategy pattern for video generation backends
# ---------------------------------------------------------------------------


@dataclass
class ClipResult:
    """Result of generating a single video clip."""
    path: str
    actual_duration_s: float
    cost: float


@dataclass
class AudioClipResult:
    """Result of generating a single audio clip."""
    path: str
    actual_duration_s: float
    cost: float


@dataclass
class SpeechClipResult:
    """Result of generating a single speech clip."""
    path: str
    duration_s: float
    offset_s: float  # shot-relative offset for placement
    cost: float


CAMERA_TERMS = {
    "slow zoom out": "slow dolly out",
    "slow zoom in": "slow dolly in",
    "zoom out": "dolly out",
    "zoom in": "dolly in",
    "follows": "tracking shot follows",
    "moves left": "pan left",
    "moves right": "pan right",
    "moves up": "crane up",
    "moves down": "crane down",
    "shaky": "handheld",
    "smooth movement": "steadicam",
}


class GenerationStrategy:
    """Base class for video generation backends."""

    name: str = "base"

    def generate(
        self,
        prompt: str,
        clips_dir: str,
        shot_index: int,
        target_duration_s: float,
        seed: int | None = None,
    ) -> list[ClipResult]:
        """Generate clip(s) for a shot.

        Returns a list because long shots may be split into multiple clips.
        """
        raise NotImplementedError

    def format_prompt(self, entry: dict) -> str:
        """Format a structured prompt entry for this strategy's model.

        Subclasses override to produce model-optimized prompts.
        """
        return format_prompt(entry)


class ReplicateWanStrategy(GenerationStrategy):
    """Replicate Wan 2.2 Fast -- fixed ~5.06s clips at $0.05 each."""

    name = "replicate-wan"
    CLIP_DURATION = 81 / 16  # ~5.0625s

    def format_prompt(self, entry: dict) -> str:
        return _format_prompt_wan(entry)

    def generate(
        self,
        prompt: str,
        clips_dir: str,
        shot_index: int,
        target_duration_s: float,
        seed: int | None = None,
    ) -> list[ClipResult]:
        import httpx
        import replicate

        clip_path = os.path.join(clips_dir, f"{shot_index:04d}.mp4")

        try:
            input_params = {
                "prompt": prompt,
                "num_frames": 81,
                "aspect_ratio": "16:9",
                "frames_per_second": 16,
            }
            if seed is not None:
                input_params["seed"] = seed

            output = replicate.run("wan-video/wan-2.2-t2v-fast", input=input_params)

            if hasattr(output, "url"):
                url = output.url
            elif isinstance(output, str):
                url = output
            else:
                url = str(output)

            resp = httpx.get(url, follow_redirects=True)
            resp.raise_for_status()
            with open(clip_path, "wb") as f:
                f.write(resp.content)

            return [ClipResult(path=clip_path, actual_duration_s=self.CLIP_DURATION, cost=0.05)]

        except Exception as e:
            print(f"  Error generating clip: {e}")
            return []


class FalSeedanceStrategy(GenerationStrategy):
    """fal.ai Seedance 1.0 Pro Fast -- 2-12s duration control at ~$0.10/clip (480p)."""

    name = "fal-seedance"
    MODEL_ID = "fal-ai/bytedance/seedance/v1/pro/fast/text-to-video"
    MIN_DURATION = 2

    def format_prompt(self, entry: dict) -> str:
        return _format_prompt_seedance(entry)
    MAX_DURATION = 12
    COST_PER_SECOND_480P = 0.02  # Approximate: ~$0.10 for 5s at 480p

    def _target_durations(self, target_s: float) -> list[int]:
        """Split target duration into a list of integer durations within 2-12s range.

        Examples:
            3.7s  -> [4]
            11.0s -> [11]
            15.0s -> [12, 3]
            19.0s -> [12, 7]
            25.0s -> [12, 12, 2]  (remainder clamped to min 2)
        """
        rounded = max(self.MIN_DURATION, min(self.MAX_DURATION, round(target_s)))
        if rounded <= self.MAX_DURATION and target_s <= self.MAX_DURATION + 0.5:
            return [rounded]

        # Split into MAX_DURATION chunks plus remainder
        parts = []
        remaining = target_s
        while remaining > self.MAX_DURATION:
            parts.append(self.MAX_DURATION)
            remaining -= self.MAX_DURATION
        # Remainder
        remainder = max(self.MIN_DURATION, round(remaining))
        parts.append(remainder)
        return parts

    def generate(
        self,
        prompt: str,
        clips_dir: str,
        shot_index: int,
        target_duration_s: float,
        seed: int | None = None,
    ) -> list[ClipResult]:
        import fal_client
        import httpx

        durations = self._target_durations(target_duration_s)
        results = []

        for part_idx, duration in enumerate(durations):
            # Single clip: 0010.mp4, split clips: 0010-01.mp4, 0010-02.mp4
            if len(durations) == 1:
                clip_name = f"{shot_index:04d}.mp4"
            else:
                clip_name = f"{shot_index:04d}-{part_idx + 1:02d}.mp4"

            clip_path = os.path.join(clips_dir, clip_name)

            try:
                arguments = {
                    "prompt": prompt,
                    "aspect_ratio": "16:9",
                    "resolution": "480p",
                    "duration": str(duration),
                }
                if seed is not None:
                    # Vary seed across parts so they don't look identical
                    arguments["seed"] = seed + part_idx

                result = fal_client.subscribe(
                    self.MODEL_ID,
                    arguments=arguments,
                    with_logs=False,
                )

                video_url = result["video"]["url"]
                resp = httpx.get(video_url, follow_redirects=True)
                resp.raise_for_status()
                with open(clip_path, "wb") as f:
                    f.write(resp.content)

                cost = duration * self.COST_PER_SECOND_480P
                results.append(ClipResult(
                    path=clip_path,
                    actual_duration_s=float(duration),
                    cost=cost,
                ))

            except Exception as e:
                print(f"  Error generating clip {clip_name}: {e}")
                return []  # Fail the whole shot if any part fails

        return results


class FalSeedanceProStrategy(FalSeedanceStrategy):
    """fal.ai Seedance 1.0 Pro (standard) -- higher quality, ~2.5x cost of Fast."""

    name = "fal-seedance-pro"
    MODEL_ID = "fal-ai/bytedance/seedance/v1/pro/text-to-video"
    COST_PER_SECOND_480P = 0.05  # Approximate: ~$0.25 for 5s at 480p


class RunPodWanStrategy(GenerationStrategy):
    """RunPod self-hosted Wan 2.2 (1.3B fp8) via ComfyUI -- ~$0.004/clip."""

    name = "runpod-wan"
    CLIP_DURATION = 81 / 16  # ~5.0625s (81 frames at 16fps)

    def format_prompt(self, entry: dict) -> str:
        return _format_prompt_wan(entry)

    GPU_TYPES = [
        ("NVIDIA GeForce RTX 4090", 0.34),
        ("NVIDIA RTX 4000 Ada Generation", 0.34),
        ("NVIDIA RTX A6000", 0.52),
        ("NVIDIA L40S", 0.54),
    ]
    DOCKER_IMAGE = "runpod/comfyui:latest"
    CONTAINER_DISK_GB = 50
    COMFYUI_PORT = 8188
    POD_READY_TIMEOUT = 600  # 10 min for image pull + model load
    COMFYUI_READY_TIMEOUT = 600  # 10 min for ComfyUI to start serving (large image + model load)
    GENERATION_TIMEOUT = 300  # 5 min per clip

    def __init__(self):
        self._pod_id: str | None = None
        self._base_url: str | None = None
        self._pod_start_time: float | None = None
        self._gpu_hourly_rate: float = self.GPU_TYPES[0][1]
        self._ssh_host: str | None = None
        self._ssh_port: int | None = None
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

        # Try GPU types in order until one is available
        pod = None
        gpu_rate = self.GPU_TYPES[0][1]  # fallback rate
        for gpu_type, rate in self.GPU_TYPES:
            print(f"  Trying {gpu_type}...")
            try:
                # Read SSH public key for model downloads
                ssh_pubkey = ""
                pubkey_path = os.path.expanduser("~/.ssh/id_ed25519.pub")
                if not os.path.exists(pubkey_path):
                    pubkey_path = os.path.expanduser("~/.ssh/id_rsa.pub")
                if os.path.exists(pubkey_path):
                    with open(pubkey_path) as f:
                        ssh_pubkey = f.read().strip()

                pod = runpod.create_pod(
                    name="lossy-comfyui",
                    image_name=self.DOCKER_IMAGE,
                    gpu_type_id=gpu_type,
                    cloud_type="COMMUNITY",
                    gpu_count=1,
                    container_disk_in_gb=self.CONTAINER_DISK_GB,
                    ports=f"{self.COMFYUI_PORT}/http,22/tcp",
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

        self._pod_id = pod["id"]
        self._gpu_hourly_rate = gpu_rate
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

        # Wait for SSH, then set up models
        self._setup_comfyui_with_models()

    COMFYUI_DIR = "/workspace/runpod-slim/ComfyUI"

    # Models to download (Comfy-Org repackaged for ComfyUI)
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

    def _ssh_cmd(self, cmd: str, timeout: int = 60) -> subprocess.CompletedProcess:
        """Run a command on the pod via SSH."""
        return subprocess.run(
            [
                "ssh",
                "-o", "StrictHostKeyChecking=no",
                "-o", "UserKnownHostsFile=/dev/null",
                "-o", "LogLevel=ERROR",
                "-o", "ServerAliveInterval=30",
                "-p", str(self._ssh_port),
                f"root@{self._ssh_host}",
                cmd,
            ],
            capture_output=True,
            text=True,
            timeout=timeout,
        )

    def _ssh_bg(self, cmd: str):
        """Run a command on the pod via SSH in the background (detached)."""
        p = subprocess.Popen(
            [
                "ssh", "-f",
                "-o", "StrictHostKeyChecking=no",
                "-o", "UserKnownHostsFile=/dev/null",
                "-o", "LogLevel=ERROR",
                "-p", str(self._ssh_port),
                f"root@{self._ssh_host}",
                cmd,
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        try:
            p.communicate(timeout=10)
        except subprocess.TimeoutExpired:
            p.kill()

    def _wait_for_comfyui(self):
        """Poll ComfyUI until it responds to /system_stats."""
        import httpx

        print("  Waiting for ComfyUI...")
        start = time.time()
        while time.time() - start < self.COMFYUI_READY_TIMEOUT:
            try:
                resp = httpx.get(f"{self._base_url}/system_stats", timeout=10)
                if resp.status_code == 200:
                    print("  ComfyUI is ready.")
                    return
            except (httpx.ConnectError, httpx.TimeoutException, httpx.ReadError):
                pass
            elapsed = int(time.time() - start)
            print(f"  Waiting for ComfyUI... ({elapsed}s)", end="\r")
            time.sleep(5)

        print(f"\n  Error: ComfyUI did not become ready within {self.COMFYUI_READY_TIMEOUT}s")
        if self._ssh_host:
            try:
                result = self._ssh_cmd("tail -50 /tmp/comfyui.log", timeout=10)
                if result.stdout:
                    print(f"  ComfyUI log:\n{result.stdout}")
            except Exception:
                pass
        self._terminate_pod()
        sys.exit(1)

    def _setup_comfyui_with_models(self):
        """Stop ComfyUI, download Wan models, restart ComfyUI."""
        import runpod

        # Get SSH info
        pod_info = runpod.get_pod(self._pod_id)
        self._ssh_host = None
        self._ssh_port = None
        for port_info in pod_info.get("runtime", {}).get("ports", []):
            if port_info["privatePort"] == 22 and port_info["isIpPublic"]:
                self._ssh_host = port_info["ip"]
                self._ssh_port = port_info["publicPort"]
                break

        if not self._ssh_host:
            print("  Warning: No SSH access. Waiting for ComfyUI without model setup.")
            self._wait_for_comfyui()
            return

        # Wait for SSH to accept connections
        print("  Waiting for SSH...")
        time.sleep(10)

        # Stop ComfyUI so we can download models before it scans
        print("  Stopping ComfyUI for model setup...")
        self._ssh_cmd('pkill -f "python main.py" || true', timeout=10)
        time.sleep(2)

        # Download models
        models_dir = f"{self.COMFYUI_DIR}/models"
        print(f"  Downloading Wan models via SSH ({self._ssh_host}:{self._ssh_port})...")

        all_ok = True
        for dest_path, url in self.WAN_MODELS:
            full_path = f"{models_dir}/{dest_path}"
            filename = os.path.basename(dest_path)

            # Check if already exists (and is non-empty)
            check = self._ssh_cmd(f'[ -s {full_path} ] && echo exists || echo missing')
            if check.returncode == 0 and "exists" in check.stdout:
                print(f"    {filename}: already exists")
                continue

            print(f"    {filename}: downloading...")
            dir_path = os.path.dirname(full_path)
            result = self._ssh_cmd(
                f"mkdir -p {dir_path} && wget -q -O {full_path} '{url}' && echo OK",
                timeout=600,
            )
            if result.returncode == 0 and "OK" in result.stdout:
                print(f"    {filename}: done")
            else:
                print(f"    {filename}: FAILED (exit {result.returncode})")
                if result.stderr:
                    print(f"      {result.stderr[:200]}")
                all_ok = False

        # Start ComfyUI with models in place — find the right Python
        print("  Starting ComfyUI...")
        find_python = self._ssh_cmd(
            f"if [ -x {self.COMFYUI_DIR}/.venv/bin/python ]; then echo .venv/bin/python; "
            f"elif command -v python3 >/dev/null; then echo python3; "
            f"else echo python; fi",
            timeout=10,
        )
        python_bin = find_python.stdout.strip() if find_python.returncode == 0 else "python3"
        print(f"  Using Python: {python_bin}")
        self._ssh_bg(
            f"cd {self.COMFYUI_DIR} && {python_bin} main.py --listen 0.0.0.0 --port 8188 "
            f"</dev/null >/tmp/comfyui.log 2>&1"
        )

        self._wait_for_comfyui()


    def _terminate_pod(self):
        """Terminate the pod if running."""
        if self._pod_id is None:
            return

        import runpod

        pod_id = self._pod_id
        self._pod_id = None  # Prevent double-terminate

        elapsed_h = (time.time() - self._pod_start_time) / 3600 if self._pod_start_time else 0
        rate = getattr(self, "_gpu_hourly_rate", self.GPU_TYPES[0][1])
        estimated_cost = elapsed_h * rate

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
                    "unet_name": "wan2.1_t2v_1.3B_fp16.safetensors",
                    "weight_dtype": "default",
                },
            },
            "2": {
                "class_type": "CLIPLoader",
                "inputs": {
                    "clip_name": "umt5_xxl_fp8_e4m3fn_scaled.safetensors",
                    "type": "wan",
                    "device": "default",
                },
            },
            "3": {
                "class_type": "VAELoader",
                "inputs": {
                    "vae_name": "wan_2.1_vae.safetensors",
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
                    "text": "\u4f4e\u8d28\u91cf, \u6a21\u7cca, \u53d8\u5f62, \u5931\u771f, \u6c34\u5370, \u6587\u5b57, \u5b57\u5e55, \u4f4e\u5206\u8fa8\u7387, \u8fc7\u66dd, \u6b20\u66dd",
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
                "class_type": "EmptyHunyuanLatentVideo",
                "inputs": {
                    "width": 848,
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
                "class_type": "SaveWEBM",
                "inputs": {
                    "filename_prefix": "lossy",
                    "fps": 16,
                    "lossless": False,
                    "quality": 80,
                    "method": "default",
                    "crf": 20,
                    "codec": "vp9",
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
            if resp.status_code != 200:
                error_detail = resp.text[:300] if resp.text else "no body"
                print(f"  ComfyUI rejected workflow ({resp.status_code}): {error_detail}")
                return []
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

            # Download output video
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

            # Save raw file, convert to MP4 via ffmpeg
            raw_ext = os.path.splitext(output_file["filename"])[1] or ".webm"
            raw_path = clip_path.replace(".mp4", raw_ext)
            with open(raw_path, "wb") as f:
                f.write(resp.content)

            result = subprocess.run(
                ["ffmpeg", "-y", "-i", raw_path,
                 "-c:v", "libx264", "-pix_fmt", "yuv420p",
                 clip_path],
                capture_output=True,
            )
            if result.returncode == 0 and os.path.exists(clip_path) and os.path.getsize(clip_path) > 0:
                os.remove(raw_path)
            else:
                # Keep raw file as fallback
                if os.path.exists(clip_path):
                    os.remove(clip_path)
                os.rename(raw_path, clip_path)

            # Cost based on wall-clock time since pod started
            elapsed_h = (time.time() - self._pod_start_time) / 3600
            rate = getattr(self, "_gpu_hourly_rate", self.GPU_TYPES[0][1])
            clips_so_far = len([f for f in os.listdir(clips_dir) if f.endswith(".mp4")])
            per_clip_cost = (elapsed_h * rate) / max(clips_so_far, 1)

            # Free cached VRAM from this generation (keep models loaded)
            try:
                httpx.post(f"{self._base_url}/free", json={"free_memory": True}, timeout=10)
            except Exception:
                pass

            return [ClipResult(path=clip_path, actual_duration_s=self.CLIP_DURATION, cost=per_clip_cost)]

        except Exception as e:
            print(f"  Error generating clip: {e}")
            return []


# ---------------------------------------------------------------------------
# Audio generation strategies
# ---------------------------------------------------------------------------


class AudioStrategy:
    """Base class for audio generation backends."""

    name: str = "base"

    def generate(
        self,
        sound_description: str,
        audio_dir: str,
        shot_index: int,
        target_duration_s: float,
        seed: int | None = None,
    ) -> list[AudioClipResult]:
        """Generate audio clip(s) for a shot.

        Returns a list because long shots may be split into multiple clips.
        """
        raise NotImplementedError


class ElevenLabsStrategy(AudioStrategy):
    """ElevenLabs Sound Effects v2 via fal.ai -- $0.002/sec, max 22s."""

    name = "elevenlabs"
    MODEL_ID = "fal-ai/elevenlabs/sound-effects/v2"
    MAX_DURATION = 22
    MIN_DURATION = 0.5
    COST_PER_SECOND = 0.002

    def _target_durations(self, target_s: float) -> list[float]:
        """Split target duration into chunks within 0.5-22s range.

        Examples:
            3.7s  -> [3.7]
            20.0s -> [20.0]
            28.0s -> [22.0, 6.0]
            50.0s -> [22.0, 22.0, 6.0]
        """
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

    def generate(
        self,
        sound_description: str,
        audio_dir: str,
        shot_index: int,
        target_duration_s: float,
        seed: int | None = None,
    ) -> list[AudioClipResult]:
        import fal_client
        import httpx

        durations = self._target_durations(target_duration_s)
        results = []

        for part_idx, duration in enumerate(durations):
            if len(durations) == 1:
                clip_name = f"{shot_index:04d}.mp3"
            else:
                clip_name = f"{shot_index:04d}-{part_idx + 1:02d}.mp3"

            clip_path = os.path.join(audio_dir, clip_name)

            try:
                # Add duration context so the model fills the full clip
                dur_int = int(round(duration))
                text = f"Continuous sound, {dur_int} seconds: {sound_description}"

                arguments = {
                    "text": text,
                    "duration_seconds": duration,
                    "prompt_influence": 0.3,
                }

                result = fal_client.subscribe(
                    self.MODEL_ID,
                    arguments=arguments,
                    with_logs=False,
                )

                audio_url = result["audio"]["url"]
                resp = httpx.get(audio_url, follow_redirects=True)
                resp.raise_for_status()
                with open(clip_path, "wb") as f:
                    f.write(resp.content)

                cost = duration * self.COST_PER_SECOND
                results.append(AudioClipResult(
                    path=clip_path,
                    actual_duration_s=duration,
                    cost=cost,
                ))

            except Exception as e:
                print(f"  Error generating audio {clip_name}: {e}")
                return []

        return results


class MMAudioStrategy(AudioStrategy):
    """MMAudio V2 text-to-audio via fal.ai -- $0.001/sec, max 30s."""

    name = "mmaudio"
    MODEL_ID = "fal-ai/mmaudio-v2/text-to-audio"
    MAX_DURATION = 30
    MIN_DURATION = 1
    COST_PER_SECOND = 0.001

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

    def generate(
        self,
        sound_description: str,
        audio_dir: str,
        shot_index: int,
        target_duration_s: float,
        seed: int | None = None,
    ) -> list[AudioClipResult]:
        import fal_client
        import httpx

        durations = self._target_durations(target_duration_s)
        results = []

        for part_idx, duration in enumerate(durations):
            if len(durations) == 1:
                clip_name = f"{shot_index:04d}.flac"
            else:
                clip_name = f"{shot_index:04d}-{part_idx + 1:02d}.flac"

            clip_path = os.path.join(audio_dir, clip_name)

            try:
                arguments = {
                    "prompt": sound_description,
                    "duration": duration,
                    "num_steps": 25,
                    "cfg_strength": 4.5,
                }
                if seed is not None:
                    arguments["seed"] = (seed + part_idx) % 65536

                result = fal_client.subscribe(
                    self.MODEL_ID,
                    arguments=arguments,
                    with_logs=False,
                )

                audio_url = result["audio"]["url"]
                resp = httpx.get(audio_url, follow_redirects=True)
                resp.raise_for_status()
                with open(clip_path, "wb") as f:
                    f.write(resp.content)

                cost = duration * self.COST_PER_SECOND
                results.append(AudioClipResult(
                    path=clip_path,
                    actual_duration_s=duration,
                    cost=cost,
                ))

            except Exception as e:
                print(f"  Error generating audio {clip_name}: {e}")
                return []

        return results


# ---------------------------------------------------------------------------
# Speech generation strategy
# ---------------------------------------------------------------------------


class SpeechStrategy:
    """TTS via fal.ai ElevenLabs Turbo v2.5."""

    MODEL_ID = "fal-ai/elevenlabs/tts/turbo-v2.5"
    COST_PER_1K_CHARS = 0.05

    def __init__(self, voice: str = "Roger"):
        self.voice = voice

    def generate(
        self,
        text: str,
        speech_dir: str,
        shot_index: int,
        line_index: int,
        offset_s: float,
    ) -> SpeechClipResult | None:
        """Generate a single TTS clip for one dialogue line."""
        import fal_client
        import httpx

        clip_name = f"{shot_index:04d}-{line_index:02d}.mp3"
        clip_path = os.path.join(speech_dir, clip_name)

        try:
            result = fal_client.subscribe(
                self.MODEL_ID,
                arguments={
                    "text": text,
                    "voice": self.voice,
                },
                with_logs=False,
            )

            audio_url = result["audio"]["url"]
            resp = httpx.get(audio_url, follow_redirects=True)
            resp.raise_for_status()
            with open(clip_path, "wb") as f:
                f.write(resp.content)

            # Get actual duration via ffprobe
            probe = subprocess.run(
                ["ffprobe", "-v", "error", "-show_entries",
                 "format=duration", "-of", "csv=p=0", clip_path],
                capture_output=True, text=True,
            )
            duration = float(probe.stdout.strip()) if probe.stdout.strip() else 0

            cost = len(text) / 1000 * self.COST_PER_1K_CHARS
            return SpeechClipResult(
                path=clip_path,
                duration_s=duration,
                offset_s=offset_s,
                cost=cost,
            )

        except Exception as e:
            print(f"  Error generating speech {clip_name}: {e}")
            return None


# ---------------------------------------------------------------------------
# Prompt formatting (strategy-independent)
# ---------------------------------------------------------------------------


def format_prompt(entry: dict) -> str:
    """Convert a structured prompt entry into a flat text prompt for video generation.

    Combines the description fields into a single cinematic prompt string
    that video generation models respond well to.
    """
    desc = entry["description"]

    parts = []

    # Lead with shot type and camera movement
    shot_type = desc.get("shot_type", "")
    camera = desc.get("camera_movement", "")
    if shot_type and camera:
        parts.append(f"Cinematic {shot_type} shot, {camera}.")
    elif shot_type:
        parts.append(f"Cinematic {shot_type} shot.")

    # Core content: action is the most important for video gen
    action = desc.get("action", "")
    if action:
        parts.append(action)

    # Subjects if not already covered by action
    subjects = desc.get("subjects", "")
    if subjects:
        if isinstance(subjects, list):
            subjects = ", ".join(subjects)
        # Only add if action doesn't already describe subjects well
        if len(str(action)) < 50:
            parts.append(subjects)

    # Visual style
    lighting = desc.get("lighting", "")
    if lighting:
        parts.append(lighting)

    palette = desc.get("color_palette", "")
    if palette:
        if isinstance(palette, list):
            palette = ", ".join(palette)
        parts.append(f"Color palette: {palette}.")

    mood = desc.get("mood", "")
    if mood:
        parts.append(f"Mood: {mood}.")

    setting = desc.get("setting", "")
    if setting:
        parts.append(setting)

    return " ".join(parts)


def _format_prompt_wan(entry: dict) -> str:
    """Wan-optimized prompt: Subject > Action > Camera > Style.

    Uses professional cinematography vocabulary, no metadata labels,
    front-loaded content. Targets ~150-200 words to stay within
    Wan's T5 encoder sweet spot (~320 tokens).
    """
    desc = entry["description"]
    parts = []

    # 1. Subject (front-loaded for T5 attention)
    subjects = desc.get("subjects", "")
    if subjects:
        if isinstance(subjects, list):
            subjects = ", ".join(subjects)
        parts.append(subjects.rstrip(".") + ".")

    # 2. Action (core content)
    action = desc.get("action", "")
    if action:
        parts.append(action)

    # 3. Camera (professional terms)
    shot_type = desc.get("shot_type", "")
    camera = desc.get("camera_movement", "")
    if camera:
        camera_lower = camera.lower()
        for casual, pro in CAMERA_TERMS.items():
            if casual in camera_lower:
                camera = camera_lower.replace(casual, pro)
                break
    if shot_type and camera:
        parts.append(f"{shot_type.title()} shot, {camera}.")
    elif shot_type:
        parts.append(f"{shot_type.title()} shot.")
    elif camera:
        parts.append(f"{camera}.")

    # 4. Style/Atmosphere (no labels, just descriptive text)
    setting = desc.get("setting", "")
    if setting:
        parts.append(setting)

    lighting = desc.get("lighting", "")
    if lighting:
        parts.append(lighting)

    palette = desc.get("color_palette", "")
    if palette:
        if isinstance(palette, list):
            palette = " and ".join(palette)
        parts.append(f"{palette} tones.")

    mood = desc.get("mood", "")
    if mood:
        parts.append(f"{mood} atmosphere.")

    return " ".join(parts)


def _format_prompt_seedance(entry: dict) -> str:
    """Seedance-optimized prompt: ~30-60 words, single action, intensity adverbs."""
    desc = entry["description"]
    parts = []

    # Subject + single action verb (Seedance responds best to concise actions)
    subjects = desc.get("subjects", "")
    if subjects:
        if isinstance(subjects, list):
            subjects = subjects[0] if subjects else ""
        parts.append(subjects.rstrip("."))

    action = desc.get("action", "")
    if action:
        # Take just the first sentence for brevity
        first_sentence = action.split(".")[0].strip()
        if first_sentence:
            parts.append(first_sentence.rstrip(".") + ".")

    # Camera as a brief modifier
    camera = desc.get("camera_movement", "")
    if camera and camera.lower() != "static":
        parts.append(camera.rstrip(".") + ".")

    # One atmosphere phrase combining mood + setting
    mood = desc.get("mood", "")
    setting = desc.get("setting", "")
    if mood and setting:
        parts.append(f"{mood.split(',')[0].strip()} {setting.rstrip('.')}")
    elif setting:
        parts.append(setting)
    elif mood:
        parts.append(mood.split(",")[0].strip())

    return " ".join(parts)


# ---------------------------------------------------------------------------
# Decode loop
# ---------------------------------------------------------------------------


def run_decode(args, strategy: GenerationStrategy):
    """Main decode loop: read prompts, generate clips, track progress."""
    output_dir = args.output_dir
    prompts_path = os.path.join(output_dir, "prompts.json")

    if not os.path.exists(prompts_path):
        print(f"Error: {prompts_path} not found. Run encoder first.")
        sys.exit(1)

    with open(prompts_path) as f:
        prompts = json.load(f)

    # Apply start index and limit
    if args.start_index:
        prompts = [p for p in prompts if p["index"] >= args.start_index]
        print(f"Starting from shot index {args.start_index}")

    if args.limit:
        prompts = prompts[:args.limit]
        print(f"Processing {len(prompts)} shots")

    clips_dir = os.path.join(output_dir, "clips", strategy.name)
    os.makedirs(clips_dir, exist_ok=True)

    # Track progress for resume (per-strategy)
    progress_path = os.path.join(output_dir, f"decode_progress_{strategy.name}.json")
    if os.path.exists(progress_path):
        with open(progress_path) as f:
            progress = json.load(f)
    else:
        progress = {"completed": [], "failed": [], "total_cost_estimate": 0.0}

    # Initialize clips metadata in progress if not present
    if "clips" not in progress:
        progress["clips"] = {}

    completed_set = set(progress["completed"])
    total = len(prompts)
    generated = 0
    errors = 0

    print(f"Generating {total} clips via {strategy.name} ({len(completed_set)} already done)...")

    for entry in prompts:
        idx = entry["index"]

        if idx in completed_set:
            continue

        # Skip if already generated (check for primary clip file or split parts)
        primary_clip = os.path.join(clips_dir, f"{idx:04d}.mp4")
        part_clip = os.path.join(clips_dir, f"{idx:04d}-01.mp4")
        if os.path.exists(primary_clip) or os.path.exists(part_clip):
            completed_set.add(idx)
            if idx not in progress["completed"]:
                progress["completed"].append(idx)
            continue

        prompt_text = strategy.format_prompt(entry)
        print(f"  Shot {idx} ({generated + 1}/{total - len(completed_set)} remaining)...")

        results = strategy.generate(prompt_text, clips_dir, idx, entry["duration_s"], seed=idx)

        if results:
            generated += 1
            progress["completed"].append(idx)
            completed_set.add(idx)
            clip_cost = sum(r.cost for r in results)
            progress["total_cost_estimate"] += clip_cost
            progress["clips"][str(idx)] = [
                {"path": os.path.basename(r.path), "duration_s": r.actual_duration_s}
                for r in results
            ]
        else:
            errors += 1
            progress["failed"].append(idx)
            # Retry once after a short wait
            print(f"  Retrying shot {idx} in 10s...")
            time.sleep(10)
            results = strategy.generate(prompt_text, clips_dir, idx, entry["duration_s"], seed=idx)
            if results:
                generated += 1
                progress["completed"].append(idx)
                completed_set.add(idx)
                clip_cost = sum(r.cost for r in results)
                progress["total_cost_estimate"] += clip_cost
                progress["clips"][str(idx)] = [
                    {"path": os.path.basename(r.path), "duration_s": r.actual_duration_s}
                    for r in results
                ]
                progress["failed"] = [f for f in progress["failed"] if f != idx]

        # Save progress every 5 clips
        if generated % 5 == 0:
            with open(progress_path, "w") as f:
                json.dump(progress, f, indent=2)

        if errors > 20:
            print("Too many errors, saving progress and stopping.")
            break

    # Final save
    with open(progress_path, "w") as f:
        json.dump(progress, f, indent=2)

    print(f"\nDone. Generated {generated} clips.")
    print(f"  Total: {len(progress['completed'])} completed, {len(progress['failed'])} failed")
    print(f"  Estimated cost: ${progress['total_cost_estimate']:.2f}")


# ---------------------------------------------------------------------------
# Audio generation loop
# ---------------------------------------------------------------------------


def run_audio(args, strategy: AudioStrategy):
    """Audio generation loop: read prompts, generate audio clips, track progress."""
    output_dir = args.output_dir
    prompts_path = os.path.join(output_dir, "prompts.json")

    if not os.path.exists(prompts_path):
        print(f"Error: {prompts_path} not found. Run encoder first.")
        sys.exit(1)

    with open(prompts_path) as f:
        prompts = json.load(f)

    # Apply start index and limit
    if args.start_index:
        prompts = [p for p in prompts if p["index"] >= args.start_index]
        print(f"Starting from shot index {args.start_index}")

    if args.limit:
        prompts = prompts[:args.limit]
        print(f"Processing {len(prompts)} shots")

    audio_dir = os.path.join(output_dir, "audio", strategy.name)
    os.makedirs(audio_dir, exist_ok=True)

    # Progress tracking (per audio strategy)
    progress_path = os.path.join(output_dir, f"audio_progress_{strategy.name}.json")
    if os.path.exists(progress_path):
        with open(progress_path) as f:
            progress = json.load(f)
    else:
        progress = {"completed": [], "failed": [], "skipped": [], "total_cost_estimate": 0.0, "clips": {}}

    completed_set = set(progress["completed"])
    skipped_set = set(progress.get("skipped", []))
    total = len(prompts)
    generated = 0
    skipped = 0
    errors = 0

    print(f"Generating audio for {total} shots via {strategy.name} "
          f"({len(completed_set)} done, {len(skipped_set)} skipped)...")

    for entry in prompts:
        idx = entry["index"]

        if idx in completed_set or idx in skipped_set:
            continue

        # Extract sound description
        sound = entry.get("description", {}).get("sound")
        if not sound:
            print(f"  Shot {idx}: no sound description, skipping")
            skipped += 1
            progress["skipped"].append(idx)
            skipped_set.add(idx)
            continue

        # Skip if already generated
        ext = ".mp3" if isinstance(strategy, ElevenLabsStrategy) else ".flac"
        primary_clip = os.path.join(audio_dir, f"{idx:04d}{ext}")
        part_clip = os.path.join(audio_dir, f"{idx:04d}-01{ext}")
        if os.path.exists(primary_clip) or os.path.exists(part_clip):
            completed_set.add(idx)
            if idx not in progress["completed"]:
                progress["completed"].append(idx)
            continue

        print(f"  Shot {idx} ({generated + 1}/{total - len(completed_set) - len(skipped_set)} remaining)...")

        results = strategy.generate(sound, audio_dir, idx, entry["duration_s"], seed=idx)

        if results:
            generated += 1
            progress["completed"].append(idx)
            completed_set.add(idx)
            clip_cost = sum(r.cost for r in results)
            progress["total_cost_estimate"] += clip_cost
            progress["clips"][str(idx)] = [
                {"path": os.path.basename(r.path), "duration_s": r.actual_duration_s}
                for r in results
            ]
        else:
            errors += 1
            progress["failed"].append(idx)
            print(f"  Retrying shot {idx} in 10s...")
            time.sleep(10)
            results = strategy.generate(sound, audio_dir, idx, entry["duration_s"], seed=idx)
            if results:
                generated += 1
                progress["completed"].append(idx)
                completed_set.add(idx)
                clip_cost = sum(r.cost for r in results)
                progress["total_cost_estimate"] += clip_cost
                progress["clips"][str(idx)] = [
                    {"path": os.path.basename(r.path), "duration_s": r.actual_duration_s}
                    for r in results
                ]
                progress["failed"] = [f for f in progress["failed"] if f != idx]

        # Save progress every 5 clips
        if generated % 5 == 0:
            with open(progress_path, "w") as f:
                json.dump(progress, f, indent=2)

        if errors > 20:
            print("Too many errors, saving progress and stopping.")
            break

    # Final save
    with open(progress_path, "w") as f:
        json.dump(progress, f, indent=2)

    print(f"\nDone. Generated {generated} audio clips, skipped {skipped}.")
    print(f"  Total: {len(progress['completed'])} completed, "
          f"{len(progress.get('skipped', []))} skipped (no sound), "
          f"{len(progress['failed'])} failed")
    print(f"  Estimated cost: ${progress['total_cost_estimate']:.2f}")


# ---------------------------------------------------------------------------
# Speech generation loop
# ---------------------------------------------------------------------------


def run_speech(args, strategy: SpeechStrategy):
    """Speech generation loop: read prompts, generate TTS clips, track progress."""
    output_dir = args.output_dir
    prompts_path = os.path.join(output_dir, "prompts.json")

    if not os.path.exists(prompts_path):
        print(f"Error: {prompts_path} not found. Run encoder first.")
        sys.exit(1)

    with open(prompts_path) as f:
        prompts = json.load(f)

    if args.start_index:
        prompts = [p for p in prompts if p["index"] >= args.start_index]
    if args.limit:
        prompts = prompts[:args.limit]

    speech_dir = os.path.join(output_dir, "speech")
    os.makedirs(speech_dir, exist_ok=True)

    progress_path = os.path.join(output_dir, "speech_progress.json")
    if os.path.exists(progress_path):
        with open(progress_path) as f:
            progress = json.load(f)
    else:
        progress = {"completed": [], "skipped": [], "failed": [],
                     "total_cost_estimate": 0.0, "clips": {}}

    completed_set = set(progress["completed"])
    skipped_set = set(progress.get("skipped", []))
    generated = 0
    skipped = 0
    errors = 0

    # Count total dialogue lines
    total_lines = sum(
        len(e.get("dialogue") or [])
        for e in prompts
        if isinstance(e.get("dialogue"), list)
           and e.get("dialogue")
           and isinstance(e["dialogue"][0], dict)
    )

    print(f"Generating speech for {len(prompts)} shots ({total_lines} lines) "
          f"via ElevenLabs TTS ({len(completed_set)} shots done)...")

    for entry in prompts:
        idx = entry["index"]
        if idx in completed_set or idx in skipped_set:
            continue

        dialogue = entry.get("dialogue")
        if not dialogue or not isinstance(dialogue, list):
            skipped += 1
            progress["skipped"].append(idx)
            skipped_set.add(idx)
            continue

        # Handle both enriched (dict) and legacy (str) formats
        if not isinstance(dialogue[0], dict):
            print(f"  Shot {idx}: dialogue not enriched (plain strings), skipping")
            skipped += 1
            progress["skipped"].append(idx)
            skipped_set.add(idx)
            continue

        shot_clips = []
        shot_ok = True
        for line_idx, line in enumerate(dialogue):
            clip_path = os.path.join(speech_dir, f"{idx:04d}-{line_idx:02d}.mp3")
            if os.path.exists(clip_path):
                shot_clips.append({
                    "path": os.path.basename(clip_path),
                    "offset_s": line["start_s"],
                })
                continue

            result = strategy.generate(
                line["text"], speech_dir, idx, line_idx, line["start_s"]
            )
            if result:
                shot_clips.append({
                    "path": os.path.basename(result.path),
                    "duration_s": result.duration_s,
                    "offset_s": result.offset_s,
                })
                progress["total_cost_estimate"] += result.cost
            else:
                errors += 1
                shot_ok = False
                break

        if shot_ok and shot_clips:
            generated += 1
            progress["completed"].append(idx)
            completed_set.add(idx)
            progress["clips"][str(idx)] = shot_clips
        elif not shot_ok:
            progress["failed"].append(idx)

        if generated % 5 == 0:
            with open(progress_path, "w") as f:
                json.dump(progress, f, indent=2)

        if errors > 20:
            print("Too many errors, saving progress and stopping.")
            break

    with open(progress_path, "w") as f:
        json.dump(progress, f, indent=2)

    print(f"\nDone. Generated speech for {generated} shots, skipped {skipped}.")
    print(f"  Estimated cost: ${progress['total_cost_estimate']:.2f}")


# ---------------------------------------------------------------------------
# FFmpeg stitcher
# ---------------------------------------------------------------------------


def _stitch_audio(output_dir: str, audio_strategy: str, prompts: list[dict],
                  start_index: int | None) -> str | None:
    """Concatenate per-shot audio clips into a single audio track.

    Returns path to the combined audio file, or None if no audio clips exist.
    Audio clips are duration-adjusted to match original shot durations using
    FFmpeg's atrim/apad filters.
    """
    audio_dir = os.path.join(output_dir, "audio", audio_strategy)
    if not os.path.exists(audio_dir):
        return None

    # Load audio progress for clip metadata
    progress_path = os.path.join(output_dir, f"audio_progress_{audio_strategy}.json")
    audio_meta = {}
    if os.path.exists(progress_path):
        with open(progress_path) as f:
            audio_meta = json.load(f).get("clips", {})

    if start_index is not None:
        prompts = [p for p in prompts if p["index"] >= start_index]

    # Build per-shot audio files, adjusting duration to match original
    adjusted_dir = os.path.join(output_dir, "audio_adjusted", audio_strategy)
    os.makedirs(adjusted_dir, exist_ok=True)

    audio_entries = []
    for entry in prompts:
        idx = entry["index"]
        idx_str = str(idx)
        target_duration = entry["duration_s"]

        if idx_str in audio_meta:
            clips = audio_meta[idx_str]
        else:
            # Try to find clips on disk by convention
            clips = []
            for ext in (".mp3", ".flac"):
                path = os.path.join(audio_dir, f"{idx:04d}{ext}")
                if os.path.exists(path):
                    clips = [{"path": f"{idx:04d}{ext}", "duration_s": target_duration}]
                    break

        if not clips:
            # Generate silence for this shot
            silence_path = os.path.join(adjusted_dir, f"{idx:04d}.wav")
            if not os.path.exists(silence_path):
                subprocess.run(
                    ["ffmpeg", "-y", "-f", "lavfi", "-i",
                     f"anullsrc=r=44100:cl=stereo",
                     "-t", str(target_duration),
                     silence_path],
                    capture_output=True,
                )
            audio_entries.append(silence_path)
            continue

        if len(clips) == 1:
            # Single clip -- trim or pad to match target duration
            clip_path = os.path.join(audio_dir, clips[0]["path"])
            adjusted_path = os.path.join(adjusted_dir, f"{idx:04d}.wav")
            if not os.path.exists(adjusted_path):
                subprocess.run(
                    ["ffmpeg", "-y", "-i", clip_path,
                     "-af", f"apad=whole_dur={target_duration},atrim=0:{target_duration}",
                     "-ar", "44100", "-ac", "2",
                     adjusted_path],
                    capture_output=True,
                )
            audio_entries.append(adjusted_path)
        else:
            # Multiple clips (split shot) -- concatenate parts, then adjust
            parts_file = os.path.join(adjusted_dir, f"{idx:04d}_parts.txt")
            with open(parts_file, "w") as f:
                for clip_info in clips:
                    clip_path = os.path.join(audio_dir, clip_info["path"])
                    f.write(f"file '{os.path.abspath(clip_path)}'\n")

            concat_path = os.path.join(adjusted_dir, f"{idx:04d}_concat.wav")
            adjusted_path = os.path.join(adjusted_dir, f"{idx:04d}.wav")
            if not os.path.exists(adjusted_path):
                subprocess.run(
                    ["ffmpeg", "-y", "-f", "concat", "-safe", "0",
                     "-i", parts_file, "-ar", "44100", "-ac", "2",
                     concat_path],
                    capture_output=True,
                )
                subprocess.run(
                    ["ffmpeg", "-y", "-i", concat_path,
                     "-af", f"apad=whole_dur={target_duration},atrim=0:{target_duration}",
                     adjusted_path],
                    capture_output=True,
                )
                if os.path.exists(concat_path):
                    os.remove(concat_path)

            audio_entries.append(adjusted_path)

    if not audio_entries:
        return None

    # Concatenate all adjusted audio into one track
    concat_file = os.path.join(output_dir, "audio_concat.txt")
    with open(concat_file, "w") as f:
        for path in audio_entries:
            f.write(f"file '{os.path.abspath(path)}'\n")

    audio_track_path = os.path.join(output_dir, f"audio_track_{audio_strategy}.wav")
    subprocess.run(
        ["ffmpeg", "-y", "-f", "concat", "-safe", "0",
         "-i", concat_file, "-c", "copy",
         audio_track_path],
        capture_output=True,
    )

    print(f"  Audio track: {audio_track_path}")
    return audio_track_path


def _stitch_speech(output_dir: str, prompts: list[dict],
                   start_index: int | None) -> str | None:
    """Build time-aligned speech track from per-line TTS clips.

    Each speech clip is placed at its SRT-derived offset within the shot
    using FFmpeg adelay, then all shots are concatenated.
    """
    speech_dir = os.path.join(output_dir, "speech")
    if not os.path.exists(speech_dir):
        return None

    progress_path = os.path.join(output_dir, "speech_progress.json")
    speech_meta = {}
    if os.path.exists(progress_path):
        with open(progress_path) as f:
            speech_meta = json.load(f).get("clips", {})

    if start_index is not None:
        prompts = [p for p in prompts if p["index"] >= start_index]

    adjusted_dir = os.path.join(output_dir, "speech_adjusted")
    os.makedirs(adjusted_dir, exist_ok=True)

    speech_entries = []
    for entry in prompts:
        idx = entry["index"]
        idx_str = str(idx)
        target_duration = entry["duration_s"]
        adjusted_path = os.path.join(adjusted_dir, f"{idx:04d}.wav")

        clips = speech_meta.get(idx_str, [])
        if not clips:
            # Silence for this shot
            if not os.path.exists(adjusted_path):
                subprocess.run(
                    ["ffmpeg", "-y", "-f", "lavfi", "-i",
                     f"anullsrc=r=44100:cl=stereo",
                     "-t", str(target_duration), adjusted_path],
                    capture_output=True,
                )
            speech_entries.append(adjusted_path)
            continue

        if not os.path.exists(adjusted_path):
            if len(clips) == 1:
                # Single line: adelay + pad/trim
                clip = clips[0]
                clip_path = os.path.join(speech_dir, clip["path"])
                delay_ms = int(clip["offset_s"] * 1000)
                subprocess.run(
                    ["ffmpeg", "-y", "-i", clip_path,
                     "-af", (f"adelay={delay_ms}|{delay_ms},"
                             f"apad=whole_dur={target_duration},"
                             f"atrim=0:{target_duration}"),
                     "-ar", "44100", "-ac", "2", adjusted_path],
                    capture_output=True,
                )
            else:
                # Multiple lines: adelay each, amix together, pad/trim
                inputs = []
                filters = []
                for i, clip in enumerate(clips):
                    clip_path = os.path.join(speech_dir, clip["path"])
                    inputs.extend(["-i", clip_path])
                    delay_ms = int(clip["offset_s"] * 1000)
                    filters.append(f"[{i}]adelay={delay_ms}|{delay_ms}[d{i}]")

                mix_inputs = "".join(f"[d{i}]" for i in range(len(clips)))
                filters.append(
                    f"{mix_inputs}amix=inputs={len(clips)}:duration=longest,"
                    f"apad=whole_dur={target_duration},"
                    f"atrim=0:{target_duration}[out]"
                )
                filter_complex = ";".join(filters)

                subprocess.run(
                    ["ffmpeg", "-y"] + inputs +
                    ["-filter_complex", filter_complex,
                     "-map", "[out]",
                     "-ar", "44100", "-ac", "2", adjusted_path],
                    capture_output=True,
                )

        speech_entries.append(adjusted_path)

    if not speech_entries:
        return None

    concat_file = os.path.join(output_dir, "speech_concat.txt")
    with open(concat_file, "w") as f:
        for path in speech_entries:
            f.write(f"file '{os.path.abspath(path)}'\n")

    speech_track_path = os.path.join(output_dir, "speech_track.wav")
    subprocess.run(
        ["ffmpeg", "-y", "-f", "concat", "-safe", "0",
         "-i", concat_file, "-c", "copy", speech_track_path],
        capture_output=True,
    )

    print(f"  Speech track: {speech_track_path}")
    return speech_track_path


def _probe_duration(clip_path: str) -> float:
    """Get video duration via ffprobe. Fallback for clips without metadata."""
    try:
        result = subprocess.run(
            [
                "ffprobe", "-v", "error",
                "-show_entries", "format=duration",
                "-of", "csv=p=0",
                clip_path,
            ],
            capture_output=True,
            text=True,
        )
        return float(result.stdout.strip())
    except (ValueError, subprocess.SubprocessError):
        return 81 / 16  # Last resort fallback


def stitch_clips(args):
    """Concatenate all clips into a single reconstructed video.

    Speed-adjusts each clip to match original shot duration using FFmpeg's
    setpts filter. Uses clip metadata from decode_progress.json when available,
    falls back to ffprobe for clips without metadata.
    """
    output_dir = args.output_dir
    strategy_name = args.strategy
    clips_dir = os.path.join(output_dir, "clips", strategy_name)
    prompts_path = os.path.join(output_dir, "prompts.json")

    if not os.path.exists(clips_dir):
        # Fall back to flat clips/ for backwards compat with old runs
        clips_dir_flat = os.path.join(output_dir, "clips")
        if os.path.exists(clips_dir_flat):
            clips_dir = clips_dir_flat
        else:
            print(f"Error: {clips_dir} not found. Run decode first.")
            sys.exit(1)

    with open(prompts_path) as f:
        prompts_full = json.load(f)

    prompts = prompts_full
    if args.start_index:
        prompts = [p for p in prompts if p["index"] >= args.start_index]

    # Load clip metadata from progress (try per-strategy, fall back to legacy)
    progress_path = os.path.join(output_dir, f"decode_progress_{strategy_name}.json")
    if not os.path.exists(progress_path):
        progress_path = os.path.join(output_dir, "decode_progress.json")
    clips_meta = {}
    if os.path.exists(progress_path):
        with open(progress_path) as f:
            progress = json.load(f)
        clips_meta = progress.get("clips", {})

    # Collect existing clips in order
    clip_entries = []
    for entry in prompts:
        idx = entry["index"]
        idx_str = str(idx)

        if idx_str in clips_meta:
            # Use metadata -- handles both single clips and splits
            for clip_info in clips_meta[idx_str]:
                clip_path = os.path.join(clips_dir, clip_info["path"])
                if os.path.exists(clip_path):
                    clip_entries.append((clip_path, entry["duration_s"], clip_info["duration_s"]))
        else:
            # Backwards compat: single clip, probe or assume old default
            clip_path = os.path.join(clips_dir, f"{idx:04d}.mp4")
            if os.path.exists(clip_path):
                clip_duration = _probe_duration(clip_path)
                clip_entries.append((clip_path, entry["duration_s"], clip_duration))

    if not clip_entries:
        print("No clips found to stitch.")
        sys.exit(1)

    print(f"Stitching {len(clip_entries)} clips...")

    # Speed-adjust each clip to match original duration, write to temp dir
    adjusted_dir = os.path.join(output_dir, "adjusted", strategy_name)
    os.makedirs(adjusted_dir, exist_ok=True)

    concat_list = []
    for clip_path, original_duration, actual_duration in clip_entries:
        basename = os.path.splitext(os.path.basename(clip_path))[0]
        adjusted_path = os.path.join(adjusted_dir, f"{basename}.mp4")

        if not os.path.exists(adjusted_path):
            # For split clips, don't speed-adjust -- they're already duration-matched
            if "-" in basename:
                speed_factor = 1.0
            else:
                speed_factor = original_duration / actual_duration if actual_duration > 0 else 1.0

            if abs(speed_factor - 1.0) < 0.05:
                # Close enough -- just copy
                subprocess.run(["cp", clip_path, adjusted_path], capture_output=True)
            else:
                subprocess.run(
                    [
                        "ffmpeg", "-i", clip_path,
                        "-filter:v", f"setpts={speed_factor}*PTS",
                        "-an", "-y", adjusted_path,
                    ],
                    capture_output=True,
                )

        concat_list.append(adjusted_path)

    # Write concat list file
    concat_file = os.path.join(output_dir, "concat.txt")
    with open(concat_file, "w") as f:
        for path in concat_list:
            f.write(f"file '{os.path.abspath(path)}'\n")

    # Concatenate
    output_path = os.path.join(output_dir, f"reconstructed_{strategy_name}.mp4")
    subprocess.run(
        [
            "ffmpeg", "-f", "concat", "-safe", "0",
            "-i", concat_file,
            "-c", "copy", "-y", output_path,
        ],
        capture_output=True,
    )

    print(f"Reconstructed film saved to {output_path}")

    # Report stats
    total_original = sum(orig_dur for _, orig_dur, _ in clip_entries)
    print(f"  Original duration: {total_original:.1f}s ({total_original / 60:.1f}min)")
    print(f"  Clips used: {len(clip_entries)}")

    # Mux audio if available (SFX and/or speech)
    audio_strategy = getattr(args, "audio_strategy", None)
    speech_voice = getattr(args, "speech_voice", None)

    if audio_strategy or speech_voice:
        # Only include shots that have video clips in the stitch
        stitched_indices = set()
        for clip_path, _, _ in clip_entries:
            basename = os.path.basename(clip_path)
            idx_str = basename.split("-")[0].split(".")[0]
            stitched_indices.add(int(idx_str))
        stitched_prompts = [p for p in prompts_full if p["index"] in stitched_indices]

        # Build SFX track
        audio_track = None
        if audio_strategy:
            audio_track = _stitch_audio(output_dir, audio_strategy, stitched_prompts, None)

        # Build speech track
        speech_track = None
        if speech_voice:
            speech_track = _stitch_speech(output_dir, stitched_prompts, None)

        # Determine final audio to mux
        audio_to_mux = None
        if audio_track and os.path.exists(audio_track) and speech_track and os.path.exists(speech_track):
            # Mix SFX + speech into combined track
            combined_path = os.path.join(output_dir, "combined_audio.wav")
            subprocess.run(
                ["ffmpeg", "-y",
                 "-i", audio_track, "-i", speech_track,
                 "-filter_complex", "amix=inputs=2:duration=longest",
                 combined_path],
                capture_output=True,
            )
            audio_to_mux = combined_path
        elif audio_track and os.path.exists(audio_track):
            audio_to_mux = audio_track
        elif speech_track and os.path.exists(speech_track):
            audio_to_mux = speech_track

        if audio_to_mux:
            muxed_path = output_path.replace(".mp4", "_with_audio.mp4")
            result = subprocess.run(
                ["ffmpeg", "-y",
                 "-i", output_path,
                 "-i", audio_to_mux,
                 "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
                 "-shortest",
                 muxed_path],
                capture_output=True,
            )
            if result.returncode == 0:
                os.replace(muxed_path, output_path)
                label = "SFX+speech" if audio_track and speech_track else ("speech" if speech_track else "SFX")
                print(f"  {label} muxed into {output_path}")
            else:
                print(f"  Warning: audio mux failed, silent video preserved")
                if os.path.exists(muxed_path):
                    os.remove(muxed_path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="lossy decoder: prompts -> video")
    parser.add_argument("output_dir", help="Output directory containing prompts.json")
    parser.add_argument("--start-index", type=int, default=None,
                        help="Skip shots before this index (e.g., 10 to skip credits)")
    parser.add_argument("--limit", type=int, default=None,
                        help="Process only first N shots (after start-index)")
    parser.add_argument("--stitch", action="store_true",
                        help="Only run the stitching step (skip generation)")
    parser.add_argument("--strategy", choices=["replicate-wan", "fal-seedance", "fal-seedance-pro", "runpod-wan"],
                        default="replicate-wan",
                        help="Video generation backend (default: replicate-wan)")
    parser.add_argument("--audio", action="store_true",
                        help="Generate audio clips (instead of video)")
    parser.add_argument("--audio-strategy", choices=["elevenlabs", "mmaudio"],
                        default="elevenlabs",
                        help="Audio generation backend (default: elevenlabs)")
    parser.add_argument("--speech", action="store_true",
                        help="Generate speech/dialogue clips (instead of video)")
    parser.add_argument("--speech-voice", default="Roger",
                        help="ElevenLabs voice name for speech (default: Roger)")
    args = parser.parse_args()

    load_env()

    if args.stitch:
        stitch_clips(args)
    elif args.audio:
        audio_strategies = {
            "elevenlabs": ElevenLabsStrategy,
            "mmaudio": MMAudioStrategy,
        }
        audio_strategy = audio_strategies[args.audio_strategy]()
        run_audio(args, audio_strategy)
    elif args.speech:
        speech_strategy = SpeechStrategy(voice=args.speech_voice)
        run_speech(args, speech_strategy)
    else:
        strategies = {
            "replicate-wan": ReplicateWanStrategy,
            "fal-seedance": FalSeedanceStrategy,
            "fal-seedance-pro": FalSeedanceProStrategy,
            "runpod-wan": RunPodWanStrategy,
        }
        strategy = strategies[args.strategy]()
        run_decode(args, strategy)


if __name__ == "__main__":
    main()
