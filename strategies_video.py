"""Video generation strategies for the lossy decoder.

Each strategy wraps a different video generation backend (Replicate, fal.ai,
RunPod self-hosted) behind a common interface. Strategies handle prompt
formatting, duration splitting, and clip download.
"""

import os
import subprocess
import time

from decode import ClipResult
from prompt_format import _format_prompt_wan, _format_prompt_seedance, format_prompt


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
    """RunPod self-hosted Wan 2.1 (1.3B fp16) via ComfyUI -- variable duration."""

    name = "runpod-wan"
    FPS = 16
    MIN_FRAMES = 33   # ~2.06s at 16fps
    MAX_FRAMES = 97    # ~6.06s at 16fps (129 times out on 1.3B GPUs)

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

    def _target_frames(self, target_s: float) -> int:
        """Convert duration to nearest valid Wan frame count (4n+1), clamped."""
        raw = round((target_s * self.FPS - 1) / 4) * 4 + 1
        return max(self.MIN_FRAMES, min(self.MAX_FRAMES, raw))

    def _target_durations(self, target_s: float) -> list[int]:
        """Split target duration into frame counts, splitting shots >MAX_FRAMES duration.

        Returns a list of frame counts (4n+1 integers). Single-element for
        shots that fit within MAX_FRAMES; multiple elements for longer shots.
        """
        max_duration = self.MAX_FRAMES / self.FPS
        min_duration = self.MIN_FRAMES / self.FPS

        if target_s <= max_duration + 0.25:  # 0.25s = half a frame-step tolerance
            return [self._target_frames(target_s)]

        parts = []
        remaining = target_s
        while remaining > max_duration:
            parts.append(self.MAX_FRAMES)
            remaining -= max_duration
        remainder_frames = self._target_frames(max(min_duration, remaining))
        parts.append(remainder_frames)
        return parts

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

    def mark_clean_exit(self):
        """Called by run_decode after successful completion."""
        self._session.mark_clean_exit()

    def _build_workflow(self, prompt: str, seed: int, length: int = 81) -> dict:
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
                    "length": length,
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

    def _generate_one_clip(
        self, prompt: str, clips_dir: str, clip_name: str, frames: int, seed: int,
    ) -> ClipResult | None:
        """Generate a single clip with the given frame count."""
        clip_path = os.path.join(clips_dir, clip_name)

        try:
            workflow = self._build_workflow(prompt, seed, length=frames)
            history = self._session.submit_workflow(workflow, timeout=300)
            if not history:
                return None

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
                print(f"  No output file found for {clip_name}")
                return None

            data = self._session.download_output(output_file)
            if not data:
                return None

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

            return ClipResult(path=clip_path, actual_duration_s=frames / self.FPS, cost=per_clip_cost)
        except Exception as e:
            print(f"  Error generating {clip_name}: {e}")
            return None

    def generate(
        self,
        prompt: str,
        clips_dir: str,
        shot_index: int,
        target_duration_s: float,
        seed: int | None = None,
    ) -> list[ClipResult]:
        self._ensure_pod()
        effective_seed = seed if seed is not None else shot_index

        frame_counts = self._target_durations(target_duration_s)
        results = []

        for part_idx, frames in enumerate(frame_counts):
            if len(frame_counts) == 1:
                clip_name = f"{shot_index:04d}.mp4"
            else:
                clip_name = f"{shot_index:04d}-{part_idx + 1:02d}.mp4"

            clip_result = self._generate_one_clip(
                prompt, clips_dir, clip_name, frames, effective_seed + part_idx,
            )
            if clip_result is None:
                return []  # Fail the whole shot if any part fails
            results.append(clip_result)

        return results
