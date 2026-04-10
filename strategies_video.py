"""Video generation strategies for the lossy decoder.

Each strategy wraps a different video generation backend (Replicate, fal.ai,
RunPod self-hosted) behind a common interface. Strategies handle prompt
formatting, duration splitting, and clip download.
"""

import os
import subprocess
import threading
import time

from clip_types import AudioClipResult, ClipResult
from prompt_format import _format_prompt_wan, _format_prompt_seedance, format_prompt, vary_prompt_for_part


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
        entry: dict | None = None,
    ) -> list[ClipResult]:
        """Generate clip(s) for a shot.

        Returns a list because long shots may be split into multiple clips.
        entry is the full prompt dict (used by RunPod for concurrent audio).
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
        entry: dict | None = None,
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
        entry: dict | None = None,
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

            # Per-part prompt: use encoded temporal segments if available, else generic cues
            temporal_segs = (entry or {}).get("temporal_segments") if len(durations) > 1 else None
            if temporal_segs:
                seg_idx = round(part_idx * (len(temporal_segs) - 1) / max(len(durations) - 1, 1))
                part_entry = dict(entry, description=temporal_segs[seg_idx])
                part_prompt = self.format_prompt(part_entry)
            else:
                part_prompt = vary_prompt_for_part(prompt, part_idx, len(durations))

            try:
                arguments = {
                    "prompt": part_prompt,
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

    def __init__(self, output_dir: str = "", keep_pod: bool = False, concurrent_audio: bool = False):
        from runpod_pod import RunPodSession
        self._session = RunPodSession(output_dir, keep_pod=keep_pod)
        self._setup_done = False
        # Concurrent audio state
        self._concurrent_audio = concurrent_audio
        self._audio_capable = False
        self._audio_thread: threading.Thread | None = None
        self._audio_results: dict[int, list[AudioClipResult]] = {}
        self._audio_failures: list[int] = []
        self._audio_dir: str | None = None
        self._pending_audio: tuple | None = None

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
            if self._concurrent_audio:
                self._setup_audio()
        else:
            print("  Warning: No SSH access. Waiting for ComfyUI without model setup.")
            self._session.wait_for_comfyui()
        self._setup_done = True

    def mark_clean_exit(self):
        """Called by run_decode after successful completion."""
        self._session.mark_clean_exit()

    # -- Concurrent audio pipelining --

    def _setup_audio(self):
        """Upload mmaudio_standalone.py, install mmaudio, download weights, check VRAM."""
        # Check GPU VRAM
        result = self._session.ssh_cmd(
            "nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits",
            timeout=10,
        )
        vram_mb = 0
        if result.returncode == 0:
            try:
                vram_mb = int(result.stdout.strip().split("\n")[0])
            except (ValueError, IndexError):
                pass
        if vram_mb < 24000:
            print(f"  Concurrent audio disabled: GPU has {vram_mb} MB VRAM (<24 GB)")
            return

        # Upload mmaudio_standalone.py
        local_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tools", "mmaudio_standalone.py")
        with open(local_path) as f:
            script_content = f.read()
        # Use base64 to safely transfer the script (avoids shell escaping issues)
        import base64
        encoded = base64.b64encode(script_content.encode()).decode()
        result = self._session.ssh_cmd(
            f"echo '{encoded}' | base64 -d > /tmp/mmaudio_standalone.py && echo OK",
            timeout=30,
        )
        if "OK" not in (result.stdout or ""):
            print(f"  Warning: Failed to upload mmaudio_standalone.py")
            return

        # Install mmaudio pip package
        print("  Installing mmaudio package on pod...")
        result = self._session.ssh_cmd(
            "pip install mmaudio 2>&1 | tail -3 && echo OK",
            timeout=300,
        )
        if "OK" not in (result.stdout or ""):
            print(f"  Warning: mmaudio pip install may have failed")
            return

        # Pre-download standard weights (~5 GB first time)
        print("  Downloading MMAudio weights...")
        result = self._session.ssh_cmd(
            'python3 -c "'
            "from mmaudio.eval_utils import all_model_cfg; "
            "all_model_cfg['large_44k_v2'].download_if_needed(); "
            "print('OK')"
            '"',
            timeout=600,
        )
        if "OK" in (result.stdout or ""):
            print("  MMAudio weights: ready")
        else:
            print("  Warning: MMAudio weight download uncertain, will retry on first use")

        self._audio_capable = True
        print(f"  Concurrent audio: enabled ({vram_mb} MB VRAM)")

    def _generate_audio_ssh(
        self, shot_index: int, sound: str, duration: float, seed: int, audio_dir: str,
    ) -> list[AudioClipResult]:
        """Generate audio for one shot via SSH to mmaudio_standalone.py. Runs in background thread."""
        from strategies_audio import _split_duration

        durations = _split_duration(duration, 5.0, 10.0)
        results = []

        for part_idx, dur in enumerate(durations):
            if len(durations) == 1:
                clip_name = f"{shot_index:04d}.flac"
            else:
                clip_name = f"{shot_index:04d}-{part_idx + 1:02d}.flac"

            remote_path = f"/tmp/audio_{shot_index:04d}_{part_idx:02d}.flac"
            local_path = os.path.join(audio_dir, clip_name)
            effective_seed = (seed + part_idx) % 65536

            # Escape single quotes in sound description for shell
            escaped_sound = sound.replace("'", "'\\''")
            result = self._session.ssh_cmd(
                f"python3 /tmp/mmaudio_standalone.py "
                f"--prompt '{escaped_sound}' "
                f"--duration {dur} "
                f"--seed {effective_seed} "
                f"--output {remote_path}",
                timeout=120,
            )

            if result.returncode != 0:
                print(f"  [audio] Error for shot {shot_index}: {(result.stderr or '')[:200]}")
                return []

            # Download FLAC via SSH binary transfer (ssh_cmd uses text=True, can't use it)
            dl = subprocess.run(
                [
                    "ssh",
                    "-o", "StrictHostKeyChecking=no",
                    "-o", "UserKnownHostsFile=/dev/null",
                    "-o", "LogLevel=ERROR",
                    "-p", str(self._session.ssh_port),
                    f"root@{self._session.ssh_host}",
                    f"cat {remote_path}",
                ],
                capture_output=True,
                timeout=30,
            )
            if dl.returncode != 0 or not dl.stdout:
                print(f"  [audio] Download failed for shot {shot_index}")
                return []

            with open(local_path, "wb") as f:
                f.write(dl.stdout)

            # Clean up remote file
            self._session.ssh_cmd(f"rm -f {remote_path}", timeout=10)

            results.append(AudioClipResult(path=local_path, actual_duration_s=dur, cost=0.0))

        return results

    def _collect_audio_thread(self):
        """Wait for the current audio thread to finish, if any."""
        if self._audio_thread is not None and self._audio_thread.is_alive():
            self._audio_thread.join(timeout=180)
            if self._audio_thread.is_alive():
                print("  [audio] Warning: audio thread still running after 180s, continuing")
        self._audio_thread = None

    def _start_audio_thread(self, shot_index: int, sound: str, duration: float, seed: int, audio_dir: str):
        """Start background audio generation. Waits for any previous audio thread first."""
        self._collect_audio_thread()

        def _run():
            try:
                results = self._generate_audio_ssh(shot_index, sound, duration, seed, audio_dir)
                if results:
                    self._audio_results[shot_index] = results
                    print(f"  [audio] Shot {shot_index}: done ({len(results)} clip(s))")
                else:
                    self._audio_failures.append(shot_index)
                    print(f"  [audio] Shot {shot_index}: failed")
            except Exception as e:
                self._audio_failures.append(shot_index)
                print(f"  [audio] Shot {shot_index}: error: {e}")

        self._audio_thread = threading.Thread(target=_run, daemon=True)
        self._audio_thread.start()

    def finish_audio(self):
        """Generate audio for the final shot and wait for completion."""
        if self._audio_capable and self._pending_audio is not None:
            pa = self._pending_audio
            self._pending_audio = None
            self._start_audio_thread(pa[0], pa[1], pa[2], pa[3], pa[4])
        self._collect_audio_thread()

    def get_audio_results(self) -> tuple[dict[int, list[AudioClipResult]], list[int]]:
        """Return accumulated audio results and failures. Call after finish_audio()."""
        return self._audio_results, self._audio_failures

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
        entry: dict | None = None,
    ) -> list[ClipResult]:
        self._ensure_pod()
        effective_seed = seed if seed is not None else shot_index

        # Fire audio for the PREVIOUS shot (queued last iteration)
        if self._audio_capable and self._pending_audio is not None:
            pa = self._pending_audio
            self._pending_audio = None
            self._start_audio_thread(pa[0], pa[1], pa[2], pa[3], pa[4])

        frame_counts = self._target_durations(target_duration_s)
        results = []

        for part_idx, frames in enumerate(frame_counts):
            if len(frame_counts) == 1:
                clip_name = f"{shot_index:04d}.mp4"
            else:
                clip_name = f"{shot_index:04d}-{part_idx + 1:02d}.mp4"

            # Per-part prompt: use encoded temporal segments if available, else generic cues
            temporal_segs = (entry or {}).get("temporal_segments") if len(frame_counts) > 1 else None
            if temporal_segs:
                seg_idx = round(part_idx * (len(temporal_segs) - 1) / max(len(frame_counts) - 1, 1))
                part_entry = dict(entry, description=temporal_segs[seg_idx])
                part_prompt = self.format_prompt(part_entry)
            else:
                part_prompt = vary_prompt_for_part(prompt, part_idx, len(frame_counts))

            clip_result = self._generate_one_clip(
                part_prompt, clips_dir, clip_name, frames, effective_seed + part_idx,
            )
            if clip_result is None:
                return []  # Fail the whole shot if any part fails
            results.append(clip_result)

        # Queue THIS shot's audio for the NEXT iteration
        if self._audio_capable and entry is not None:
            sound = entry.get("description", {}).get("sound")
            if sound:
                from strategies_audio import filter_speech_from_sound
                sound = filter_speech_from_sound(sound)
            if sound:
                if self._audio_dir is None:
                    self._audio_dir = os.path.join(
                        os.path.dirname(clips_dir), "..", "audio", "runpod-mmaudio-pipelined"
                    )
                    os.makedirs(self._audio_dir, exist_ok=True)
                self._pending_audio = (shot_index, sound, target_duration_s, effective_seed, self._audio_dir)

        return results


class RunPodWanEnrichedStrategy(RunPodWanStrategy):
    """RunPod Wan T2V with canonical character identity injected into prompts.

    Same model and pod setup as RunPodWanStrategy. Only difference: format_prompt()
    prepends canonical character descriptions so the model has stronger identity
    signals. No reference image conditioning (that's RunPodVaceStrategy).

    Used as a middle condition in continuity A/B/C tests to isolate whether prompt
    enrichment alone reduces character drift vs. VACE reference conditioning.
    """

    name = "runpod-wan-enriched"

    def __init__(
        self,
        output_dir: str = "",
        keep_pod: bool = False,
        concurrent_audio: bool = False,
        character_shot_map: dict | None = None,
        characters_data: dict | None = None,
    ):
        super().__init__(output_dir, keep_pod, concurrent_audio)
        self._character_shot_map = character_shot_map or {}
        self._characters_by_name = {
            c["name"]: c for c in (characters_data or {}).get("characters", [])
        }

    def format_prompt(self, entry: dict) -> str:
        """Prepend canonical character identity to prompt when available."""
        base = super().format_prompt(entry)
        shot_idx = entry.get("index")
        if shot_idx is None:
            return base
        char_names = self._character_shot_map.get(shot_idx, [])
        if not char_names:
            return base
        identity_parts = []
        for name in char_names:
            char = self._characters_by_name.get(name)
            if char:
                identity_parts.append(f"{char['display_name']}: {char['description']}")
        if not identity_parts:
            return base
        return " ".join(identity_parts) + " " + base


class RunPodWan22Strategy(RunPodWanStrategy):
    """RunPod self-hosted Wan 2.2 TI2V-5B fp16 via ComfyUI -- 720P at 24fps, variable duration.

    Unified T2V + I2V single dense model (~10 GB fp16, ~24 GB VRAM). Uses the
    new wan2.2_vae.safetensors. Fits on RTX 4090 class pods (same tier as 2.1).
    T2V mode only -- I2V support (start_image conditioning) can be added later.
    """

    name = "runpod-wan22"
    FPS = 24
    MIN_FRAMES = 49   # ~2.04s at 24fps (4*12+1)
    MAX_FRAMES = 97   # ~4.04s at 24fps (4*24+1); conservative for 24GB VRAM headroom

    WAN_MODELS = [
        (
            "vae/wan2.2_vae.safetensors",
            "https://huggingface.co/Comfy-Org/Wan_2.2_ComfyUI_Repackaged/resolve/main/split_files/vae/wan2.2_vae.safetensors",
        ),
        (
            "text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors",
            "https://huggingface.co/Comfy-Org/Wan_2.1_ComfyUI_repackaged/resolve/main/split_files/text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors",
        ),
        (
            "diffusion_models/wan2.2_ti2v_5B_fp16.safetensors",
            "https://huggingface.co/Comfy-Org/Wan_2.2_ComfyUI_Repackaged/resolve/main/split_files/diffusion_models/wan2.2_ti2v_5B_fp16.safetensors",
        ),
    ]

    def _build_workflow(self, prompt: str, seed: int, length: int = 97) -> dict:
        """Build ComfyUI API-format workflow JSON for Wan 2.2 TI2V-5B T2V."""
        return {
            "1": {
                "class_type": "UNETLoader",
                "inputs": {
                    "unet_name": "wan2.2_ti2v_5B_fp16.safetensors",
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
                    "vae_name": "wan2.2_vae.safetensors",
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
                "class_type": "Wan22ImageToVideoLatent",
                "inputs": {
                    "width": 1280,
                    "height": 704,
                    "length": length,
                    "batch_size": 1,
                    "vae": ["3", 0],
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
                    "fps": 24,
                    "lossless": False,
                    "quality": 80,
                    "method": "default",
                    "crf": 20,
                    "codec": "vp9",
                    "images": ["9", 0],
                },
            },
        }


class RunPodVaceStrategy(RunPodWanStrategy):
    """RunPod self-hosted Wan 2.1 VACE-1.3B -- reference-conditioned video generation."""

    name = "runpod-vace"

    VACE_MODELS = [
        (
            "vae/wan_2.1_vae.safetensors",
            "https://huggingface.co/Comfy-Org/Wan_2.1_ComfyUI_repackaged/resolve/main/split_files/vae/wan_2.1_vae.safetensors",
        ),
        (
            "text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors",
            "https://huggingface.co/Comfy-Org/Wan_2.1_ComfyUI_repackaged/resolve/main/split_files/text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors",
        ),
        (
            "diffusion_models/wan2.1_vace_1.3B_fp16.safetensors",
            "https://huggingface.co/Comfy-Org/Wan_2.1_ComfyUI_repackaged/resolve/main/split_files/diffusion_models/wan2.1_vace_1.3B_fp16.safetensors",
        ),
        # T2V model needed for shots without character references (fallback workflow)
        (
            "diffusion_models/wan2.1_t2v_1.3B_fp16.safetensors",
            "https://huggingface.co/Comfy-Org/Wan_2.1_ComfyUI_repackaged/resolve/main/split_files/diffusion_models/wan2.1_t2v_1.3B_fp16.safetensors",
        ),
    ]

    def __init__(
        self,
        output_dir: str = "",
        keep_pod: bool = False,
        concurrent_audio: bool = False,
        portraits: dict | None = None,
        character_shot_map: dict | None = None,
        characters_data: dict | None = None,
    ):
        super().__init__(output_dir, keep_pod, concurrent_audio)
        self._portraits = portraits or {}
        self._character_shot_map = character_shot_map or {}
        self._uploaded_portraits: dict = {}  # name -> remote filename
        self._characters_data = characters_data or {}
        # Build name -> character lookup for prompt enrichment
        self._characters_by_name = {
            c["name"]: c for c in self._characters_data.get("characters", [])
        }

    def format_prompt(self, entry: dict) -> str:
        """Enrich prompt with canonical character identity when available."""
        base = super().format_prompt(entry)
        shot_idx = entry.get("index")
        if shot_idx is None:
            return base
        char_names = self._character_shot_map.get(shot_idx, [])
        if not char_names:
            return base
        identity_parts = []
        for name in char_names:
            char = self._characters_by_name.get(name)
            if char:
                identity_parts.append(f"{char['display_name']}: {char['description']}")
        if not identity_parts:
            return base
        return " ".join(identity_parts) + " " + base

    def _ensure_pod(self):
        if self._setup_done:
            return
        self._session.ensure_pod()
        if self._session.ssh_host:
            self._session.ssh_cmd('pkill -f "python main.py" || true', timeout=10)
            time.sleep(2)
            print("  Waiting for SSH...")
            time.sleep(10)
            self._session.download_models(self.VACE_MODELS)
            self._upload_portraits()
            self._session.restart_comfyui()
            if self._concurrent_audio:
                self._setup_audio()
        else:
            print("  Warning: No SSH access.")
            self._session.wait_for_comfyui()
        self._setup_done = True

    def _upload_portraits(self):
        """Upload character portrait images to the pod's ComfyUI input directory."""
        if not self._portraits:
            return
        from runpod_pod import COMFYUI_DIR
        remote_dir = f"{COMFYUI_DIR}/input"
        local_paths = list(self._portraits.values())
        result = self._session.scp_to(local_paths, remote_dir, timeout=30)
        if result.returncode == 0:
            for name in self._portraits:
                self._uploaded_portraits[name] = f"{name}.png"
            print(f"  Uploaded {len(self._portraits)} portraits")
        else:
            print(f"  Warning: Failed to upload portraits: {result.stderr.strip()}")

    def _build_vace_workflow(self, prompt: str, seed: int, length: int, reference_image: str) -> dict:
        """Build ComfyUI API-format workflow for VACE reference-to-video."""
        return {
            "1": {
                "class_type": "UNETLoader",
                "inputs": {
                    "unet_name": "wan2.1_vace_1.3B_fp16.safetensors",
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
                "class_type": "LoadImage",
                "inputs": {
                    "image": reference_image,
                },
            },
            "8": {
                "class_type": "WanVaceToVideo",
                "inputs": {
                    "positive": ["4", 0],
                    "negative": ["5", 0],
                    "vae": ["3", 0],
                    "width": 848,
                    "height": 480,
                    "length": length,
                    "batch_size": 1,
                    "strength": 1.0,
                    "reference_image": ["7", 0],
                },
            },
            "9": {
                "class_type": "KSampler",
                "inputs": {
                    "seed": seed,
                    "steps": 20,
                    "cfg": 6.0,
                    "sampler_name": "uni_pc",
                    "scheduler": "simple",
                    "denoise": 1.0,
                    "model": ["6", 0],
                    "positive": ["8", 0],
                    "negative": ["8", 1],
                    "latent_image": ["8", 2],
                },
            },
            "10": {
                "class_type": "TrimVideoLatent",
                "inputs": {
                    "samples": ["9", 0],
                    "trim_amount": ["8", 3],
                },
            },
            "11": {
                "class_type": "VAEDecode",
                "inputs": {
                    "samples": ["10", 0],
                    "vae": ["3", 0],
                },
            },
            "12": {
                "class_type": "SaveWEBM",
                "inputs": {
                    "filename_prefix": "lossy",
                    "fps": 16,
                    "lossless": False,
                    "quality": 80,
                    "method": "default",
                    "crf": 20,
                    "codec": "vp9",
                    "images": ["11", 0],
                },
            },
        }

    def _generate_one_clip(
        self, prompt: str, clips_dir: str, clip_name: str, frames: int, seed: int,
        reference_image: str | None = None,
    ) -> ClipResult | None:
        """Generate a single clip, optionally with VACE reference image."""
        clip_path = os.path.join(clips_dir, clip_name)

        try:
            if reference_image:
                workflow = self._build_vace_workflow(prompt, seed, length=frames, reference_image=reference_image)
            else:
                workflow = self._build_workflow(prompt, seed, length=frames)

            history = self._session.submit_workflow(workflow, timeout=300)
            if not history:
                return None

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
        entry: dict | None = None,
    ) -> list[ClipResult]:
        self._ensure_pod()
        effective_seed = seed if seed is not None else shot_index

        # Fire audio for the PREVIOUS shot (same pipelining as parent)
        if self._audio_capable and self._pending_audio is not None:
            pa = self._pending_audio
            self._pending_audio = None
            self._start_audio_thread(pa[0], pa[1], pa[2], pa[3], pa[4])

        # Look up reference image for this shot
        char_names = self._character_shot_map.get(shot_index, [])
        reference_image = None
        if char_names:
            primary_char = char_names[0]
            reference_image = self._uploaded_portraits.get(primary_char)

        frame_counts = self._target_durations(target_duration_s)
        results = []

        for part_idx, frames in enumerate(frame_counts):
            if len(frame_counts) == 1:
                clip_name = f"{shot_index:04d}.mp4"
            else:
                clip_name = f"{shot_index:04d}-{part_idx + 1:02d}.mp4"

            clip_result = self._generate_one_clip(
                prompt, clips_dir, clip_name, frames, effective_seed + part_idx,
                reference_image=reference_image,
            )
            if clip_result is None:
                return []
            results.append(clip_result)

        # Queue audio (same pipelining as parent)
        if self._audio_capable and entry is not None:
            sound = entry.get("description", {}).get("sound")
            if sound:
                from strategies_audio import filter_speech_from_sound
                sound = filter_speech_from_sound(sound)
            if sound:
                if self._audio_dir is None:
                    self._audio_dir = os.path.join(
                        os.path.dirname(clips_dir), "..", "audio", "runpod-mmaudio-pipelined"
                    )
                    os.makedirs(self._audio_dir, exist_ok=True)
                self._pending_audio = (shot_index, sound, target_duration_s, effective_seed, self._audio_dir)

        return results
