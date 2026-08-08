"""Video generation strategies for the lossy decoder.

Each strategy wraps a different video generation backend (Replicate, fal.ai,
RunPod self-hosted) behind a common interface. Strategies handle prompt
formatting, duration splitting, and clip download.
"""

import os
import subprocess
import tempfile
import threading
import time

import manifest
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

    def expected_part_count(self, target_duration_s: float) -> int:
        """How many clip files a shot of this duration splits into.

        Resume needs this to tell a finished shot from one that was cut off
        partway through its parts. Strategies that split by a different rule
        override ``_target_durations`` and inherit a correct count from it.
        """
        splitter = getattr(self, "_target_durations", None)
        if splitter is None:
            return 1
        return len(splitter(target_duration_s))

    def finish_audio(self) -> None:
        """Flush any pipelined audio generation. No-op unless overridden."""
        pass

    def get_audio_results(self) -> tuple[dict[int, list[AudioClipResult]], list[int]]:
        """Return accumulated pipelined audio results and failures. No-op default: none."""
        return {}, []

    def mark_clean_exit(self) -> None:
        """Signal a clean run completion, e.g. for pod-lifecycle bookkeeping. No-op default."""
        pass


class CharacterIdentityMixin(GenerationStrategy):
    """Shared character-identity prompt enrichment for RunPod strategies.

    Prepends canonical character descriptions ("Name: description") verbatim
    ahead of the base-formatted prompt for characters present in a shot.
    Identical tokens hold continuity, not more detail — the same string is
    prepended for a character in every shot, never reworded. Subclasses must
    call _init_character_identity() in __init__ and appear before their
    GenerationStrategy base in the MRO so super().format_prompt() reaches it.
    """

    def _init_character_identity(
        self,
        character_shot_map: dict | None = None,
        characters_data: dict | None = None,
    ) -> None:
        self._character_shot_map = character_shot_map or {}
        self._characters_by_name = {
            c["name"]: c for c in (characters_data or {}).get("characters", [])
        }

    def format_prompt(self, entry: dict) -> str:
        """Prepend the locked verbatim character identity blocks, then base."""
        return self._prepend_identity(entry, super().format_prompt(entry))

    def _prepend_identity(self, entry: dict, base: str) -> str:
        """Prepend identity blocks to an already-formatted prompt.

        Split out from format_prompt so a strategy whose base prompt does not
        come from its MRO parent (LTX-2 takes plain natural language, not the
        Wan-specific formatting its base class would apply) can still opt into
        identity enrichment.
        """
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

        clip_path = manifest.clip_path(clips_dir, shot_index)

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

            return [ClipResult(path=clip_path, actual_duration_s=manifest.probe_duration(clip_path), cost=0.05)]

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
            clip_name = manifest.clip_filename(
                shot_index, None if len(durations) == 1 else part_idx + 1, ".mp4")

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
                    actual_duration_s=manifest.probe_duration(clip_path),
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

    def __init__(self, output_dir: str = "", keep_pod: bool | int = False, concurrent_audio: bool = False):
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
        # Long-shot chaining (I2V): seed each split part from the previous part's
        # last frame so a long shot stays continuous instead of jump-cutting.
        # Default on; set LOSSY_CHAIN=0 to disable. Only used when the strategy's
        # model supports image conditioning (see _supports_i2v).
        self._chaining = os.environ.get("LOSSY_CHAIN", "1").lower() not in ("0", "false", "no", "off")

    # Whether this strategy's model accepts a start_image (I2V). Base Wan 2.1
    # T2V can't chain; RunPodWan22Strategy (TI2V-5B) overrides this to True.
    _supports_i2v = False

    # Frames dropped either side of an internal seam to cut the model's
    # slow-in/slow-out envelope. Zero disables trimming entirely, which is the
    # default -- only strategies measured to need it opt in.
    SEAM_TRIM_HEAD = 0
    SEAM_TRIM_TAIL = 0

    # A failed clip is retried rather than failing its whole shot. The first
    # submission after a fresh pod boot has been observed to fail (ComfyUI
    # rejecting the workflow, or a 404 with no body) even though
    # wait_for_comfyui()'s /system_stats probe already returned 200 -- that
    # probe proves the server is up, not that the model is loaded and
    # sampleable, and there is a window between the two. One retry after a
    # short delay clears it; without it generate() discards an entire shot on
    # a pod that is already paid for and about to work.
    CLIP_ATTEMPTS = 2
    CLIP_RETRY_DELAY_S = 20

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

        When SEAM_TRIM_* is set, each part loses frames to trimming, so parts
        cover less screen time and a long shot needs proportionally more of
        them -- see _usable_frames_per_part.
        """
        max_duration = self._usable_frames_per_part() / self.FPS
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

    def _usable_frames_per_part(self) -> int:
        """Frames a split part contributes after seam trimming.

        A part is generated at MAX_FRAMES but an interior one is trimmed at
        both ends, so only the middle survives into the film.
        """
        return self.MAX_FRAMES - self.SEAM_TRIM_HEAD - self.SEAM_TRIM_TAIL

    def _trim_seam_frames(self, clip: ClipResult, part_idx: int, n_parts: int) -> None:
        """Drop the eased-in/eased-out frames at an internal seam, in place.

        The model renders each part with a slow-in/slow-out envelope: measured
        over chained parts, motion runs at ~3.9 in the first frames and ~4.6 in
        the last against ~8.2 mid-clip, and some parts end on a frozen frame.
        Chaining removes the *content* jump at a seam but not this *motion*
        one, so a long shot decelerates and re-accelerates every part boundary
        -- a rhythmic hitch through any elongated scene.

        Only interior edges are trimmed. The shot's true first and last frames
        keep their natural ease, since a real cut lands there anyway.
        """
        if n_parts <= 1:
            return
        head = 0 if part_idx == 0 else self.SEAM_TRIM_HEAD
        tail = 0 if part_idx == n_parts - 1 else self.SEAM_TRIM_TAIL
        if not head and not tail:
            return

        total = round(clip.actual_duration_s * self.FPS)
        keep_last = total - tail - 1
        if keep_last <= head:
            return  # too short to trim safely -- leave it alone

        trimmed = clip.path + ".trim.mp4"
        r = subprocess.run(
            ["ffmpeg", "-y", "-v", "error", "-i", clip.path,
             "-vf", f"select='between(n\\,{head}\\,{keep_last})',setpts=N/FRAME_RATE/TB",
             "-an", trimmed],
            capture_output=True,
        )
        if r.returncode != 0 or not os.path.exists(trimmed) or os.path.getsize(trimmed) == 0:
            print(f"  Seam trim failed for {os.path.basename(clip.path)}; keeping untrimmed")
            if os.path.exists(trimmed):
                os.remove(trimmed)
            return
        os.replace(trimmed, clip.path)
        clip.actual_duration_s = manifest.probe_duration(clip.path)

    def _ensure_pod(self):
        """Bring a working pod up, re-provisioning if one fails to come alive.

        Inherited by every RunPod video strategy: setup is the fragile part
        (bad hosts, no capacity), and without a retry each failure stalls the
        run until a human notices.
        """
        if self._setup_done:
            return
        self._session.with_setup_retry(self._setup_pod_once)

    def _setup_pod_once(self):
        if self._setup_done:
            return
        self._session.ensure_pod()
        if self._session.ssh_host:
            self._session.ssh_cmd('pkill -f "main.py" || true', timeout=10)
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
            print("  Warning: Failed to upload mmaudio_standalone.py")
            return

        # Install mmaudio pip package
        print("  Installing mmaudio package on pod...")
        result = self._session.ssh_cmd(
            "pip install mmaudio 2>&1 | tail -3 && echo OK",
            timeout=300,
        )
        if "OK" not in (result.stdout or ""):
            print("  Warning: mmaudio pip install may have failed")
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
            clip_name = manifest.audio_clip_filename(
                shot_index, None if len(durations) == 1 else part_idx + 1)

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

    def _build_workflow(self, prompt: str, seed: int, length: int = 81, start_image: str | None = None) -> dict:
        """Build ComfyUI API-format workflow JSON for Wan T2V.

        Base Wan 2.1 is T2V-only and ignores ``start_image``; the Wan 2.2
        (TI2V) subclass overrides this to wire image conditioning for chaining.
        """
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

    def _extract_last_frame(self, clip_path: str, out_path: str) -> bool:
        """Write a clip's final frame to out_path (PNG). Returns success.

        `-sseof -N -frames:v 1` looks right but takes the FIRST frame in the
        trailing window, not the last -- with -0.2 that is ~5 frames early at
        25fps. Handing the model that frame as "continue from here" made every
        chained part restart 4 frames back and replay them: motion visibly
        undone at each join, measured across four joins as a best match 4
        frames before the end (diff ~2.5) against 8-19 for the true final
        frame.

        Dropping -frames:v 1 and keeping -update 1 writes every frame in the
        window, each overwriting the last, so the file ends up holding the
        clip's actual final frame.
        """
        r = subprocess.run(
            ["ffmpeg", "-y", "-sseof", "-0.5", "-i", clip_path,
             "-update", "1", out_path],
            capture_output=True,
        )
        return r.returncode == 0 and os.path.exists(out_path) and os.path.getsize(out_path) > 0

    def _upload_start_frame(self, clip_path: str, shot_index: int, part_idx: int) -> str | None:
        """Extract a clip's last frame and upload it to the pod's ComfyUI input
        dir for I2V chaining. Returns the remote filename (for LoadImage) or None."""
        if not self._session.ssh_host:
            return None
        from runpod_pod import COMFYUI_DIR
        name = f"chain_{shot_index:04d}_{part_idx:02d}.png"
        local = os.path.join(tempfile.gettempdir(), name)
        if not self._extract_last_frame(clip_path, local):
            return None
        # Chaining is an enhancement, never a reason to lose a run. A stalled
        # scp used to raise TimeoutExpired straight through the generate loop
        # and abort the whole render 84 clips in; degrade to an unchained part
        # (one visible seam) instead.
        try:
            result = self._session.scp_to([local], f"{COMFYUI_DIR}/input", timeout=60)
        except subprocess.TimeoutExpired:
            print("  Chain: start-frame upload timed out; generating this part unchained")
            return None
        except Exception as e:
            print(f"  Chain: start-frame upload failed ({e}); generating this part unchained")
            return None
        if result.returncode != 0:
            print(f"  Chain: failed to upload start frame: {result.stderr.strip()[:120]}")
            return None
        return name

    def _build_clip_workflow(self, prompt: str, seed: int, frames: int, start_image: str | None = None) -> dict:
        """Choose the ComfyUI workflow for one clip.

        Subclasses override to pick an alternate workflow (e.g. VACE
        reference-conditioned generation) while reusing _generate_one_clip.
        """
        return self._build_workflow(prompt, seed, length=frames, start_image=start_image)

    def _generate_one_clip(
        self, prompt: str, clips_dir: str, clip_name: str, frames: int, seed: int,
        start_image: str | None = None,
    ) -> ClipResult | None:
        """Generate a single clip, retrying a failed attempt (see CLIP_ATTEMPTS).

        The seed is held constant across attempts: a retry is meant to be the
        same clip generated again, not a different one, so a run that resumes
        after a failure lands on the same footage it would have had.
        """
        for attempt in range(1, self.CLIP_ATTEMPTS + 1):
            result = self._attempt_one_clip(
                prompt, clips_dir, clip_name, frames, seed, start_image=start_image)
            if result is not None:
                return result
            if attempt < self.CLIP_ATTEMPTS:
                print(f"  {clip_name}: attempt {attempt} failed, "
                      f"retrying in {self.CLIP_RETRY_DELAY_S}s "
                      f"[attempt {attempt + 1}/{self.CLIP_ATTEMPTS}]...")
                time.sleep(self.CLIP_RETRY_DELAY_S)
        return None

    def _attempt_one_clip(
        self, prompt: str, clips_dir: str, clip_name: str, frames: int, seed: int,
        start_image: str | None = None,
    ) -> ClipResult | None:
        """Generate a single clip with the given frame count. One attempt."""
        clip_path = os.path.join(clips_dir, clip_name)

        try:
            workflow = self._build_clip_workflow(prompt, seed, frames, start_image=start_image)
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

            return ClipResult(path=clip_path, actual_duration_s=manifest.probe_duration(clip_path), cost=per_clip_cost)
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
        # Chain split parts via I2V when the model supports it (Wan 2.2 TI2V):
        # each part after the first is seeded with the previous part's last frame.
        chaining = self._supports_i2v and self._chaining and len(frame_counts) > 1
        start_image = None

        for part_idx, frames in enumerate(frame_counts):
            clip_name = manifest.clip_filename(
                shot_index, None if len(frame_counts) == 1 else part_idx + 1, ".mp4")

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
                start_image=start_image,
            )
            if clip_result is None:
                return []  # Fail the whole shot if any part fails

            # Trim before chaining reads the last frame: conditioning the next
            # part on a full-motion frame rather than a decelerating (sometimes
            # frozen) one is exactly what keeps the motion continuous.
            self._trim_seam_frames(clip_result, part_idx, len(frame_counts))
            results.append(clip_result)

            # Seed the next split part from this part's last frame.
            if chaining and part_idx < len(frame_counts) - 1:
                start_image = self._upload_start_frame(clip_result.path, shot_index, part_idx)

        # Queue THIS shot's audio for the NEXT iteration
        if self._audio_capable and entry is not None:
            sound = entry.get("description", {}).get("sound")
            if sound:
                from strategies_audio import filter_speech_from_sound
                sound = filter_speech_from_sound(sound)
            if sound:
                if self._audio_dir is None:
                    self._audio_dir = manifest.audio_dir(
                        os.path.dirname(os.path.dirname(clips_dir)), "runpod-mmaudio-pipelined")
                    os.makedirs(self._audio_dir, exist_ok=True)
                self._pending_audio = (shot_index, sound, target_duration_s, effective_seed, self._audio_dir)

        return results


class RunPodWanEnrichedStrategy(CharacterIdentityMixin, RunPodWanStrategy):
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
        keep_pod: bool | int = False,
        concurrent_audio: bool = False,
        character_shot_map: dict | None = None,
        characters_data: dict | None = None,
    ):
        super().__init__(output_dir, keep_pod, concurrent_audio)
        self._init_character_identity(character_shot_map, characters_data)


class RunPodWan22Strategy(CharacterIdentityMixin, RunPodWanStrategy):
    """RunPod self-hosted Wan 2.2 TI2V-5B fp16 via ComfyUI -- 720P at 24fps, variable duration.

    Unified T2V + I2V single dense model (~10 GB fp16, ~24 GB VRAM). Uses the
    new wan2.2_vae.safetensors. Fits on RTX 4090 class pods (same tier as 2.1).
    Unified T2V + I2V: split long-shot parts are chained via start_image
    conditioning (see _supports_i2v and _build_workflow) so a long shot stays
    continuous instead of jump-cutting.
    """

    name = "runpod-wan22"
    _supports_i2v = True  # accepts a start_image — enables long-shot chaining
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

    def __init__(
        self,
        output_dir: str = "",
        keep_pod: bool | int = False,
        concurrent_audio: bool = False,
        character_shot_map: dict | None = None,
        characters_data: dict | None = None,
    ):
        super().__init__(output_dir, keep_pod, concurrent_audio)
        self._init_character_identity(character_shot_map, characters_data)

    def _build_workflow(self, prompt: str, seed: int, length: int = 97, start_image: str | None = None) -> dict:
        """Build ComfyUI API-format workflow JSON for Wan 2.2 TI2V-5B.

        With ``start_image`` (a filename already in the pod's ComfyUI input dir),
        the first frame is conditioned on it via a LoadImage node — used to chain
        split long-shot parts so the shot stays continuous.
        """
        wf = {
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
        if start_image:
            # Condition the first frame on the previous part's last frame (I2V chaining).
            wf["11"] = {"class_type": "LoadImage", "inputs": {"image": start_image}}
            wf["7"]["inputs"]["start_image"] = ["11", 0]
        return wf


class RunPodLtx2Strategy(CharacterIdentityMixin, RunPodWanStrategy):
    """RunPod self-hosted LTX-2.3 (22B, distilled FP8) via ComfyUI.

    Spike strategy for the LTX-2 vs Wan 2.2 TI2V-5B trial (docs/research
    0022-ltx2-vs-wan22-trial). Live-render verified working (all 5 shots of
    the trial's benchmark slice generated successfully).

    LTX-2's distilled-FP8 checkpoint loads as an audio-video *joint* model at
    the architecture level (comfy/ldm/lightricks/av_model.py) -- its
    text-conditioning path unconditionally expects an AV-shaped latent, so
    even this video-only strategy has to build and run the audio-latent
    scaffold (LTXVEmptyLatentAudio -> LTXVConcatAVLatent -> KSampler ->
    LTXVSeparateAVLatent) around the sampler; the audio latent is generated
    but never decoded or saved. See the trial write-up's "Native audio"
    section for what it would take to actually decode and use that audio.

    Forces an RTX A6000 (48GB) pod since LTX-2 needs 32GB+ VRAM even at FP8,
    which rules out the 24GB RTX 4090 that's normally tried first.
    """

    name = "runpod-ltx2"
    FPS = 25
    MIN_FRAMES = 25    # 1s
    MAX_FRAMES = 121   # ~4.8s -- LTX-2 native clip length before quality degrades
    _supports_i2v = True  # accepts a start_image via LTXVImgToVideo -- chains split parts

    # Matches runpod-wan22's render size. The original trial hardcoded 768x512
    # as a cautious placeholder while debugging the graph, which made the
    # speed/cost comparison unfair (~2.3x fewer pixels than Wan22). Both
    # dimensions must stay divisible by 32 for the LTX VAE.
    WIDTH = 1280
    HEIGHT = 704

    # Measured over chained parts of the dress-rehearsal segment: motion sits
    # at ~3.9 across the first frames and ~4.6 across the last, against ~8.2
    # mid-clip, with ramps running ~18 and ~13 frames. Some parts end on a
    # frozen frame (frame-to-frame delta 0.12 against ~5 typical), which is
    # the most visible half of the hitch.
    #
    # Seam trimming is DISABLED: it was tried against the residual hitch at
    # part boundaries and measured no benefit. Surveyed across every
    # multi-part shot of the rehearsal segment (seam = mean abs frame delta
    # across the boundary, against normal frame-to-frame motion):
    #
    #   unchained            seam 44.6   motion 5.24   8.5x
    #   chained              seam 10.4   motion 4.67   2.2x
    #   chained + tail-trim  seam 11.5   motion 5.12   2.2x
    #
    # Chaining does the work; trimming 12 frames off the tail left the seam
    # fractionally worse and still produced near-frozen final frames (4/34
    # vs 5/26), while costing frames and forcing extra parts per shot.
    #
    # If this is revisited: HEAD MUST STAY 0 while chaining. Continuity lives
    # in the first frames -- they are the ones conditioned on the previous
    # part's last frame -- and an 18-frame head trim measured seams of 25-43,
    # undoing chaining entirely. Any trim must also run before
    # _upload_start_frame; retrofitting one onto generated clips breaks the
    # correspondence and measured 23.1.
    SEAM_TRIM_HEAD = 0
    SEAM_TRIM_TAIL = 0

    # LTX-2 needs 32GB+ VRAM at FP8 (docs.ltx.io) -- skip the 24GB RTX 4090 in
    # the default fallback chain, it would OOM. All of these are 48GB, ordered
    # by price.
    #
    # The chain is deliberately long: a rehearsal attempt died with A6000
    # "no longer any instances available" and L40S erroring in the same
    # breath, which stalls a run outright when only two pools are tried. A40
    # costs about the same as the A6000 and draws from a different pool, so
    # widening this is nearly free insurance for a multi-hour run.
    LTX_GPU_TYPES = [
        ("NVIDIA RTX A6000", 0.33),
        ("NVIDIA A40", 0.35),
        ("NVIDIA L40", 0.69),
        ("NVIDIA RTX 6000 Ada Generation", 0.74),
        ("NVIDIA L40S", 0.79),
    ]

    CHECKPOINT_NAME = "ltx-2.3-22b-distilled-fp8.safetensors"
    # fp8-scaled quantized Gemma encoder (~12GB) rather than the 24.4GB bf16
    # full weights -- the 22B video/audio checkpoint already uses ~22GB, and
    # a 48GB A6000 needs headroom for both loaded at once.
    TEXT_ENCODER_NAME = "gemma_3_12B_it_fp8_scaled.safetensors"

    LTX_MODELS = [
        (
            f"checkpoints/{CHECKPOINT_NAME}",
            f"https://huggingface.co/Lightricks/LTX-2.3-fp8/resolve/main/{CHECKPOINT_NAME}",
        ),
        (
            f"text_encoders/{TEXT_ENCODER_NAME}",
            f"https://huggingface.co/Comfy-Org/ltx-2/resolve/main/split_files/text_encoders/{TEXT_ENCODER_NAME}",
        ),
    ]

    CUSTOM_NODE_REPO = "https://github.com/Lightricks/ComfyUI-LTXVideo.git"

    def __init__(
        self,
        output_dir: str = "",
        keep_pod: bool | int = False,
        character_shot_map: dict | None = None,
        characters_data: dict | None = None,
    ):
        # No concurrent-audio pipelining for this spike -- LTX-2's own audio
        # path is evaluated separately (see class docstring).
        super().__init__(output_dir, keep_pod, concurrent_audio=False)
        self._init_character_identity(character_shot_map, characters_data)
        from runpod_pod import RunPodSession
        self._session = RunPodSession(output_dir, keep_pod=keep_pod, gpu_types=self.LTX_GPU_TYPES)

    def format_prompt(self, entry: dict) -> str:
        # LTX-2 takes plain natural-language prompts (no Wan-specific
        # formatting), so build the base directly rather than through the MRO,
        # then let the identity mixin prepend canonical character descriptions.
        # Without this the model has no idea who anyone is: an early rehearsal
        # pass rendered Han and Luke as generic modern men in a warehouse.
        return self._prepend_identity(entry, format_prompt(entry))

    def _install_custom_nodes(self):
        """Clone ComfyUI-LTXVideo into custom_nodes and install its requirements."""
        from runpod_pod import COMFYUI_DIR
        custom_nodes_dir = f"{COMFYUI_DIR}/custom_nodes"
        check = self._session.ssh_cmd(f"[ -d {custom_nodes_dir}/ComfyUI-LTXVideo ] && echo exists || echo missing")
        if check.returncode == 0 and "exists" in check.stdout:
            print("  ComfyUI-LTXVideo: already installed")
            return
        print("  Installing ComfyUI-LTXVideo custom nodes...")
        result = self._session.ssh_cmd(
            f"cd {custom_nodes_dir} && git clone --depth 1 {self.CUSTOM_NODE_REPO} 2>&1 | tail -5 "
            f"&& pip install -r ComfyUI-LTXVideo/requirements.txt 2>&1 | tail -10 && echo OK",
            timeout=300,
        )
        if "OK" not in (result.stdout or ""):
            print(f"  Warning: ComfyUI-LTXVideo install may have failed: {(result.stdout or '')[-500:]}")

    def _setup_pod_once(self):
        if self._setup_done:
            return
        self._session.ensure_pod()
        if self._session.ssh_host:
            self._session.ssh_cmd('pkill -f "main.py" || true', timeout=10)
            time.sleep(2)
            print("  Waiting for SSH...")
            time.sleep(10)
            self._install_custom_nodes()
            # The 22B FP8 checkpoint is ~22GB -- well beyond what fits in the
            # default 600s SSH timeout on a middling connection.
            self._session.download_models(self.LTX_MODELS, timeout=1800)
            self._session.restart_comfyui()
        else:
            print("  Warning: No SSH access. Waiting for ComfyUI without model setup.")
            self._session.wait_for_comfyui()
        self._setup_done = True

    def _target_frames(self, target_s: float) -> int:
        raw = round(target_s * self.FPS)
        return max(self.MIN_FRAMES, min(self.MAX_FRAMES, raw))

    def _build_workflow(self, prompt: str, seed: int, length: int = 97, start_image: str | None = None) -> dict:
        """Build a minimal T2V ComfyUI API-format workflow for LTX-2.3 distilled.

        LTX-2's checkpoint is an audio-video *joint* model at the architecture
        level (comfy/ldm/lightricks/av_model.py) -- its text-conditioning path
        unconditionally expects an AV-shaped latent, so even a video-only
        render has to build the audio latent and concat/separate it around
        the sampler (LTXVEmptyLatentAudio -> LTXVConcatAVLatent -> KSampler ->
        LTXVSeparateAVLatent). The audio latent is generated but discarded --
        never decoded or saved -- since this trial doesn't use LTX-2 audio.
        The text encoder must be loaded via LTXAVTextEncoderLoader (not the
        generic CLIPLoader): it merges the standalone gemma safetensors file
        with weights that live in the main checkpoint itself.
        """
        wf = {
            "1": {
                "class_type": "CheckpointLoaderSimple",
                "inputs": {"ckpt_name": self.CHECKPOINT_NAME},
            },
            "2": {
                "class_type": "LTXAVTextEncoderLoader",
                "inputs": {"text_encoder": self.TEXT_ENCODER_NAME, "ckpt_name": self.CHECKPOINT_NAME, "device": "default"},
            },
            "3": {
                "class_type": "CLIPTextEncode",
                "inputs": {"text": prompt, "clip": ["2", 0]},
            },
            "4": {
                "class_type": "CLIPTextEncode",
                "inputs": {"text": "worst quality, blurry, distorted, low resolution, watermark, subtitles", "clip": ["2", 0]},
            },
            "5": {
                "class_type": "EmptyLTXVLatentVideo",
                "inputs": {"width": self.WIDTH, "height": self.HEIGHT, "length": length, "batch_size": 1},
            },
            "6": {
                "class_type": "LTXVConditioning",
                "inputs": {"positive": ["3", 0], "negative": ["4", 0], "frame_rate": self.FPS},
            },
            # Nodes 14/15 replace node 5 when chaining -- see the tail of this method.
            "7": {
                "class_type": "LTXVAudioVAELoader",
                "inputs": {"ckpt_name": self.CHECKPOINT_NAME},
            },
            "8": {
                "class_type": "LTXVEmptyLatentAudio",
                "inputs": {"frames_number": length, "frame_rate": self.FPS, "batch_size": 1, "audio_vae": ["7", 0]},
            },
            "9": {
                "class_type": "LTXVConcatAVLatent",
                "inputs": {"video_latent": ["5", 0], "audio_latent": ["8", 0]},
            },
            "10": {
                "class_type": "KSampler",
                "inputs": {
                    "seed": seed,
                    "steps": 8,
                    "cfg": 1.0,
                    "sampler_name": "euler_ancestral",
                    "scheduler": "simple",
                    "denoise": 1.0,
                    "model": ["1", 0],
                    "positive": ["6", 0],
                    "negative": ["6", 1],
                    "latent_image": ["9", 0],
                },
            },
            "11": {
                "class_type": "LTXVSeparateAVLatent",
                "inputs": {"av_latent": ["10", 0]},
            },
            "12": {
                "class_type": "VAEDecode",
                "inputs": {"samples": ["11", 0], "vae": ["1", 2]},
            },
            "13": {
                "class_type": "SaveWEBM",
                "inputs": {
                    "filename_prefix": "lossy",
                    "fps": self.FPS,
                    "lossless": False,
                    "quality": 80,
                    "method": "default",
                    "crf": 20,
                    "codec": "vp9",
                    "images": ["12", 0],
                },
            },
        }
        if start_image:
            # I2V chaining for split parts: condition this part's first frame on
            # the previous part's last frame, so a long shot stays continuous
            # instead of jump-cutting every ~4.8s. Without this, 52% of a
            # typical segment's runtime sits in split shots that re-roll the
            # scene at every part boundary (see research/0023).
            #
            # LTXVImgToVideo supersedes EmptyLTXVLatentVideo: it returns the
            # image-conditioned positive/negative AND the latent, so the
            # conditioning has to flow through it before LTXVConditioning
            # stamps the frame rate, and the audio concat takes its latent.
            wf["14"] = {"class_type": "LoadImage", "inputs": {"image": start_image}}
            wf["15"] = {
                "class_type": "LTXVImgToVideo",
                "inputs": {
                    "positive": ["3", 0],
                    "negative": ["4", 0],
                    "vae": ["1", 2],
                    "image": ["14", 0],
                    "width": self.WIDTH,
                    "height": self.HEIGHT,
                    "length": length,
                    "batch_size": 1,
                    "strength": 1.0,
                },
            }
            wf["6"]["inputs"]["positive"] = ["15", 0]
            wf["6"]["inputs"]["negative"] = ["15", 1]
            wf["9"]["inputs"]["video_latent"] = ["15", 2]
        return wf


class RunPodVaceStrategy(CharacterIdentityMixin, RunPodWanStrategy):
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
        keep_pod: bool | int = False,
        concurrent_audio: bool = False,
        portraits: dict | None = None,
        character_shot_map: dict | None = None,
        characters_data: dict | None = None,
    ):
        super().__init__(output_dir, keep_pod, concurrent_audio)
        self._portraits = portraits or {}
        self._uploaded_portraits: dict = {}  # name -> remote filename
        self._init_character_identity(character_shot_map, characters_data)

    def _setup_pod_once(self):
        if self._setup_done:
            return
        self._session.ensure_pod()
        if self._session.ssh_host:
            self._session.ssh_cmd('pkill -f "main.py" || true', timeout=10)
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

    def _build_clip_workflow(self, prompt: str, seed: int, frames: int, start_image: str | None = None) -> dict:
        """Use VACE reference conditioning when a reference image is given,
        else fall back to the parent's plain T2V workflow."""
        if start_image:
            return self._build_vace_workflow(prompt, seed, length=frames, reference_image=start_image)
        return super()._build_clip_workflow(prompt, seed, frames)

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
            clip_name = manifest.clip_filename(
                shot_index, None if len(frame_counts) == 1 else part_idx + 1, ".mp4")

            clip_result = self._generate_one_clip(
                prompt, clips_dir, clip_name, frames, effective_seed + part_idx,
                start_image=reference_image,
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
                    self._audio_dir = manifest.audio_dir(
                        os.path.dirname(os.path.dirname(clips_dir)), "runpod-mmaudio-pipelined")
                    os.makedirs(self._audio_dir, exist_ok=True)
                self._pending_audio = (shot_index, sound, target_duration_s, effective_seed, self._audio_dir)

        return results
