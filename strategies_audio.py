"""Audio and speech generation strategies for the lossy decoder.

Each strategy wraps a different audio generation backend (ElevenLabs, MMAudio,
RunPod self-hosted) behind a common interface. SpeechStrategy handles TTS
for dialogue lines.
"""

import os
import subprocess
import sys
import time

from clip_types import AudioClipResult, SpeechClipResult


def _split_duration(target: float, min_val: float, max_val: float) -> list[float]:
    """Split target duration into chunks within [min_val, max_val].

    Returns a list of float durations. Single-element if target fits within
    max_val; multiple elements for longer targets, with remainder clamped
    to min_val.
    """
    clamped = max(min_val, target)
    if clamped <= max_val:
        return [clamped]

    parts = []
    remaining = clamped
    while remaining > max_val:
        parts.append(float(max_val))
        remaining -= max_val
    parts.append(max(min_val, remaining))
    return parts


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
        return _split_duration(target_s, self.MIN_DURATION, self.MAX_DURATION)

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
        return _split_duration(target_s, self.MIN_DURATION, self.MAX_DURATION)

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
        return _split_duration(target_s, self.MIN_DURATION, self.MAX_DURATION)

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
