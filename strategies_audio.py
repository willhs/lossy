"""Audio and speech generation strategies for the lossy decoder.

Each strategy wraps a different audio generation backend (ElevenLabs, MMAudio,
RunPod self-hosted) behind a common interface. SpeechStrategy handles TTS
for dialogue lines.
"""

import os
import re
import subprocess
import sys
import time

import manifest
from clip_types import AudioClipResult, SpeechClipResult


# ---------------------------------------------------------------------------
# Speech-cue filter for MMAudio prompts
# ---------------------------------------------------------------------------

_SPEECH_RE = re.compile(
    r"\bvoice(?:s)?\b"
    r"|\bspeaking\b|\bspeaks\b|\bspoken\b|\bspeech\b"
    r"|\bdialogue\b"
    r"|\btalking\b|\btalks\b"
    r"|\bconversation\b"
    r"|\bsays\b|\bsaying\b"
    r"|\bshout(?:s|ing)?\b"
    r"|\bwhisper(?:s|ing)?\b"
    r"|\bscream(?:s|ing)?\b"
    r"|\byell(?:s|ing)?\b"
    r"|\bmurmur(?:s|ing)?\b"
    r"|\bnarrat(?:es?|ing|ion)\b"
    r"|\bvocal\b",
    re.IGNORECASE,
)

# Split into clauses at major descriptive boundaries (not bare commas)
_CLAUSE_SPLIT_RE = re.compile(
    r"(?<=\.)\s+"
    r"|(?<=;)\s+"
    r"|,\s+(?=accompanied\s+by\b)"
    r"|,\s+(?=followed\s+by\b)"
    r"|,\s+(?=along\s+with\b)"
    r"|,\s+(?=set\s+against\b)"
    r"|,\s+(?=creating\s)"
    r"|,\s+(?=interspersed\b)",
    re.IGNORECASE,
)

# Inline removal patterns for speech items within a clause (order matters)
_SPEECH_INLINE_RE = re.compile(
    # " and [possessive] [adj,adj] voice/dialogue [verb]" — and-joined speech clause
    r"\s+and\s+(?:[\w'-]+\s+)?[\w,\s-]*?\b(?:voice(?:s)?|dialogue|conversation)\b"
    r"(?:\s+[\w'-]+(?:ing|out)\b)?[^,.]*"
    # ", [and] [the] [0-3 adj] speech_verb [of/from/in X] [rest]" — list item with speech verb
    # (?!voice\b) prevents "voice" from being consumed as a prefix word (it's not an adjective)
    r"|,\s+(?:and\s+)?(?:the\s+)?(?:(?!voice\b)[\w'-]+\s+){0,3}"
    r"\b(?:shouting|whispering|screaming|yelling|murmuring|speaking|talking|saying|narrating|vocal)\b"
    r"(?:\s+(?:of|from|at|in)\s+[\w\s]+?)?[^,.]*(?=[,.]|$)"
    # ", [and] [determiner/possessive] [adj] voice/dialogue [verb]" — list item with voice
    # (requires determiner or possessive to avoid consuming unrelated items)
    r"|,\s+(?:and\s+)?(?:(?:the|a|an|his|her|its|their)\s+|[\w'-]+(?:'s)\s+)"
    r"[\w,\s-]*?\b(?:voice(?:s)?|dialogue|conversation)\b"
    r"(?:\s+[\w'-]+(?:ing|out)\b)?[^,.]*(?=[,.]|$)"
    # Start-of-clause "[possessive] voice [verb], " or " and " — speech before other content
    r"|^(?:[\w'-]+\s+)?[\w,\s-]*?\b(?:voice(?:s)?|dialogue|conversation)\b"
    r"(?:\s+[\w'-]+(?:ing|out)\b)?[^,.]*(?:,\s+|\s+and\s+)"
    # Start-of-clause "speech_verb [of/in X], " — speech verb before other content
    r"|^(?:the\s+)?(?:(?!voice\b)[\w'-]+\s+){0,3}"
    r"\b(?:shouting|whispering|screaming|yelling|murmuring|speaking|talking|saying|narrating|vocal)\b"
    r"(?:\s+(?:of|from|at|in)\s+[\w\s]+?)?[^,.]*(?:,\s+|\s+and\s+)",
    re.IGNORECASE,
)


def filter_speech_from_sound(sound_description: str) -> str | None:
    """Remove speech/dialogue/voice references from a sound description.

    MMAudio is an SFX/ambiance model that produces poor output with speech cues.
    Splits on clause boundaries, removes speech items inline where possible,
    and drops entire clauses when speech is the main subject.

    Returns the filtered description, or None if nothing meaningful remains.
    """
    clauses = _CLAUSE_SPLIT_RE.split(sound_description)
    kept = []
    modified = False

    for clause in clauses:
        if not _SPEECH_RE.search(clause):
            kept.append(clause)
            continue

        modified = True
        # Try removing speech items inline within this clause
        cleaned = _SPEECH_INLINE_RE.sub("", clause)
        if cleaned != clause:
            cleaned = _cleanup_text(cleaned)
            if cleaned and len(cleaned) >= 5 and not _SPEECH_RE.search(cleaned):
                kept.append(cleaned)
        # Otherwise the whole clause is speech-dominated — drop it

    if not kept:
        return None

    # Rejoin: use period-space if previous clause ends with period, else comma
    result = kept[0]
    for clause in kept[1:]:
        if result.endswith("."):
            result += " " + clause
        else:
            result += ", " + clause

    result = _cleanup_text(result)

    if not result or len(result) < 5:
        return None

    # Capitalize first letter when we've changed the start of the text
    if modified and result[0].islower():
        result = result[0].upper() + result[1:]

    return result


def _cleanup_text(text: str) -> str:
    """Clean up artifacts from removed speech segments."""
    text = text.strip()
    text = re.sub(r"^[,;\s]+", "", text)
    text = re.sub(r"[,;\s]+$", "", text)
    text = re.sub(r",\s*,", ",", text)
    text = re.sub(r";\s*,", ";", text)
    text = re.sub(r"\.\s*\.", ".", text)
    text = re.sub(r"\s{2,}", " ", text)
    # Remove orphaned "The sound of" with no subject after it
    text = re.sub(r"^[Tt]he\s+sound\s+of\s*$", "", text)
    return text.strip()


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
            clip_name = manifest.audio_clip_filename(
                shot_index, None if len(durations) == 1 else part_idx + 1, ".mp3")

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
    """MMAudio V2 text-to-audio via fal.ai -- $0.001/sec, max 10s."""

    name = "mmaudio"
    MODEL_ID = "fal-ai/mmaudio-v2/text-to-audio"
    MAX_DURATION = 10
    MIN_DURATION = 5
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
        filtered = filter_speech_from_sound(sound_description)
        if filtered is None:
            print(f"  Shot {shot_index}: sound is entirely speech, skipping MMAudio")
            return []
        if filtered != sound_description:
            print(f"  Shot {shot_index}: filtered speech cues from sound description")
        sound_description = filtered

        import fal_client
        import httpx

        durations = self._target_durations(target_duration_s)
        results = []

        for part_idx, duration in enumerate(durations):
            clip_name = manifest.audio_clip_filename(
                shot_index, None if len(durations) == 1 else part_idx + 1, ".flac")

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
    MAX_DURATION = 10
    MIN_DURATION = 5
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
            self._session.ssh_cmd('pkill -f "main.py" || true', timeout=10)
            time.sleep(2)

            print("  Installing ComfyUI-MMAudio custom nodes...")
            # Use the same Python that ComfyUI runs with for pip install
            pip_cmd = (
                f"if [ -x {COMFYUI_DIR}/.venv/bin/pip ]; then "
                f"  {COMFYUI_DIR}/.venv/bin/pip install -r ComfyUI-MMAudio/requirements.txt; "
                f"else pip install -r ComfyUI-MMAudio/requirements.txt; fi"
            )
            result = self._session.ssh_cmd(
                f"cd {custom_nodes_dir} && "
                f"git clone https://github.com/kijai/ComfyUI-MMAudio && "
                f"{pip_cmd} && echo OK",
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

        # Verify MMAudio nodes actually loaded
        self._verify_mmaudio_nodes()

    def _verify_mmaudio_nodes(self):
        """Check that MMAudio custom nodes loaded in ComfyUI."""
        import httpx

        try:
            resp = httpx.get(
                f"{self._session.base_url}/object_info/MMAudioModelLoader",
                timeout=15,
            )
            if resp.status_code == 200:
                print("  MMAudio nodes: verified")
                return
        except Exception:
            pass

        # Nodes didn't load — dump ComfyUI log for diagnosis
        print("  Error: MMAudio nodes not found in ComfyUI.")
        if self._session.ssh_host:
            result = self._session.ssh_cmd("tail -30 /tmp/comfyui.log", timeout=10)
            if result.stdout:
                print(f"  ComfyUI log:\n{result.stdout}")
        self._session.terminate()
        sys.exit(1)

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
        filtered = filter_speech_from_sound(sound_description)
        if filtered is None:
            print(f"  Shot {shot_index}: sound is entirely speech, skipping MMAudio")
            return []
        if filtered != sound_description:
            print(f"  Shot {shot_index}: filtered speech cues from sound description")
        sound_description = filtered

        self._ensure_pod()

        durations = self._target_durations(target_duration_s)
        results = []

        for part_idx, duration in enumerate(durations):
            clip_name = manifest.audio_clip_filename(
                shot_index, None if len(durations) == 1 else part_idx + 1, ".flac")

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


class ReplicateMusicGenStrategy(AudioStrategy):
    """MusicGen via Replicate API — generates music for music-bucket shots only.

    Reads shot.description.music (set by encode stage) as the prompt.
    Cost: ~$0.002/sec at stereo-large. Max 30s per clip.
    """

    name = "musicgen"
    uses_music_field = True
    MODEL = "meta/musicgen:671ac645ce5e552cc63a54a2bbff63fcf798043055d2dac5fc9e36a837eedcfb"
    MAX_DURATION = 30
    MIN_DURATION = 2
    COST_PER_SECOND = 0.002

    def generate(
        self,
        sound_description: str,
        audio_dir: str,
        shot_index: int,
        target_duration_s: float,
        seed: int | None = None,
    ) -> list[AudioClipResult]:
        import replicate
        import httpx

        duration = int(min(max(target_duration_s, self.MIN_DURATION), self.MAX_DURATION))
        effective_seed = seed if seed is not None else shot_index

        clip_path = manifest.audio_clip_path(audio_dir, shot_index, None, ".wav")

        try:
            output = replicate.run(
                self.MODEL,
                input={
                    "prompt": sound_description,
                    "model_version": "stereo-large",
                    "duration": duration,
                    "output_format": "wav",
                    "seed": effective_seed,
                },
            )
            resp = httpx.get(str(output), follow_redirects=True)
            resp.raise_for_status()
            with open(clip_path, "wb") as f:
                f.write(resp.content)
            return [AudioClipResult(
                path=clip_path,
                actual_duration_s=duration,
                cost=duration * self.COST_PER_SECOND,
            )]
        except Exception as e:
            print(f"  Error generating music for shot {shot_index}: {e}")
            return []


class RunPodMusicGenStrategy(AudioStrategy):
    """MusicGen via self-hosted ComfyUI on RunPod — ~$0/marginal.

    Installs ComfyUI-MusicGen custom nodes on first run.
    Reads shot.description.music as the generation prompt.
    """

    name = "runpod-musicgen"
    uses_music_field = True
    MAX_DURATION = 30
    MIN_DURATION = 2
    GENERATION_TIMEOUT = 180  # 3 min per clip

    MUSICGEN_MODELS = [
        (
            "musicgen/musicgen_large.safetensors",
            "https://huggingface.co/facebookresearch/musicgen-large/resolve/main/musicgen_large.safetensors",
        ),
    ]

    def __init__(self, output_dir: str = ""):
        from runpod_pod import RunPodSession
        self._session = RunPodSession(output_dir)
        self._setup_done = False

    def _target_durations(self, target_s: float) -> list[float]:
        return _split_duration(target_s, self.MIN_DURATION, self.MAX_DURATION)

    def _ensure_pod(self):
        if self._setup_done:
            return
        self._session.ensure_pod()
        self._install_musicgen()
        self._setup_done = True

    def _install_musicgen(self):
        """Install ComfyUI-MusicGen custom nodes and download models."""
        from runpod_pod import COMFYUI_DIR

        if not self._session.ssh_host:
            print("  Warning: No SSH access. Assuming MusicGen is pre-installed.")
            self._session.wait_for_comfyui()
            return

        custom_nodes_dir = f"{COMFYUI_DIR}/custom_nodes"

        check = self._session.ssh_cmd(
            f'[ -d {custom_nodes_dir}/ComfyUI-MusicGen ] && echo exists || echo missing'
        )
        if check.returncode == 0 and "exists" in check.stdout:
            print("  ComfyUI-MusicGen: already installed")
        else:
            self._session.ssh_cmd('pkill -f "main.py" || true', timeout=10)
            time.sleep(2)

            print("  Installing ComfyUI-MusicGen custom nodes...")
            pip_cmd = (
                f"if [ -x {COMFYUI_DIR}/.venv/bin/pip ]; then "
                f"  {COMFYUI_DIR}/.venv/bin/pip install -r ComfyUI-MusicGen/requirements.txt; "
                f"else pip install -r ComfyUI-MusicGen/requirements.txt; fi"
            )
            result = self._session.ssh_cmd(
                f"cd {custom_nodes_dir} && "
                f"git clone https://github.com/GentlemanHu/ComfyUI-MusicGen && "
                f"{pip_cmd} && echo OK",
                timeout=300,
            )
            if result.returncode != 0 or "OK" not in result.stdout:
                print(f"  Error installing MusicGen nodes: {result.stderr[:300]}")
                self._session.terminate()
                sys.exit(1)
            print("  ComfyUI-MusicGen: installed")

        self._session.download_models(self.MUSICGEN_MODELS)
        self._session.restart_comfyui()
        self._verify_musicgen_nodes()

    def _verify_musicgen_nodes(self):
        """Check that MusicGen custom nodes loaded in ComfyUI."""
        import httpx

        try:
            resp = httpx.get(
                f"{self._session.base_url}/object_info/MusicGenNode",
                timeout=15,
            )
            if resp.status_code == 200:
                print("  MusicGen nodes: verified")
                return
        except Exception:
            pass

        print("  Error: MusicGen nodes not found in ComfyUI.")
        if self._session.ssh_host:
            result = self._session.ssh_cmd("tail -30 /tmp/comfyui.log", timeout=10)
            if result.stdout:
                print(f"  ComfyUI log:\n{result.stdout}")
        self._session.terminate()
        sys.exit(1)

    def _build_workflow(self, prompt: str, duration: float, seed: int) -> dict:
        """Build ComfyUI API-format workflow for MusicGen text-to-audio."""
        return {
            "1": {
                "class_type": "MusicGenNode",
                "inputs": {
                    "model": "musicgen_large",
                    "text": prompt,
                    "duration": duration,
                    "seed": seed,
                    "top_k": 250,
                    "top_p": 0.0,
                    "temperature": 1.0,
                    "cfg_coef": 3.0,
                },
            },
            "2": {
                "class_type": "SaveAudio",
                "inputs": {
                    "filename_prefix": "lossy_music",
                    "audio": ["1", 0],
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
            clip_name = manifest.audio_clip_filename(
                shot_index, None if len(durations) == 1 else part_idx + 1, ".wav")

            clip_path = os.path.join(audio_dir, clip_name)
            effective_seed = (seed if seed is not None else shot_index) + part_idx

            try:
                workflow = self._build_workflow(sound_description, duration, effective_seed)
                history = self._session.submit_workflow(workflow, timeout=self.GENERATION_TIMEOUT)
                if not history:
                    print(f"  Failed to generate music for shot {shot_index}")
                    return []

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

                results.append(AudioClipResult(
                    path=clip_path,
                    actual_duration_s=duration,
                    cost=0.0,
                ))

                self._session.free_vram()

            except Exception as e:
                print(f"  Error generating music {clip_name}: {e}")
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

        clip_name = manifest.speech_clip_filename(shot_index, line_index)
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
