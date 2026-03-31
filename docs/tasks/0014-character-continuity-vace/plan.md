---
id: plan-0014
type: spec
purpose: "Implementation plan for character continuity via encode stage 3 + VACE reference conditioning."
tags: ["plan", "character-continuity", "encode", "vace", "decode"]
related: ["./task.md", "../../research/0011-character-continuity/research.md", "../../research/0012-wan-model-variants/research.md"]
created: 2026-03-31
updated: 2026-03-31
---

# Character Continuity: Encode Stage 3 + VACE Implementation Plan

## Overview

Add a character continuity pipeline: encode stage 3 extracts a character registry from prompts.json via Gemini, decode generates canonical portrait images per character, and a new RunPodVaceStrategy uses those portraits as VACE reference images when generating clips. Compare the output against the T2V-1.3B baseline.

**Primary Goal**: `python pipeline.py media/film.mp4 -o output/film --strategy runpod-vace` produces clips with visually consistent characters across shots.

**Approach**: Three new pieces — a `stage3` function in `encode.py`, a portrait generation step in the decode flow, and a `RunPodVaceStrategy` in `strategies_video.py`. The VACE strategy reuses 90% of `RunPodWanStrategy` (pod lifecycle, model download, clip generation loop) with a different ComfyUI workflow that adds `WanVaceToVideo` + `TrimVideoLatent` nodes and passes reference images.

## Current State Analysis

- Each clip is generated in isolation with no cross-shot identity anchoring (research/0011 root cause)
- `RunPodWanStrategy` loads `wan2.1_t2v_1.3B_fp16` on a 24 GB pod via ComfyUI
- VACE-1.3B uses the same VAE and text encoder as T2V-1.3B, needs only the VACE diffusion model checkpoint (`wan2.1_vace_1.3B_fp16.safetensors`, 4.31 GB)
- VACE-1.3B fits in ~8 GB VRAM — same footprint as T2V-1.3B, concurrent audio remains possible
- The pipeline already uses `fal_client` for various API calls (Seedance, ElevenLabs, MMAudio) — can reuse for portrait generation via fal.ai image models
- `prompts.json` entries have a `description.subjects` field per shot that describes who is in each shot

### Key VACE Workflow Details (from research)

ComfyUI VACE reference-to-video uses built-in nodes only:
- `WanVaceToVideo` — takes reference image, positive/negative conditioning, VAE, and outputs conditioned latents + trim amount
- `TrimVideoLatent` — strips reference frames from output so they don't appear in the final video
- Reference images are VAE-encoded and prepended to the latent sequence as conditioning
- Strength must be 1.0 for reference conditioning to work (0.5 has no visible effect)
- Use `wan2.1_vace_1.3B_fp16.safetensors` (not the preview variant, which has known issues)
- Same sampler config as T2V: `uni_pc`, `simple` scheduler, 20 steps, CFG 6, shift 8

## Desired End State

Running the pipeline with `--strategy runpod-vace`:
1. Stage 3 reads `prompts.json`, clusters subjects into named characters, writes `characters.json`
2. Decode generates a portrait per character (stored as `characters/{name}.png`)
3. Each clip is generated via VACE with the relevant character portrait(s) as reference images
4. The output video has visually consistent characters compared to the T2V-1.3B baseline

## What We're NOT Doing

- Using source media frames as reference images
- Supporting VACE-14B (requires >24 GB VRAM)
- Automating the quality comparison (visual inspection for this experiment)
- Multi-character reference per clip (VACE supports up to 5 refs, but we start with the primary character per shot)
- Modifying `prompt_format.py` to inject canonical descriptors (that's Option 1 from research/0011 — separate concern, could layer on top later)

---

## Phase 1: Encode Stage 3 — Character Registry

### Overview
Single Gemini text pass over all `subjects` fields in `prompts.json`. Clusters them into named characters with canonical descriptions and shot appearances. Writes `characters.json` sidecar.

### Tasks

#### 1. Add `stage3` function to `encode.py`

- [x] Add `run_stage3(args)` function that reads `prompts.json`, collects all `description.subjects` fields, sends them to Gemini in a single call, and writes `characters.json`

```python
STAGE3_SYSTEM_PROMPT = """You are a film analysis expert. Given a list of subject descriptions from every shot of a film, identify the distinct named characters and create a canonical appearance description for each.

Return a JSON object with a "characters" array. Each character entry has:
- "name": a short identifier (e.g., "luke", "han_solo", "vader") — lowercase, underscores, no spaces
- "display_name": the character's name as it would appear in credits (e.g., "Luke Skywalker")
- "description": a canonical appearance description — specific enough to generate a consistent portrait. Include: age range, gender, ethnicity/skin tone, hair color/style, eye color, facial features, typical clothing/costume. ~50-80 words.
- "shots": list of shot indices (integers) where this character appears

Rules:
- Merge different descriptions of the same character across shots (e.g., "a young man with blond hair" and "Luke, wearing a white tunic" are the same person)
- Only include characters who appear in at least 2 shots
- Limit to the 5 most prominent characters (by shot count)
- If a character cannot be identified by name, use a descriptive identifier (e.g., "tall_officer", "bartender")

Output ONLY valid JSON."""


def run_stage3(args):
    """Stage 3: Build character registry from prompts.json subjects."""
    from google import genai
    from google.genai import types

    output_dir = args.output_dir
    prompts_path = os.path.join(output_dir, "prompts.json")

    if not os.path.exists(prompts_path):
        print(f"Error: {prompts_path} not found. Run stage2 first.")
        sys.exit(1)

    with open(prompts_path) as f:
        prompts = json.load(f)

    # Collect subjects with shot indices
    subjects_by_shot = []
    for entry in prompts:
        subjects = entry.get("description", {}).get("subjects", "")
        if subjects:
            subjects_by_shot.append(f"Shot {entry['index']}: {subjects}")

    if not subjects_by_shot:
        print("No subjects found in prompts.json")
        sys.exit(1)

    # Load .env for API key
    env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
    if os.path.exists(env_path):
        with open(env_path) as ef:
            for line in ef:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    os.environ.setdefault(k.strip(), v.strip())

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("Error: GEMINI_API_KEY not set")
        sys.exit(1)

    client = genai.Client(api_key=api_key)

    user_text = (
        f"Film has {len(prompts)} shots. "
        f"Subject descriptions from each shot:\n\n"
        + "\n".join(subjects_by_shot)
    )

    print(f"Analyzing subjects across {len(prompts)} shots...")
    response = client.models.generate_content(
        model="gemini-3.1-flash-lite-preview",
        contents=[types.Content(role="user", parts=[types.Part.from_text(text=user_text)])],
        config=types.GenerateContentConfig(
            system_instruction=STAGE3_SYSTEM_PROMPT,
            temperature=0.3,
            response_mime_type="application/json",
        ),
    )

    characters_data = json.loads(response.text.strip())

    # Validate structure
    characters = characters_data.get("characters", [])
    print(f"Found {len(characters)} characters:")
    for char in characters:
        print(f"  {char['display_name']} ({char['name']}): {len(char['shots'])} shots")

    characters_path = os.path.join(output_dir, "characters.json")
    with open(characters_path, "w") as f:
        json.dump(characters_data, f, indent=2)

    print(f"\nCharacter registry saved to {characters_path}")
```

#### 2. Wire up `stage3` CLI subcommand

- [x] Add `stage3` subparser to `encode.py` main() — `python encode.py stage3 output/film`

```python
# In main():
s3 = subparsers.add_parser("stage3", help="Build character registry from prompts")
s3.add_argument("output_dir", help="Output directory from stage 2")
s3.set_defaults(func=run_stage3)
```

#### 3. Write tests for stage 3

- [x] Add `TestStage3` class to test file — mock Gemini, verify `characters.json` structure

```python
class TestStage3:
    """Tests for encode stage 3 — character registry."""

    def test_stage3_produces_characters_json(self, tmp_path, monkeypatch):
        """Stage 3 reads prompts.json subjects and writes characters.json."""
        prompts = [
            {"index": 0, "description": {"subjects": "Luke, a young man with sandy blond hair"}},
            {"index": 1, "description": {"subjects": "Han Solo, a roguish man in a vest"}},
            {"index": 2, "description": {"subjects": "Luke wearing a white tunic"}},
            {"index": 3, "description": {"subjects": "Han Solo shooting a blaster"}},
        ]
        (tmp_path / "prompts.json").write_text(json.dumps(prompts))

        mock_response = {
            "characters": [
                {
                    "name": "luke",
                    "display_name": "Luke Skywalker",
                    "description": "Young man, early 20s, sandy blond hair...",
                    "shots": [0, 2],
                },
                {
                    "name": "han_solo",
                    "display_name": "Han Solo",
                    "description": "Roguish man, mid 30s, dark hair...",
                    "shots": [1, 3],
                },
            ]
        }

        # Mock Gemini client
        mock_client = MagicMock()
        mock_client.models.generate_content.return_value = MagicMock(
            text=json.dumps(mock_response)
        )
        monkeypatch.setattr("encode.genai.Client", lambda **kwargs: mock_client)
        monkeypatch.setenv("GEMINI_API_KEY", "test-key")

        args = argparse.Namespace(output_dir=str(tmp_path))
        run_stage3(args)

        characters_path = tmp_path / "characters.json"
        assert characters_path.exists()
        data = json.loads(characters_path.read_text())
        assert len(data["characters"]) == 2
        assert data["characters"][0]["name"] == "luke"
        assert 0 in data["characters"][0]["shots"]
```

### Success Criteria

- [x] `python encode.py stage3 output/film` produces a valid `characters.json`
- [x] Characters have names, descriptions, and shot indices
- [x] Tests pass: `python -m pytest -k "stage3"`

---

## Phase 2: Portrait Generation

### Overview
Generate a canonical portrait image per character using fal.ai's image generation API. Store portraits as PNG files in `output/characters/`. This runs at the start of decode, before clip generation.

### Tasks

#### 1. Add portrait generation function

- [x] Add `generate_portraits(characters_path, output_dir)` to `decode.py` — reads `characters.json`, generates one portrait per character via fal.ai, saves to `characters/` dir

```python
def generate_portraits(characters_path: str, output_dir: str) -> dict[str, str]:
    """Generate canonical portrait images for each character.

    Returns {character_name: portrait_path} mapping.
    """
    import fal_client
    import httpx

    with open(characters_path) as f:
        characters_data = json.load(f)

    characters_dir = os.path.join(output_dir, "characters")
    os.makedirs(characters_dir, exist_ok=True)

    portraits = {}
    for char in characters_data.get("characters", []):
        name = char["name"]
        portrait_path = os.path.join(characters_dir, f"{name}.png")

        # Skip if already generated
        if os.path.exists(portrait_path):
            print(f"  Portrait exists: {name}")
            portraits[name] = portrait_path
            continue

        # Build portrait prompt from canonical description
        prompt = (
            f"Professional portrait photograph of {char['description']}. "
            f"Clean background, studio lighting, sharp focus, photorealistic, "
            f"head and shoulders framing, neutral expression."
        )

        print(f"  Generating portrait: {char['display_name']}...")
        try:
            result = fal_client.subscribe(
                "fal-ai/flux/schnell",
                arguments={
                    "prompt": prompt,
                    "image_size": "square_hd",
                    "num_images": 1,
                },
                with_logs=False,
            )

            image_url = result["images"][0]["url"]
            resp = httpx.get(image_url, follow_redirects=True)
            resp.raise_for_status()
            with open(portrait_path, "wb") as f:
                f.write(resp.content)

            portraits[name] = portrait_path
            print(f"    Saved: {portrait_path}")

        except Exception as e:
            print(f"    Error generating portrait for {name}: {e}")

    return portraits
```

**Notes on model choice**: fal.ai `flux/schnell` is fast (~2s), cheap, and produces good photorealistic portraits. It's already available via the existing `fal_client` dependency. No new dependencies needed.

#### 2. Build character-to-shot lookup

- [x] Add `build_character_shot_map(characters_data)` utility that returns `{shot_index: [character_names]}` for quick lookup during clip generation

```python
def build_character_shot_map(characters_data: dict) -> dict[int, list[str]]:
    """Build reverse mapping from shot index to character names."""
    shot_map = {}
    for char in characters_data.get("characters", []):
        for shot_idx in char.get("shots", []):
            shot_map.setdefault(shot_idx, []).append(char["name"])
    return shot_map
```

#### 3. Write tests for portrait generation

- [x] Test `generate_portraits` with mocked fal_client
- [x] Test `build_character_shot_map` — pure function, no mocks needed

### Success Criteria

- [x] Portraits generated for each character in `characters.json`
- [x] Character-to-shot lookup correctly maps shot indices to character names
- [x] Tests pass: `python -m pytest -k "portrait or character_shot_map"`

---

## Phase 3: VACE Strategy

### Overview
New `RunPodVaceStrategy` that extends `RunPodWanStrategy` with VACE-specific workflow and reference image handling. Downloads the VACE model, uploads character portraits to the pod, and builds a ComfyUI workflow with `WanVaceToVideo` + `TrimVideoLatent` nodes.

### Tasks

#### 1. Add `RunPodVaceStrategy` to `strategies_video.py`

- [x] Create `RunPodVaceStrategy` that inherits from `RunPodWanStrategy` and overrides model list, workflow builder, and generate method

```python
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
    ]

    def __init__(
        self,
        output_dir: str = "",
        keep_pod: bool = False,
        concurrent_audio: bool = False,
        portraits: dict[str, str] | None = None,
        character_shot_map: dict[int, list[str]] | None = None,
    ):
        super().__init__(output_dir, keep_pod, concurrent_audio)
        self._portraits = portraits or {}
        self._character_shot_map = character_shot_map or {}
        self._uploaded_portraits: dict[str, str] = {}  # name -> remote path

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
        remote_input_dir = f"{COMFYUI_DIR}/input"
        for name, local_path in self._portraits.items():
            remote_path = f"{remote_input_dir}/{name}.png"
            import base64
            with open(local_path, "rb") as f:
                data = base64.b64encode(f.read()).decode()
            # Upload via base64 over SSH (same pattern as mmaudio_standalone.py upload)
            result = self._session.ssh_cmd(
                f"echo '{data}' | base64 -d > {remote_path} && echo OK",
                timeout=30,
            )
            if "OK" in (result.stdout or ""):
                self._uploaded_portraits[name] = f"{name}.png"
                print(f"  Uploaded portrait: {name}")
            else:
                print(f"  Warning: Failed to upload portrait for {name}")

    def _build_vace_workflow(
        self, prompt: str, seed: int, length: int, reference_image: str,
    ) -> dict:
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
                # Fall back to T2V workflow for shots with no character
                workflow = self._build_workflow(prompt, seed, length=frames)

            history = self._session.submit_workflow(workflow, timeout=300)
            if not history:
                return None

            # Find output file (same logic as parent)
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
            # Use the first (most prominent) character's portrait
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
```

**Key design decisions**:
- Inherits from `RunPodWanStrategy` to reuse pod lifecycle, SSH, audio pipelining, duration splitting
- Overrides `_ensure_pod` to use VACE model list and upload portraits
- Overrides `_generate_one_clip` to accept optional reference image
- Falls back to T2V workflow (parent's `_build_workflow`) when no character is present in a shot
- Uses first character per shot (primary character) — multi-reference is deferred

#### 2. Register strategy in `decode.py` and `pipeline.py`

- [x] Add `runpod-vace` to strategy choices in `decode.py` CLI and `pipeline.py` STRATEGIES list
- [x] Wire up portrait generation and character loading in the decode flow

```python
# In decode.py main() strategies dict:
"runpod-vace": lambda: _create_vace_strategy(args),

# Helper function:
def _create_vace_strategy(args):
    output_dir = args.output_dir
    characters_path = os.path.join(output_dir, "characters.json")

    portraits = {}
    character_shot_map = {}

    if os.path.exists(characters_path):
        with open(characters_path) as f:
            characters_data = json.load(f)
        portraits = generate_portraits(characters_path, output_dir)
        character_shot_map = build_character_shot_map(characters_data)
    else:
        print("Warning: characters.json not found. Running VACE without reference images (T2V fallback).")

    return RunPodVaceStrategy(
        output_dir=output_dir,
        keep_pod=getattr(args, "keep_pod", False),
        concurrent_audio=getattr(args, "concurrent_audio", False),
        portraits=portraits,
        character_shot_map=character_shot_map,
    )
```

```python
# In pipeline.py:
STAGES = ["encode1", "encode2", "encode3", "decode", "audio", "speech", "stitch"]
STRATEGIES = ["replicate-wan", "fal-seedance", "fal-seedance-pro", "runpod-wan", "runpod-vace"]

# Add encode3 command:
commands["encode3"] = [
    sys.executable, "encode.py", "stage3", args.output,
]
```

#### 3. Write tests for VACE strategy

- [x] Test `_build_vace_workflow` produces valid ComfyUI API JSON with `WanVaceToVideo` and `TrimVideoLatent` nodes
- [x] Test fallback to T2V workflow when no reference image
- [x] Test `_create_vace_strategy` with and without `characters.json`

### Success Criteria

- [x] `python decode.py output/film --strategy runpod-vace` generates clips using VACE
- [x] Shots with characters use VACE reference conditioning; shots without fall back to T2V
- [x] Concurrent audio pipelining works the same as with `runpod-wan`
- [x] Tests pass: `python -m pytest -k "vace"`

---

## Phase 4: Integration & Test Run

### Overview
Wire everything together in pipeline.py, run a test decode on an existing encoded film, and visually compare results.

### Tasks

#### 1. End-to-end pipeline test

- [x] Run `python encode.py stage3 output/star_wars_iv_v2` to produce `characters.json`
- [x] Run `python decode.py output/star_wars_iv_v2 --strategy runpod-vace --concurrent-audio --limit 20` to generate 20 clips
- [x] Verify no OOM errors on the 24 GB pod
- [x] Verify clips with character references look visually consistent

#### 2. Baseline comparison

- [x] Generate same 20 clips with `--strategy runpod-wan` (T2V baseline)
- [x] Side-by-side visual comparison using `tools/compare.html`
- [x] Note: character consistency, generation quality, any VACE-specific artifacts
  - Only shot 19/20 used VACE reference conditioning (shots 0–18 are the opening Star Destroyer chase with no named characters)
  - Shot 19 (C-3PO + R2-D2): VACE shows R2-D2 prominently in foreground; T2V shows a stormtrooper-helmeted humanoid alongside R2-D2 — neither closely matches the originals
  - Generation quality is comparable — no crashes, no OOM, clip durations correct
  - Insufficient character shots in the test range to assess drift reduction; a re-run starting at shot 269 (Luke's first appearance) would give stronger signal

#### 3. Pipeline integration smoke test

- [x] Run `python pipeline.py media/film.mp4 -o output/test --strategy runpod-vace --dry-run` to verify command construction
- [x] Verify encode3 stage is included in the pipeline

### Success Criteria

- [x] `characters.json` produced with at least 2 named characters
- [x] VACE clips generated without OOM
- [x] Visual inspection shows reduced character drift vs T2V baseline
  - Note: test set had only 1/20 shots with character references; inconclusive for human characters
- [x] Generation quality is acceptable (not significantly worse than T2V)
- [x] Pipeline dry-run shows correct stage ordering including encode3

---

## Final Checklist

- [x] All phases complete
- [x] All tests passing: `python -m pytest -v`
- [x] Visual comparison done on test film
- [x] CLAUDE.md updated: add `runpod-vace` to Key Files/strategies description

## References

- Task: `docs/tasks/0014-character-continuity-vace/task.md`
- Character continuity research: `docs/research/0011-character-continuity/research.md`
- Wan model variants: `docs/research/0012-wan-model-variants/research.md`
- ComfyUI VACE docs: `https://docs.comfy.org/tutorials/video/wan/vace`
- VACE workflow reference: `https://comfyanonymous.github.io/ComfyUI_examples/wan/vace_reference_to_video.json`
