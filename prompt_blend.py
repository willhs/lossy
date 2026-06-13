"""Decode-time LLM blending of character identity into shot prompts.

Expansion is a decode concern (decompressing the dictionary+reference codec).
This module makes that expansion smart via Gemini instead of dumb concatenation.

The output (blended_prompts.json) is a derived decode cache — NOT part of the
compressed representation. shots.json and characters.json are never modified.
"""

import hashlib
import json
import os

from prompt_format import _format_prompt_wan


BLEND_SYSTEM_PROMPT = """You are a video generation prompt writer for the Wan T2V model.

You will receive a base shot description and one or more character identity blocks.
Rewrite them into ONE coherent prompt that weaves character appearance and bearing
naturally into the action of the shot.

Requirements:
- Single paragraph, no labels, no "Character:" or "Shot:" prefixes
- Character identity (appearance, clothing, bearing) woven into the action naturally
- Preserve: shot type, camera movement, setting, lighting, color palette, mood
- Target 150-200 words to stay within Wan's T5 encoder sweet spot (~320 tokens)
- Do NOT write separate identity and action paragraphs — synthesize them into one
- Output the prompt text only, nothing else
"""


def _load_env():
    env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
    if os.path.exists(env_path):
        with open(env_path) as ef:
            for line in ef:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    os.environ.setdefault(k.strip(), v.strip())


def _input_hash(base: str, identity_blocks: list[str]) -> str:
    content = base + "|" + "|".join(sorted(identity_blocks))
    return hashlib.sha256(content.encode()).hexdigest()


def build_blended_prompts(
    output_dir: str,
    shots: list[dict],
    characters_data: dict,
    character_shot_map: dict,
) -> dict:
    """Build (or load from cache) blended prompts for shots with assigned characters.

    Returns dict[int, str] mapping shot index to blended prompt.
    Shots without characters are omitted — strategies fall back to base/static-prepend.
    """
    if not character_shot_map or not characters_data:
        return {}

    characters_by_name = {
        c["name"]: c for c in characters_data.get("characters", [])
    }
    shots_by_index = {s["index"]: s for s in shots}

    cache_path = os.path.join(output_dir, "blended_prompts.json")
    cache: dict = {}
    if os.path.exists(cache_path):
        try:
            with open(cache_path) as f:
                cache = json.load(f)
        except (json.JSONDecodeError, OSError):
            cache = {}

    _load_env()
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("Warning: GEMINI_API_KEY not set — skipping blended prompts (static prepend fallback).")
        return {}

    from google import genai  # noqa: PLC0415
    client = genai.Client(api_key=api_key)

    result: dict[int, str] = {}
    cache_dirty = False

    for shot_idx, char_names in character_shot_map.items():
        entry = shots_by_index.get(shot_idx)
        if entry is None:
            continue

        identity_blocks = []
        for name in char_names:
            char = characters_by_name.get(name)
            if char:
                identity_blocks.append(f"{char['display_name']}: {char['description']}")

        if not identity_blocks:
            continue

        base = _format_prompt_wan(entry)
        h = _input_hash(base, identity_blocks)

        cached = cache.get(str(shot_idx))
        if cached and cached.get("hash") == h:
            result[shot_idx] = cached["prompt"]
            continue

        identity_text = "\n".join(identity_blocks)
        user_content = f"{base}\n\nCharacter identities:\n{identity_text}"

        try:
            response = client.models.generate_content(
                model="gemini-2.5-flash-lite",
                contents=user_content,
                config={"system_instruction": BLEND_SYSTEM_PROMPT, "temperature": 0.3},
            )
            blended = (response.text or "").strip()
        except Exception as e:
            print(f"Warning: Gemini blend failed for shot {shot_idx}: {e} — using static prepend fallback.")
            continue

        result[shot_idx] = blended
        cache[str(shot_idx)] = {"hash": h, "prompt": blended}
        cache_dirty = True

    if cache_dirty:
        with open(cache_path, "w") as f:
            json.dump(cache, f, indent=2)

    return result
