#!/usr/bin/env python3
"""
voice_casting.py: propose a per-character ElevenLabs voice map from characters.json.

Casting is Will's call, not this script's — it PROPOSES a default mapping
with reasoning and (optionally) renders a short sample per character so it
can be judged by ear, then writes a plain, hand-editable voice_map.json.
Re-casting later is just editing that file; nothing here is authoritative.

Usage:
    python voice_casting.py output/star_wars_iv
    python voice_casting.py output/star_wars_iv --sample   # + one short clip per voice
"""

import argparse
import json
import os
import sys

import manifest
from config import ENCODE_MODEL, load_env

# A small, known-good set of fal.ai ElevenLabs Turbo v2.5 premade voice names.
# This is the "data-driven, trivially editable" voice bank the task asks for —
# add/remove/rename entries freely; nothing else in the pipeline hardcodes
# voice names except this list and the "Roger" CLI default.
CURATED_VOICE_BANK = {
    "Roger": "adult male, warm, measured — current default/narrator voice",
    "Adam": "adult male, deep, authoritative",
    "Charlie": "young adult male, casual, energetic",
    "George": "middle-aged male, calm, gravelly",
    "Callum": "adult male, intense, gritty",
    "Sarah": "adult female, soft, professional",
    "Alice": "adult female, crisp, confident, British accent",
    "Matilda": "young adult female, warm, friendly",
    "Laura": "adult female, upbeat, expressive",
    "Charlotte": "adult female, sultry, low",
}

VOICE_CASTING_SYSTEM_PROMPT = """You are a casting director for an audiobook/film dub. You are given a curated bank of available TTS voices (with a short tone/age/gender description each) and a character registry (with each character's canonical visual description).

For each character, pick the single best-fitting voice from the bank, and give a one-sentence reason. Also pick a "narrator" voice for dialogue lines that can't be attributed to a specific character.

Return a JSON object: {"casting": {"<character_name>": {"voice": "<bank name>", "reasoning": "..."}, ...}, "narrator": {"voice": "<bank name>", "reasoning": "..."}}

Rules:
- Only use voice names from the provided bank — do not invent new ones.
- Different characters should get different voices where the bank allows it; only repeat a voice if you run out of good fits.
- Base the pick on the character's description (age, gender, tone implied by role), not the character's name.

Output ONLY valid JSON."""


def propose_casting(client, characters: list[dict]) -> dict:
    from google.genai import types

    bank_text = "\n".join(f"- {name}: {desc}" for name, desc in CURATED_VOICE_BANK.items())
    registry_text = "\n".join(
        f"- {c['name']} ({c['display_name']}): {c['description']}" for c in characters
    )
    user_text = f"Voice bank:\n{bank_text}\n\nCharacter registry:\n{registry_text}"

    response = client.models.generate_content(
        model=ENCODE_MODEL,
        contents=[types.Content(role="user", parts=[types.Part.from_text(text=user_text)])],
        config=types.GenerateContentConfig(
            system_instruction=VOICE_CASTING_SYSTEM_PROMPT,
            temperature=0.4,
            response_mime_type="application/json",
        ),
    )
    return json.loads(response.text.strip())


def _sample_line_for(character_name: str, output_dir: str) -> str:
    """First real attributed line for this character (speakers.json), else a generic sample."""
    speakers = manifest.load_speakers(output_dir)
    if speakers:
        try:
            _, dialog = manifest.load_shots(output_dir)
        except (FileNotFoundError, ValueError):
            dialog = []
        for idx, speaker in sorted(speakers.items()):
            if speaker == character_name and idx < len(dialog):
                return dialog[idx]["text"]
    return f"Hello, this is a sample of my voice for {character_name.replace('_', ' ')}."


def main():
    parser = argparse.ArgumentParser(description="Propose a per-character ElevenLabs voice map")
    parser.add_argument("output_dir", help="Output directory containing characters.json")
    parser.add_argument("--sample", action="store_true",
                         help="Generate one short ElevenLabs clip per proposed voice")
    args = parser.parse_args()

    load_env()
    output_dir = args.output_dir

    characters_file = manifest.characters_path(output_dir)
    if not os.path.exists(characters_file):
        print(f"Error: {characters_file} not found. Run `encode.py stage3` first.")
        sys.exit(1)

    with open(characters_file) as f:
        characters = json.load(f).get("characters", [])
    if not characters:
        print("No characters in characters.json; nothing to cast.")
        sys.exit(0)

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("Error: GEMINI_API_KEY not set")
        sys.exit(1)

    from google import genai
    client = genai.Client(api_key=api_key)

    print(f"Proposing casting for {len(characters)} characters against a "
          f"{len(CURATED_VOICE_BANK)}-voice bank...")
    proposal = propose_casting(client, characters)
    casting = proposal.get("casting", {})
    narrator = proposal.get("narrator", {"voice": "Roger", "reasoning": "default"})

    voice_map = {name: entry["voice"] for name, entry in casting.items()
                 if entry.get("voice") in CURATED_VOICE_BANK}
    voice_map["narrator"] = narrator["voice"] if narrator.get("voice") in CURATED_VOICE_BANK else "Roger"

    voice_map_file = manifest.voice_map_path(output_dir)
    with open(voice_map_file, "w") as f:
        json.dump(voice_map, f, indent=2)
    print(f"\nProposed voice map saved to {voice_map_file} "
          f"(PROPOSAL ONLY — edit freely, this is not a final casting call)")

    casting_md_path = os.path.join(output_dir, "voice_casting.md")
    with open(casting_md_path, "w") as f:
        f.write("# Proposed voice casting (NOT FINAL — edit voice_map.json to re-cast)\n\n")
        for char in characters:
            entry = casting.get(char["name"])
            if not entry:
                continue
            f.write(f"- **{char['display_name']}** (`{char['name']}`) -> "
                    f"**{entry['voice']}** — {entry.get('reasoning', '')}\n")
        f.write(f"- **narrator** (unattributed lines) -> **{narrator['voice']}** — "
                f"{narrator.get('reasoning', '')}\n")
    print(f"Reasoning written to {casting_md_path}")

    if args.sample:
        from strategies_audio import SpeechStrategy

        samples_dir = os.path.join(output_dir, "voice_samples")
        os.makedirs(samples_dir, exist_ok=True)
        total_cost = 0.0

        for i, (name, voice) in enumerate(voice_map.items()):
            text = (_sample_line_for(name, output_dir) if name != "narrator"
                    else "This is the narrator voice for unattributed dialogue lines.")
            strategy = SpeechStrategy(voice=voice)
            result = strategy.generate(text, samples_dir, shot_index=9000 + i,
                                        line_index=0, offset_s=0.0)
            if result:
                final_path = os.path.join(samples_dir, f"{name}.mp3")
                os.replace(result.path, final_path)
                total_cost += result.cost
                print(f"  {name} ({voice}): {final_path}")
            else:
                print(f"  {name} ({voice}): generation failed")

        print(f"\nSample generation cost: ${total_cost:.4f}")


if __name__ == "__main__":
    main()
