"""One-off: generate MusicGen demo clips for a handful of iconic music shots.

Reuses the production encode music-prompt, fal Seedance video strategy, and
Replicate MusicGen audio strategy for specific shot indices, then muxes each
into a self-contained demo clip for the blog's Music section.

Usage: python tools/gen_music_demos.py 81 405 406 670
"""
import json
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from encode import MUSIC_SYSTEM_PROMPT, extract_shot_audio  # noqa: E402
from strategies_video import FalSeedanceStrategy  # noqa: E402
from strategies_audio import ReplicateMusicGenStrategy  # noqa: E402
from config import ENCODE_MODEL, load_env  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "output", "star_wars_iv_v2")
STAGE = os.path.join(OUT, "music_demos")
KEYFRAMES = os.path.join(OUT, "keyframes")
WAV = os.path.join(OUT, "audio.wav")


def shot_keyframes(idx):
    pre = f"{idx:04d}-"
    files = sorted(f for f in os.listdir(KEYFRAMES) if f.startswith(pre) and f.endswith(".jpg"))
    return [os.path.join(KEYFRAMES, f) for f in files]


def describe_music(client, types, shot):
    """Run the production music prompt for one shot -> description dict with 'music'."""
    idx = shot["index"]
    parts = []
    for fpath in shot_keyframes(idx):
        with open(fpath, "rb") as f:
            parts.append(types.Part.from_bytes(data=f.read(), mime_type="image/jpeg"))

    audio_parts = []
    shot_audio = extract_shot_audio(WAV, shot["start_s"], shot["end_s"])
    if shot_audio:
        audio_parts = [types.Part.from_bytes(data=shot_audio, mime_type="audio/wav")]

    context = [
        f"Duration: {shot['duration_s']:.1f}s.",
        f"These are {len(parts)} uniformly-sampled frames from the shot, in chronological order.",
        "The audio for this shot is also provided. Use it to fill the 'music' field.",
        "Analyze the frames and return the JSON description.",
    ]
    user_content = audio_parts + parts + [types.Part.from_text(text="\n".join(context))]
    resp = client.models.generate_content(
        model=ENCODE_MODEL,
        contents=[types.Content(role="user", parts=user_content)],
        config=types.GenerateContentConfig(
            system_instruction=MUSIC_SYSTEM_PROMPT,
            temperature=0.3,
            response_mime_type="application/json",
        ),
    )
    return json.loads(resp.text.strip())


def mux(idx, clip_paths, music_wav, dur):
    os.makedirs(STAGE, exist_ok=True)
    concat_txt = os.path.join(STAGE, f"{idx:04d}_concat.txt")
    with open(concat_txt, "w") as f:
        for c in clip_paths:
            f.write(f"file '{c}'\n")
    out_mp4 = os.path.join(STAGE, f"{idx:04d}.mp4")
    subprocess.run([
        "ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", concat_txt,
        "-i", music_wav, "-map", "0:v:0", "-map", "1:a:0",
        "-c:v", "libx264", "-crf", "26", "-preset", "slow", "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-b:a", "128k", "-movflags", "+faststart", "-shortest",
        out_mp4,
    ], check=True, capture_output=True)
    poster = os.path.join(STAGE, f"{idx:04d}-poster.jpg")
    ss = min(max(dur * 0.6, 1), dur)
    subprocess.run([
        "ffmpeg", "-y", "-ss", str(ss), "-i", out_mp4, "-frames:v", "1", "-q:v", "3", poster,
    ], check=True, capture_output=True)
    return out_mp4, poster


def main():
    load_env()
    from google import genai
    from google.genai import types

    indices = [int(x) for x in sys.argv[1:]] or [81, 405, 406, 670]
    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    shots = {s["index"]: s for s in json.load(open(os.path.join(OUT, "shots.json")))["shots"]}

    vid = FalSeedanceStrategy()
    mus = ReplicateMusicGenStrategy()
    os.makedirs(STAGE, exist_ok=True)
    clips_dir = os.path.join(STAGE, "clips")
    audio_dir = os.path.join(STAGE, "audio")
    os.makedirs(clips_dir, exist_ok=True)
    os.makedirs(audio_dir, exist_ok=True)

    summary = []
    for idx in indices:
        shot = shots[idx]
        dur = shot["duration_s"]
        print(f"\n=== shot {idx} ({dur:.1f}s) — {shot['description'].get('subjects','')[:60]}")

        desc = describe_music(client, types, shot)
        # Keep the original curated visual description for the video; only take the
        # freshly-listened 'music' field (and 'sound' if present) from the music pass.
        merged = dict(shot["description"])
        for k in ("music", "sound"):
            if desc.get(k):
                merged[k] = desc[k]
        entry = dict(shot, description=merged)
        print("  music:", merged.get("music"))

        prompt = vid.format_prompt(entry)
        print("  video prompt:", prompt[:120])
        clips = vid.generate(prompt, clips_dir, idx, dur, seed=idx, entry=entry)
        if not clips:
            print("  !! video generation failed, skipping")
            continue

        music_clips = mus.generate(merged["music"], audio_dir, idx, dur, seed=idx)
        if not music_clips:
            print("  !! music generation failed, skipping")
            continue

        out_mp4, poster = mux(idx, [c.path for c in clips], music_clips[0].path, dur)
        print(f"  -> {out_mp4}")
        summary.append({"index": idx, "music": merged.get("music"),
                        "subjects": merged.get("subjects"), "mp4": out_mp4, "poster": poster})

    with open(os.path.join(STAGE, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print("\nDONE:", json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
