#!/usr/bin/env python3
"""Generate a short film-style scene from a written description, using Wan 2.2 5B
on fal.ai, with last-frame -> first-frame chaining for graceful long shots.

Self-contained: needs only `fal-client` (pip install fal-client), `ffmpeg`, and a
FAL_KEY env var. Works from any directory.

Usage:
    FAL_KEY=... python3 generate.py scene.json
    python3 generate.py scene.json --env /path/to/.env   # load FAL_KEY from a .env

Scene spec (JSON):
{
  "output": "scene.mp4",
  "resolution": "580p",            # 580p | 720p   (optional, default 580p)
  "aspect_ratio": "16:9",          # optional
  "num_frames": 121,               # ~5s @24fps   (optional)
  "fps": 24,                       # optional
  "seed": 100,                     # base seed; +1 per clip (optional)
  "negative": "text, watermark, blurry, distorted, deformed, duplicated",
  "shots": [
    {"prompt": "Wide shot of ...", "continue": ["...the action continues..."]},
    {"prompt": "Close-up of ..."}
  ],
  "audio": {"prompt": "soft ambience ...", "volume": 0.4}   # optional MMAudio bed
}

Each shot's first prompt is text->video. Each string in "continue" is an
image->video clip seeded with the previous clip's LAST frame (the chaining trick),
so a long shot holds together across the cut.
"""
import argparse, json, os, subprocess, tempfile, urllib.request, sys

T2V = "fal-ai/wan/v2.2-5b/text-to-video"
I2V = "fal-ai/wan/v2.2-5b/image-to-video"
MM = "fal-ai/mmaudio-v2/text-to-audio"
DEFAULT_NEG = "text, watermark, subtitles, logo, blurry, distorted, deformed, duplicated, extra limbs"


def load_env(path):
    for line in open(path):
        line = line.strip()
        if "=" in line and not line.startswith("#"):
            k, v = line.split("=", 1)
            os.environ.setdefault(k, v.strip().strip('"').strip("'"))


def ffprobe_dur(path):
    out = subprocess.check_output(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", path])
    return float(out.strip())


def main():
    ap = argparse.ArgumentParser(description="Generate a chained scene via Wan 2.2 5B on fal.")
    ap.add_argument("spec", help="Path to scene JSON spec")
    ap.add_argument("--env", help="Path to a .env file to load FAL_KEY from")
    ap.add_argument("--workdir", help="Where to keep intermediate clips (default: temp; reused if present)")
    args = ap.parse_args()

    if args.env:
        load_env(args.env)
    if not os.environ.get("FAL_KEY"):
        sys.exit("FAL_KEY not set. Export it or pass --env path/to/.env")

    import fal_client

    spec = json.load(open(args.spec))
    out = spec["output"]
    common = dict(
        num_frames=spec.get("num_frames", 121),
        frames_per_second=spec.get("fps", 24),
        resolution=spec.get("resolution", "580p"),
    )
    neg = spec.get("negative", DEFAULT_NEG)
    seed = spec.get("seed", 100)
    work = args.workdir or tempfile.mkdtemp(prefix="scene-")
    os.makedirs(work, exist_ok=True)
    print(f"workdir: {work}")

    def gen(model, a, path):
        if os.path.exists(path):
            print("  skip (exists)", path); return path
        r = fal_client.subscribe(model, arguments=a, with_logs=False)
        url = r["video"]["url"] if isinstance(r.get("video"), dict) else r["video"]
        urllib.request.urlretrieve(url, path)
        print("  saved", path)
        return path

    clips = []
    idx = 0
    for si, shot in enumerate(spec["shots"]):
        idx += 1
        a = os.path.join(work, f"c{idx:02d}.mp4")
        print(f"shot {si+1} clip {idx} (T2V) seed={seed}")
        gen(T2V, dict(prompt=shot["prompt"], negative_prompt=neg, seed=seed,
                      aspect_ratio=spec.get("aspect_ratio", "16:9"), **common), a)
        clips.append(a); seed += 1
        prev = a
        for cont in shot.get("continue", []):
            idx += 1
            b = os.path.join(work, f"c{idx:02d}.mp4")
            if not os.path.exists(b):
                frame = os.path.join(work, f"c{idx-1:02d}_last.jpg")
                subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-sseof", "-0.1",
                                "-i", prev, "-frames:v", "1", frame], check=True)
                img_url = fal_client.upload_file(frame)
                print(f"shot {si+1} clip {idx} (I2V chained) seed={seed}")
                gen(I2V, dict(prompt=cont, negative_prompt=neg, seed=seed, image_url=img_url,
                              aspect_ratio="auto", **common), b)
            clips.append(b); seed += 1; prev = b

    # concat all clips -> silent mp4
    silent = os.path.join(work, "_silent.mp4")
    inputs = []
    for c in clips:
        inputs += ["-i", c]
    fc = "".join(f"[{i}:v]" for i in range(len(clips))) + f"concat=n={len(clips)}:v=1:a=0,format=yuv420p[v]"
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", *inputs,
                    "-filter_complex", fc, "-map", "[v]",
                    "-c:v", "libx264", "-crf", "28", "-preset", "slow",
                    "-movflags", "+faststart", "-an", silent], check=True)
    dur = ffprobe_dur(silent)

    audio = spec.get("audio")
    if audio:
        print(f"generating MMAudio bed: {audio['prompt'][:60]}...")
        amb = os.path.join(work, "_amb.mp3")
        r = fal_client.subscribe(MM, arguments={"prompt": audio["prompt"], "duration": 10}, with_logs=False)
        url = r["audio"]["url"] if isinstance(r.get("audio"), dict) else r["audio"]
        urllib.request.urlretrieve(url, amb)
        vol = audio.get("volume", 0.4)
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error",
                        "-i", silent, "-stream_loop", "-1", "-i", amb,
                        "-filter_complex",
                        f"[1:a]aresample=44100,aformat=channel_layouts=stereo,volume={vol},alimiter=limit=0.95[a]",
                        "-map", "0:v", "-map", "[a]", "-c:v", "copy", "-c:a", "aac", "-b:a", "128k",
                        "-t", str(dur), "-movflags", "+faststart", out], check=True)
    else:
        subprocess.run(["cp", silent, out], check=True)

    print(f"\nDONE -> {out}  ({dur:.1f}s, {len(clips)} clips)")


if __name__ == "__main__":
    main()
