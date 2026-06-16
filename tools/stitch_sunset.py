"""Faithfully stitch shots 403+405+406 the way the lossy decoder (stitch.py) would.

Demonstrates that the decoder treats each shot's music INDEPENDENTLY: per-shot
MusicGen, each padded/trimmed to the shot's original duration, then hard-
concatenated with NO cross-shot crossfade. So a single musical cue spanning the
binary-sunset scene comes back as three unrelated fragments.
"""
import json
import os
import subprocess

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "output", "star_wars_iv_v2")
STAGE = os.path.join(OUT, "music_demos")
SHOTS = {s["index"]: s for s in json.load(open(os.path.join(OUT, "shots.json")))["shots"]}
SEQ = [403, 405, 406]
WORK = os.path.join(STAGE, "sunset_seq")
os.makedirs(WORK, exist_ok=True)


def clips_for(idx):
    cdir = os.path.join(STAGE, "clips")
    single = os.path.join(cdir, f"{idx:04d}.mp4")
    if os.path.exists(single):
        return [single]
    parts = sorted(os.path.join(cdir, f) for f in os.listdir(cdir)
                   if f.startswith(f"{idx:04d}-") and f.endswith(".mp4"))
    return parts


def adjust_video(idx, target):
    """Mirror stitch.py: split clips -> native concat; single clip -> setpts to target."""
    out = os.path.join(WORK, f"{idx:04d}_v.mp4")
    clips = clips_for(idx)
    if len(clips) == 1:
        actual = float(subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "csv=p=0", clips[0]], capture_output=True, text=True).stdout.strip())
        sf = target / actual if actual > 0 else 1.0
        if abs(sf - 1.0) < 0.05:
            subprocess.run(["cp", clips[0], out], check=True, capture_output=True)
        else:
            subprocess.run(["ffmpeg", "-y", "-i", clips[0], "-filter:v",
                            f"setpts={sf}*PTS", "-an", out], check=True, capture_output=True)
    else:
        # split parts: concat natively (speed_factor 1.0 each, per stitch.py)
        lst = os.path.join(WORK, f"{idx:04d}_v.txt")
        with open(lst, "w") as f:
            for c in clips:
                f.write(f"file '{c}'\n")
        subprocess.run(["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", lst,
                        "-an", "-c:v", "libx264", "-crf", "23", "-pix_fmt", "yuv420p", out],
                       check=True, capture_output=True)
    return out


def adjust_audio(idx, target):
    """Mirror stitch.py single-clip path: apad to whole_dur, atrim to target."""
    src = os.path.join(STAGE, "audio", f"{idx:04d}.wav")
    out = os.path.join(WORK, f"{idx:04d}_a.wav")
    subprocess.run(["ffmpeg", "-y", "-i", src, "-af",
                    f"apad=whole_dur={target},atrim=0:{target}", out],
                   check=True, capture_output=True)
    return out


def main():
    vids, auds, total = [], [], 0.0
    for idx in SEQ:
        target = SHOTS[idx]["duration_s"]
        total += target
        vids.append(adjust_video(idx, target))
        auds.append(adjust_audio(idx, target))
        print(f"shot {idx}: padded to {target:.2f}s")

    # Re-encode video segments to a uniform format so concat is clean
    norm = []
    for v in vids:
        n = v.replace("_v.mp4", "_vn.mp4")
        subprocess.run(["ffmpeg", "-y", "-i", v, "-c:v", "libx264", "-crf", "23",
                        "-pix_fmt", "yuv420p", "-r", "24", "-vf", "scale=864:480", "-an", n],
                       check=True, capture_output=True)
        norm.append(n)

    vlist = os.path.join(WORK, "v.txt")
    with open(vlist, "w") as f:
        for v in norm:
            f.write(f"file '{v}'\n")
    vcat = os.path.join(WORK, "video.mp4")
    subprocess.run(["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", vlist,
                    "-c:v", "libx264", "-crf", "23", "-pix_fmt", "yuv420p", vcat],
                   check=True, capture_output=True)

    alist = os.path.join(WORK, "a.txt")
    with open(alist, "w") as f:
        for a in auds:
            f.write(f"file '{a}'\n")
    acat = os.path.join(WORK, "audio.wav")
    subprocess.run(["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", alist, acat],
                   check=True, capture_output=True)

    final = os.path.join(STAGE, "sw-sunset-sequence-musicgen.mp4")
    subprocess.run(["ffmpeg", "-y", "-i", vcat, "-i", acat, "-map", "0:v:0", "-map", "1:a:0",
                    "-c:v", "libx264", "-crf", "26", "-preset", "slow", "-pix_fmt", "yuv420p",
                    "-c:a", "aac", "-b:a", "128k", "-movflags", "+faststart", "-shortest", final],
                   check=True, capture_output=True)
    poster = os.path.join(STAGE, "sw-sunset-sequence-musicgen-poster.jpg")
    subprocess.run(["ffmpeg", "-y", "-ss", "16", "-i", final, "-frames:v", "1", "-q:v", "3", poster],
                   check=True, capture_output=True)
    dur = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                          "-of", "csv=p=0", final], capture_output=True, text=True).stdout.strip()
    print(f"\nFINAL: {final}  ({dur}s, expected ~{total:.1f}s)")


if __name__ == "__main__":
    main()
