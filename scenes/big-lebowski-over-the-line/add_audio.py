#!/usr/bin/env python3
"""Add ambience + dialogue audio to both renders of the scene (Wan/fal and LTX-2).

Ambience: one MMAudio bed via fal.ai (fal-ai/mmaudio-v2/text-to-audio), looped
under the whole scene -- same pattern the generate-scene skill's own "audio"
field uses.

Dialogue: a handful of short lines via fal's ElevenLabs Turbo v2.5 TTS
(fal-ai/elevenlabs/tts/turbo-v2.5), one voice per speaking character, cast
from lossy's curated voice bank (voice_casting.py). Muxed in at each line's
clip-start offset, computed separately per video since Wan/fal and LTX-2
clips run different lengths. Not lip-synced -- neither video was generated
with mouth movement matched to these specific words -- this is a soundtrack
pass, the same tradeoff the generate-scene skill documents for dialogue
scenes ("generate lines separately and mux by hand").

Usage:
    cd lossy
    .venv/bin/python3 scenes/big-lebowski-over-the-line/add_audio.py
"""
import os
import subprocess
import sys
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))


def load_env(path):
    for line in open(path):
        line = line.strip()
        if "=" in line and not line.startswith("#"):
            k, v = line.split("=", 1)
            os.environ.setdefault(k, v.strip().strip('"').strip("'"))


load_env(os.path.join(ROOT, ".env"))
sys.path.insert(0, ROOT)

AUDIO_DIR = os.path.join(HERE, "audio")
os.makedirs(AUDIO_DIR, exist_ok=True)

AMBIENCE_PROMPT = (
    "distant retro bowling alley ambience, pins crashing occasionally, "
    "muffled crowd chatter, bowling balls rolling on wooden lanes, "
    "electromechanical pinsetter hum, no music, no speech"
)

# (clip_index 0-13, delay-after-clip-start seconds, voice, text)
# clip order: 0 foul, 1 rise, 2 draw (chained), 3 close-up shout, 4 calming, 5 bewildered,
# 6 standoff, 7 cops reaction, 8 insisting close-up, 9 smokey marks pt1, 10 smokey marks pt2
# (chained), 11 walter lowers gun, 12 dude exasperated, 13 closing wide
DIALOGUE = [
    (1, 0.3, "Callum", "Over the line!"),
    (3, 0.3, "Callum", "Smokey, you're entering a world of pain."),
    (4, 0.3, "George", "Walter, ya can't do that, man."),
    (6, 0.3, "Roger", "Okay, okay! It's just a game!"),
    (7, 0.3, "George", "They're calling the cops, man!"),
    (8, 0.3, "Callum", "You think I'm fucking around? Mark it zero!"),
    (10, 0.3, "Roger", "All right, it's fucking zero. You happy, you crazy fuck?"),
    (11, 0.3, "Callum", "It's a league game, Smokey."),
]

VIDEOS = [
    {
        "name": "wan",
        "video": os.path.join(HERE, "over-the-line.mp4"),
        "durations": [5.045329] * 14,
        "output": os.path.join(HERE, "over-the-line-with-audio.mp4"),
    },
    {
        "name": "ltx2",
        "video": os.path.join(HERE, "over-the-line-ltx2.mp4"),
        "durations": [4.52] * 14,
        "output": os.path.join(HERE, "over-the-line-ltx2-with-audio.mp4"),
    },
]


def fal_download(model_id, arguments, out_path, url_key="audio"):
    import fal_client
    r = fal_client.subscribe(model_id, arguments=arguments, with_logs=False)
    url = r[url_key]["url"] if isinstance(r.get(url_key), dict) else r[url_key]
    urllib.request.urlretrieve(url, out_path)


def generate_ambience():
    path = os.path.join(AUDIO_DIR, "ambience.mp3")
    if os.path.exists(path):
        print(f"ambience: skip (exists) {path}")
        return path
    print("generating ambience bed...")
    fal_download("fal-ai/mmaudio-v2/text-to-audio",
                 {"prompt": AMBIENCE_PROMPT, "duration": 10}, path)
    return path


def generate_dialogue():
    paths = []
    for clip_idx, delay, voice, text in DIALOGUE:
        safe = text[:20].replace(" ", "_").replace("'", "").replace(",", "").replace("!", "")
        path = os.path.join(AUDIO_DIR, f"line_{clip_idx:02d}_{safe}.mp3")
        if os.path.exists(path):
            print(f"line (clip {clip_idx}): skip (exists) {path}")
        else:
            print(f"generating line (clip {clip_idx}, {voice}): {text!r}")
            fal_download("fal-ai/elevenlabs/tts/turbo-v2.5",
                         {"text": text, "voice": voice}, path)
        paths.append((clip_idx, delay, path))
    return paths


def probe_duration(path):
    out = subprocess.check_output(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", path])
    return float(out.strip())


def mux(video_info, ambience_path, dialogue):
    video_path = video_info["video"]
    durations = video_info["durations"]
    output_path = video_info["output"]

    offsets = []
    t = 0.0
    for d in durations:
        offsets.append(t)
        t += d
    total_duration = t

    inputs = ["-i", video_path, "-stream_loop", "-1", "-i", ambience_path]
    for _, _, path in dialogue:
        inputs += ["-i", path]

    filters = ["[1:a]aresample=44100,aformat=channel_layouts=stereo,volume=0.3,"
               "alimiter=limit=0.95[amb]"]
    mix_labels = ["[amb]"]
    for i, (clip_idx, delay, _) in enumerate(dialogue):
        delay_ms = round((offsets[clip_idx] + delay) * 1000)
        in_idx = i + 2  # 0=video, 1=ambience, 2..=dialogue
        label = f"sp{i}"
        filters.append(
            f"[{in_idx}:a]aresample=44100,aformat=channel_layouts=stereo,"
            f"adelay={delay_ms}|{delay_ms},volume=1.4[{label}]")
        mix_labels.append(f"[{label}]")

    filters.append(
        "".join(mix_labels) + f"amix=inputs={len(mix_labels)}:duration=first:"
        f"dropout_transition=0,alimiter=limit=0.9[aout]")

    filter_complex = ";".join(filters)

    cmd = ["ffmpeg", "-y", "-loglevel", "error", *inputs,
           "-filter_complex", filter_complex,
           "-map", "0:v", "-map", "[aout]",
           "-c:v", "copy", "-c:a", "aac", "-b:a", "128k",
           "-t", str(total_duration), "-movflags", "+faststart", output_path]
    subprocess.run(cmd, check=True)
    dur = probe_duration(output_path)
    print(f"DONE -> {output_path} ({dur:.1f}s)")


def main():
    ambience_path = generate_ambience()
    dialogue = generate_dialogue()
    for video_info in VIDEOS:
        if not os.path.exists(video_info["video"]):
            print(f"skip {video_info['name']}: {video_info['video']} not found")
            continue
        mux(video_info, ambience_path, dialogue)


if __name__ == "__main__":
    main()
