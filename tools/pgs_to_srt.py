"""OCR a bitmap (PGS) subtitle track from a video into an SRT file.

Blu-ray rips carry subtitles as PGS images, which `ffmpeg -map 0:s:0 out.srt`
cannot convert, so encode stage 2 would see no dialogue. This renders the
chosen subtitle stream over a black canvas with ffmpeg, groups consecutive
frames that show the same subtitle, and OCRs each group once with tesseract.
Needs `ffmpeg` and `tesseract` on PATH; no Python deps beyond numpy/Pillow.

    uv run python tools/pgs_to_srt.py media/ep.mkv -o output/ep/subtitles.srt --stream 1

--stream is the 0-based index among subtitle streams (ffprobe -select_streams s).
Timing resolution is 1/--fps seconds.
"""

import argparse
import hashlib
import subprocess
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image


def _probe(video: str) -> tuple[int, int]:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "s:0", "-show_entries",
         "stream=width,height", "-of", "csv=p=0", video],
        capture_output=True, text=True, check=True).stdout.strip().split(",")
    return int(out[0]), int(out[1])


def _srt_time(s: float) -> str:
    ms = round(s * 1000)
    return f"{ms // 3600000:02}:{ms // 60000 % 60:02}:{ms // 1000 % 60:02},{ms % 1000:03}"


def _ocr(img: np.ndarray, box: tuple[int, int, int, int]) -> str:
    y0, y1, x0, x1 = box
    pad = 12
    crop = img[max(0, y0 - pad):y1 + pad, max(0, x0 - pad):x1 + pad]
    # subtitles are bright on black here; tesseract wants dark text on white
    pil = Image.fromarray(255 - np.pad(crop, 20)).convert("L")
    with tempfile.NamedTemporaryFile(suffix=".png") as f:
        pil.save(f.name)
        res = subprocess.run(["tesseract", f.name, "-", "--psm", "6"],
                             capture_output=True, text=True)
    lines = [ln.strip() for ln in res.stdout.splitlines() if ln.strip()]
    # tesseract reads a capital I in this sans font as a pipe; real subtitles never contain one
    return "\n".join(lines).replace("|", "I")


def pgs_to_srt(video: str, out_path: str, stream: int, fps: float) -> int:
    w, h = _probe(video)
    duration = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", video],
        capture_output=True, text=True, check=True).stdout.strip()
    # the canvas is infinite, so bound it or the filter graph never ends
    cmd = ["ffmpeg", "-v", "error", "-f", "lavfi", "-t", duration,
           "-i", f"color=c=black:s={w}x{h}:r={fps}",
           "-i", video, "-filter_complex", f"[0:v][1:s:{stream}]overlay=shortest=1,format=gray",
           "-f", "rawvideo", "-"]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE)
    assert proc.stdout is not None
    frame_bytes = w * h
    cues: list[tuple[float, float, str]] = []
    cur = None  # (start_idx, digest, img, box)
    idx = 0

    def close(end_idx):
        nonlocal cur
        if cur is None:
            return
        text = _ocr(cur[2], cur[3])
        if text:
            cues.append((cur[0] / fps, end_idx / fps, text))
        cur = None

    while True:
        buf = proc.stdout.read(frame_bytes)
        if len(buf) < frame_bytes:
            break
        img = np.frombuffer(buf, np.uint8).reshape(h, w)
        mask = img > 100
        if mask.sum() < 200:
            close(idx)
        else:
            ys, xs = np.where(mask)
            box = (ys.min(), ys.max() + 1, xs.min(), xs.max() + 1)
            digest = hashlib.md5(img[box[0]:box[1], box[2]:box[3]].tobytes()).hexdigest()
            if cur is None or cur[1] != digest:
                close(idx)
                cur = (idx, digest, img.copy(), box)
        idx += 1
        if idx % 1000 == 0:
            print(f"  {idx / fps:.0f}s of {float(duration):.0f}s, {len(cues)} cues", flush=True)
    close(idx)
    proc.wait()

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for n, (a, b, t) in enumerate(cues, 1):
            f.write(f"{n}\n{_srt_time(a)} --> {_srt_time(b)}\n{t}\n\n")
    return len(cues)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    ap.add_argument("video")
    ap.add_argument("-o", "--output", required=True)
    ap.add_argument("--stream", type=int, default=0, help="0-based subtitle stream index")
    ap.add_argument("--fps", type=float, default=5.0)
    a = ap.parse_args()
    print(f"{pgs_to_srt(a.video, a.output, a.stream, a.fps)} cues -> {a.output}")
