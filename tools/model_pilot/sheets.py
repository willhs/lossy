"""Side-by-side contact sheets and comparison clips for the pilot (research 0025).

Contact sheet row per shot: original keyframe, then two frames (at 30% and 70%
of the clip) from each model, so both look and motion are visible on paper.
Comparison clip per shot: models side by side at a common height, trimmed to
the shot's own duration, labelled; all shots concatenated into one file.

    uv run python tools/model_pilot/sheets.py output/pilot0025 ltx23 ltx25-base k6 \
        --sheets-out docs/research/0025-video-model-refresh-nov-2026
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import tempfile

from PIL import Image, ImageDraw, ImageFont

TILE_W, TILE_H = 320, 176
LABEL_H = 22
ROWS_PER_SHEET = 10


def _probe_duration(path: str) -> float:
    r = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                        "-of", "csv=p=0", path], capture_output=True, text=True)
    try:
        return float(r.stdout.strip())
    except ValueError:
        return 0.0


def _frame(path: str, t: float) -> Image.Image | None:
    with tempfile.NamedTemporaryFile(suffix=".jpg") as tmp:
        r = subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", f"{t:.3f}", "-i", path,
                            "-frames:v", "1", tmp.name], capture_output=True)
        if r.returncode or not os.path.getsize(tmp.name):
            return None
        return Image.open(tmp.name).convert("RGB").copy()


def _fit(img: Image.Image | None) -> Image.Image:
    tile = Image.new("RGB", (TILE_W, TILE_H), (40, 0, 0))
    if img is None:
        return tile
    img = img.copy()
    img.thumbnail((TILE_W, TILE_H))
    tile.paste(img, ((TILE_W - img.width) // 2, (TILE_H - img.height) // 2))
    return tile


def contact_sheets(root: str, models: list[str], out_dir: str) -> list[str]:
    bench = json.load(open(os.path.join(root, "benchmark.json")))
    font = ImageFont.load_default()
    cols = 1 + 2 * len(models)
    headers = ["original"] + [f"{m} {p}" for m in models for p in ("30%", "70%")]
    paths = []
    shots = bench["shots"]
    for sheet_i in range(0, len(shots), ROWS_PER_SHEET):
        chunk = shots[sheet_i:sheet_i + ROWS_PER_SHEET]
        sheet = Image.new("RGB", (cols * TILE_W, LABEL_H + len(chunk) * (TILE_H + LABEL_H)), (16, 16, 16))
        d = ImageDraw.Draw(sheet)
        for c, h in enumerate(headers):
            d.text((c * TILE_W + 6, 5), h, fill=(255, 255, 255), font=font)
        for r, shot in enumerate(chunk):
            y = LABEL_H + r * (TILE_H + LABEL_H)
            d.text((6, y + 4), f"#{shot['index']}  {shot['category']}  {shot['duration_s']:.1f}s",
                   fill=(255, 220, 120), font=font)
            y += LABEL_H
            kf = shot.get("keyframe")
            sheet.paste(_fit(Image.open(os.path.join(root, kf)).convert("RGB") if kf else None), (0, y))
            for mi, m in enumerate(models):
                clip = os.path.join(root, "clips", m, f"{shot['index']:04d}.mp4")
                dur = _probe_duration(clip) if os.path.exists(clip) else 0
                for pi, frac in enumerate((0.3, 0.7)):
                    img = _frame(clip, dur * frac) if dur else None
                    sheet.paste(_fit(img), ((1 + 2 * mi + pi) * TILE_W, y))
        p = os.path.join(out_dir, f"contact_sheet_{sheet_i // ROWS_PER_SHEET + 1}.jpg")
        sheet.save(p, quality=82)
        paths.append(p)
    return paths


def comparison_clip(root: str, models: list[str], out_path: str, height: int = 288) -> str:
    bench = json.load(open(os.path.join(root, "benchmark.json")))
    tmpdir = tempfile.mkdtemp(prefix="pilot_cmp_")
    parts = []
    w = height * 16 // 9
    for shot in bench["shots"]:
        idx, dur = shot["index"], min(shot["duration_s"], 4.8)
        inputs, filters = [], []
        for i, m in enumerate(models):
            clip = os.path.join(root, "clips", m, f"{idx:04d}.mp4")
            if os.path.exists(clip):
                inputs += ["-stream_loop", "-1", "-i", clip]
            else:
                inputs += ["-f", "lavfi", "-i", f"color=c=black:s={w}x{height}:r=25"]
            filters.append(
                f"[{i}:v]scale={w}:{height}:force_original_aspect_ratio=decrease,"
                f"pad={w}:{height}:(ow-iw)/2:(oh-ih)/2,fps=25,trim=duration={dur:.3f},setpts=PTS-STARTPTS,"
                f"drawtext=text='{m}':x=8:y=8:fontsize=18:fontcolor=white:box=1:boxcolor=black@0.6[v{i}]")
        stack = "".join(f"[v{i}]" for i in range(len(models)))
        filters.append(f"{stack}hstack=inputs={len(models)},"
                       f"drawtext=text='#{idx} {shot['category'].replace(':', ' - ')}':x=8:y=h-28:fontsize=18:"
                       f"fontcolor=yellow:box=1:boxcolor=black@0.6[out]")
        part = os.path.join(tmpdir, f"{idx:04d}.mp4")
        subprocess.run(["ffmpeg", "-v", "error", "-y", *inputs, "-filter_complex", ";".join(filters),
                        "-map", "[out]", "-t", f"{dur:.3f}", "-an", "-c:v", "libx264", "-crf", "26",
                        "-pix_fmt", "yuv420p", part], check=True)
        parts.append(part)
    lst = os.path.join(tmpdir, "list.txt")
    with open(lst, "w") as f:
        f.writelines(f"file '{p}'\n" for p in parts)
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "concat", "-safe", "0", "-i", lst,
                    "-c", "copy", out_path], check=True)
    return out_path


def main():
    ap = argparse.ArgumentParser(description="Contact sheets + comparison clip for the pilot")
    ap.add_argument("root")
    ap.add_argument("models", nargs="+")
    ap.add_argument("--sheets-out", required=True)
    ap.add_argument("--clip-out", default=None)
    a = ap.parse_args()
    os.makedirs(a.sheets_out, exist_ok=True)
    for p in contact_sheets(a.root, a.models, a.sheets_out):
        print("sheet:", p)
    clip = comparison_clip(a.root, a.models, a.clip_out or os.path.join(a.root, "comparison.mp4"))
    print("clip:", clip)


if __name__ == "__main__":
    main()
