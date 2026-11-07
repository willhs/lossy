"""Measure chaining seams in a clips directory (the research 0023 method).

For every multi-part shot (NNNN-01.mp4, NNNN-02.mp4, ...) take the mean absolute
gray-level difference between the true final frame of part N and the first frame
of part N+1 (the seam), and divide by the mean frame-to-frame difference inside
the parts (normal motion). 0023 measured 8.5x unchained and 2.2x chained.

    uv run python tools/seam_ratio.py output/film/clips/runpod-ltx25
"""

import glob
import os
import re
import subprocess
import sys

import numpy as np

PART_RE = re.compile(r"^(\d+)-(\d+)\.mp4$")


def gray_frames(path: str, size: tuple[int, int] = (320, 176)) -> np.ndarray:
    """All frames of a clip as float gray arrays, shape (n, h, w)."""
    w, h = size
    r = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", path, "-vf", f"scale={w}:{h},format=gray",
         "-f", "rawvideo", "-"],
        capture_output=True, check=True,
    )
    return np.frombuffer(r.stdout, dtype=np.uint8).reshape(-1, h, w).astype(np.float32)


def shot_parts(clips_dir: str) -> dict[int, list[str]]:
    shots: dict[int, list[str]] = {}
    for p in sorted(glob.glob(os.path.join(clips_dir, "*.mp4"))):
        m = PART_RE.match(os.path.basename(p))
        if m:
            shots.setdefault(int(m.group(1)), []).append(p)
    return {k: v for k, v in shots.items() if len(v) > 1}


def measure(clips_dir: str) -> dict:
    seams, motions = [], []
    per_shot = {}
    for shot, parts in shot_parts(clips_dir).items():
        frames = [gray_frames(p) for p in parts]
        for f in frames:
            motions.append(np.abs(np.diff(f, axis=0)).mean(axis=(1, 2)))
        shot_seams = [float(np.abs(a[-1] - b[0]).mean()) for a, b in zip(frames, frames[1:])]
        seams += shot_seams
        per_shot[shot] = [round(s, 2) for s in shot_seams]
    if not seams:
        return {"shots": 0}
    motion = float(np.concatenate(motions).mean())
    seam = float(np.mean(seams))
    return {"shots": len(per_shot), "seams": len(seams), "seam": round(seam, 2),
            "motion": round(motion, 2), "ratio": round(seam / motion, 2), "per_shot": per_shot}


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    import json
    print(json.dumps(measure(sys.argv[1]), indent=1))
