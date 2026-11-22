"""Fidelity yardstick: can each reconstructed shot be matched back to its source shot? (research 0033)

For every source shot, embed one midpoint frame of the reconstruction and one of the source (cut
fresh from the video), then ask: is the true source frame the nearest of N=10 candidates?
Distractors come from three pools: the 9 nearest shots in time (shot level), 9 random shots from
the same film (scene level) and 9 frames from other films (film level). Chance is 10% at each
level. The ceiling swaps the reconstruction for a different frame (the quarter point) of the same
source shot.

Shots are the source's own cuts, from the same detector `encode.py stage1 -d content` uses
(ContentDetector, threshold 27), so on the encoded file they reproduce the encode's shots.
Source time maps to the reconstruction's timeline as t_rec = a * t_src + b: identity by default
(the stitched reconstruction keeps the source's shot timing); --map handles a re-timed copy (a
PAL 25 fps copy is a = 25/23.976). --scan checks the map by embedding similarity over offsets.
Check that SOURCE is the encoded episode (research 0033 found a mislabelled copy).

  uv run --extra fidelity python tools/fidelity.py SOURCE RECON --others media/a.mp4 ... -o DIR [--scan]
"""
import argparse
import itertools
import json
import math
import os
import random
import subprocess

import cv2
import numpy as np

# open_clip_torch 4.2.2, torch 2.15.0 (uv.lock); HF snapshot f9e2bfdc4807cd0fbadf67301e45193d6f330c20
MODEL = "hf-hub:timm/ViT-SO400M-14-SigLIP2"
PAL = 25 / (24000 / 1001)
N, DRAWS, SEED = 10, 50, 0


def write_json(path, obj):
    with open(path, "w") as fh:
        json.dump(obj, fh, indent=1)


def duration(video):
    return float(subprocess.check_output(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                                          "-of", "csv=p=0", video]))


def cuts(video, cache):
    if os.path.exists(cache):
        with open(cache) as fh:
            return json.load(fh)
    from scenedetect import ContentDetector, detect
    out = [s[0].get_seconds() for s in detect(video, ContentDetector(), show_progress=False)[1:]]
    write_json(cache, out)
    return out


def frame(cap, t):
    cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
    ok, img = cap.read()
    if not ok:
        raise RuntimeError(f"no frame at {t:.2f}s")
    h, w = img.shape[:2]  # centre square crop: the two videos have different aspect ratios
    s = min(h, w)
    return img[(h - s) // 2:(h + s) // 2, (w - s) // 2:(w + s) // 2]


_model = None


def embed(images):
    global _model
    import open_clip
    import torch
    from PIL import Image
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    if _model is None:
        model, pre = open_clip.create_model_from_pretrained(MODEL)
        _model = (model.eval().to(dev), pre)
    model, pre = _model
    out = []
    with torch.no_grad():
        for i in range(0, len(images), 16):
            batch = torch.stack([pre(Image.fromarray(cv2.cvtColor(im, cv2.COLOR_BGR2RGB)))
                                 for im in images[i:i + 16]]).to(dev)
            out.append(torch.nn.functional.normalize(model.encode_image(batch), dim=-1).cpu().numpy())
    return np.concatenate(out)


def scan(source, recon, a, out):
    """Mean cosine of recon frames (every 2 s) vs the source frame the map points at, over offsets b."""
    sd, rd = duration(source), duration(recon)
    st, rt = np.arange(0, sd, 0.5), np.arange(1, rd - 1, 2.0)
    scap, rcap = cv2.VideoCapture(source), cv2.VideoCapture(recon)
    S = embed([frame(scap, t) for t in st])
    R = embed([frame(rcap, t) for t in rt])
    sim = R @ S.T
    curve = {}
    for b in np.arange(-15, 15.01, 0.5):
        j = np.rint(((rt - b) / a) / 0.5).astype(int)
        ok = (j >= 0) & (j < len(st))
        curve[round(float(b), 2)] = round(float(sim[np.flatnonzero(ok), j[ok]].mean()), 4)
    write_json(os.path.join(out, f"scan_a{a:.5f}.json"), curve)
    best = max(curve, key=curve.get)
    print(f"scan a={a:.5f}: best b={best} cos={curve[best]}; b=0 cos={curve.get(0.0)}; "
          f"median over b={np.median(list(curve.values())):.4f}")


def top1(q, truth, distractor_sets):
    """Fraction of distractor sets where the truth beats all N-1 distractors (ties count as misses)."""
    t = q @ truth
    return float(np.mean([t > (q @ d.T).max() for d in distractor_sets]))


def wilson(k, n, z=1.96):
    p = k / n
    c = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    m = (p + z * z / (2 * n)) / (1 + z * z / n)
    return [round(m - c, 3), round(m + c, 3)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("source")
    ap.add_argument("recon")
    ap.add_argument("--others", nargs="+", required=True, help="videos from other films (film-level pool)")
    ap.add_argument("--map", nargs=2, type=float, metavar=("A", "B"), help="t_rec = A*t_src + B")
    ap.add_argument("--scan", action="store_true", help="check the map by embedding similarity over offsets")
    ap.add_argument("-o", "--out", required=True)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    rng = random.Random(SEED)
    sd, rd = duration(args.source), duration(args.recon)
    a, b = args.map or (1.0, 0.0)
    print(f"map t_rec = {a:.5f} * t_src + {b:.3f}; durations src {sd:.2f}s rec {rd:.2f}s")
    if args.scan:
        for sa in sorted({1.0, PAL, a}):
            scan(args.source, args.recon, sa, args.out)

    bounds = [0.0] + cuts(args.source, os.path.join(args.out, "cuts_source.json")) + [sd]
    shots = [(s, e) for s, e in itertools.pairwise(bounds) if 0 <= a * s + b and a * e + b <= rd]
    n = len(shots)
    print(f"{n} source shots on the reconstruction timeline")
    scap, rcap = cv2.VideoCapture(args.source), cv2.VideoCapture(args.recon)
    rec = [frame(rcap, a * (s + e) / 2 + b) for s, e in shots]
    src = [frame(scap, (s + e) / 2) for s, e in shots]
    ceil = [frame(scap, s + (e - s) / 4) for s, e in shots]
    other = []
    for v in args.others:
        cap = cv2.VideoCapture(v)
        d = cap.get(cv2.CAP_PROP_FRAME_COUNT) / cap.get(cv2.CAP_PROP_FPS)
        other += [frame(cap, rng.uniform(0.05, 0.95) * d) for _ in range(max(1, 400 // len(args.others)))]
    E = embed(rec + src + ceil + other)
    R, S, C, O = E[:n], E[n:2 * n], E[2 * n:3 * n], E[3 * n:]

    levels = {"shot": [], "scene": [], "film": []}
    for i in range(n):
        near = sorted((j for j in range(n) if j != i), key=lambda j: abs(j - i))[:N - 1]
        rest = [j for j in range(n) if j != i]
        levels["shot"].append([S[near]])
        levels["scene"].append([S[rng.sample(rest, N - 1)] for _ in range(DRAWS)])
        levels["film"].append([O[rng.sample(range(len(O)), N - 1)] for _ in range(DRAWS)])
    res = {"model": MODEL, "shots": n, "map": [a, b], "chance": 1 / N, "other_frames": len(O), "levels": {}}
    for lv, sets in levels.items():
        got = [top1(R[i], S[i], sets[i]) for i in range(n)]
        cei = [top1(C[i], S[i], sets[i]) for i in range(n)]
        res["levels"][lv] = {"top1": round(float(np.mean(got)), 3), "ci95": wilson(sum(got), n),
                             "ceiling": round(float(np.mean(cei)), 3)}
    sim, csim = R @ S.T, C @ S.T
    rank = (sim > np.diag(sim)[:, None]).sum(1)  # 0 = the true source is the nearest of all n shots
    res["all_shots"] = {"top1": round(float(np.mean(rank == 0)), 3), "top5": round(float(np.mean(rank < 5)), 3),
                        "median_rank": int(np.median(rank)),
                        "ceiling_top1": round(float(np.mean((csim > np.diag(csim)[:, None]).sum(1) == 0)), 3)}
    res["cosine"] = {"recon_vs_source": round(float(np.diag(sim).mean()), 3),
                     "ceiling": round(float(np.diag(csim).mean()), 3),
                     "recon_vs_other_films": round(float((R @ O.T).mean()), 3)}
    margin = [float(R[i] @ S[i] - (R[i] @ levels["shot"][i][0].T).max()) for i in range(n)]
    order = np.argsort(margin)
    pick = lambda ix: [{"shot": int(i), "src_s": round(shots[i][0], 2), "rec_s": round(a * shots[i][0] + b, 2),
                        "rank_of_all": int(rank[i]), "margin": round(margin[i], 3)} for i in ix]
    res["best"], res["worst"] = pick(order[::-1][:10]), pick(order[:10])
    write_json(os.path.join(args.out, "results.json"), {**res, "per_shot": pick(range(n))})

    th = 128  # contact sheet: rows of source | reconstruction; best 10 left, worst 10 right

    def cell(i):
        row = np.hstack([cv2.resize(src[i], (th, th)), cv2.resize(rec[i], (th, th))])
        cv2.putText(row, f"#{i} {margin[i]:+.2f}", (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
        return row
    sheet = np.hstack([np.vstack([cell(i) for i in order[::-1][:10]]), np.full((10 * th, 10, 3), 255, np.uint8),
                       np.vstack([cell(i) for i in order[:10]])])
    cv2.imwrite(os.path.join(args.out, "contact_sheet.jpg"), sheet, [cv2.IMWRITE_JPEG_QUALITY, 80])
    print(json.dumps({k: v for k, v in res.items() if k not in ("best", "worst")}, indent=1))


if __name__ == "__main__":
    main()
