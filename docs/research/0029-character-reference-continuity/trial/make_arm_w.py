"""Render research 0029's bake-off arm W: Wan 2.1 VACE-14B reference-to-video, hosted on fal.

Usage (from the repo root; RUN_DIR as for make_refs.py, with sheets/ and trial_manifest.json):
    uv run python docs/research/0029-character-reference-continuity/trial/make_arm_w.py RUN_DIR \
        [--price-only] [--only SHOT[,SHOT]] [--resolution 480p] [--cap USD]

Same text as arm T (the manifest's T prompt, no name in it) and the same klein sheets as
arm K, passed as VACE's `ref_image_urls` (one per character in the shot). No first frame.
Clips land in RUN_DIR/clips/fal-wan-vace-ab/W/NNNN.mp4 and are resumable. Frame counts follow
arm T's duration at VACE's 16 fps (4n+1). fal bills $/video second at 16 fps (480p $0.04,
580p $0.06, 720p $0.08, page read 2026-11-20); every call is appended to refs_cost.jsonl with
its wall seconds, and --cap stops before a call that would pass the cap on this arm.
"""
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import httpx

sys.path.insert(0, os.getcwd())
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config  # noqa: E402
import make_refs  # noqa: E402

MODEL = "fal-ai/wan-vace-14b"
PRICE_PER_S = {"480p": 0.04, "580p": 0.06, "720p": 0.08}
FPS = 16


def vace_frames(ltx_frames: int, min_frames: int) -> int:
    n = round(ltx_frames / 24 * FPS)
    n = max(n, min_frames)
    return (n - 1) // 4 * 4 + 1


def main(run_dir: str, argv: list[str]) -> None:
    res = argv[argv.index("--resolution") + 1] if "--resolution" in argv else "480p"
    cap = float(argv[argv.index("--cap") + 1]) if "--cap" in argv else 1e9
    only = set(argv[argv.index("--only") + 1].split(",")) if "--only" in argv else None
    min_frames = int(argv[argv.index("--min-frames") + 1]) if "--min-frames" in argv else 17
    plan = json.load(open(os.path.join(run_dir, "trial_plan.json")))
    t_jobs = {int(j["shot"]): j for j in json.load(open(os.path.join(run_dir, "trial_manifest.json")))["jobs"]
              if j["arm"] == "T"}
    shots = [s for s in plan["shots"] if only is None or str(s["shot"]) in only]
    frames = {s["shot"]: vace_frames(int(t_jobs[s["shot"]]["frames"]), min_frames) for s in shots}
    est = sum(frames[s["shot"]] / FPS * PRICE_PER_S[res] for s in shots)
    print(f"arm W {res}: {len(shots)} clips, {sum(frames.values())/FPS:.1f} video-s, estimate ${est:.2f}")
    if "--price-only" in argv:
        return

    config.load_env()
    import fal_client

    out_dir = os.path.join(run_dir, "clips", "fal-wan-vace-ab", "W")
    os.makedirs(out_dir, exist_ok=True)
    sheet_urls = {n: fal_client.upload_file(os.path.join(run_dir, "sheets", f"{n}.png")) for n in plan["identity"]}
    todo = [s for s in shots if not os.path.exists(os.path.join(out_dir, f"{s['shot']:04d}.mp4"))]
    budget, picked = cap, []
    for s in todo:   # the cap is checked up front: calls run in parallel, so it cannot be checked per call
        usd = frames[s["shot"]] / FPS * PRICE_PER_S[res]
        if usd > budget:
            print(f"cap ${cap:.2f} reached: shot {s['shot']:04d} onward skipped")
            break
        budget -= usd
        picked.append(s)

    def one(s: dict) -> float:
        usd = frames[s["shot"]] / FPS * PRICE_PER_S[res]
        args = {"prompt": t_jobs[s["shot"]]["prompt"], "ref_image_urls": [sheet_urls[c] for c in s["characters"]],
                "num_frames": frames[s["shot"]], "frames_per_second": FPS, "resolution": res,
                "aspect_ratio": "16:9", "seed": s["shot"]}
        t0 = time.time()
        try:
            r = fal_client.subscribe(MODEL, arguments=args, with_logs=False)
            video = httpx.get(r["video"]["url"], follow_redirects=True, timeout=300)
            video.raise_for_status()
        except Exception as e:  # a failed shot is reported, not retried blindly (the call may have billed)
            print(f"shot {s['shot']:04d} FAILED: {e}", flush=True)
            return 0.0
        open(os.path.join(out_dir, f"{s['shot']:04d}.mp4"), "wb").write(video.content)
        wall = time.time() - t0
        make_refs._log(run_dir, {"kind": "clip", "arm": "W", "shot": s["shot"], "model": MODEL, "resolution": res,
                                 "frames": frames[s["shot"]], "usd": round(usd, 4), "wall_s": round(wall, 1)})
        print(f"shot {s['shot']:04d} ({'+'.join(s['characters'])}) {wall:.0f}s ${usd:.2f}", flush=True)
        return usd

    with ThreadPoolExecutor(max_workers=8) as ex:
        spent = sum(ex.map(one, picked))
    print(f"arm W spend this run: ${spent:.2f}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2:])
