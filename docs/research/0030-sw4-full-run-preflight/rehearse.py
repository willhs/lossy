"""No-spend rehearsal of the Star Wars IV full run (research 0030).

Runs the REAL decode.main / run_decode / RunPodLtx25Strategy.generate and the REAL
tools/capped_run.main, with only the two paid edges replaced:

  * RunPodSession -> FakeSession: no pod, no ssh. It records a "pod" in a shared
    fake-account file when the strategy provisions, and removes it on a normal exit
    (atexit), like the real session. A SIGKILLed child leaves its pod "billing".
  * _generate_one_clip -> writes a tiny real mp4 with the requested frame count
    (64x36, so ffprobe/stitch see true durations).
  * capped_run._gql -> answers from the same fake-account file: balance falls at
    --rate $/s for every second a pod is alive; podTerminate removes the pod.

No network call is made and no key is read for a paid call.

    uv run python docs/research/0030-sw4-full-run-preflight/rehearse.py plan <encode_dir>
    uv run python .../rehearse.py decode-child [--kill-at-shot N] [--kill-in-save K] -- <decode.py args>
    uv run python .../rehearse.py supervise --account F [--rate R] -- <capped_run.py args>
"""

import argparse
import atexit
import json
import os
import shutil
import signal
import subprocess
import sys
import time

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))

ACCOUNT = os.environ.get("REHEARSE_ACCOUNT", "")       # fake RunPod account file
TEMPLATES = os.environ.get("REHEARSE_TEMPLATES", "/tmp/rehearse_templates")
FPS = 25


# ----------------------------------------------------------------- fake account

def _load_account() -> dict:
    try:
        with open(ACCOUNT) as f:
            return json.load(f)
    except (OSError, ValueError):
        return {"start_balance": 30.0, "rate": 0.001, "pods": []}


def _save_account(acc: dict) -> None:
    tmp = ACCOUNT + ".tmp"
    with open(tmp, "w") as f:
        json.dump(acc, f)
    os.replace(tmp, ACCOUNT)


def _balance(acc: dict) -> float:
    now = time.time()
    used = sum((p.get("ended") or now) - p["created"] for p in acc["pods"])
    return acc["start_balance"] - acc["rate"] * used


# ----------------------------------------------------------------- fake session

class FakeSession:
    """Stands in for runpod_pod.RunPodSession. Unknown methods are recorded no-ops."""

    ssh_host = None

    def __init__(self, output_dir="", keep_pod=False, gpu_types=None, **_):
        self.output_dir = output_dir
        self.pod_id = None
        self.disk_gb = 0
        self.calls: list[str] = []
        atexit.register(self._cleanup)

    def with_setup_retry(self, fn):
        return fn()

    def ensure_pod(self):
        if self.pod_id:
            return
        acc = _load_account()
        self.pod_id = f"fake-{os.getpid()}-{len(acc['pods'])}"
        acc["pods"].append({"id": self.pod_id, "created": time.time(), "ended": None})
        _save_account(acc)
        print(f"  [fake] pod {self.pod_id} up", flush=True)

    def _cleanup(self):
        if not self.pod_id or not ACCOUNT:
            return
        acc = _load_account()
        for p in acc["pods"]:
            if p["id"] == self.pod_id and not p["ended"]:
                p["ended"] = time.time()
        _save_account(acc)
        print(f"  [fake] pod {self.pod_id} terminated on exit", flush=True)
        self.pod_id = None

    def __getattr__(self, name):
        def noop(*a, **kw):
            self.calls.append(name)
            return None
        return noop


def _template(frames: int) -> str:
    os.makedirs(TEMPLATES, exist_ok=True)
    path = os.path.join(TEMPLATES, f"t{frames}.mp4")
    if not os.path.exists(path):
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i",
                        f"color=c=gray:s=64x36:r={FPS}", "-frames:v", str(frames),
                        "-c:v", "libx264", "-pix_fmt", "yuv420p", path], check=True)
    return path


def _marker(name: str) -> str:
    return os.path.join(os.path.dirname(ACCOUNT) or ".", f".{name}.fired")


def install_fakes(kill_at_shot: int | None, kill_in_save: int | None, delay: float) -> None:
    import runpod_pod
    import strategies_video
    from clip_types import ClipResult
    import decode

    runpod_pod.RunPodSession = FakeSession

    def fake_clip(self, prompt, clips_dir, clip_name, frames, seed, start_image=None):
        shot = int(clip_name.split("-")[0].split(".")[0])
        part = int(clip_name.split("-")[1].split(".")[0]) if "-" in clip_name else 1
        if kill_at_shot is not None and shot == kill_at_shot and part == 2 \
                and not os.path.exists(_marker("kill_shot")):
            open(_marker("kill_shot"), "w").close()
            print(f"  [fake] SIGKILL at shot {shot} part {part} (part 1 already on disk)", flush=True)
            os.kill(os.getpid(), signal.SIGKILL)
        time.sleep(delay)
        path = os.path.join(clips_dir, clip_name)
        shutil.copyfile(_template(frames), path)
        return ClipResult(path=path, actual_duration_s=frames / FPS, cost=0.0)

    strategies_video.RunPodWanStrategy._generate_one_clip = fake_clip
    strategies_video.RunPodWanStrategy._upload_start_frame = lambda self, *a, **kw: "fake_start.png"

    if kill_in_save is not None:
        real_dump = json.dump
        count = {"n": 0}

        def dump(obj, fp, **kw):
            if "decode_progress" in getattr(fp, "name", ""):   # .json, or .json.tmp once atomic
                count["n"] += 1
                if count["n"] == kill_in_save and not os.path.exists(_marker("kill_save")):
                    open(_marker("kill_save"), "w").close()
                    text = json.dumps(obj, **kw)
                    fp.write(text[: len(text) // 2])
                    fp.flush()
                    print(f"  [fake] SIGKILL halfway through progress save #{count['n']}", flush=True)
                    os.kill(os.getpid(), signal.SIGKILL)
            return real_dump(obj, fp, **kw)

        decode.json.dump = dump


# ----------------------------------------------------------------- plan

def plan(encode_dir: str, out_json: str | None) -> None:
    """Every shot's parts, frame counts and prompt for runpod-ltx25, with each part's
    ComfyUI graph built (the step that rejects a bad frame count). No pod, no network."""
    import manifest
    from decode import build_character_shot_map
    from strategies_video import RunPodLtx25Strategy

    prompts, _ = manifest.load_shots(encode_dir)
    with open(manifest.characters_path(encode_dir)) as f:
        chars = json.load(f)
    probe = object.__new__(RunPodLtx25Strategy)
    probe._init_character_identity(build_character_shot_map(chars), chars,
                                   manifest.load_style_prefix(encode_dir))
    # Personal names only: keep_name characters (Vader, C-3PO, stormtrooper...) keep
    # their name on purpose (strategies_video._all_character_names).
    people = [c for c in chars.get("characters", []) if not c.get("keep_name")]
    names = sorted(({c["name"].replace("_", " ").lower() for c in people}
                    | {c.get("display_name", "").lower() for c in people}) - {""})
    rows, built, named = [], 0, []
    for e in sorted(prompts, key=lambda p: p["index"]):
        frames = probe._target_durations(e["duration_s"])
        text = probe.format_prompt(e)
        for i, n in enumerate(frames):
            part_prompt = probe._part_prompt(text, e, i, len(frames))
            probe._build_workflow(part_prompt, e["index"] + i, length=n,
                                  start_image="x.png" if i else None)
            built += 1
        low = text.lower()
        hit = [n for n in names if len(n) > 3 and n in low]
        if hit:
            named.append((e["index"], hit))
        rows.append({"index": e["index"], "duration_s": e["duration_s"], "frames": frames,
                     "prompt_chars": len(text)})
    n = len(rows)
    parts = sum(len(r["frames"]) for r in rows)
    split = sum(1 for r in rows if len(r["frames"]) > 1)
    src_s = sum(r["duration_s"] for r in rows)
    gen_s = sum(sum(r["frames"]) for r in rows) / FPS
    # Seinfeld fit (research 0026): ~46 s fixed + ~16 s per video-second, per shot.
    gpu_h = (46 * n + 16 * src_s) / 3600
    summary = {
        "shots": n, "index_range": [rows[0]["index"], rows[-1]["index"]],
        "parts": parts, "split_shots": split, "graphs_built": built,
        "source_minutes": round(src_s / 60, 2), "generated_minutes": round(gen_s / 60, 2),
        "max_prompt_chars": max(r["prompt_chars"] for r in rows),
        "shots_with_character_name_in_prompt": len(named), "named_examples": named[:8],
        "style_prefix": probe.style_prefix or None,
        "gpu_hours_seinfeld_fit": round(gpu_h, 1),
        "cost_usd_at": {f"${r}/h": round(gpu_h * r, 2) for r in (0.33, 0.49, 0.59, 0.69, 0.79)},
    }
    print(json.dumps(summary, indent=2))
    if out_json:
        with open(out_json, "w") as f:
            json.dump({"summary": summary, "shots": rows}, f)


# ----------------------------------------------------------------- supervise

def supervise(cap_args: list[str]) -> int:
    import capped_run

    def fake_gql(query: str) -> dict:
        acc = _load_account()
        if "podTerminate" in query:
            pid = query.split('podId:"')[1].split('"')[0]
            for p in acc["pods"]:
                if p["id"] == pid and not p["ended"]:
                    p["ended"] = time.time()
            _save_account(acc)
            return {"podTerminate": None}
        live = [{"id": p["id"]} for p in acc["pods"] if not p["ended"]]
        return {"myself": {"clientBalance": _balance(acc), "pods": live}}

    capped_run._gql = fake_gql
    capped_run.CHILD_GRACE_S = 10
    sys.argv = ["capped_run.py"] + cap_args
    return capped_run.main()


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("plan")
    p.add_argument("encode_dir")
    p.add_argument("--out")
    c = sub.add_parser("decode-child")
    c.add_argument("--kill-at-shot", type=int)
    c.add_argument("--kill-in-save", type=int)
    c.add_argument("--delay", type=float, default=0.0)
    c.add_argument("rest", nargs=argparse.REMAINDER)
    s = sub.add_parser("supervise")
    s.add_argument("rest", nargs=argparse.REMAINDER)
    a = ap.parse_args()
    if a.cmd == "plan":
        plan(a.encode_dir, a.out)
        return 0
    rest = a.rest[1:] if a.rest and a.rest[0] == "--" else a.rest
    if a.cmd == "decode-child":
        install_fakes(a.kill_at_shot, a.kill_in_save, a.delay)
        import decode
        sys.argv = ["decode.py"] + rest
        decode.main()
        return 0
    return supervise(rest)


if __name__ == "__main__":
    sys.exit(main())
