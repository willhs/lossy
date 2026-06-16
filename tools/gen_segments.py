"""Orchestrate the full self-hosted decode of several scene clips on ONE RunPod pod.

Brings up a single ComfyUI pod, writes its state into every clip's output dir so
each strategy reconnects to the same pod, then runs the requested stages across
all clips:

    video  -> Wan 2.2 TI2V-5B (runpod-wan22)
    audio  -> MMAudio V2       (runpod-mmaudio)   SFX/ambience, all shots
    music  -> MusicGen large   (runpod-musicgen)  music-bucket shots only
    speech -> ElevenLabs TTS   (local/API, no pod)
    stitch -> ffmpeg mux       (local, no pod)

The pod is kept alive between invocations (state file persists) so this can be
run stage-by-stage. Pass --terminate on the final run to tear the pod down.

Usage:
    python tools/gen_segments.py --clips sw_r2_leia lotr_moria phantom_podrace --stages video
    python tools/gen_segments.py --clips ... --stages audio music
    python tools/gen_segments.py --clips ... --stages speech stitch --terminate
"""
import argparse
import os
import shutil
import sys
import types

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

from decode import (  # noqa: E402
    load_env, run_decode, run_audio, run_speech, stitch_clips,
    _create_wan22_strategy,
)
from strategies_audio import RunPodMMAudioStrategy, RunPodMusicGenStrategy  # noqa: E402
from runpod_pod import RunPodSession  # noqa: E402


def mkargs(clip, **kw):
    a = types.SimpleNamespace(
        output_dir=f"output/{clip}",
        start_index=None, limit=None,
        keep_pod=True, concurrent_audio=False,
        strategy="runpod-wan22",
        audio_strategy="runpod-mmaudio",
        music_strategy="runpod-musicgen",
        speech_voice="Roger",
    )
    a.__dict__.update(kw)
    return a


def keep(session):
    """Mark a session so its atexit handler never terminates the shared pod."""
    session.keep_pod = True
    session.mark_clean_exit()
    return session


def skip_setup_if_loaded(strat, probe_node):
    """If the pod's ComfyUI already has this custom node loaded, skip the
    strategy's install+restart (which races: ComfyUI answers /system_stats
    before custom nodes finish loading, so /prompt 400s during the window).
    No-op on a fresh pod, where the normal install path runs."""
    import httpx
    strat._session.ensure_pod()
    try:
        # ComfyUI returns 200 with an empty {} body for UNKNOWN nodes, so a
        # non-empty body keyed by the node name is the real "loaded" signal.
        r = httpx.get(f"{strat._session.base_url}/object_info/{probe_node}", timeout=15)
        if r.status_code == 200 and probe_node in (r.json() or {}):
            strat._setup_done = True
            print(f"  {probe_node} already loaded — skipping install/restart")
    except Exception:
        pass


def ensure_shared_pod(clips):
    """Create or reconnect ONE pod; mirror its state file into every clip dir."""
    master = RunPodSession(output_dir=f"output/{clips[0]}", keep_pod=True)
    master.ensure_pod()           # reconnect via state file, else create
    keep(master)
    master.write_state()
    for c in clips[1:]:
        os.makedirs(f"output/{c}", exist_ok=True)
        shutil.copy(master.state_path, f"output/{c}/runpod_pod.json")
    print(f"  Shared pod {master.pod_id} ready: {master.base_url}")
    return master


def stage_video(clip):
    args = mkargs(clip)
    strat = _create_wan22_strategy(args)
    keep(strat._session)
    run_decode(args, strat)
    keep(strat._session)


def stage_audio(clip):
    args = mkargs(clip)
    strat = RunPodMMAudioStrategy(output_dir=args.output_dir)
    keep(strat._session)
    skip_setup_if_loaded(strat, "MMAudioModelLoader")
    run_audio(args, strat)
    keep(strat._session)


def stage_music(clip):
    args = mkargs(clip)
    strat = RunPodMusicGenStrategy(output_dir=args.output_dir)
    keep(strat._session)
    skip_setup_if_loaded(strat, "MusicGenNode")
    run_audio(args, strat)
    keep(strat._session)


def stage_speech(clip):
    from strategies_audio import SpeechStrategy
    args = mkargs(clip)
    run_speech(args, SpeechStrategy(voice=args.speech_voice))


def stage_stitch(clip):
    args = mkargs(clip)
    stitch_clips(args)


POD_STAGES = {"video": stage_video, "audio": stage_audio, "music": stage_music}
LOCAL_STAGES = {"speech": stage_speech, "stitch": stage_stitch}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clips", nargs="+", required=True)
    ap.add_argument("--stages", nargs="+", required=True,
                    choices=["video", "audio", "music", "speech", "stitch"])
    ap.add_argument("--terminate", action="store_true",
                    help="Tear down the pod after these stages complete")
    args = ap.parse_args()
    load_env()

    needs_pod = any(s in POD_STAGES for s in args.stages)
    master = ensure_shared_pod(args.clips) if needs_pod else None

    try:
        # Run stage-major, clip-minor so a model loads once then sweeps all clips.
        for stage in args.stages:
            fn = POD_STAGES.get(stage) or LOCAL_STAGES[stage]
            for clip in args.clips:
                print(f"\n===== {stage.upper()} :: {clip} =====")
                fn(clip)
    finally:
        if args.terminate and master is not None:
            print("\n  Tearing down shared pod...")
            master.keep_pod = False
            master.terminate()
            for c in args.clips:
                sp = f"output/{c}/runpod_pod.json"
                if os.path.exists(sp):
                    os.remove(sp)
        elif master is not None:
            master.write_state()
            print(f"\n  Pod {master.pod_id} left alive. Re-run with --terminate to stop it.")


if __name__ == "__main__":
    main()
