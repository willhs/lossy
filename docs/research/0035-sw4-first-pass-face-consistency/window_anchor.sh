#!/usr/bin/env bash
# Research 0035: research 0030's window (shots 1284-1321) in Will's chosen setup (LTX-2.5
# one-stage 640x352, keyframe-first with klein on the pod, native audio, stitched at 1280x704),
# then the same window with the shot's keyframe re-anchored later in the clip.
#
#   ARM=base                     the 0030/0034 setup, flag off (draws sheets + keyframes)
#   ARM=end100                   --keyframe-anchor-end 1.0
#   ARM=mid50end50               --keyframe-anchor-mid 0.5 --keyframe-anchor-end 0.5
#
# An anchor arm is a copy of base with only the keyframe shots' clips and progress removed, so
# its text shots are base's clips byte for byte and its sheets and keyframes are base's (klein
# cache hits): the arms differ in the anchor alone. Run base first, then the arms (in parallel
# is safe: capped_run only touches pods its own run's journal claims).
#
#   WS=/abs/workspace BASELINE=28.19 ARM=base bash docs/research/0035-.../window_anchor.sh
#
# One spend cap (CAP, default $6) for the whole task, measured from BASELINE (the balance when
# the task started), so no arm or re-run can take the total past it. Idempotent: decode, speech
# and stitch resume from their progress files.
set -euo pipefail

WS=${WS:?absolute workspace path, holding runs/base and face_models}
BASELINE=${BASELINE:?balance the task cap is measured from}
ARM=${ARM:?base, end100 or mid50end50}
CAP=${CAP:-6}
START=1284
LIMIT=38
KF=runpod-ltx25-keyframe
STYLE="Live-action 1977 science-fiction film footage of real actors and practical sets, photographed on 35mm film."
WINDOW=(--start-index "$START" --limit "$LIMIT")
PY=(uv run python)
RUN=$WS/runs/$ARM
EV=$WS/evidence/$ARM
export LOSSY_FACE_MODELS=$WS/face_models
mkdir -p "$EV"

case $ARM in
  base) ANCHOR=() ;;
  end100) ANCHOR=(--keyframe-anchor-end 1.0) ;;
  mid50end50) ANCHOR=(--keyframe-anchor-mid 0.5 --keyframe-anchor-end 0.5) ;;
  *) echo "unknown ARM $ARM" >&2; exit 2 ;;
esac

balance() {
  "${PY[@]}" -c 'import sys; sys.path.insert(0, "tools"); import config; config.load_env()
from capped_run import balance_and_pods as b
bal, pods = b(); print(f"{bal:.2f} {len(pods)}")'
}

if [ "$ARM" != base ] && [ ! -d "$RUN" ]; then
  # Fork base: everything but the keyframe shots' clips, and those shots out of the progress file.
  rsync -a --exclude runpod_journal --exclude '*.mp4' "$WS/runs/base/" "$RUN/"
  rsync -a "$WS/runs/base/clips/" "$RUN/clips/"
  rsync -a "$WS/runs/base/speech/" "$RUN/speech/"
  "${PY[@]}" - "$RUN" "$KF" <<'EOF'
import json, os, sys
run, kf = sys.argv[1:]
shots = sorted(int(f[3:7]) for f in os.listdir(os.path.join(run, "keyframes")) if f.startswith("kf_") and f.endswith(".png"))
path = os.path.join(run, f"decode_progress_{kf}.json")
p = json.load(open(path))
p["completed"] = [s for s in p["completed"] if s not in shots]
json.dump(p, open(path, "w"), indent=1)
clips = os.path.join(run, "clips", kf)
gone = [f for f in os.listdir(clips) if f.endswith(".mp4") and int(f[:4]) in shots]
for f in gone:
    os.remove(os.path.join(clips, f))
print(f"forked base: {len(shots)} keyframe shots to re-render ({shots}); removed {len(gone)} clips")
EOF
fi

read -r NOW PODS < <(balance)
echo "RunPod balance \$$NOW (task baseline \$$BASELINE), $PODS pod(s) on the account" | tee "$EV/balance_start.txt"

# 1. Speech for the window with the recast voices (fal ElevenLabs, cents). base only; arms copy it.
if [ "$ARM" = base ]; then
  "${PY[@]}" decode.py "$RUN" --speech "${WINDOW[@]}" 2>&1 | tee "$EV/speech.log"
fi

# 2. Video under the task cap.
"${PY[@]}" tools/capped_run.py --cap "$CAP" --margin 0.5 --baseline "$BASELINE" --output-dir "$RUN" \
  --progress "$RUN/decode_progress_$KF.json" --total-shots "$LIMIT" --no-project \
  -- "${PY[@]}" decode.py "$RUN" --strategy "$KF" --keyframe-generator pod --ltx-one-stage \
  ${ANCHOR[@]+"${ANCHOR[@]}"} --style-prefix "$STYLE" "${WINDOW[@]}" 2>&1 | tee "$EV/video.log"

# 3. Stitch with each clip's own LTX audio, Lanczos to 1280x704 ($0, local ffmpeg).
"${PY[@]}" decode.py "$RUN" --strategy "$KF" --stitch "${WINDOW[@]}" --audio-strategy native \
  --scale 1280x704 2>&1 | tee "$EV/stitch.log"

# 4. Pod hygiene: no pods, journal clean.
read -r FINAL PODS < <(balance)
{
  echo "task baseline \$$BASELINE  before this arm \$$NOW  end \$$FINAL  pods left: $PODS"
  "${PY[@]}" pod_journal.py list "$RUN" || true
} | tee "$EV/hygiene.txt"
