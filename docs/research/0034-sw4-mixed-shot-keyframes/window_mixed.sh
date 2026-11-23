#!/usr/bin/env bash
# Research 0034: re-render research 0030's window (shots 1284-1321) with --keyframe-mixed-shots,
# in Will's chosen setup (LTX-2.5 one-stage 640x352, native audio, stitched at 1280x704).
#
# Needs RUNPOD_API_KEY and FAL_KEY (the project's .env loader, a Manager job with secrets, or
# `op run --env-file .env.tpl -- bash <this file>`). Run from the repo root. RUN is an absolute
# path to a scratch copy of the encode with 0030's photographic characters.json, shot fixes,
# recast voice map and speaker fixes applied; never the primary checkout.
#
#   RUN=/abs/scratch/star_wars_iv_v2_mixed EV=/abs/evidence MODELS=/abs/face_models \
#     BASELINE=28.76 bash docs/research/0034-sw4-mixed-shot-keyframes/window_mixed.sh
#
# One spend cap (CAP, default $6) for the whole task, measured from BASELINE (the balance
# when the task started), so a re-run or a later step can never take the total past it.
# Idempotent: decode, speech and stitch resume from their progress files.
set -euo pipefail

RUN=${RUN:?absolute path of the scratch encode copy}
EV=${EV:?absolute path of the evidence dir}
MODELS=${MODELS:?dir holding yunet.onnx and sface.onnx}
BASELINE=${BASELINE:?balance the cap of this task is measured from}
CAP=${CAP:-6}
START=1284
LIMIT=38
KF=runpod-ltx25-keyframe
STYLE="Live-action 1977 science-fiction film footage of real actors and practical sets, photographed on 35mm film."
WINDOW=(--start-index "$START" --limit "$LIMIT")
PY=(uv run python)
export LOSSY_FACE_MODELS=$MODELS
mkdir -p "$EV"

balance() {
  "${PY[@]}" -c 'import sys; sys.path.insert(0, "tools"); import config; config.load_env()
from capped_run import balance_and_pods as b
bal, pods = b(); print(f"{bal:.2f} {len(pods)}")'
}

read -r NOW PODS < <(balance)
echo "RunPod balance \$$NOW (task baseline \$$BASELINE), $PODS pod(s) on the account" | tee "$EV/balance_start.txt"

# 1. Speech for the window with the recast voices (fal ElevenLabs, cents).
"${PY[@]}" decode.py "$RUN" --speech "${WINDOW[@]}" 2>&1 | tee "$EV/speech.log"

# 2. Video: keyframe-first with mixed shots, klein on the pod, one-stage 640x352.
"${PY[@]}" tools/capped_run.py --cap "$CAP" --margin 0.5 --baseline "$BASELINE" --output-dir "$RUN" \
  --progress "$RUN/decode_progress_$KF.json" --total-shots "$LIMIT" --no-project \
  -- "${PY[@]}" decode.py "$RUN" --strategy "$KF" --keyframe-generator pod --keyframe-mixed-shots \
  --ltx-one-stage --style-prefix "$STYLE" "${WINDOW[@]}" 2>&1 | tee "$EV/video.log"

# 3. Stitch with each clip's own LTX audio, Lanczos to 1280x704 ($0, local ffmpeg).
"${PY[@]}" decode.py "$RUN" --strategy "$KF" --stitch "${WINDOW[@]}" --audio-strategy native \
  --scale 1280x704 2>&1 | tee "$EV/stitch.log"

# 4. Pod hygiene: no pods, journal clean.
read -r FINAL PODS < <(balance)
{
  echo "task baseline \$$BASELINE  before this run \$$NOW  end \$$FINAL  this run \$$(awk -v a="$NOW" -v b="$FINAL" 'BEGIN{printf "%.2f", a - b}')  pods left: $PODS"
  "${PY[@]}" pod_journal.py list "$RUN" || true
} | tee "$EV/hygiene.txt"
