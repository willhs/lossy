#!/usr/bin/env bash
# The 2-minute mid-film test before the 30 Nov run (task 462c92a3), all four arms.
#
# Needs RUNPOD_API_KEY and FAL_KEY in the environment (a Manager job with
# secrets=true, or `op run --env-file .env.tpl -- bash <this file>` on the Mac),
# plus LOSSY_FACE_MODELS (yunet.onnx + sface.onnx) for the keyframe strategy.
# Run from the repo root. RUN is an absolute path to a scratch copy of the encode
# with the recast voice map, speaker fixes and photographic characters.json
# applied (research.md, "Recast and 2-minute test"); never the primary checkout.
#
#   RUN=/abs/scratch/star_wars_iv_v2 EV=/abs/evidence bash docs/research/0030-sw4-full-run-preflight/window_test.sh
#
# Idempotent: decode, speech, audio and stitch all resume from their progress
# files, so a re-run after a stop picks up where it left off. One spend cap
# (CAP, default $5) covers every paid step: each capped_run is measured from
# the balance read in step 0.
set -euo pipefail

RUN=${RUN:?absolute path of the scratch encode copy}
EV=${EV:?absolute path of the evidence dir}
LO=${RUN}_lowres
SRC=${SRC:-/Users/will/projects/lossy/media/star_wars_iv.mp4}
CAP=${CAP:-5}
MIN_BALANCE=${MIN_BALANCE:-3}
START=1284
LIMIT=38
KF=runpod-ltx25-keyframe
STYLE="Live-action 1977 science-fiction film footage of real actors and practical sets, photographed on 35mm film."
WINDOW=(--start-index "$START" --limit "$LIMIT")
PY=(uv run python)
mkdir -p "$EV"

balance() {
  "${PY[@]}" -c 'import sys; sys.path.insert(0, "tools"); from capped_run import balance_and_pods as b
bal, pods = b(); print(f"{bal:.2f} {len(pods)}")'
}

capped() {  # capped <output dir> <progress json or ""> -- <decode args...>
  local out=$1 progress=$2; shift 3
  "${PY[@]}" tools/capped_run.py --cap "$CAP" --margin 0.5 --baseline "$BASELINE" --output-dir "$out" \
    ${progress:+--progress "$progress" --total-shots "$LIMIT"} --no-project \
    -- "${PY[@]}" decode.py "$out" "$@"
}

# 0. Balance gate: never rent below MIN_BALANCE; Will tops up, never this script.
read -r BASELINE PODS < <(balance)
echo "RunPod balance \$$BASELINE, $PODS pod(s) on the account" | tee "$EV/balance_start.txt"
if awk -v b="$BASELINE" -v m="$MIN_BALANCE" 'BEGIN{exit !(b < m)}'; then
  echo "needs a RunPod top-up (balance \$$BASELINE < \$$MIN_BALANCE); nothing rented" | tee -a "$EV/balance_start.txt"
  exit 10
fi

# 1. Recast speech for the window only (fal ElevenLabs, cents).
"${PY[@]}" decode.py "$RUN" --speech "${WINDOW[@]}" 2>&1 | tee "$EV/speech.log"

# 2. Planned arm: keyframe-first, klein on the pod, full resolution.
capped "$RUN" "$RUN/decode_progress_$KF.json" -- --strategy "$KF" --keyframe-generator pod \
  --style-prefix "$STYLE" "${WINDOW[@]}" 2>&1 | tee "$EV/video_planned.log"

# 3. MMAudio ambience for the window (planned arm's audio).
capped "$RUN" "" -- --audio --audio-strategy runpod-mmaudio "${WINDOW[@]}" 2>&1 | tee "$EV/mmaudio.log"

# 4. Low-res arm (Arm B): same encode, seeds (shot index) and keyframes, one-stage 640x352.
#    Its own dir, because clips and progress are keyed by strategy name, not resolution.
mkdir -p "$LO"
rsync -a --exclude clips --exclude adjusted --exclude 'audio_adjusted' --exclude 'decode_progress_*' \
  --exclude runpod_journal --exclude '*.mp4' "$RUN/" "$LO/"
capped "$LO" "$LO/decode_progress_$KF.json" -- --strategy "$KF" --keyframe-generator pod --ltx-one-stage \
  --style-prefix "$STYLE" "${WINDOW[@]}" 2>&1 | tee "$EV/video_lowres.log"

# 5. Four stitches ($0, local ffmpeg). Arm A is the same clips with their own LTX audio.
for dir in "$RUN" "$LO"; do
  scale=(); [ "$dir" = "$LO" ] && scale=(--scale 1280x704)
  for audio in runpod-mmaudio native; do
    "${PY[@]}" decode.py "$dir" --strategy "$KF" --stitch "${WINDOW[@]}" --audio-strategy "$audio" "${scale[@]}"
  done
done 2>&1 | tee "$EV/stitch.log"

# 6. Source cut, side-by-side and contact sheet for the planned arm.
read -r T0 T1 < <("${PY[@]}" -c "import json; s={x['index']:x for x in json.load(open('$RUN/shots.json'))['shots']}
print(s[$START]['start_s'], s[$START + $LIMIT - 1]['end_s'])")
DUR=$(awk -v a="$T0" -v b="$T1" 'BEGIN{print b - a}')
FILM=$(basename "$RUN")
PLANNED="$RUN/${FILM}_reconstructed_${KF}+runpod-mmaudio.mp4"
LOWRES="$LO/$(basename "$LO")_reconstructed_${KF}+runpod-mmaudio.mp4"
ffmpeg -y -v error -ss "$T0" -t "$DUR" -i "$SRC" -vf scale=1280:704,setsar=1 -c:a aac "$EV/source_window.mp4"
ffmpeg -y -v error -i "$EV/source_window.mp4" -i "$PLANNED" \
  -filter_complex "[0:v]scale=640:352[a];[1:v]scale=640:352[b];[a][b]hstack" -map 1:a? -shortest "$EV/side_by_side.mp4"
ffmpeg -y -v error -i "$PLANNED" -vf "fps=1/3,scale=320:-2,tile=6x7" -frames:v 1 "$EV/contact_sheet.jpg"
ffmpeg -y -v error -i "$PLANNED" -i "$LOWRES" \
  -filter_complex "[0:v]fps=1/10,scale=640:352[a];[1:v]fps=1/10,scale=640:352[b];[a][b]hstack,tile=1x6" \
  -frames:v 1 "$EV/res_compare.jpg"

# 7. Pod hygiene: no pods, journals clean.
read -r FINAL PODS < <(balance)
{
  echo "start \$$BASELINE  end \$$FINAL  spent \$$(awk -v a="$BASELINE" -v b="$FINAL" 'BEGIN{printf "%.2f", a - b}')  pods left: $PODS"
  "${PY[@]}" pod_journal.py list "$RUN" || true
  "${PY[@]}" pod_journal.py list "$LO" || true
} | tee "$EV/hygiene.txt"
