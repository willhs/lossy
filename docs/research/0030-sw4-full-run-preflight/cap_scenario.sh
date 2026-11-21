#!/usr/bin/env bash
# Research 0030: the $27 cap firing mid-run under the REAL tools/capped_run.py,
# then a resume. Fake account (rehearse.py): no key, no pod, no spend.
#   bash cap_scenario.sh <encode_dir> <scratch_dir>
set -u
ENC=$1; R=$2
H="$(cd "$(dirname "$0")" && pwd)/rehearse.py"
rm -rf "$R"; mkdir -p "$R"; cp -p "$ENC/shots.json" "$ENC/characters.json" "$R/"
export REHEARSE_ACCOUNT=$R/account.json
P=(--progress "$R/decode_progress_runpod-ltx25.json" --total-shots 300 --no-project --poll 1)
CHILD=(uv run python "$H" decode-child --delay 0.05 -- "$R" --strategy runpod-ltx25 --limit 300)

echo '{"start_balance": 30.0, "rate": 2.0, "pods": []}' > "$R/account.json"   # $2/s: the line is hit in ~13 s
env -u RUNPOD_API_KEY uv run python "$H" supervise -- --cap 27 --margin 0.5 "${P[@]}" -- "${CHILD[@]}"
echo "run1 rc=$?"
uv run python -c "import json;p=json.load(open('$R/decode_progress_runpod-ltx25.json'));print('after cap: completed',len(p['completed']),'failed',p['failed'])"
echo "clips on disk: $(ls "$R/clips/runpod-ltx25" | wc -l)"
echo "live fake pods after run1: $(python3 -c "import json;print([p['id'] for p in json.load(open('$R/account.json'))['pods'] if not p['ended']])")"

echo '{"start_balance": 30.0, "rate": 0.0001, "pods": []}' > "$R/account.json"
env -u RUNPOD_API_KEY uv run python "$H" supervise -- --cap 27 --margin 0.5 "${P[@]}" -- "${CHILD[@]}"
echo "run2 rc=$?"
uv run python -c "import json;p=json.load(open('$R/decode_progress_runpod-ltx25.json'));print('after resume: completed',len(set(p['completed'])),'failed',p['failed'])"
