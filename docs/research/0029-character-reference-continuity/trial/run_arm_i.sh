#!/usr/bin/env bash
# Research 0029 arm I (LTX-2.5 Ingredients IC-LoRA), end to end: ONE command, run from anywhere.
#
#   docs/research/0029-character-reference-continuity/trial/run_arm_i.sh RUN_DIR [CAP_USD]
#
# RUN_DIR holds trial_plan.json, trial_manifest.json and sheets/<name>.png (see
# make_ingredients_arm.py). CAP_USD is the hard RunPod spend cap (default 1.00).
# Step 1 is free and stops with the fix if the Hugging Face gate is closed, before any pod.
# Step 2 spends: it is the only paid step, under capped_run's cap.
set -euo pipefail
RUN_DIR=${1:?usage: run_arm_i.sh RUN_DIR [CAP_USD]}
CAP=${2:-1.00}
cd "$(dirname "$0")/../../../.."

uv run python docs/research/0029-character-reference-continuity/trial/make_ingredients_arm.py "$RUN_DIR"
uv run python tools/capped_run.py --cap "$CAP" --margin 0.05 --no-project -- \
    uv run python decode.py "$RUN_DIR" --strategy runpod-ltx25-ingredients \
    --ab-manifest "$RUN_DIR/trial_manifest_I.json"
