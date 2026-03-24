#!/usr/bin/env bash
# Usage: ./tools/vram_benchmark.sh <output_dir>
# Requires: an active pod (run decode.py first with --keep-pod, or create manually)
# Reads connection info from <output_dir>/runpod_pod.json

set -euo pipefail

OUTPUT_DIR="${1:?Usage: $0 <output_dir>}"
POD_STATE="$OUTPUT_DIR/runpod_pod.json"

if [ ! -f "$POD_STATE" ]; then
    echo "Error: $POD_STATE not found. Start a pod first."
    exit 1
fi

SSH_HOST=$(python3 -c "import json; print(json.load(open('$POD_STATE'))['ssh_host'])")
SSH_PORT=$(python3 -c "import json; print(json.load(open('$POD_STATE'))['ssh_port'])")
BASE_URL=$(python3 -c "import json; print(json.load(open('$POD_STATE'))['base_url'])")

SSH_CMD="ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR -p $SSH_PORT root@$SSH_HOST"

echo "=== VRAM Benchmark ==="
echo "Pod: $SSH_HOST:$SSH_PORT"
echo ""

# Baseline VRAM
BASELINE=$($SSH_CMD "nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits" | tr -d '[:space:]')
echo "Baseline VRAM: ${BASELINE} MiB"

# Start VRAM logging
$SSH_CMD "nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -l 1 > /tmp/vram_log.csv &"

echo ""
echo "--- Wan Video Generation ---"
echo "Submit a single Wan workflow via ComfyUI API and monitor peak VRAM."
echo "(Run this while a video clip is generating)"
echo ""

echo "Instructions:"
echo "  1. In another terminal, generate 1 video clip:"
echo "     python decode.py <output_dir> --strategy runpod-wan --limit 1 --keep-pod"
echo "  2. While it generates, this script monitors VRAM"
echo "  3. After completion, press Enter here"
read -p "Press Enter when video generation is complete..."

# Read peak from log
VIDEO_PEAK=$($SSH_CMD "sort -n /tmp/vram_log.csv | tail -1" | tr -d '[:space:]')
echo "Wan peak VRAM: ${VIDEO_PEAK} MiB"

# Reset log
$SSH_CMD "curl -s -X POST $BASE_URL/free -H 'Content-Type: application/json' -d '{\"free_memory\": true}' > /dev/null 2>&1 || true"
sleep 3
$SSH_CMD "> /tmp/vram_log.csv"

echo ""
echo "--- MMAudio Audio Generation ---"
read -p "Now generate 1 audio clip and press Enter when done..."

AUDIO_PEAK=$($SSH_CMD "sort -n /tmp/vram_log.csv | tail -1" | tr -d '[:space:]')
echo "MMAudio peak VRAM: ${AUDIO_PEAK} MiB"

# Cleanup
$SSH_CMD "pkill -f 'nvidia-smi.*-l' || true"

echo ""
echo "=== Results ==="
echo "Baseline:        ${BASELINE} MiB"
echo "Wan peak:        ${VIDEO_PEAK} MiB"
echo "MMAudio peak:    ${AUDIO_PEAK} MiB"
echo "Sum (concurrent): $((VIDEO_PEAK + AUDIO_PEAK - BASELINE)) MiB"
echo "RTX 4090 total:  24576 MiB"
echo "Headroom:        $((24576 - VIDEO_PEAK - AUDIO_PEAK + BASELINE)) MiB"
