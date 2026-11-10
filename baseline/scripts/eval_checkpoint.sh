#!/usr/bin/env bash
# Serve one fine-tuned checkpoint, evaluate it on the test suites, stop the server. From the ME295 root:
#   bash baseline/scripts/eval_checkpoint.sh <step> [gpu] [port] [extra evaluate.py args...]
# e.g. bash baseline/scripts/eval_checkpoint.sh 2500 1 8000 --suites nominal,random -n 20
set -euo pipefail
STEP=$1; GPU=${2:-1}; PORT=${3:-8000}; shift $(( $# < 3 ? $# : 3 ))
CKPT=baseline/checkpoints/pi05_librarian_lora/v1/$STEP
PY=baseline/third_party/openpi/.venv/bin/python
LOG=baseline/results/eval/server_step${STEP}_port${PORT}.log   # one log per server instance
mkdir -p baseline/results/eval
[ -d "$CKPT" ] || { echo "no checkpoint $CKPT"; exit 1; }
CUDA_VISIBLE_DEVICES=$GPU XLA_PYTHON_CLIENT_PREALLOCATE=false \
  $PY -m baseline.openpi_ext.cli serve pi05_librarian_lora --dir "$CKPT" --port "$PORT" > "$LOG" 2>&1 &
SERVER=$!
trap 'kill $SERVER 2>/dev/null || true' EXIT
until grep -q "server listening" "$LOG"; do
  kill -0 $SERVER 2>/dev/null || { echo "server died:"; tail -20 "$LOG"; exit 1; }
  sleep 5
done
python baseline/scripts/evaluate.py --policy pi05_ft --port "$PORT" --tag "step$STEP" "$@"
