#!/usr/bin/env bash
# Launch OVCR diagnostic probe sweep on 2 local GPUs (~10h budget).
set -euo pipefail

ROOT="/home/ma-user/work/dataset/cwr_dataset_wulann/AHA-WAM-anygrasp"
OUT="/home/ma-user/work/dataset/cwr_dataset_wulann/aha-wam-runs/robotwin/ovcr改进"
LOG="/tmp/ovcr_diag_sweep.log"
PYTHON="${PYTHON:-/opt/huawei/miniconda/envs/python39/bin/python}"

cd "$ROOT"
export PYTHONPATH="${ROOT}/src${PYTHONPATH:+:$PYTHONPATH}"

echo "python: $PYTHON"
"$PYTHON" -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), 'n', torch.cuda.device_count())"

mkdir -p "$OUT"
nohup "$PYTHON" -u experiments/robotwin/run_ovcr_diag_sweep.py \
  --gpus 0,1 \
  --hours-per-gpu 10 \
  --num-episodes 5 \
  --action-horizon 64 \
  --cpp 2 \
  --output-dir "$OUT" \
  > "$LOG" 2>&1 &

echo "launched pid=$!"
echo "log: $LOG"
echo "out: $OUT"
