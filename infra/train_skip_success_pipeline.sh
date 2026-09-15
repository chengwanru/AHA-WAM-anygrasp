#!/usr/bin/env bash
# =============================================================================
# ONE JOB: FSM-skip success FULL collect → Success-BC finetune (4000 steps)
# Default is FULL (8 tasks × 20 ep), NOT smoke.
# Platform: pick THIS entry. Mounts: wulann~wulann4.
# Timeout: ≥48h recommended (lavapipe collect + train).
# =============================================================================
set -euo pipefail

REVISION="2026-09-11-skip-success-pipeline-v3-full"
echo "train_skip_success_pipeline.sh revision: ${REVISION}"
echo "EXPERIMENT=SKIP_SUCCESS_PIPELINE (FULL collect then train)"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

COLLECT_SH="${SCRIPT_DIR}/train_collect_skip_success.sh"
TRAIN_SH="${SCRIPT_DIR}/train_finetune_skip_success_bc.sh"

if [[ ! -f "${COLLECT_SH}" ]]; then
    echo "ERROR: missing ${COLLECT_SH}"
    exit 1
fi
if [[ ! -f "${TRAIN_SH}" ]]; then
    echo "ERROR: missing ${TRAIN_SH}"
    exit 1
fi

if [[ -d "/opt/huawei/dataset/cwr_dataset_wulann/AHA-WAM-anygrasp" ]]; then
    DATA="/opt/huawei/dataset"
else
    DATA="/home/ma-user/work/dataset"
fi
# FULL path only — do not auto-skip using smoke leftover
export SKIP_PHASE_BATCH=full
LEROBOT="${DATA}/cwr_dataset_wulann4/aha-wam-runs/robotwin/skip_success_collect_full/lerobot_success"

echo "COLLECT_SH=${COLLECT_SH}"
echo "TRAIN_SH=${TRAIN_SH}"
echo "LEROBOT_EXPECT=${LEROBOT}"
echo "SKIP_PHASE_BATCH=full (forced)"
echo "NOTE: success-BC data = FSM-skip SUCCESS rollouts only; NOT all robotwin2.0 demos (~27.5k)."
echo "      full collect attempts ≈ 8 tasks × 20 ep = 160; kept successes depend on SR."

# ---------- Stage 1: collect (skip only if FULL lerobot exists) ----------
if [[ -f "${LEROBOT}/meta/info.json" ]]; then
    echo "[pipeline] full lerobot_success already exists → skip collect"
    echo "[pipeline] (delete ${LEROBOT} to force re-collect)"
    python3 - <<PY
import json
info=json.load(open("${LEROBOT}/meta/info.json"))
print(f"[pipeline] existing full dataset: episodes={info.get('total_episodes')} frames={info.get('total_frames')}")
PY
else
    echo "================================================================================"
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] PIPELINE STAGE 1/2: COLLECT (full)"
    echo "================================================================================"
    bash "${COLLECT_SH}"
    if [[ ! -f "${LEROBOT}/meta/info.json" ]]; then
        echo "ERROR: collect finished but missing ${LEROBOT}/meta/info.json"
        exit 1
    fi
fi

# ---------- Stage 2: train ----------
echo "================================================================================"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] PIPELINE STAGE 2/2: TRAIN"
echo "================================================================================"
bash "${TRAIN_SH}"

echo "[$(date '+%Y-%m-%d %H:%M:%S')] PIPELINE DONE"
echo "ckpt under: ${DATA}/cwr_dataset_wulann4/aha-wam-runs/train/skip_success_bc_full_8x4k/"
echo "progress:   ${DATA}/cwr_dataset_wulann4/aha-wam-runs/train/skip_success_bc_full_8x4k/progress/"
