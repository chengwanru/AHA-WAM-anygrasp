#!/usr/bin/env bash
# RoboTwin AHA-WAM sweep 入口（由 train_mtp.sh 调用）
# 变量由 train_mtp.sh 写死/导出；此处只做兜底，无需你手动 set。
set -euo pipefail

export WANDB_MODE="${WANDB_MODE:-offline}"
export TORCH_NCCL_ENABLE_MONITORING="${TORCH_NCCL_ENABLE_MONITORING:-0}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-true}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-16}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"

# 卡数：优先平台 MA_NUM_GPUS，其次 train_mtp 导出的 NNPU，默认 8（全量）
export NNPU="${MA_NUM_GPUS:-${NNPU:-8}}"
NUM_EPISODES="${NUM_EPISODES:-40}"

if [[ -d "/opt/huawei/dataset" ]]; then
    DEFAULT_OUTPUT_DIR="/opt/huawei/dataset/cwr_dataset_wulann/aha-wam-runs/robotwin_ahawam_sweep_20tasks_40eps"
    _FFMPEG_BIN="/opt/huawei/dataset/cwr_dataset_wulann/bin"
else
    DEFAULT_OUTPUT_DIR="/home/ma-user/work/dataset/cwr_dataset_wulann/aha-wam-runs/robotwin_ahawam_sweep_20tasks_40eps"
    _FFMPEG_BIN="/home/ma-user/work/dataset/cwr_dataset_wulann/bin"
fi
if [[ -x "${_FFMPEG_BIN}/ffmpeg" ]]; then
    export PATH="${_FFMPEG_BIN}:${PATH}"
fi
OUTPUT_DIR="${1:-${DEFAULT_OUTPUT_DIR}}"
mkdir -p "${OUTPUT_DIR}"

echo "FINAL OUTPUT_DIR: ${OUTPUT_DIR}"
echo "FINAL NNPU: ${NNPU}"
echo "FINAL NUM_EPISODES: ${NUM_EPISODES}"

python_exec="${PYTHON:-python3}"
echo "Using python exec: ${python_exec}"
"${python_exec}" scripts/run_robotwin_sweep.py \
  --num_episodes "${NUM_EPISODES}" \
  --num_gpus "${NNPU}" \
  --output_dir "${OUTPUT_DIR}" \
  --timeout_s "${JOB_TIMEOUT_S:-43200}"
