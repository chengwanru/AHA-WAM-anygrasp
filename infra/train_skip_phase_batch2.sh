#!/usr/bin/env bash
# CUDA 8×GPU — skip_phase batch2（10 个与 batch1 不同类型 task，与 batch1 并行跑）
# 提交：算法包 cwr_wulan2，入口选 train_skip_phase_batch2.sh（不要用 train_skip_phase.sh / train_mtp.sh）
# 需要另一台 8 卡机器，勿与 batch1 skip_phase / ah-cpp sweep 抢 GPU。
set -euo pipefail
export SKIP_PHASE_BATCH=batch2
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
exec "${SCRIPT_DIR}/train_skip_phase.sh"
