#!/usr/bin/env bash
# CUDA 8×GPU — random_skip eval batch2
set -euo pipefail
export SKIP_PHASE_BATCH=batch2
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
exec "${SCRIPT_DIR}/train_eval_random_skip.sh"
