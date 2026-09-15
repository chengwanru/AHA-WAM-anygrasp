#!/usr/bin/env bash
# 8×GPU — CPP1 AB arm B (adapt step_010000 @ ah64/cpp1/always)
export CPP1_AB_ARM=adapt
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
exec bash "${SCRIPT_DIR}/train_eval_cpp1_ab.sh" "$@"
