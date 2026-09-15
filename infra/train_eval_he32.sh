#!/usr/bin/env bash
# 8×GPU — HE adapt eval only, He=32
export HE_ADAPT_ARM=32
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
exec bash "${SCRIPT_DIR}/train_eval_he_adapt.sh" "$@"
