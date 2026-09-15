#!/usr/bin/env bash
# 8×GPU — HE adapt arm He=32 (train then eval)
export HE_ADAPT_ARM=32
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
exec bash "${SCRIPT_DIR}/train_finetune_he_adapt.sh" "$@"
