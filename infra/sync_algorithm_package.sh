#!/usr/bin/env bash
# Copy submit-entry scripts from this git repo (infra/) into the platform
# algorithm package. Source of truth is always AHA-WAM-anygrasp; the algorithm
# folder is only what the cluster job launcher executes.
#
# New platform: algorithm/cwr_wulan2
# Old platform: algorithm/cwr_wulan_algorithm
#
# Usage (after git pull on the target machine):
#   bash infra/sync_algorithm_package.sh
#   ALG_PKG=/path/to/cwr_wulan2 bash infra/sync_algorithm_package.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

detect_alg_pkg() {
    local cand
    if [[ -n "${ALG_PKG:-}" ]]; then
        echo "${ALG_PKG}"
        return 0
    fi
    for cand in \
        "/home/ma-user/work/algorithm/cwr_wulan2" \
        "/opt/huawei/work/algorithm/cwr_wulan2" \
        "/home/ma-user/work/algorithm/cwr_wulan_algorithm" \
        "/opt/huawei/work/algorithm/cwr_wulan_algorithm"
    do
        if [[ -d "${cand}" ]]; then
            echo "${cand}"
            return 0
        fi
    done
    return 1
}

if ! ALG="$(detect_alg_pkg)"; then
    echo "ERROR: no algorithm package directory found."
    echo "  New platform expected: /home/ma-user/work/algorithm/cwr_wulan2"
    echo "  Old platform expected: /home/ma-user/work/algorithm/cwr_wulan_algorithm"
    echo "  Or set ALG_PKG=/abs/path"
    exit 1
fi

echo "repo:    ${REPO_ROOT}"
echo "alg pkg: ${ALG}"

# Files that the cluster entrypoint actually runs (must sit at package ROOT).
COPY_INFRA=(
    cluster_paths.sh
    watch_train_progress.py
    train_finetune_he_adapt.sh
    train_finetune_he8.sh
    train_finetune_he32.sh
    train_eval_he_adapt.sh
    train_eval_he8.sh
    train_eval_he32.sh
    SUBMIT_FINETUNE_HE_ADAPT.md
)

OPTIONAL_INFRA=(
    train_eval_cpp1_ab.sh
    train_eval_cpp1_ab_released.sh
    train_eval_cpp1_ab_adapt.sh
    SUBMIT_EVAL_CPP1_AB.md
    train_finetune_cpp1_ditkv.sh
    SUBMIT_FINETUNE_CPP1_DITKV.md
    train_probe_vulkan.sh
    SUBMIT_PROBE_VULKAN.md
)

copy_one() {
    local src="$1" dest="$2" required="$3"
    if [[ ! -f "${src}" ]]; then
        if [[ "${required}" == "1" ]]; then
            echo "ERROR: missing ${src}"
            exit 1
        fi
        echo "skip (absent): $(basename "${src}")"
        return 0
    fi
    cp -f "${src}" "${dest}"
    if [[ "${src}" == *.sh || "${src}" == *.py ]]; then
        chmod +x "${dest}" 2>/dev/null || true
    fi
    echo "copied $(basename "${src}") -> ${dest}"
}

for name in "${COPY_INFRA[@]}"; do
    copy_one "${SCRIPT_DIR}/${name}" "${ALG}/${name}" 1
done

if [[ "${SYNC_OPTIONAL:-0}" == "1" ]]; then
    echo "--- optional older entries (SYNC_OPTIONAL=1) ---"
    for name in "${OPTIONAL_INFRA[@]}"; do
        copy_one "${SCRIPT_DIR}/${name}" "${ALG}/${name}" 0
    done
fi

if [[ -f "${REPO_ROOT}/docs/EXPERIMENT_REGISTRY.md" ]]; then
    copy_one "${REPO_ROOT}/docs/EXPERIMENT_REGISTRY.md" "${ALG}/EXPERIMENT_REGISTRY.md" 1
fi
if [[ -f "${REPO_ROOT}/docs/PLATFORM_MIGRATION.md" ]]; then
    copy_one "${REPO_ROOT}/docs/PLATFORM_MIGRATION.md" "${ALG}/PLATFORM_MIGRATION.md" 0
fi

# Tiny stats JSON (not the 14G pt). Helps eval if ckpt dir has no stats yet.
mkdir -p "${ALG}/infra/assets"
if [[ -f "${SCRIPT_DIR}/assets/robotwin_dataset_stats.json" ]]; then
    cp -f "${SCRIPT_DIR}/assets/robotwin_dataset_stats.json" \
        "${ALG}/infra/assets/robotwin_dataset_stats.json"
    echo "copied robotwin_dataset_stats.json -> ${ALG}/infra/assets/"
fi

echo ""
echo "DONE. Platform submit package should select entry: train_finetune_he_adapt.sh"
echo "  (algorithm package name: $(basename "${ALG}"))"
echo "Code + Hydra yamls still live in the git repo under dataset/.../AHA-WAM-anygrasp"
echo "See docs/PLATFORM_MIGRATION.md"
