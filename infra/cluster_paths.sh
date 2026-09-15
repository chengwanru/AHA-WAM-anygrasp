#!/usr/bin/env bash
# Shared cluster / explore path detection for AHA-WAM jobs.
# Source this file (do not execute). Sets:
#   DATA, AHA_PLATFORM, AHA_WAM_CODE_DIR, RUNS_ROOT, DATASET_ROOT,
#   TEXT_EMBEDS_DIR, CKPT_DIR, INIT_CKPT, NORM_STATS, WHEELS_DIR,
#   SAPIEN_LIBS, NVIDIA_DRIVER_DIR, AHAWAM_ENV_PREFIX
#
# Platforms:
#   wulan_aha — new: dataset/cwr_wulan_aha + algorithm/cwr_wulan2
#   wulann    — old: dataset/cwr_dataset_wulann + algorithm/cwr_wulan_algorithm
#
# If this file was already sourced in the current shell, it is a no-op.

if [[ -n "${AHA_CLUSTER_PATHS_LOADED:-}" ]]; then
    return 0 2>/dev/null || true
fi

_aha_has_repo() {
    local root="$1" folder="$2"
    [[ -d "${root}/${folder}/AHA-WAM-anygrasp" ]]
}

DATA=""
if _aha_has_repo /opt/huawei/dataset cwr_wulan_aha || _aha_has_repo /opt/huawei/dataset cwr_dataset_wulann; then
    DATA="/opt/huawei/dataset"
elif _aha_has_repo /home/ma-user/work/dataset cwr_wulan_aha || _aha_has_repo /home/ma-user/work/dataset cwr_dataset_wulann; then
    DATA="/home/ma-user/work/dataset"
else
    echo "ERROR: cannot find AHA-WAM-anygrasp under /opt/huawei/dataset or /home/ma-user/work/dataset"
    echo "  expected .../cwr_wulan_aha/AHA-WAM-anygrasp  (new platform)"
    echo "       or .../cwr_dataset_wulann/AHA-WAM-anygrasp (old platform)"
    return 1 2>/dev/null || exit 1
fi
export DATA

# Prefer the new-platform repo when both exist.
if [[ -d "${DATA}/cwr_wulan_aha/AHA-WAM-anygrasp" ]]; then
    export AHA_PLATFORM="wulan_aha"
    export AHA_WAM_CODE_DIR="${DATA}/cwr_wulan_aha/AHA-WAM-anygrasp"
    export RUNS_ROOT="${DATA}/cwr_wulan_aha/aha-wam-runs"
    export AHAWAM_ENV_PREFIX="${DATA}/cwr_wulan_aha/envs/ahawam"
else
    export AHA_PLATFORM="wulann"
    export AHA_WAM_CODE_DIR="${DATA}/cwr_dataset_wulann/AHA-WAM-anygrasp"
    export RUNS_ROOT="${DATA}/cwr_dataset_wulann4/aha-wam-runs"
    export AHAWAM_ENV_PREFIX="${DATA}/cwr_dataset_wulann4/envs/ahawam"
fi

DATASET_ROOT=""
for _cand in \
    "${DATA}/cwr_dataset_wulann2/robotwin2.0" \
    "${DATA}/cwr_wulan_aha/robotwin2.0" \
    "${DATA}/cwr_wulan_aha/data/robotwin2.0"
do
    if [[ -d "${_cand}" ]]; then
        DATASET_ROOT="${_cand}"
        break
    fi
done
export DATASET_ROOT

TEXT_EMBEDS_DIR=""
for _cand in \
    "${DATA}/cwr_dataset_wulann4/text_embeds_cache/robotwin" \
    "${DATA}/cwr_wulan_aha/text_embeds_cache/robotwin"
do
    if [[ -d "${_cand}" ]]; then
        TEXT_EMBEDS_DIR="${_cand}"
        break
    fi
done
if [[ -z "${TEXT_EMBEDS_DIR}" ]]; then
    if [[ "${AHA_PLATFORM}" == "wulan_aha" ]]; then
        TEXT_EMBEDS_DIR="${DATA}/cwr_wulan_aha/text_embeds_cache/robotwin"
    else
        TEXT_EMBEDS_DIR="${DATA}/cwr_dataset_wulann4/text_embeds_cache/robotwin"
    fi
fi
export TEXT_EMBEDS_DIR

CKPT_DIR=""
for _cand in \
    "${AHA_WAM_CODE_DIR}/checkpoints/AHA-WAM-RoboTwin2.0" \
    "${DATA}/cwr_wulan_aha/checkpoints/AHA-WAM-RoboTwin2.0" \
    "${DATA}/cwr_dataset_wulann/AHA-WAM-anygrasp/checkpoints/AHA-WAM-RoboTwin2.0"
do
    if [[ -f "${_cand}/robotwin_ahawam.pt" ]]; then
        CKPT_DIR="${_cand}"
        break
    fi
done
if [[ -z "${CKPT_DIR}" ]]; then
    CKPT_DIR="${AHA_WAM_CODE_DIR}/checkpoints/AHA-WAM-RoboTwin2.0"
fi
export CKPT_DIR
export INIT_CKPT="${CKPT_DIR}/robotwin_ahawam.pt"

NORM_STATS=""
for _cand in \
    "${CKPT_DIR}/dataset_stats.json" \
    "${AHA_WAM_CODE_DIR}/infra/assets/robotwin_dataset_stats.json" \
    "${DATA}/cwr_wulan_aha/checkpoints/AHA-WAM-RoboTwin2.0/dataset_stats.json"
do
    if [[ -f "${_cand}" ]]; then
        NORM_STATS="${_cand}"
        break
    fi
done
export NORM_STATS

WHEELS_DIR=""
for _cand in \
    "${DATA}/cwr_wulan_aha/wheels" \
    "${DATA}/cwr_dataset_wulann/wheels" \
    "${DATA}/cwr_dataset_wulann4/wheels"
do
    if [[ -d "${_cand}" ]]; then
        WHEELS_DIR="${_cand}"
        break
    fi
done
export WHEELS_DIR

SAPIEN_LIBS=""
for _cand in \
    "${DATA}/cwr_wulan_aha/sapien-runtime-libs" \
    "${DATA}/cwr_dataset_wulann/sapien-runtime-libs"
do
    if [[ -d "${_cand}" ]]; then
        SAPIEN_LIBS="${_cand}"
        break
    fi
done
export SAPIEN_LIBS

NVIDIA_DRIVER_DIR=""
for _cand in \
    "${DATA}/cwr_wulan_aha/nvidia-driver-libs/nvidia-535.183.01" \
    "${DATA}/cwr_dataset_wulann/nvidia-driver-libs/nvidia-535.183.01"
do
    if [[ -d "${_cand}" ]]; then
        NVIDIA_DRIVER_DIR="${_cand}"
        break
    fi
done
export NVIDIA_DRIVER_DIR

export AHA_CLUSTER_PATHS_LOADED=1
echo "cluster_paths: platform=${AHA_PLATFORM} DATA=${DATA}"
echo "cluster_paths: AHA_WAM_CODE_DIR=${AHA_WAM_CODE_DIR}"
echo "cluster_paths: RUNS_ROOT=${RUNS_ROOT}"
echo "cluster_paths: DATASET_ROOT=${DATASET_ROOT:-<missing>}"
echo "cluster_paths: INIT_CKPT=${INIT_CKPT}"
echo "cluster_paths: NORM_STATS=${NORM_STATS:-<missing>}"
