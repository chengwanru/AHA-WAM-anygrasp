#!/usr/bin/env bash
# =============================================================================
# EVAL: same-budget RANDOM skip vs MAIN-2 FSM (supervision-signal experiment)
# - ckpt: released robotwin_ahawam.pt (NOT skip_v2 finetune)
# - mode: random_skip ONLY (reuse MAIN-2 baseline + skip_phase numbers)
# - 19 MAIN-2 tasks × 20 ep (≈5h wall on 2×8 GPUs)
# - py>=3.10 + PyPI sapien + abs Vulkan ICD
# =============================================================================
# 提交：两台 8 卡并行
#   算法包 cwr_wulan_algorithm
#   入口1: train_eval_random_skip.sh
#   入口2: train_eval_random_skip_batch2.sh
# 挂载：wulann + wulann2 + wulann3 + wulann4（必须有 wulann4/envs/ahawam）
# 超时：≥6h（目标约 5h 出齐）
set -euo pipefail
# Requires AHA-WAM-anygrasp: eval_robotwin_single abs-ICD + test_render exit(1) + sweep job_complete fix

echo "train_eval_random_skip.sh revision: 2026-09-09-random-skip-lavapipe-v12-epidx"
# ===================== 写死配置（你不用 set）=====================
SKIP_PHASE_BATCH="${SKIP_PHASE_BATCH:-batch1}"
SMOKE=0
SMOKE_NUM_EPISODES=1
# 20 ep: ~half of MAIN-2 40ep wall; fits ~5h on 2×8 with headroom
FULL_NUM_EPISODES=20
export FIXED_ACTION_HORIZON=64
export FIXED_CPP=2
export SKIP_PHASE_NEAR_THRESH_M=0.10
export SKIP_PHASE_MAX_CONSECUTIVE_SKIPS=1
# p=1.0 + max_consec=1 ⇒ same consecutive budget as FSM when always eligible
export RANDOM_SKIP_P="${RANDOM_SKIP_P:-1.0}"
export RANDOM_SKIP_SEED="${RANDOM_SKIP_SEED:-0}"
export OVCR_DIAG_MODE=baseline
# Success-only: skip mp4 encode/write (policy RGB still needed for VLA)
export ROBOTWIN_EVAL_VIDEO_LOG=0
export SKIP_PHASE_BATCH

# Released ckpt (cluster path; explore fallback applied later)
export SKIP_PHASE_CKPT="${SKIP_PHASE_CKPT:-/opt/huawei/dataset/cwr_dataset_wulann/AHA-WAM-anygrasp/checkpoints/AHA-WAM-RoboTwin2.0/robotwin_ahawam.pt}"
export SKIP_PHASE_DATASET_STATS="${SKIP_PHASE_DATASET_STATS:-/opt/huawei/dataset/cwr_dataset_wulann/AHA-WAM-anygrasp/checkpoints/AHA-WAM-RoboTwin2.0/dataset_stats.json}"
EVAL_MODES=(random_skip)

case "${SKIP_PHASE_BATCH}" in
batch1)
    HARDCODED_OUTPUT_DIR="/opt/huawei/dataset/cwr_dataset_wulann4/aha-wam-runs/robotwin/video_dit_random_skip_smoke_v12_lvp"
    HARDCODED_OUTPUT_DIR_FULL="/opt/huawei/dataset/cwr_dataset_wulann4/aha-wam-runs/robotwin/video_dit_random_skip_20eps_v12_lvp"
    SKIP_PHASE_TASKS=(
        handover_mic hanging_mug move_stapler_pad place_bread_basket place_mouse_pad
        place_object_basket stack_blocks_three stack_blocks_two click_bell
    )
    ;;
batch2)
    echo "train_eval_random_skip.sh revision: 2026-09-09-random-skip-lavapipe-v12-epidx-batch2"
    HARDCODED_OUTPUT_DIR="/opt/huawei/dataset/cwr_dataset_wulann4/aha-wam-runs/robotwin/video_dit_random_skip_smoke_v12_lvp_batch2"
    HARDCODED_OUTPUT_DIR_FULL="/opt/huawei/dataset/cwr_dataset_wulann4/aha-wam-runs/robotwin/video_dit_random_skip_20eps_v12_lvp_batch2"
    SKIP_PHASE_TASKS=(
        pick_dual_bottles pick_diverse_bottles place_a2b_left place_a2b_right move_can_pot
        stack_bowls_three handover_block lift_pot press_stapler turn_switch
    )
    ;;
*)
    echo "ERROR: unknown SKIP_PHASE_BATCH=${SKIP_PHASE_BATCH} (expected batch1|batch2)"
    exit 1
    ;;
esac
# ================================================================

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "${SCRIPT_DIR}"
echo "code dir: $(pwd)"

# ---------- 路径（自动探测，无需手动 set）----------
if [[ -d "/opt/huawei/dataset/cwr_dataset_wulann/AHA-WAM-anygrasp" ]]; then
    export DATA="/opt/huawei/dataset"
    export AHA_WAM_CODE_DIR="${DATA}/cwr_dataset_wulann/AHA-WAM-anygrasp"
else
    export DATA="/home/ma-user/work/dataset"
    export AHA_WAM_CODE_DIR="${DATA}/cwr_dataset_wulann/AHA-WAM-anygrasp"
fi

# Prefer shared offline wheels (sapien 3.0.3 is NOT on Huawei mirror; needs cp310+)
_wheels_has_sapien() {
    local d="$1"
    local f
    [[ -d "$d" ]] || return 1
    for f in "$d"/sapien-3.0.3*.whl; do
        [[ -f "$f" ]] || continue
        if python3 -c "import zipfile; z=zipfile.ZipFile('$f'); raise SystemExit(0 if z.testzip() is None else 1)" 2>/dev/null; then
            return 0
        fi
    done
    return 1
}

WHEELS_DIR=""
for _w in \
    "${DATA}/cwr_dataset_wulann/wheels" \
    "${DATA}/cwr_dataset_wulann4/wheels" \
    "${DATA}/cwr_wulan_aha/wheels"
do
    if _wheels_has_sapien "${_w}"; then
        WHEELS_DIR="${_w}"
        break
    fi
    if [[ -z "${WHEELS_DIR}" && -d "${_w}" ]]; then
        WHEELS_DIR="${_w}"
    fi
done
echo "AHA_WAM_CODE_DIR: ${AHA_WAM_CODE_DIR}"
echo "WHEELS_DIR: ${WHEELS_DIR:-<missing>}"
echo "SKIP_PHASE_BATCH=${SKIP_PHASE_BATCH} tasks=${#SKIP_PHASE_TASKS[@]}"
echo "FIXED ah=${FIXED_ACTION_HORIZON} cpp=${FIXED_CPP} near=${SKIP_PHASE_NEAR_THRESH_M} max_consec_skip=${SKIP_PHASE_MAX_CONSECUTIVE_SKIPS} OVCR=${OVCR_DIAG_MODE} RANDOM_SKIP_P=${RANDOM_SKIP_P} SEED=${RANDOM_SKIP_SEED}"

# Explore-machine path fallback for ckpt/stats
if [[ ! -d "/opt/huawei/dataset" ]]; then
    SKIP_PHASE_CKPT="${SKIP_PHASE_CKPT/\/opt\/huawei\/dataset/${DATA}}"
    SKIP_PHASE_DATASET_STATS="${SKIP_PHASE_DATASET_STATS/\/opt\/huawei\/dataset/${DATA}}"
    export SKIP_PHASE_CKPT SKIP_PHASE_DATASET_STATS
fi
echo "SKIP_PHASE_CKPT=${SKIP_PHASE_CKPT}"
echo "SKIP_PHASE_DATASET_STATS=${SKIP_PHASE_DATASET_STATS}"

# ---------- 运行环境变量（CUDA）----------
export WANDB_MODE="offline"
export TORCH_NCCL_ENABLE_MONITORING=0
export TOKENIZERS_PARALLELISM=true
export OMP_NUM_THREADS=16
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

# 平台注入；没有时给默认值，避免 set -u 报错
export NNPU="${MA_NUM_GPUS:-8}"
export NNODES="${MA_NUM_HOSTS:-1}"
export NODE_RANK="${VC_TASK_INDEX:-0}"
MASTER_HOST="${VC_WORKER_HOSTS:-127.0.0.1}"
export MASTER_ADDR="${MASTER_HOST%%,*}"
export MASTER_PORT="6070"

if [[ "${SMOKE}" == "1" ]]; then
    export NUM_EPISODES="${SMOKE_NUM_EPISODES}"
    OUTPUT_DIR="${HARDCODED_OUTPUT_DIR}"
else
    export NUM_EPISODES="${FULL_NUM_EPISODES}"
    OUTPUT_DIR="${HARDCODED_OUTPUT_DIR_FULL}"
fi
# 探索环境路径回退
if [[ ! -d "/opt/huawei/dataset" ]]; then
    OUTPUT_DIR="${OUTPUT_DIR/\/opt\/huawei\/dataset/${DATA}}"
fi

echo "SMOKE=${SMOKE} NUM_EPISODES=${NUM_EPISODES}"
echo "NNPU=${NNPU} NNODES=${NNODES} NODE_RANK=${NODE_RANK} MASTER_ADDR=${MASTER_ADDR}"
echo "OUTPUT_DIR=${OUTPUT_DIR}"
# 40 eps ≈ 6h+；默认 12h，避免跑到 38/40 被杀掉
export JOB_TIMEOUT_S="${JOB_TIMEOUT_S:-86400}"  # 24h — lavapipe; was 6h for GPU
echo "JOB_TIMEOUT_S=${JOB_TIMEOUT_S}"

# ---------- pip：华为源默认；sapien 等缺包时再走 PyPI ----------
export PIP_INDEX_URL="http://repo.myhuaweicloud.com/repository/pypi/simple"
export PIP_TRUSTED_HOST="repo.myhuaweicloud.com"
export PIP_EXTRA_INDEX_URL=""
export PIP_DISABLE_PIP_VERSION_CHECK=1
export PIP_DEFAULT_TIMEOUT=120
export PIP_CONFIG_FILE="/dev/null"
PYPI_INDEX_URL="https://pypi.org/simple"

# ---------- 强制 Python >=3.10（sapien 3.0.3 无 cp39 wheel；禁止镜像默认 3.9）----------
resolve_python() {
    local candidates=(
        "${DATA}/cwr_dataset_wulann4/envs/ahawam/bin/python"
        "/home/ma-user/anaconda3/envs/ahawam/bin/python"
        "${HOME}/anaconda3/envs/ahawam/bin/python"
    )
    local c
    for c in "${candidates[@]}"; do
        if [[ -x "$c" ]] && "$c" -c 'import sys; raise SystemExit(0 if sys.version_info[:2] >= (3, 10) else 1)' 2>/dev/null; then
            echo "$c"
            return 0
        fi
    done
    return 1
}

PYTHON=""
if PYTHON="$(resolve_python)"; then
    echo "Found Python>=3.10: ${PYTHON}"
else
    echo "ERROR: need Python>=3.10 for sapien==3.0.3 (image python3.9 cannot install it)"
    echo "Mount wulann4 (envs/ahawam) or create that env, then resubmit."
    exit 1
fi
export PYTHON
echo "Using python: ${PYTHON} ($("${PYTHON}" -V 2>&1))"
"${PYTHON}" - <<'PY'
import sys
assert sys.version_info[:2] >= (3, 10), f"need Python>=3.10, got {sys.version}"
print("python version gate: ok")
PY
# sapien 无头渲染（对齐 setup_env.sh）：
#   方案A：NVIDIA ICD（需要创建时注入 graphics + 可用 /dev/nvidia-modeset）
#   方案B：modeset 不可用时自动 lavapipe（软件 Vulkan；全量会很慢）
SAPIEN_LIBS="${DATA}/cwr_dataset_wulann/sapien-runtime-libs"
NVIDIA_DRIVER_DIR="${DATA}/cwr_dataset_wulann/nvidia-driver-libs/nvidia-535.183.01"
# 强制覆盖平台注入的 compute,utility（仅环境变量；modeset 仍依赖创建时 toolkit）
export NVIDIA_DRIVER_CAPABILITIES="compute,utility,graphics,display,video"
echo "NVIDIA_DRIVER_CAPABILITIES=${NVIDIA_DRIVER_CAPABILITIES} (forced)"
if [[ ! -e /dev/nvidia-modeset ]]; then
    mknod /dev/nvidia-modeset c 195 254 2>/dev/null || true
    chmod 666 /dev/nvidia-modeset 2>/dev/null || true
fi
AHAWAM_VULKAN_MODE="nvidia"
_MODESET_OK=0
if [[ -e /dev/nvidia-modeset ]] && "${PYTHON}" -c "import os; os.open('/dev/nvidia-modeset', os.O_RDWR)" >/dev/null 2>&1; then
    _MODESET_OK=1
    echo "nvidia-modeset: RDWR ok"
else
    echo "WARNING: nvidia-modeset missing/unusable — will try lavapipe (方案B)"
fi
_ld_add=()
if [[ -d "${NVIDIA_DRIVER_DIR}" ]]; then
    _ld_add+=("${NVIDIA_DRIVER_DIR}")
    echo "LD_LIBRARY_PATH += nvidia-535.183.01"
fi
if [[ -d "${SAPIEN_LIBS}" ]]; then
    _ld_add+=("${SAPIEN_LIBS}")
    echo "LD_LIBRARY_PATH += sapien-runtime-libs"
fi
# conda/env lib (ahawam) often needed for loader deps
_CONDA_LIB="$(cd "$(dirname "${PYTHON}")/../lib" && pwd)"
if [[ -d "${_CONDA_LIB}" ]]; then
    _ld_add+=("${_CONDA_LIB}")
    echo "LD_LIBRARY_PATH += ${_CONDA_LIB}"
fi
if [[ ${#_ld_add[@]} -gt 0 ]]; then
    _joined="$(IFS=:; echo "${_ld_add[*]}")"
    export LD_LIBRARY_PATH="${_joined}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
fi
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"
# 离线加载 DiffSynth/Wan 权重，禁止任务内下载
export DIFFSYNTH_SKIP_DOWNLOAD=true
export DIFFSYNTH_MODEL_BASE_PATH="${AHA_WAM_CODE_DIR}/checkpoints"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

# ---------- 离线安装依赖（不覆盖镜像 CUDA torch；只写 /tmp user site）----------
export PYTHONUSERBASE="/tmp/ahawam_user"
mkdir -p "${PYTHONUSERBASE}"
PYVER="$("${PYTHON}" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
export PATH="$(dirname "${PYTHON}"):${PYTHONUSERBASE}/bin:${PATH}"
export PYTHONPATH="${PYTHONUSERBASE}/lib/python${PYVER}/site-packages:${AHA_WAM_CODE_DIR}:${AHA_WAM_CODE_DIR}/src${PYTHONPATH:+:${PYTHONPATH}}"

_pip() {
    "${PYTHON}" -m pip "$@"
}

_need_import() {
    ! "${PYTHON}" -c "import $1" >/dev/null 2>&1
}

_install_robotwin_stack() {
    local mode="$1"  # wheels | hybrid | pypi
    # Must cover every third-party import on RoboTwin startup:
    # script/eval_policy.py -> envs -> utils (incl. pkl2hdf5/h5py) -> _base_task path
    local common=(
        "numpy==1.26.4"
        "opencv-python==4.8.1.78"
        transforms3d lxml pyperclip importlib_resources
        "sapien==3.0.3"
        "open3d==0.18.0"
        "mplib==0.2.1" toppra scipy
        gymnasium farama-notifications cloudpickle
        trimesh yourdfpy networkx
        "h5py==3.16.0"
        imageio imageio-ffmpeg
        # preflight also imports these (often already in ahawam; install if missing)
        PyYAML hydra-core omegaconf einops
        "transformers==4.49.0" av pandas pyarrow
        pillow wandb rich
    )
    if [[ "${mode}" == "wheels" ]]; then
        _pip install --user --no-index --find-links "${WHEELS_DIR}" "${common[@]}"
    elif [[ "${mode}" == "hybrid" ]]; then
        # Prefer local wheels (esp. sapien); fill gaps from PyPI (Huawei lacks sapien 3.0.3)
        _pip install --user --find-links "${WHEELS_DIR}" --index-url "${PYPI_INDEX_URL}" "${common[@]}"
    else
        _pip install --user --index-url "${PYPI_INDEX_URL}" "${common[@]}"
    fi
}

echo "--- install RoboTwin/sapien deps ---"
if _wheels_has_sapien "${WHEELS_DIR:-/nonexistent}"; then
    echo "WHEELS_DIR=${WHEELS_DIR} (valid sapien wheel) -> hybrid install"
    if [[ -f "${AHA_WAM_CODE_DIR}/requirements.txt" ]]; then
        _pip install --user --find-links "${WHEELS_DIR}" --index-url "${PYPI_INDEX_URL}" \
            -r "${AHA_WAM_CODE_DIR}/requirements.txt" || true
    fi
    _install_robotwin_stack hybrid
else
    echo "No valid local sapien-3.0.3 wheel; install RoboTwin stack from PyPI (not Huawei)"
    if [[ -f "${AHA_WAM_CODE_DIR}/requirements.txt" ]]; then
        _pip install --user --index-url "${PIP_INDEX_URL}" --trusted-host "${PIP_TRUSTED_HOST}" \
            -r "${AHA_WAM_CODE_DIR}/requirements.txt" || true
    fi
    _install_robotwin_stack pypi
fi

# Hard gate: do not continue without sapien
if ! "${PYTHON}" -c 'import sapien; assert sapien.__version__.startswith("3.0")' >/dev/null 2>&1; then
    echo "ERROR: sapien==3.0.x import failed after install"
    "${PYTHON}" -c 'import sapien' || true
    exit 1
fi
echo "sapien import: ok ($("${PYTHON}" -c 'import sapien; print(sapien.__version__)'))"
# Vulkan/EGL setup after sapien is installed
SAPIEN_VK_DIR="$("${PYTHON}" - <<'PY'
import sapien
from pathlib import Path
print(Path(sapien.__file__).resolve().parent / "vulkan_library")
PY
)"
ICD_DIR="/tmp/ahawam_vulkan_icd"
mkdir -p "${ICD_DIR}"

_ensure_lavapipe_icd() {
    local candidates=(
        "/usr/share/vulkan/icd.d/lvp_icd.x86_64.json"
        "/usr/share/vulkan/icd.d/lvp_icd.json"
        "${_CONDA_LIB%/lib}/share/vulkan/icd.d/lvp_icd.x86_64.json"
        "${DATA}/cwr_dataset_wulann/vulkan-icd/lvp_icd.x86_64.json"
    )
    local c
    for c in "${candidates[@]}"; do
        if [[ -f "${c}" ]]; then
            echo "${c}"
            return 0
        fi
    done
    echo "--- install mesa-vulkan-drivers (lavapipe) ---" >&2
    if command -v apt-get >/dev/null 2>&1; then
        if [[ "$(id -u)" -eq 0 ]]; then
            apt-get update -qq || true
            DEBIAN_FRONTEND=noninteractive apt-get install -y -qq mesa-vulkan-drivers || true
        elif command -v sudo >/dev/null 2>&1; then
            sudo -n apt-get update -qq 2>/dev/null || true
            sudo -n env DEBIAN_FRONTEND=noninteractive apt-get install -y -qq mesa-vulkan-drivers 2>/dev/null || true
        fi
    fi
    for c in "${candidates[@]}"; do
        if [[ -f "${c}" ]]; then
            echo "${c}"
            return 0
        fi
    done
    return 1
}

if [[ "${_MODESET_OK}" -eq 1 ]]; then
    AHAWAM_VULKAN_MODE="nvidia"
    if [[ -f "${NVIDIA_DRIVER_DIR}/libGLX_nvidia.so.0" && -f "${NVIDIA_DRIVER_DIR}/libEGL_nvidia.so.0" ]]; then
        cat > "${ICD_DIR}/nvidia_icd_abs.json" <<EOF
{
  "file_format_version": "1.0.0",
  "ICD": {
    "library_path": "${NVIDIA_DRIVER_DIR}/libGLX_nvidia.so.0",
    "api_version": "1.3.242"
  }
}
EOF
        cat > "${ICD_DIR}/10_nvidia_abs.json" <<EOF
{
  "file_format_version": "1.0.0",
  "ICD": {
    "library_path": "${NVIDIA_DRIVER_DIR}/libEGL_nvidia.so.0"
  }
}
EOF
        export VK_ICD_FILENAMES="${ICD_DIR}/nvidia_icd_abs.json"
        export __EGL_VENDOR_LIBRARY_FILENAMES="${ICD_DIR}/10_nvidia_abs.json"
    else
        echo "WARNING: nvidia GLX/EGL libs missing; falling back to sapien ICD"
        [[ -f "${SAPIEN_VK_DIR}/nvidia_icd.json" ]] && export VK_ICD_FILENAMES="${SAPIEN_VK_DIR}/nvidia_icd.json"
        [[ -f "${SAPIEN_VK_DIR}/10_nvidia.json" ]] && export __EGL_VENDOR_LIBRARY_FILENAMES="${SAPIEN_VK_DIR}/10_nvidia.json"
    fi
    if [[ -f "${SAPIEN_VK_DIR}/libvulkan.so.1.3.224" ]]; then
        export SAPIEN_VULKAN_LIBRARY_PATH="${SAPIEN_VK_DIR}/libvulkan.so.1.3.224"
    fi
    echo "Vulkan: NVIDIA ICD (方案A)"
else
    AHAWAM_VULKAN_MODE="lavapipe"
    # Best-effort system deps (libLLVM etc.) even when we use bundled ICD
    if command -v apt-get >/dev/null 2>&1; then
        echo "--- lavapipe: ensure mesa-vulkan-drivers (system deps) ---"
        if [[ "$(id -u)" -eq 0 ]]; then
            apt-get update -qq || true
            DEBIAN_FRONTEND=noninteractive apt-get install -y -qq mesa-vulkan-drivers libvulkan1 || true
        elif command -v sudo >/dev/null 2>&1; then
            sudo -n apt-get update -qq 2>/dev/null || true
            sudo -n env DEBIAN_FRONTEND=noninteractive apt-get install -y -qq mesa-vulkan-drivers libvulkan1 2>/dev/null || true
        fi
    fi
    LVP_ICD="$(_ensure_lavapipe_icd)" || LVP_ICD=""
    if [[ -z "${LVP_ICD}" || ! -f "${LVP_ICD}" ]]; then
        echo "ERROR: lavapipe ICD not found and mesa-vulkan-drivers install failed"
        echo "Need either usable /dev/nvidia-modeset (graphics at container create) or lavapipe ICD"
        exit 1
    fi
    # Rewrite ICD with absolute library_path under THIS job's DATA mount
    # (bundled json may point at /home/... while cluster uses /opt/huawei/...)
    LVP_SO=""
    for _cand in \
        "/usr/lib/x86_64-linux-gnu/libvulkan_lvp.so" \
        "${DATA}/cwr_dataset_wulann/vulkan-icd/libvulkan_lvp.so" \
        "$(dirname "${LVP_ICD}")/libvulkan_lvp.so"
    do
        if [[ -f "${_cand}" ]]; then LVP_SO="${_cand}"; break; fi
    done
    if [[ -n "${LVP_SO}" ]]; then
        # Persist under DATA so child eval workers (not just /tmp) resolve the same ICD.
        mkdir -p "${DATA}/cwr_dataset_wulann/vulkan-icd"
        cat > "${DATA}/cwr_dataset_wulann/vulkan-icd/lvp_icd_runtime.json" <<EOF
{
  "file_format_version": "1.0.0",
  "ICD": {
    "library_path": "${LVP_SO}",
    "api_version": "1.1.255"
  }
}
EOF
        cat > "${ICD_DIR}/lvp_icd_abs.json" <<EOF
{
  "file_format_version": "1.0.0",
  "ICD": {
    "library_path": "${LVP_SO}",
    "api_version": "1.1.255"
  }
}
EOF
        LVP_ICD="${DATA}/cwr_dataset_wulann/vulkan-icd/lvp_icd_runtime.json"
        export LD_LIBRARY_PATH="$(dirname "${LVP_SO}"):${LD_LIBRARY_PATH}"
    fi
    # Fail early if lavapipe .so cannot resolve deps
    if [[ -n "${LVP_SO}" ]] && command -v ldd >/dev/null 2>&1; then
        if ldd "${LVP_SO}" 2>/dev/null | grep -qi 'not found'; then
            echo "ERROR: lavapipe .so has unresolved deps:"
            ldd "${LVP_SO}" | grep -i 'not found' || true
            echo "apt-get install mesa-vulkan-drivers on the job image, or open graphics at create"
            exit 1
        fi
    fi
    export VK_ICD_FILENAMES="${LVP_ICD}"
    unset __EGL_VENDOR_LIBRARY_FILENAMES || true
    # Prefer system/libvulkan; sapien bundled loader is ok as fallback
    if [[ -f /usr/lib/x86_64-linux-gnu/libvulkan.so.1 ]]; then
        export SAPIEN_VULKAN_LIBRARY_PATH="/usr/lib/x86_64-linux-gnu/libvulkan.so.1"
    elif [[ -f "${SAPIEN_VK_DIR}/libvulkan.so.1.3.224" ]]; then
        export SAPIEN_VULKAN_LIBRARY_PATH="${SAPIEN_VK_DIR}/libvulkan.so.1.3.224"
    fi
    echo "Vulkan: lavapipe (方案B software) ICD=${LVP_ICD} SO=${LVP_SO:-?}"
    if [[ "${SMOKE}" != "1" ]]; then
        echo "WARNING: FULL eval on lavapipe is VERY slow (CPU render)."
        echo "WARNING: expect multi-x wall time vs GPU Vulkan; watch first episode ETA in sweep_master.log."
        echo "WARNING: platform timeout should be >= JOB_TIMEOUT_S (ft default 48h / random 24h)."
    fi
fi
export AHAWAM_VULKAN_MODE

# lavapipe: drop NVIDIA userspace GL/Vulkan stubs (they segfault on URDF without modeset)
if [[ "${AHAWAM_VULKAN_MODE}" == "lavapipe" ]]; then
    export SAPIEN_DISABLE_RAYTRACING=1
    export ROBOTWIN_MPLIB_NO_SAPIEN_WORLD=1
    export ROBOTWIN_MPLIB_STUB=1
    _new_ld=""
    IFS=':' read -r -a _ld_parts <<< "${LD_LIBRARY_PATH:-}"
    for _p in "${_ld_parts[@]}"; do
        [[ -z "${_p}" ]] && continue
        [[ "${_p}" == *nvidia-driver-libs* ]] && continue
        _new_ld="${_new_ld:+${_new_ld}:}${_p}"
    done
    export LD_LIBRARY_PATH="/usr/lib/x86_64-linux-gnu${_new_ld:+:${_new_ld}}"
    echo "lavapipe: stripped nvidia-driver-libs from LD_LIBRARY_PATH"
fi

echo "AHAWAM_VULKAN_MODE=${AHAWAM_VULKAN_MODE}"
echo "SAPIEN_DISABLE_RAYTRACING=${SAPIEN_DISABLE_RAYTRACING:-0}"
echo "SAPIEN_VULKAN_LIBRARY_PATH=${SAPIEN_VULKAN_LIBRARY_PATH:-<empty>}"
echo "VK_ICD_FILENAMES=${VK_ICD_FILENAMES:-<empty>}"
echo "__EGL_VENDOR_LIBRARY_FILENAMES=${__EGL_VENDOR_LIBRARY_FILENAMES:-<empty>}"

# RoboTwin eval_policy.py calls bare "ffmpeg" (not imageio's ffmpeg-linux-*).
# Ship a stable name under dataset/.../bin and put it first on PATH.
FFMPEG_BIN_DIR="${DATA}/cwr_dataset_wulann/bin"
FFMPEG_STORE="${DATA}/cwr_dataset_wulann/ffmpeg-bin"
mkdir -p "${FFMPEG_BIN_DIR}" "${FFMPEG_STORE}"
FFMPEG_EXE=""
if [[ -x "${FFMPEG_BIN_DIR}/ffmpeg" ]]; then
    FFMPEG_EXE="${FFMPEG_BIN_DIR}/ffmpeg"
elif [[ -x "${FFMPEG_STORE}/ffmpeg-linux-x86_64-v7.0.2" ]]; then
    ln -sfn "../ffmpeg-bin/ffmpeg-linux-x86_64-v7.0.2" "${FFMPEG_BIN_DIR}/ffmpeg"
    FFMPEG_EXE="${FFMPEG_BIN_DIR}/ffmpeg"
else
    _img_ff="$("${PYTHON}" -c 'import imageio_ffmpeg; print(imageio_ffmpeg.get_ffmpeg_exe())' 2>/dev/null || true)"
    if [[ -n "${_img_ff}" && -x "${_img_ff}" ]]; then
        cp -f "${_img_ff}" "${FFMPEG_STORE}/$(basename "${_img_ff}")"
        chmod +x "${FFMPEG_STORE}/$(basename "${_img_ff}")"
        ln -sfn "../ffmpeg-bin/$(basename "${_img_ff}")" "${FFMPEG_BIN_DIR}/ffmpeg"
        FFMPEG_EXE="${FFMPEG_BIN_DIR}/ffmpeg"
    fi
fi
if [[ -z "${FFMPEG_EXE}" || ! -x "${FFMPEG_EXE}" ]]; then
    echo "ERROR: ffmpeg binary missing. Expected ${FFMPEG_BIN_DIR}/ffmpeg"
    exit 1
fi
export PATH="${FFMPEG_BIN_DIR}:${PATH}"
echo "ffmpeg: ${FFMPEG_EXE} (PATH includes ${FFMPEG_BIN_DIR})"
command -v ffmpeg >/dev/null || { echo "ERROR: ffmpeg not on PATH"; exit 1; }

# ---------- 冒烟检查（缺任一关键依赖直接失败，避免空跑 205 jobs）----------
echo "LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-<empty>}"
"${PYTHON}" - <<PY
import os, sys, shutil, subprocess
from pathlib import Path
print("executable:", sys.executable)
import torch
print(f"torch={torch.__version__} cuda={torch.cuda.is_available()} n_gpu={torch.cuda.device_count()}")
assert torch.cuda.is_available(), "CUDA torch required (do not overwrite image torch)"

import sapien
print(f"sapien={sapien.__version__}")
_mods_req = [
    "open3d", "mplib", "gymnasium", "cv2", "yaml", "trimesh", "yourdfpy",
    "toppra", "scipy", "h5py", "imageio",
]
for _m in _mods_req:
    try:
        __import__(_m)
        print(f"{_m}: ok")
    except Exception as _e:
        raise ImportError(f"preflight missing module {_m!r}: {_e}") from _e
for name in ["hydra", "omegaconf", "transformers", "PIL", "av", "numpy", "pandas", "wandb", "transforms3d", "lxml"]:
    try:
        __import__(name)
        print(f"{name}: ok")
    except Exception as _e:
        raise ImportError(f"preflight missing module {name!r}: {_e}") from _e

# ffmpeg must resolve as the bare name used by RoboTwin eval_policy.py
ff = shutil.which("ffmpeg")
assert ff, "ffmpeg not found on PATH (RoboTwin needs bare 'ffmpeg')"
subprocess.run([ff, "-version"], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
print(f"ffmpeg which: {ff}")

# ckpt / stats / wan assets — released robotwin ckpt for random_skip experiment
data = Path(os.environ["DATA"])
ckpt = Path(os.environ["SKIP_PHASE_CKPT"])
stats = Path(os.environ.get("SKIP_PHASE_DATASET_STATS") or "")
if not stats.is_file():
    # fallback: released stats next to official ckpt
    stats = Path(os.environ["AHA_WAM_CODE_DIR"]) / "checkpoints/AHA-WAM-RoboTwin2.0/dataset_stats.json"
t5 = Path(os.environ["AHA_WAM_CODE_DIR"]) / "checkpoints/DiffSynth-Studio/Wan-Series-Converted-Safetensors/models_t5_umt5-xxl-enc-bf16.safetensors"
vae = Path(os.environ["AHA_WAM_CODE_DIR"]) / "checkpoints/DiffSynth-Studio/Wan-Series-Converted-Safetensors/Wan2.2_VAE.safetensors"
tok = Path(os.environ["AHA_WAM_CODE_DIR"]) / "checkpoints/Wan-AI/Wan2.1-T2V-1.3B/google/umt5-xxl"
for p in [ckpt, stats, t5, vae, tok]:
    assert p.exists(), f"missing required asset: {p}"
    print(f"asset ok: {p}")

# RoboTwin task_config (cwd=RoboTwin root looks for ./task_config/demo_randomized.yml)
tc = Path(os.environ["AHA_WAM_CODE_DIR"]) / "third_party/RoboTwin/task_config/demo_randomized.yml"
assert tc.is_file(), f"missing RoboTwin task_config: {tc}"
print(f"asset ok: {tc}")
emb = Path(os.environ["AHA_WAM_CODE_DIR"]) / "third_party/RoboTwin/assets/embodiments/aloha-agilex"
assert emb.is_dir(), f"missing embodiment assets: {emb}"
print(f"asset ok: {emb}")
print(f"EVAL CKPT (released): {ckpt}")

# RoboTwin: same import path as eval_policy.py + all task modules + policy bridge
rt = Path(os.environ["AHA_WAM_CODE_DIR"]) / "third_party/RoboTwin"
sys.path.insert(0, str(rt))
sys.path.insert(0, str(rt / "description/utils"))
sys.path.insert(0, str(Path(os.environ["AHA_WAM_CODE_DIR"]) / "experiments/robotwin/ahawam_policy"))
_old_cwd = os.getcwd()
os.chdir(rt)
from envs import CONFIGS_PATH
from envs.utils import pkl2hdf5  # noqa: F401 — forces h5py chain
import envs._base_task  # noqa: F401
import envs.robot.robot  # noqa: F401
import envs.camera.camera  # noqa: F401
print("envs.CONFIGS_PATH", CONFIGS_PATH)
print("envs.utils.pkl2hdf5 / _base_task / robot / camera: ok")
print("NOTE: CuroboPlanner warning above is optional and can be ignored")

task_files = sorted(
    p for p in (rt / "envs").glob("*.py")
    if p.name not in ("__init__.py",) and not p.name.startswith("_")
)
for p in task_files:
    __import__(f"envs.{p.stem}")
print(f"all task modules: {len(task_files)} ok")

import generate_episode_instructions  # noqa: F401
import deploy_policy  # noqa: F401
print("generate_episode_instructions + deploy_policy: ok")

# Sapien renderer smoke — match RoboTwin setup_scene (no RT, no shadow on lavapipe)
try:
    print("preflight SAPIEN_DISABLE_RAYTRACING=", os.environ.get("SAPIEN_DISABLE_RAYTRACING"))
    print("preflight ROBOTWIN_EVAL_VIDEO_LOG=", os.environ.get("ROBOTWIN_EVAL_VIDEO_LOG"))
    renderer = sapien.render.SapienRenderer()
    print("SapienRenderer: ok mode=", os.environ.get("AHAWAM_VULKAN_MODE"))
    engine = sapien.Engine()
    engine.set_renderer(renderer)
    scene = engine.create_scene(sapien.SceneConfig())
    print("create_scene: ok")
    scene.add_ground(0)
    scene.set_ambient_light([0.5, 0.5, 0.5])
    # Must match _base_task lavapipe path: shadow=False (shadow=True SEGVs on lavapipe)
    scene.add_directional_light([0, 0.5, -1], [0.5, 0.5, 0.5], shadow=False)
    scene.add_point_light([1, 0, 1.8], [1, 1, 1], shadow=False)
    print("lights shadow=False: ok")
    # Same crash site as eval: aloha dual-arm URDF under lavapipe (+ bad LD path)
    urdf = Path(os.environ["AHA_WAM_CODE_DIR"]) / (
        "third_party/RoboTwin/assets/embodiments/aloha-agilex/urdf/arx5_description_isaac.urdf"
    )
    if not urdf.is_file():
        # fallback common names
        cand = list((Path(os.environ["AHA_WAM_CODE_DIR"]) / "third_party/RoboTwin/assets/embodiments/aloha-agilex").rglob("*.urdf"))
        urdf = cand[0] if cand else None
    if urdf is None or not Path(urdf).is_file():
        raise FileNotFoundError("aloha urdf missing for preflight")
    print("preflight URDF load:", urdf)
    rt_root = Path(os.environ["AHA_WAM_CODE_DIR"]) / "third_party/RoboTwin"
    _cwd = os.getcwd()
    os.chdir(rt_root)
    try:
        loader = scene.create_urdf_loader()
        loader.fix_root_link = True
        ent = loader.load(str(urdf))
        print("preflight URDF load: ok", type(ent))
    finally:
        os.chdir(_cwd)
    print("sapien.Scene: ok (lavapipe-safe preflight incl. URDF)")
except Exception as e:
    import glob
    print("ERROR: SapienRenderer/scene preflight failed:", repr(e))
    print("AHAWAM_VULKAN_MODE=", os.environ.get("AHAWAM_VULKAN_MODE"))
    print("NVIDIA_DRIVER_CAPABILITIES=", os.environ.get("NVIDIA_DRIVER_CAPABILITIES"))
    print("VK_ICD_FILENAMES=", os.environ.get("VK_ICD_FILENAMES"))
    print("__EGL_VENDOR_LIBRARY_FILENAMES=", os.environ.get("__EGL_VENDOR_LIBRARY_FILENAMES"))
    print("SAPIEN_VULKAN_LIBRARY_PATH=", os.environ.get("SAPIEN_VULKAN_LIBRARY_PATH"))
    print("LD_LIBRARY_PATH=", os.environ.get("LD_LIBRARY_PATH"))
    print("/dev/nvidia*:", sorted(glob.glob("/dev/nvidia*")))
    print(
        "HINT: need usable /dev/nvidia-modeset (graphics at create) OR lavapipe ICD "
        "(/usr/share/vulkan/icd.d/lvp_icd*.json via mesa-vulkan-drivers)"
    )
    raise
os.chdir(_old_cwd)
PY

nvidia-smi || true

# 多卡启动前先建好 policy 软链，避免并发 unlink 竞态
POLICY_SRC="${AHA_WAM_CODE_DIR}/experiments/robotwin/ahawam_policy"
POLICY_LINK="${AHA_WAM_CODE_DIR}/third_party/RoboTwin/policy/ahawam_policy"
if [[ -d "${POLICY_SRC}" && -d "$(dirname "${POLICY_LINK}")" ]]; then
    ln -sfn "${POLICY_SRC}" "${POLICY_LINK}"
    echo "policy symlink: ${POLICY_LINK} -> ${POLICY_SRC}"
    ls -l "${POLICY_LINK}" || true
    # skip_phase FSM must be visible through the policy package
    if [[ ! -f "${POLICY_SRC}/skip_phase_fsm.py" ]]; then
        echo "ERROR: missing ${POLICY_SRC}/skip_phase_fsm.py"
        exit 1
    fi
    if [[ ! -f "${AHA_WAM_CODE_DIR}/experiments/robotwin/run_skip_phase_sweep.py" ]]; then
        echo "ERROR: missing run_skip_phase_sweep.py"
        exit 1
    fi
    "${PYTHON}" - <<'PY'
import importlib.util
import os
import sys
from pathlib import Path

code_dir = Path(os.environ["AHA_WAM_CODE_DIR"])
fsm_path = code_dir / "experiments/robotwin/ahawam_policy/skip_phase_fsm.py"
sweep_path = code_dir / "experiments/robotwin/run_skip_phase_sweep.py"
assert fsm_path.is_file(), f"missing {fsm_path}"
assert sweep_path.is_file(), f"missing {sweep_path}"

# Register in sys.modules before exec_module (safe for any future dataclasses too).
spec = importlib.util.spec_from_file_location("skip_phase_fsm", fsm_path)
mod = importlib.util.module_from_spec(spec)
sys.modules["skip_phase_fsm"] = mod
assert spec.loader is not None
spec.loader.exec_module(mod)

assert "click_bell" in mod.TASK_PHASE_TARGETS, "click_bell missing from TASK_PHASE_TARGETS"
batch = os.environ.get("SKIP_PHASE_BATCH", "batch1")
if batch == "batch2":
    expected = [
        "pick_dual_bottles", "pick_diverse_bottles", "place_a2b_left", "place_a2b_right",
        "move_can_pot", "stack_bowls_three", "handover_block", "lift_pot", "press_stapler",
        "turn_switch",
    ]
else:
    expected = [
        "handover_mic", "hanging_mug", "move_stapler_pad", "place_bread_basket",
        "place_mouse_pad", "place_object_basket", "stack_blocks_three",
        "stack_blocks_two", "click_bell",
    ]
missing = [t for t in expected if t not in mod.TASK_PHASE_TARGETS]
assert not missing, f"TASK_PHASE_TARGETS missing: {missing}"
fsm = mod.SkipPhaseFSM(task_name=expected[0], near_thresh_m=0.10)
assert fsm.template == "pick_place"
assert fsm.near_thresh_m == 0.10, fsm.near_thresh_m
fsm2 = mod.SkipPhaseFSM(task_name="click_bell")
assert fsm2.template == "contact"
fsm2.set_task_name("handover_mic")
assert fsm2.template == "pick_place"
# deploy_policy budget knob must exist in env for sweep child jobs
assert os.environ.get("SKIP_PHASE_MAX_CONSECUTIVE_SKIPS") == "1"
assert os.environ.get("OVCR_DIAG_MODE") == "baseline"
print(
    f"skip_phase_fsm ok | batch={batch} tasks={len(expected)} "
    f"| near={fsm.near_thresh_m} max_consec_skip=1 OVCR=baseline"
)
PY
else
    echo "WARNING: cannot create policy symlink (src or parent missing)"
fi

mkdir -p "${OUTPUT_DIR}"
echo "FINAL OUTPUT_DIR=${OUTPUT_DIR}"
echo "EVAL CKPT=${SKIP_PHASE_CKPT}"
echo "RANDOM_SKIP_P=${RANDOM_SKIP_P} RANDOM_SKIP_SEED=${RANDOM_SKIP_SEED}"
echo "EVAL STATS=${SKIP_PHASE_DATASET_STATS}"

# ---------- 只跑 skip_phase（ft ckpt）；baseline 用 MAIN-2 released 结果 ----------
export SKIP_PHASE_OUTPUT_DIR="${OUTPUT_DIR}"
SWEEP_PY="${AHA_WAM_CODE_DIR}/experiments/robotwin/run_skip_phase_sweep.py"
echo "launch: ${SWEEP_PY} batch=${SKIP_PHASE_BATCH} modes=${EVAL_MODES[*]} ckpt=${SKIP_PHASE_CKPT}"
echo "NOTE: mode=random_skip on released robotwin_ahawam.pt (control; not ft skip_phase)."
"${PYTHON}" -u "${SWEEP_PY}" \
    --output_dir "${OUTPUT_DIR}" \
    --num_episodes "${NUM_EPISODES}" \
    --num_gpus "${NNPU}" \
    --timeout_s "${JOB_TIMEOUT_S}" \
    --ckpt "${SKIP_PHASE_CKPT}" \
    --dataset_stats_path "${SKIP_PHASE_DATASET_STATS}" \
    --disable_baseline_reuse \
    --modes "${EVAL_MODES[@]}" \
    --tasks "${SKIP_PHASE_TASKS[@]}"
