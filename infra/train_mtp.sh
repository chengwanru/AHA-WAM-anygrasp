#!/usr/bin/env bash
# CUDA GPU 任务入口（A100/A800）——单机多卡 RoboTwin AHA-WAM sweep
# 提交时无需手动 set 任何变量；平台会注入 MA_NUM_GPUS 等。
#
# 全量：平台选 1 机 x 8 卡；SMOKE=0（已写死）
# 冒烟：把 SMOKE 改回 1，并选 4 卡即可
set -euo pipefail

echo "train_mtp.sh revision: 2026-09-03-fix-pipe-deadlock-skip"
# ===================== 写死配置（你不用 set）=====================
# 1=冒烟（少 episode），0=全量
SMOKE=0
# 冒烟 / 全量 episode 数
SMOKE_NUM_EPISODES=1
FULL_NUM_EPISODES=40
# 输出目录（训练环境挂载路径；脚本也会自动适配探索环境路径）
HARDCODED_OUTPUT_DIR="/opt/huawei/dataset/cwr_wulan_aha/aha-wam-runs/robotwin_ahawam_smoke"
HARDCODED_OUTPUT_DIR_FULL="/opt/huawei/dataset/cwr_wulan_aha/aha-wam-runs/robotwin_ahawam_sweep_20tasks_40eps"
# ================================================================

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "${SCRIPT_DIR}"
echo "code dir: $(pwd)"

# ---------- 路径（自动探测，无需手动 set）----------
if [[ -d "/opt/huawei/dataset/cwr_wulan_aha/AHA-WAM-anygrasp" ]]; then
    export DATA="/opt/huawei/dataset"
    export AHA_WAM_CODE_DIR="${DATA}/cwr_wulan_aha/AHA-WAM-anygrasp"
else
    export DATA="/home/ma-user/work/dataset"
    export AHA_WAM_CODE_DIR="${DATA}/cwr_wulan_aha/AHA-WAM-anygrasp"
fi

if [[ -f "${SCRIPT_DIR}/scripts/run_robotwin_sweep.sh" ]]; then
    export CWR_WULAN2_DIR="${SCRIPT_DIR}"
elif [[ -d "${SCRIPT_DIR}/cwr_wulan2" && -f "${SCRIPT_DIR}/cwr_wulan2/scripts/run_robotwin_sweep.sh" ]]; then
    export CWR_WULAN2_DIR="${SCRIPT_DIR}/cwr_wulan2"
elif [[ -d "/opt/huawei/schedule-train/algorithm/cwr_wulan2" ]]; then
    export CWR_WULAN2_DIR="/opt/huawei/schedule-train/algorithm/cwr_wulan2"
else
    export CWR_WULAN2_DIR="${SCRIPT_DIR}"
fi

WHEELS_DIR="${DATA}/cwr_wulan_aha/wheels"
echo "AHA_WAM_CODE_DIR: ${AHA_WAM_CODE_DIR}"
echo "CWR_WULAN2_DIR: ${CWR_WULAN2_DIR}"
echo "WHEELS_DIR: ${WHEELS_DIR}"

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
export JOB_TIMEOUT_S="${JOB_TIMEOUT_S:-43200}"
echo "JOB_TIMEOUT_S=${JOB_TIMEOUT_S}"

# ---------- 强制只用华为源，禁用 NGC ----------
export PIP_INDEX_URL="http://repo.myhuaweicloud.com/repository/pypi/simple"
export PIP_TRUSTED_HOST="repo.myhuaweicloud.com"
export PIP_EXTRA_INDEX_URL=""
export PIP_DISABLE_PIP_VERSION_CHECK=1
export PIP_DEFAULT_TIMEOUT=120
export PIP_CONFIG_FILE="/dev/null"

PYTHON="$(command -v python3 || command -v python)"
export PYTHON
echo "Using python: ${PYTHON} ($("${PYTHON}" -V 2>&1))"

# sapien 无头渲染：必须用完整 nvidia-535.183.01（不是精简 lib/）
# 与 eval_robotwin_single.py 保持一致；ICD 在 sapien 安装后再指向其 vulkan_library
SAPIEN_LIBS="${DATA}/cwr_wulan_aha/sapien-runtime-libs"
NVIDIA_DRIVER_DIR="${DATA}/cwr_wulan_aha/nvidia-driver-libs/nvidia-535.183.01"
_ld_add=()
if [[ -d "${NVIDIA_DRIVER_DIR}" ]]; then
    _ld_add+=("${NVIDIA_DRIVER_DIR}")
    echo "LD_LIBRARY_PATH += nvidia-535.183.01"
fi
if [[ -d "${SAPIEN_LIBS}" ]]; then
    _ld_add+=("${SAPIEN_LIBS}")
    echo "LD_LIBRARY_PATH += sapien-runtime-libs"
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

# ---------- 离线安装依赖（不覆盖镜像 CUDA torch）----------
export PYTHONUSERBASE="/tmp/ahawam_user"
mkdir -p "${PYTHONUSERBASE}"
PYVER="$("${PYTHON}" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
export PATH="${PYTHONUSERBASE}/bin:${PATH}"
export PYTHONPATH="${PYTHONUSERBASE}/lib/python${PYVER}/site-packages:${AHA_WAM_CODE_DIR}:${AHA_WAM_CODE_DIR}/src${PYTHONPATH:+:${PYTHONPATH}}"

_pip() {
    "${PYTHON}" -m pip "$@"
}

_need_import() {
    ! "${PYTHON}" -c "import $1" >/dev/null 2>&1
}

_install_robotwin_stack() {
    local mode="$1"  # wheels | network
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
    )
    if [[ "${mode}" == "wheels" ]]; then
        _pip install --user --no-index --find-links "${WHEELS_DIR}" "${common[@]}"
    else
        _pip install --user --index-url "${PIP_INDEX_URL}" --trusted-host "${PIP_TRUSTED_HOST}" "${common[@]}"
    fi
}

echo "--- install deps from offline wheels (if present) ---"
if [[ -d "${WHEELS_DIR}" ]]; then
    if [[ -f "${AHA_WAM_CODE_DIR}/requirements.txt" ]]; then
        echo "installing requirements.txt from wheels ..."
        # Non-fatal: some AHA extras may be absent; RoboTwin stack below is hard-required.
        _pip install --user --no-index --find-links "${WHEELS_DIR}" \
            -r "${AHA_WAM_CODE_DIR}/requirements.txt" || true
    fi
    echo "installing RoboTwin/sapien stack from wheels (hard-required) ..."
    _install_robotwin_stack wheels
else
    echo "WARNING: wheels dir missing: ${WHEELS_DIR}"
    echo "fallback: Huawei mirror only (no NGC)"
    if [[ -f "${AHA_WAM_CODE_DIR}/requirements.txt" ]]; then
        _pip install --user --index-url "${PIP_INDEX_URL}" --trusted-host "${PIP_TRUSTED_HOST}" \
            -r "${AHA_WAM_CODE_DIR}/requirements.txt" || true
    fi
    _install_robotwin_stack network
fi

# sapien 自带 vulkan ICD / loader（安装后才能定位）
SAPIEN_VK_DIR="$("${PYTHON}" - <<'PY'
import sapien
from pathlib import Path
print(Path(sapien.__file__).resolve().parent / "vulkan_library")
PY
)"
if [[ -d "${SAPIEN_VK_DIR}" ]]; then
    if [[ -f "${SAPIEN_VK_DIR}/libvulkan.so.1.3.224" ]]; then
        export SAPIEN_VULKAN_LIBRARY_PATH="${SAPIEN_VK_DIR}/libvulkan.so.1.3.224"
    fi
    if [[ -f "${SAPIEN_VK_DIR}/nvidia_icd.json" ]]; then
        export VK_ICD_FILENAMES="${SAPIEN_VK_DIR}/nvidia_icd.json"
    fi
    if [[ -f "${SAPIEN_VK_DIR}/10_nvidia.json" ]]; then
        export __EGL_VENDOR_LIBRARY_FILENAMES="${SAPIEN_VK_DIR}/10_nvidia.json"
    fi
    echo "SAPIEN_VULKAN_LIBRARY_PATH=${SAPIEN_VULKAN_LIBRARY_PATH:-<empty>}"
    echo "VK_ICD_FILENAMES=${VK_ICD_FILENAMES:-<empty>}"
fi

# RoboTwin eval_policy.py calls bare "ffmpeg" (not imageio's ffmpeg-linux-*).
# Ship a stable name under dataset/.../bin and put it first on PATH.
FFMPEG_BIN_DIR="${DATA}/cwr_wulan_aha/bin"
FFMPEG_STORE="${DATA}/cwr_wulan_aha/ffmpeg-bin"
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
import open3d, mplib, gymnasium, cv2, yaml, trimesh, yourdfpy, toppra, scipy, h5py, imageio
print(f"open3d={open3d.__version__}")
print(f"mplib={getattr(mplib, '__version__', 'ok')}")
print(f"gymnasium={gymnasium.__version__}")
print(f"cv2={cv2.__version__}")
print(f"h5py={h5py.__version__}")
print(f"imageio={imageio.__version__}")
for name in ["hydra", "omegaconf", "transformers", "PIL", "av", "numpy", "pandas", "wandb", "transforms3d", "lxml"]:
    __import__(name)
    print(f"{name}: ok")

# ffmpeg must resolve as the bare name used by RoboTwin eval_policy.py
ff = shutil.which("ffmpeg")
assert ff, "ffmpeg not found on PATH (RoboTwin needs bare 'ffmpeg')"
subprocess.run([ff, "-version"], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
print(f"ffmpeg which: {ff}")

# ckpt / stats / wan assets
data = Path(os.environ["DATA"])
ckpt = data / "cwr_wulan_model/aha-wam/checkpoints/AHA-WAM-RoboTwin2.0/robotwin_ahawam.pt"
stats = ckpt.with_name("dataset_stats.json")
t5 = Path(os.environ["AHA_WAM_CODE_DIR"]) / "checkpoints/DiffSynth-Studio/Wan-Series-Converted-Safetensors/models_t5_umt5-xxl-enc-bf16.safetensors"
vae = Path(os.environ["AHA_WAM_CODE_DIR"]) / "checkpoints/DiffSynth-Studio/Wan-Series-Converted-Safetensors/Wan2.2_VAE.safetensors"
tok = Path(os.environ["AHA_WAM_CODE_DIR"]) / "checkpoints/Wan-AI/Wan2.1-T2V-1.3B/google/umt5-xxl"
for p in [ckpt, stats, t5, vae, tok]:
    assert p.exists(), f"missing required asset: {p}"
    print(f"asset ok: {p}")

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

# Sapien renderer (真实无头渲染冒烟)
renderer = sapien.render.SapienRenderer()
print("SapienRenderer: ok")
scene = sapien.Scene()
print("sapien.Scene: ok")
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
else
    echo "WARNING: cannot create policy symlink (src or parent missing)"
fi

mkdir -p "${OUTPUT_DIR}"
echo "FINAL OUTPUT_DIR=${OUTPUT_DIR}"

# ---------- 启动 sweep ----------
cd "${CWR_WULAN2_DIR}"
echo "sweep root: $(pwd)"
bash scripts/run_robotwin_sweep.sh "${OUTPUT_DIR}"
