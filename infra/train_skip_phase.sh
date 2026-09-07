#!/usr/bin/env bash
# CUDA GPU 任务入口——baseline vs skip_phase（固定 ah64/cpp2，与 ah/cpp sweep 隔离）
# 提交：算法包仍选 cwr_wulan2，但入口脚本选 train_skip_phase.sh（不要用 train_mtp.sh）。
# 不要与正在跑的 robotwin_ahawam_sweep_20tasks_40eps 抢同一台机器的 GPU。
set -euo pipefail

echo "train_skip_phase.sh revision: 2026-09-03-skip-phase-v2-budget-ovcr"
# ===================== 写死配置（你不用 set）=====================
SKIP_PHASE_BATCH="${SKIP_PHASE_BATCH:-batch1}"
SMOKE=0
SMOKE_NUM_EPISODES=1
FULL_NUM_EPISODES=40
# 固定与旧 skip_approach 一致；不扫 ah/cpp
export FIXED_ACTION_HORIZON=64
export FIXED_CPP=2
export SKIP_PHASE_NEAR_THRESH_M=0.10
export SKIP_PHASE_MAX_CONSECUTIVE_SKIPS=1
export OVCR_DIAG_MODE=baseline
export SKIP_PHASE_BATCH

case "${SKIP_PHASE_BATCH}" in
batch1)
    HARDCODED_OUTPUT_DIR="/opt/huawei/dataset/cwr_wulan_aha/aha-wam-runs/robotwin/video_dit_skip_phase_smoke_v2"
    HARDCODED_OUTPUT_DIR_FULL="/opt/huawei/dataset/cwr_wulan_aha/aha-wam-runs/robotwin/video_dit_skip_phase_40eps_v2"
    SKIP_PHASE_TASKS=(
        handover_mic hanging_mug move_stapler_pad place_bread_basket place_mouse_pad
        place_object_basket put_bottles_dustbin stack_blocks_three stack_blocks_two click_bell
    )
    ;;
batch2)
    echo "train_skip_phase.sh revision: 2026-09-03-skip-phase-v2-batch2"
    HARDCODED_OUTPUT_DIR="/opt/huawei/dataset/cwr_wulan_aha/aha-wam-runs/robotwin/video_dit_skip_phase_smoke_v2_batch2"
    HARDCODED_OUTPUT_DIR_FULL="/opt/huawei/dataset/cwr_wulan_aha/aha-wam-runs/robotwin/video_dit_skip_phase_40eps_v2_batch2"
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
if [[ -d "/opt/huawei/dataset/cwr_wulan_aha/AHA-WAM-anygrasp" ]]; then
    export DATA="/opt/huawei/dataset"
    export AHA_WAM_CODE_DIR="${DATA}/cwr_wulan_aha/AHA-WAM-anygrasp"
else
    export DATA="/home/ma-user/work/dataset"
    export AHA_WAM_CODE_DIR="${DATA}/cwr_wulan_aha/AHA-WAM-anygrasp"
fi

WHEELS_DIR="${DATA}/cwr_wulan_aha/wheels"
echo "AHA_WAM_CODE_DIR: ${AHA_WAM_CODE_DIR}"
echo "WHEELS_DIR: ${WHEELS_DIR}"
echo "SKIP_PHASE_BATCH=${SKIP_PHASE_BATCH} tasks=${#SKIP_PHASE_TASKS[@]}"
echo "FIXED ah=${FIXED_ACTION_HORIZON} cpp=${FIXED_CPP} near=${SKIP_PHASE_NEAR_THRESH_M} max_consec_skip=${SKIP_PHASE_MAX_CONSECUTIVE_SKIPS} OVCR=${OVCR_DIAG_MODE}"

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
export JOB_TIMEOUT_S="${JOB_TIMEOUT_S:-64800}"
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
        "place_mouse_pad", "place_object_basket", "put_bottles_dustbin", "stack_blocks_three",
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

# ---------- 启动 skip_phase vs baseline（固定 ah64_cpp2）----------
# 不设置会改变 ah/cpp sweep 的环境；本进程 VIDEO_DIT_MODE 由子 job 单独设置。
export SKIP_PHASE_OUTPUT_DIR="${OUTPUT_DIR}"
SWEEP_PY="${AHA_WAM_CODE_DIR}/experiments/robotwin/run_skip_phase_sweep.py"
echo "launch: ${SWEEP_PY} batch=${SKIP_PHASE_BATCH}"
"${PYTHON}" -u "${SWEEP_PY}" \
    --output_dir "${OUTPUT_DIR}" \
    --num_episodes "${NUM_EPISODES}" \
    --num_gpus "${NNPU}" \
    --timeout_s "${JOB_TIMEOUT_S}" \
    --tasks "${SKIP_PHASE_TASKS[@]}"
