#!/usr/bin/env bash
# Cheap local 2×A100 verify: skip_v2 FT ckpt, FSM skip_phase vs train proxy.
# Same machine / lavapipe / ckpt / tasks / episode count.
# Usage: bash infra/eval_skip_phase_proxy_local.sh
set -euo pipefail

echo "eval_skip_phase_proxy_local.sh revision: 2026-09-14-local-2gpu-proxy-ab"

DATA="/home/ma-user/work/dataset"
AHA_WAM_CODE_DIR="${DATA}/cwr_dataset_wulann/AHA-WAM-anygrasp"
WHEELS_DIR="${DATA}/cwr_dataset_wulann/wheels"
PYTHON="${PYTHON:-/home/ma-user/anaconda3/envs/ahawam/bin/python}"

export DATA AHA_WAM_CODE_DIR
export FIXED_ACTION_HORIZON=64
export FIXED_CPP=2
export SKIP_PHASE_NEAR_THRESH_M=0.10
export SKIP_PHASE_MAX_CONSECUTIVE_SKIPS=1
export SKIP_PHASE_PLACE_PROGRESS=0.85
export OVCR_DIAG_MODE=baseline
export ROBOTWIN_EVAL_VIDEO_LOG=0
export WANDB_MODE=offline
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export DIFFSYNTH_SKIP_DOWNLOAD=true
export DIFFSYNTH_MODEL_BASE_PATH="${AHA_WAM_CODE_DIR}/checkpoints"
export PYOPENGL_PLATFORM=egl
export NVIDIA_DRIVER_CAPABILITIES="compute,utility,graphics,display,video"

NUM_EPISODES="${NUM_EPISODES:-8}"
NUM_GPUS="${NUM_GPUS:-2}"
JOB_TIMEOUT_S="${JOB_TIMEOUT_S:-28800}"
TASKS=(${TASKS:-pick_dual_bottles press_stapler})
EVAL_MODES=(skip_phase skip_phase_proxy)

export SKIP_PHASE_CKPT="${SKIP_PHASE_CKPT:-${DATA}/cwr_dataset_wulann4/aha-wam-runs/train/skip_v2_ft_8x10h/checkpoints/weights/step_004000.pt}"
export SKIP_PHASE_DATASET_STATS="${SKIP_PHASE_DATASET_STATS:-${DATA}/cwr_dataset_wulann4/aha-wam-runs/train/skip_v2_ft_8x10h/dataset_stats.json}"
OUTPUT_DIR="${OUTPUT_DIR:-${DATA}/cwr_dataset_wulann4/aha-wam-runs/robotwin/skip_v2_proxy_align_local_8eps}"

echo "PYTHON=${PYTHON} ($("${PYTHON}" -V 2>&1))"
echo "CKPT=${SKIP_PHASE_CKPT}"
echo "STATS=${SKIP_PHASE_DATASET_STATS}"
echo "OUTPUT_DIR=${OUTPUT_DIR}"
echo "TASKS=${TASKS[*]} MODES=${EVAL_MODES[*]} eps=${NUM_EPISODES} gpus=${NUM_GPUS}"

[[ -x "${PYTHON}" ]] || { echo "ERROR: python missing: ${PYTHON}"; exit 1; }
[[ -f "${SKIP_PHASE_CKPT}" ]] || { echo "ERROR: ckpt missing: ${SKIP_PHASE_CKPT}"; exit 1; }
[[ -f "${SKIP_PHASE_DATASET_STATS}" ]] || { echo "ERROR: stats missing: ${SKIP_PHASE_DATASET_STATS}"; exit 1; }

export PYTHONUSERBASE="/tmp/ahawam_user"
mkdir -p "${PYTHONUSERBASE}"
PYVER="$("${PYTHON}" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
export PATH="$(dirname "${PYTHON}"):${PYTHONUSERBASE}/bin:${PATH}"
export PYTHONPATH="${PYTHONUSERBASE}/lib/python${PYVER}/site-packages:${AHA_WAM_CODE_DIR}:${AHA_WAM_CODE_DIR}/src${PYTHONPATH:+:${PYTHONPATH}}"

SAPIEN_WHEEL="${WHEELS_DIR}/sapien-3.0.3-cp310-cp310-manylinux_2_28_x86_64.whl"
if ! "${PYTHON}" -c 'import sapien; assert sapien.__version__.startswith("3.0")' >/dev/null 2>&1; then
    echo "--- pip install sapien from local wheel ---"
    [[ -f "${SAPIEN_WHEEL}" ]] || { echo "ERROR: missing ${SAPIEN_WHEEL}"; exit 1; }
    "${PYTHON}" -m pip install --user --index-url https://pypi.org/simple "${SAPIEN_WHEEL}"
fi
echo "sapien: $("${PYTHON}" -c 'import sapien; print(sapien.__version__, sapien.__file__)')"

# Need-import extras used by RoboTwin eval
for mod in open3d mplib gymnasium cv2 h5py trimesh yourdfpy yaml hydra omegaconf; do
    if ! "${PYTHON}" -c "import ${mod}" >/dev/null 2>&1; then
        echo "ERROR: missing python module ${mod} (expected in ${PYTHONUSERBASE})"
        exit 1
    fi
done

# Vulkan: this box has no /dev/nvidia-modeset → lavapipe
AHAWAM_VULKAN_MODE="lavapipe"
ICD_DIR="/tmp/ahawam_vulkan_icd_proxy_local"
mkdir -p "${ICD_DIR}"
LVP_SO=""
for _cand in \
    "/usr/lib/x86_64-linux-gnu/libvulkan_lvp.so" \
    "${DATA}/cwr_dataset_wulann/vulkan-icd/libvulkan_lvp.so"
do
    if [[ -f "${_cand}" ]]; then LVP_SO="${_cand}"; break; fi
done
[[ -n "${LVP_SO}" ]] || { echo "ERROR: libvulkan_lvp.so not found"; exit 1; }
cat > "${ICD_DIR}/lvp_icd_abs.json" <<EOF
{
  "file_format_version": "1.0.0",
  "ICD": {
    "library_path": "${LVP_SO}",
    "api_version": "1.1.255"
  }
}
EOF
export VK_ICD_FILENAMES="${ICD_DIR}/lvp_icd_abs.json"
unset __EGL_VENDOR_LIBRARY_FILENAMES || true
if [[ -f /usr/lib/x86_64-linux-gnu/libvulkan.so.1 ]]; then
    export SAPIEN_VULKAN_LIBRARY_PATH="/usr/lib/x86_64-linux-gnu/libvulkan.so.1"
fi
export AHAWAM_VULKAN_MODE
export SAPIEN_DISABLE_RAYTRACING=1
export ROBOTWIN_MPLIB_NO_SAPIEN_WORLD=1
export ROBOTWIN_MPLIB_STUB=1

# Strip nvidia GL stubs (segfault on URDF without modeset)
_CONDA_LIB="$(cd "$(dirname "${PYTHON}")/../lib" && pwd)"
SAPIEN_LIBS="${DATA}/cwr_dataset_wulann/sapien-runtime-libs"
_ld_parts=("/usr/lib/x86_64-linux-gnu" "$(dirname "${LVP_SO}")" "${SAPIEN_LIBS}" "${_CONDA_LIB}")
_new_ld=""
for _p in "${_ld_parts[@]}"; do
    [[ -d "${_p}" ]] || continue
    _new_ld="${_new_ld:+${_new_ld}:}${_p}"
done
export LD_LIBRARY_PATH="${_new_ld}"
echo "Vulkan: lavapipe ICD=${VK_ICD_FILENAMES} SO=${LVP_SO}"
echo "LD_LIBRARY_PATH=${LD_LIBRARY_PATH}"

FFMPEG_BIN_DIR="${DATA}/cwr_dataset_wulann/bin"
if [[ -x "${FFMPEG_BIN_DIR}/ffmpeg" ]]; then
    export PATH="${FFMPEG_BIN_DIR}:${PATH}"
fi
command -v ffmpeg >/dev/null || echo "WARNING: ffmpeg not on PATH (video log is off)"

POLICY_SRC="${AHA_WAM_CODE_DIR}/experiments/robotwin/ahawam_policy"
POLICY_LINK="${AHA_WAM_CODE_DIR}/third_party/RoboTwin/policy/ahawam_policy"
ln -sfn "${POLICY_SRC}" "${POLICY_LINK}"
echo "policy symlink: ${POLICY_LINK} -> ${POLICY_SRC}"

echo "--- sapien renderer preflight ---"
export DATA AHA_WAM_CODE_DIR
"${PYTHON}" - <<'PY'
import os, glob
from pathlib import Path
import sapien
print("sapien", sapien.__version__, "mode", os.environ.get("AHAWAM_VULKAN_MODE"))
print("VK_ICD_FILENAMES", os.environ.get("VK_ICD_FILENAMES"))
renderer = sapien.render.SapienRenderer()
engine = sapien.Engine()
engine.set_renderer(renderer)
scene = engine.create_scene(sapien.SceneConfig())
scene.add_ground(0)
scene.set_ambient_light([0.5, 0.5, 0.5])
scene.add_directional_light([0, 0.5, -1], [0.5, 0.5, 0.5], shadow=False)
print("SapienRenderer+scene: ok")
code = Path(os.environ["AHA_WAM_CODE_DIR"])
urdf = code / "third_party/RoboTwin/assets/embodiments/aloha-agilex/urdf/arx5_description_isaac.urdf"
rt = code / "third_party/RoboTwin"
cwd = os.getcwd()
os.chdir(rt)
try:
    loader = scene.create_urdf_loader()
    loader.fix_root_link = True
    ent = loader.load(str(urdf))
    print("URDF load: ok", type(ent))
finally:
    os.chdir(cwd)
print("/dev/nvidia*:", sorted(glob.glob("/dev/nvidia*")))
PY

mkdir -p "${OUTPUT_DIR}"
SWEEP_PY="${AHA_WAM_CODE_DIR}/experiments/robotwin/run_skip_phase_sweep.py"
echo "launch sweep: ${SWEEP_PY}"
echo "NOTE: GPU0/1 run skip_phase vs skip_phase_proxy in parallel per task."
exec "${PYTHON}" -u "${SWEEP_PY}" \
    --output_dir "${OUTPUT_DIR}" \
    --num_episodes "${NUM_EPISODES}" \
    --num_gpus "${NUM_GPUS}" \
    --timeout_s "${JOB_TIMEOUT_S}" \
    --ckpt "${SKIP_PHASE_CKPT}" \
    --dataset_stats_path "${SKIP_PHASE_DATASET_STATS}" \
    --disable_baseline_reuse \
    --modes "${EVAL_MODES[@]}" \
    --tasks "${TASKS[@]}"
