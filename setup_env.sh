#!/usr/bin/env bash
# AHA-WAM-anygrasp environment activation for this platform
#
# Usage:
#   source setup_env.sh              # prefer NVIDIA Vulkan (needs graphics-capable container)
#   AHAWAM_VULKAN=lavapipe source setup_env.sh   # software Vulkan smoke only
#
# After weights are local, optional offline mode:
#   export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 DIFFSYNTH_SKIP_DOWNLOAD=true
# Do NOT set those before Wan/tokenizer downloads finish.

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
AHA_ENV="/home/ma-user/anaconda3/envs/ahawam"
CWR_AHA="/home/ma-user/work/dataset/cwr_dataset_wulann"
NV_LIBS="$CWR_AHA/nvidia-driver-libs/nvidia-535.183.01"
SAPIEN_LIBS="$CWR_AHA/sapien-runtime-libs"
SAPIEN_VK="$AHA_ENV/lib/python3.10/site-packages/sapien/vulkan_library"
LVP_ICD="/usr/share/vulkan/icd.d/lvp_icd.x86_64.json"

source /home/ma-user/anaconda3/etc/profile.d/conda.sh
conda activate ahawam
cd "$REPO_ROOT"

export PATH="$CWR_AHA/bin:$PATH"
export LD_LIBRARY_PATH="$NV_LIBS:$SAPIEN_LIBS:$AHA_ENV/lib:${LD_LIBRARY_PATH:-}"
export NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics,display,video
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-0}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-0}"
export DIFFSYNTH_SKIP_DOWNLOAD="${DIFFSYNTH_SKIP_DOWNLOAD:-false}"
export DIFFSYNTH_MODEL_BASE_PATH="${DIFFSYNTH_MODEL_BASE_PATH:-$REPO_ROOT/checkpoints}"
export DIFFSYNTH_DOWNLOAD_SOURCE="${DIFFSYNTH_DOWNLOAD_SOURCE:-modelscope}"

AHAWAM_VULKAN="${AHAWAM_VULKAN:-auto}"
_modeset_ok=0
if [[ -e /dev/nvidia-modeset ]] && python -c "import os; os.open('/dev/nvidia-modeset', os.O_RDWR)" 2>/dev/null; then
  _modeset_ok=1
fi

if [[ "$AHAWAM_VULKAN" == "lavapipe" ]] || { [[ "$AHAWAM_VULKAN" == "auto" ]] && [[ "$_modeset_ok" -eq 0 ]]; }; then
  export VK_ICD_FILENAMES="$LVP_ICD"
  unset SAPIEN_VULKAN_LIBRARY_PATH __EGL_VENDOR_LIBRARY_FILENAMES || true
  echo "Vulkan: lavapipe (CPU software) — smoke only, not for full eval"
else
  # Ensure nvidia-modeset node exists when graphics-capable
  if [[ ! -e /dev/nvidia-modeset ]]; then
    mknod /dev/nvidia-modeset c 195 254 2>/dev/null || true
    chmod 666 /dev/nvidia-modeset 2>/dev/null || true
  fi
  export VK_ICD_FILENAMES="${NV_LIBS}/nvidia_icd_abs.json"
  export __EGL_VENDOR_LIBRARY_FILENAMES="${NV_LIBS}/10_nvidia_abs.json"
  export SAPIEN_VULKAN_LIBRARY_PATH="${SAPIEN_VK}/libvulkan.so.1.3.224"
  echo "Vulkan: NVIDIA ICD"
fi

echo "ahawam env ready @ $REPO_ROOT"
echo "  torch: $($AHA_ENV/bin/python -c 'import torch; print(torch.__version__, torch.cuda.is_available())')"
echo "  ckpt:  $REPO_ROOT/checkpoints/AHA-WAM-RoboTwin2.0/robotwin_ahawam.pt"
if [[ "$_modeset_ok" -eq 0 && "$AHAWAM_VULKAN" != "lavapipe" ]]; then
  echo "  WARN: /dev/nvidia-modeset not usable — rebuild container with NVIDIA_DRIVER_CAPABILITIES=...graphics... (方案A)"
  echo "        or: AHAWAM_VULKAN=lavapipe source setup_env.sh (方案B smoke)"
fi
