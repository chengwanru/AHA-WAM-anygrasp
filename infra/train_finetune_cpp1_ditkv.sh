#!/usr/bin/env bash
# =============================================================================
# CPP1_DITKV ADAPT — Mot Video DiT+KV train path, cpp=1 always-refresh (§4.2a path-3)
# NOT skip_v2 proxy. NOT offset. NOT success-BC. TRAIN ONLY (no robotwin eval).
# Output: .../aha-wam-runs/train/cpp1_ditkv_adapt_16x30h/
# =============================================================================
# 平台：算法包 cwr_wulan_algorithm，入口本脚本。
# 资源：2 机 × 8 卡 = 16 GPU（若单机 16 卡：改 NPROC=16 NUM_MACHINES=1）。
# 超时：平台填 ≥30h；脚本 JOB_TIMEOUT_S=108000。
# 挂载：wulann + wulann2 + wulann3 + wulann4
# =============================================================================
set -euo pipefail

REVISION="2026-09-11-cpp1-ditkv-16x30h-v3-hostfile"
echo "train_finetune_cpp1_ditkv.sh revision: ${REVISION}"
echo "EXPERIMENT=CPP1_DITKV_ADAPT (train only; no robotwin eval in this job)"
echo "REGISTRY: docs/EXPERIMENT_REGISTRY.md — keep weights isolated from skip_v2/offset/baseline"

# ===================== 全部写死（平台不用 set 覆盖）=====================
# 禁止默认可变 smoke：正式训必须 SMOKE=0。
SMOKE=0
SMOKE_MAX_STEPS=30
# 30h 上限预算：16 卡、~8–11s/step → ~10000 step ≈ 22–30h（含启动余量）。
FULL_MAX_STEPS=10000
SMOKE_SAVE_EVERY=10
FULL_SAVE_EVERY=1000
LOG_EVERY=10

# 默认 2×8=16；单机 16 卡时改成 NPROC_PER_NODE=16 NUM_MACHINES=1
NPROC_PER_NODE=8
NUM_MACHINES=2
# MACHINE_RANK 下面从 VC_TASK_INDEX 读；这里只是 fallback
MACHINE_RANK=0
MASTER_PORT=29503
JOB_TIMEOUT_S=108000

TASK_NAME="robotwin_ahawam_cpp1_ditkv"
# 全局 batch ≈ 2*16*4=128（与 skip_v2 的 4*8*4 同量级）；DiT 通路偏省显存用 bs=2
BATCH_SIZE="2"
GRAD_ACCUM="4"
NUM_WORKERS="4"
MOT_CHECKPOINT_MIXED_ATTN="true"

# 输出分区：主线 cpp1 适配；勿写入 skip_v2/offset/success_bc 目录。
# 当前正式 16 卡 run 目录名保持 cpp1_ditkv_adapt_16x30h（已在跑）；未来新 run 用 _r2 后缀。
HARDCODED_OUTPUT_DIR="/opt/huawei/dataset/cwr_dataset_wulann4/aha-wam-runs/train/_aborted_or_partial/cpp1_ditkv_adapt_smoke"
HARDCODED_OUTPUT_DIR_FULL="/opt/huawei/dataset/cwr_dataset_wulann4/aha-wam-runs/train/cpp1_ditkv_adapt_16x30h"
# ==================================================================

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "${SCRIPT_DIR}"

# ---------- 路径自探测（/opt vs 探索机）----------
if [[ -d "/opt/huawei/dataset/cwr_dataset_wulann/AHA-WAM-anygrasp" ]]; then
    DATA="/opt/huawei/dataset"
elif [[ -d "/home/ma-user/work/dataset/cwr_dataset_wulann/AHA-WAM-anygrasp" ]]; then
    DATA="/home/ma-user/work/dataset"
else
    echo "ERROR: cannot find AHA-WAM-anygrasp under /opt/huawei/dataset or /home/ma-user/work/dataset"
    exit 1
fi
export DATA
export AHA_WAM_CODE_DIR="${DATA}/cwr_dataset_wulann/AHA-WAM-anygrasp"
DATASET_ROOT="${DATA}/cwr_dataset_wulann2/robotwin2.0"
TEXT_EMBEDS_DIR="${DATA}/cwr_dataset_wulann4/text_embeds_cache/robotwin"
CKPT_DIR="${AHA_WAM_CODE_DIR}/checkpoints/AHA-WAM-RoboTwin2.0"
INIT_CKPT="${CKPT_DIR}/robotwin_ahawam.pt"
NORM_STATS="${CKPT_DIR}/dataset_stats.json"
WHEELS_DIR="${DATA}/cwr_dataset_wulann/wheels"

if [[ "${SMOKE}" == "1" ]]; then
    MAX_STEPS="${SMOKE_MAX_STEPS}"
    SAVE_EVERY="${SMOKE_SAVE_EVERY}"
    OUTPUT_DIR="${HARDCODED_OUTPUT_DIR}"
    RUN_TAG="smoke"
else
    MAX_STEPS="${FULL_MAX_STEPS}"
    SAVE_EVERY="${FULL_SAVE_EVERY}"
    OUTPUT_DIR="${HARDCODED_OUTPUT_DIR_FULL}"
    RUN_TAG="full"
fi
if [[ ! -d "/opt/huawei/dataset" ]]; then
    OUTPUT_DIR="${OUTPUT_DIR/\/opt\/huawei\/dataset/${DATA}}"
fi

# 禁止 silent-smoke：正式训不得落到 *_smoke* 目录
if [[ "${SMOKE}" != "1" ]] && [[ "${OUTPUT_DIR}" == *smoke* ]]; then
    echo "ERROR: full train OUTPUT_DIR looks like smoke: ${OUTPUT_DIR}"
    exit 1
fi
if [[ "${SMOKE}" == "1" ]]; then
    echo "ERROR: SMOKE=1 is blocked for cpp1_ditkv entry (set SMOKE=0)."
    exit 1
fi
if [[ "${TASK_NAME}" != "robotwin_ahawam_cpp1_ditkv" ]]; then
    echo "ERROR: TASK_NAME must be robotwin_ahawam_cpp1_ditkv, got ${TASK_NAME}"
    exit 1
fi

mkdir -p "${OUTPUT_DIR}" "${TEXT_EMBEDS_DIR}"
RUN_TS="$(date +%Y%m%d_%H%M%S)"
JOB_LOG="${OUTPUT_DIR}/train_finetune_cpp1_ditkv_${RUN_TAG}_${RUN_TS}.log"
PROGRESS_LOG="${OUTPUT_DIR}/progress.log"

# 全程 tee 到 dataset 盘日志（平台 stdout 也可能有一份）
exec > >(tee -a "${JOB_LOG}" "${PROGRESS_LOG}") 2>&1

log_step() {
    echo ""
    echo "================================================================================"
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] STEP: $*"
    echo "================================================================================"
}

log_step "start revision=${REVISION}"
echo "JOB_LOG=${JOB_LOG}"
echo "PROGRESS_LOG=${PROGRESS_LOG}"
echo "code dir: $(pwd)"
echo "hostname=$(hostname 2>/dev/null || true)"
echo "whoami=$(whoami 2>/dev/null || true)"

# 平台注入值只打印，不覆盖写死配置
echo "--- platform inject (for debug only; NOT used to override) ---"
echo "MA_NUM_GPUS=${MA_NUM_GPUS:-<unset>} MA_NUM_HOSTS=${MA_NUM_HOSTS:-<unset>}"
echo "VC_TASK_INDEX=${VC_TASK_INDEX:-<unset>} VC_WORKER_HOSTS=${VC_WORKER_HOSTS:-<unset>}"

# 分布式：默认 2×8；MACHINE_RANK / MASTER 读平台（多机必须）
if [[ -n "${VC_TASK_INDEX:-}" ]]; then
    MACHINE_RANK="${VC_TASK_INDEX}"
fi
if [[ -n "${MA_NUM_HOSTS:-}" ]] && [[ "${MA_NUM_HOSTS}" =~ ^[0-9]+$ ]] && [[ "${MA_NUM_HOSTS}" -gt 0 ]]; then
    # 平台声明的 host 数优先（避免写死 2 与实际不符）
    if [[ "${MA_NUM_HOSTS}" -eq 1 ]]; then
        # 单机：若报 16 卡则用满
        if [[ -n "${MA_NUM_GPUS:-}" ]] && [[ "${MA_NUM_GPUS}" =~ ^[0-9]+$ ]] && [[ "${MA_NUM_GPUS}" -ge 1 ]]; then
            NPROC_PER_NODE="${MA_NUM_GPUS}"
        fi
        NUM_MACHINES=1
        MACHINE_RANK=0
    else
        NUM_MACHINES="${MA_NUM_HOSTS}"
    fi
fi
export NNPU="${NPROC_PER_NODE}"
export NNODES="${NUM_MACHINES}"
export NODE_RANK="${MACHINE_RANK}"
MASTER_HOST="${VC_WORKER_HOSTS:-127.0.0.1}"
export MASTER_ADDR="${MASTER_HOST%%,*}"
export MASTER_PORT
export JOB_TIMEOUT_S
TOTAL_GPUS=$((NPROC_PER_NODE * NUM_MACHINES))
echo "RESOLVED: NPROC_PER_NODE=${NPROC_PER_NODE} NUM_MACHINES=${NUM_MACHINES} MACHINE_RANK=${MACHINE_RANK} TOTAL_GPUS=${TOTAL_GPUS}"
if [[ "${TOTAL_GPUS}" -lt 8 ]]; then
    echo "ERROR: expected >=8 GPUs for this job, got TOTAL_GPUS=${TOTAL_GPUS}"
    exit 1
fi

# ---- DeepSpeed hostfile (required for multi-node; without it DS silently uses 8 local GPUs) ----
# Must live on shared NFS (OUTPUT_DIR), same path on every rank — NOT /tmp with $$.
if [[ "${NUM_MACHINES}" -gt 1 ]]; then
    if [[ -z "${VC_WORKER_HOSTS:-}" ]]; then
        echo "ERROR: multi-node needs VC_WORKER_HOSTS to build DeepSpeed hostfile"
        exit 1
    fi
    mkdir -p "${OUTPUT_DIR}"
    DEEPSPEED_HOSTFILE="${OUTPUT_DIR}/deepspeed_hostfile.txt"
    IFS=',' read -r -a _hosts <<< "${VC_WORKER_HOSTS}"
    if [[ "${#_hosts[@]}" -lt "${NUM_MACHINES}" ]]; then
        echo "ERROR: VC_WORKER_HOSTS has ${#_hosts[@]} entries but NUM_MACHINES=${NUM_MACHINES}"
        echo "VC_WORKER_HOSTS=${VC_WORKER_HOSTS}"
        exit 1
    fi
    {
        for ((_i=0; _i<NUM_MACHINES; _i++)); do
            _h="$(echo "${_hosts[_i]}" | tr -d '[:space:]')"
            echo "${_h} slots=${NPROC_PER_NODE}"
        done
    } > "${DEEPSPEED_HOSTFILE}"
    export DEEPSPEED_HOSTFILE
    echo "DEEPSPEED_HOSTFILE=${DEEPSPEED_HOSTFILE}"
    cat "${DEEPSPEED_HOSTFILE}"
else
    unset DEEPSPEED_HOSTFILE || true
fi

echo "--- hardcoded run config ---"
echo "SMOKE=${SMOKE} RUN_TAG=${RUN_TAG}"
echo "TASK_NAME=${TASK_NAME}"
echo "MAX_STEPS=${MAX_STEPS} SAVE_EVERY=${SAVE_EVERY} LOG_EVERY=${LOG_EVERY}"
echo "TRAIN_BUDGET_HINT=~30h cap / ~10000 steps on 16 GPU (max_steps=${MAX_STEPS}, timeout=${JOB_TIMEOUT_S}s)"
echo "BATCH_SIZE=${BATCH_SIZE:-yaml_default} GRAD_ACCUM=${GRAD_ACCUM:-1} NUM_WORKERS=${NUM_WORKERS:-yaml_default}"
echo "NPROC_PER_NODE=${NPROC_PER_NODE} NNODES=${NNODES} NODE_RANK=${NODE_RANK}"
echo "MASTER_ADDR=${MASTER_ADDR} MASTER_PORT=${MASTER_PORT}"
echo "JOB_TIMEOUT_S=${JOB_TIMEOUT_S}"
echo "DATA=${DATA}"
echo "AHA_WAM_CODE_DIR=${AHA_WAM_CODE_DIR}"
echo "DATASET_ROOT=${DATASET_ROOT}"
echo "TEXT_EMBEDS_DIR=${TEXT_EMBEDS_DIR}"
echo "INIT_CKPT=${INIT_CKPT}"
echo "NORM_STATS=${NORM_STATS}"
echo "OUTPUT_DIR=${OUTPUT_DIR}"
echo "WHEELS_DIR=${WHEELS_DIR}"

# ---------- 运行环境变量 ----------
export WANDB_MODE="offline"
export TORCH_NCCL_ENABLE_MONITORING=0
# CUDA/NCCL stability (success-BC 曾 misaligned address / CUMEM)
export NCCL_CUMEM_ENABLE="${NCCL_CUMEM_ENABLE:-0}"
export NCCL_CUMEM_HOST_ENABLE="${NCCL_CUMEM_HOST_ENABLE:-0}"
export TORCH_NCCL_AVOID_RECORD_STREAMS="${TORCH_NCCL_AVOID_RECORD_STREAMS:-1}"
export TOKENIZERS_PARALLELISM=true
export OMP_NUM_THREADS=8
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export DIFFSYNTH_SKIP_DOWNLOAD=true
export DIFFSYNTH_MODEL_BASE_PATH="${AHA_WAM_CODE_DIR}/checkpoints"

export PIP_INDEX_URL="http://repo.myhuaweicloud.com/repository/pypi/simple"
export PIP_TRUSTED_HOST="repo.myhuaweicloud.com"
export PIP_EXTRA_INDEX_URL=""
export PIP_DISABLE_PIP_VERSION_CHECK=1
export PIP_DEFAULT_TIMEOUT=120
export PIP_CONFIG_FILE="/dev/null"

# ---------- 强制 Python >=3.10（ahawam 要求；禁止用镜像默认 3.9）----------
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
    echo "No Python>=3.10 env found; creating on dataset disk..."
    if [[ ! -f /home/ma-user/anaconda3/etc/profile.d/conda.sh ]] && [[ ! -f "${HOME}/anaconda3/etc/profile.d/conda.sh" ]]; then
        echo "ERROR: conda not found; cannot create Python 3.10 env"
        exit 1
    fi
    # shellcheck disable=SC1091
    source /home/ma-user/anaconda3/etc/profile.d/conda.sh 2>/dev/null || source "${HOME}/anaconda3/etc/profile.d/conda.sh"
    ENV_PREFIX="${DATA}/cwr_dataset_wulann4/envs/ahawam"
    if [[ ! -x "${ENV_PREFIX}/bin/python" ]]; then
        conda create -p "${ENV_PREFIX}" python=3.10 -y
    fi
    PYTHON="${ENV_PREFIX}/bin/python"
fi
export PYTHON
echo "Using python: ${PYTHON} ($("${PYTHON}" -V 2>&1))"
"${PYTHON}" - <<'PY'
import sys
assert sys.version_info[:2] >= (3, 10), f"need Python>=3.10, got {sys.version}"
print("python version gate: ok")
PY

SAPIEN_LIBS="${DATA}/cwr_dataset_wulann/sapien-runtime-libs"
NVIDIA_DRIVER_DIR="${DATA}/cwr_dataset_wulann/nvidia-driver-libs/nvidia-535.183.01"
_ld_add=()
if [[ -d "${NVIDIA_DRIVER_DIR}" ]]; then
    _ld_add+=("${NVIDIA_DRIVER_DIR}")
fi
if [[ -d "${SAPIEN_LIBS}" ]]; then
    _ld_add+=("${SAPIEN_LIBS}")
fi
if [[ ${#_ld_add[@]} -gt 0 ]]; then
    _joined="$(IFS=:; echo "${_ld_add[*]}")"
    export LD_LIBRARY_PATH="${_joined}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
fi
echo "LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-<empty>}"

# ---------- 依赖：共享盘 env 已预装则绝不 pip 写 NFS（并发写会 SIGBUS）----------
log_step "install training deps"
export PYTHONUSERBASE="/tmp/ahawam_user"
mkdir -p "${PYTHONUSERBASE}"
PYVER="$("${PYTHON}" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
export PATH="$(dirname "${PYTHON}"):${PYTHONUSERBASE}/bin:${PATH}"
# 先不把 ahawam 放进 PYTHONPATH，避免 pip 把它当成已装包并打印假冲突
USER_SITE="${PYTHONUSERBASE}/lib/python${PYVER}/site-packages"
export PYTHONPATH="${USER_SITE}${PYTHONPATH:+:${PYTHONPATH}}"

_pip() {
    # 缺包时只装到 /tmp，禁止写入共享 NFS env
    "${PYTHON}" -m pip install --user "$@"
}

# Huawei mirror: accelerate max 1.10.1 (no 1.12.0). deepspeed 0.18.7 OK.
DEPS=(
    "accelerate==1.10.1"
    "deepspeed==0.18.7"
    "hydra-core==1.3.2"
    "omegaconf==2.3.0"
    "einops==0.8.1"
    "safetensors==0.5.3"
    "transformers==4.49.0"
    "datasets==3.6.0"
    "pandas==2.2.3"
    "numpy==1.26.4"
    "tqdm==4.66.5"
    "pillow"
    "imageio"
    "imageio-ffmpeg"
    "wandb"
    "rich"
    "termcolor"
    "jsonlines"
    "huggingface-hub"
    "packaging"
    "regex"
    "boto3==1.35.99"
    "modelscope==1.34.0"
    "sentencepiece"
    "ftfy"
    "pyarrow"
    "av"
    "gitpython"
)
OPTIONAL_DEPS=()

# ahawam 仅 PYTHONPATH；其它包在 env 内
check_critical_imports() {
    AHA_SRC="${AHA_WAM_CODE_DIR}/src" AHA_ROOT="${AHA_WAM_CODE_DIR}" "${PYTHON}" - <<'PY'
import importlib, os, sys
sys.path.insert(0, os.environ["AHA_ROOT"])
sys.path.insert(0, os.environ["AHA_SRC"])
# Hard imports for train.py + precompute_text_embeds + video(pyav) path
need = [
    "torch", "torchvision", "accelerate", "deepspeed", "hydra", "omegaconf",
    "einops", "safetensors", "transformers", "datasets", "pandas", "numpy",
    "tqdm", "PIL", "imageio", "imageio_ffmpeg", "wandb", "rich", "termcolor",
    "jsonlines", "huggingface_hub", "packaging", "regex", "boto3", "modelscope",
    "av", "pyarrow", "git", "sentencepiece", "ftfy", "ahawam",
]
for name in need:
    mod = "PIL" if name == "PIL" else name
    importlib.import_module(mod)
    print(f"import {name}: ok")
# train / text-embeds entry surfaces
import ahawam.utils.misc  # noqa: F401
from ahawam.runtime import run_training  # noqa: F401
from ahawam.datasets.lerobot.robot_video_dataset import DEFAULT_PROMPT  # noqa: F401
from ahawam.models.wan22.wan_video_text_encoder import HuggingfaceTokenizer  # noqa: F401
from ahawam.datasets.lerobot.lerobot.datasets.video_utils import get_safe_default_codec
print("import ahawam.utils.misc: ok")
print("import ahawam.runtime.run_training: ok")
print("import text/video path: ok")
print("video_codec_default=", get_safe_default_codec())
import torch
assert torch.cuda.is_available(), "CUDA torch required"
print(f"torch={torch.__version__} n_gpu={torch.cuda.device_count()}")
PY
}

# 轻量缺包补到 /tmp，不碰共享 NFS env
ensure_light_pkgs_to_tmp() {
    local need_pkgs=(
        boto3 modelscope sentencepiece ftfy av pyarrow gitpython
        imageio imageio-ffmpeg termcolor jsonlines rich wandb
        matplotlib
    )
    local miss=()
    local p mod
    for p in "${need_pkgs[@]}"; do
        case "$p" in
            gitpython) mod=git ;;
            imageio-ffmpeg) mod=imageio_ffmpeg ;;
            *) mod="${p%%==*}" ;;
        esac
        if ! "${PYTHON}" -c "import ${mod}" 2>/dev/null; then
            miss+=("$p")
        fi
    done
    if [[ ${#miss[@]} -eq 0 ]]; then
        echo "light pkgs already present"
        return 0
    fi
    echo "installing missing light pkgs to /tmp: ${miss[*]}"
    _pip --index-url "${PIP_INDEX_URL}" --trusted-host "${PIP_TRUSTED_HOST}" "${miss[@]}" || true
}

IMPORTS_OK=0
if check_critical_imports; then
    IMPORTS_OK=1
    echo "critical imports already ok -> skip heavy pip (avoid NFS rewrite / SIGBUS)"
    ensure_light_pkgs_to_tmp
else
    echo "critical imports missing/broken -> install into /tmp only (not NFS env)"
    if ! "${PYTHON}" -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
        echo "torch+cuda missing; installing cu121 torch into --user /tmp"
        _pip torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121
    fi
    if [[ -d "${WHEELS_DIR}" ]]; then
        echo "install from wheels -> --user: ${WHEELS_DIR}"
        _pip --no-index --find-links "${WHEELS_DIR}" "${DEPS[@]}" || true
    else
        echo "install from Huawei mirror -> --user"
        _pip --index-url "${PIP_INDEX_URL}" --trusted-host "${PIP_TRUSTED_HOST}" "${DEPS[@]}" || true
    fi
    ensure_light_pkgs_to_tmp
fi

echo "[$(date '+%Y-%m-%d %H:%M:%S')] deps stage finished; verifying imports (retry on NFS SIGBUS)"
# 正式启用 ahawam 路径
export PYTHONPATH="${AHA_WAM_CODE_DIR}/src:${AHA_WAM_CODE_DIR}:${USER_SITE}${PYTHONPATH:+:${PYTHONPATH}}"

# NFS 上读半截 .so 会 Bus error；短暂重试即可
set +e
IMPORT_RC=1
for attempt in 1 2 3 4 5 6 7 8; do
    echo "import attempt ${attempt}/8 ..."
    check_critical_imports
    IMPORT_RC=$?
    if [[ "${IMPORT_RC}" -eq 0 ]]; then
        break
    fi
    echo "import failed rc=${IMPORT_RC}; sleep 15s (possible NFS settle)"
    sleep 15
done
set -e
if [[ "${IMPORT_RC}" -ne 0 ]]; then
    echo "ERROR: critical imports failed after retries (rc=${IMPORT_RC})."
    echo "If Bus error: shared env was being written concurrently; re-submit after env settles."
    exit "${IMPORT_RC}"
fi
echo "deps import gate: ok (skip_pip=${IMPORTS_OK})"

# ---------- 前置检查 ----------
log_step "preflight"
echo "--- regression checklist (prior offset-job failures) ---"
echo "1. Python>=3.10 via NFS ahawam env (not image 3.9)"
echo "2. skip heavy pip when imports ok (avoid NFS SIGBUS)"
echo "3. text cache must be COMPLETE (~921032), else refuse train"
echo "4. video symlinks must resolve (mount wulann2+3+4)"
echo "5. OOM guards: batch_size=${BATCH_SIZE} accum=${GRAD_ACCUM} mot_checkpoint_mixed_attn=${MOT_CHECKPOINT_MIXED_ATTN}"
echo "6. task=${TASK_NAME} freeze_video_dit=true cpp1_ditkv_train=true cpp=1 (NOT skip_v2/offset)"
echo "7. THIS JOB = TRAIN ONLY (eval cpp1 always separately after)"
echo "8. NCCL_CUMEM_ENABLE=${NCCL_CUMEM_ENABLE} TOTAL_GPUS≈${TOTAL_GPUS}"
echo "9. text cache must be COMPLETE; multi-node will NOT run precompute (refuse if incomplete)"
"${PYTHON}" - <<PY
from pathlib import Path
print("executable:", __import__("sys").executable)
import torch
print(f"torch={torch.__version__} cuda={torch.cuda.is_available()} n_gpu={torch.cuda.device_count()}")
assert torch.cuda.is_available(), "CUDA torch required"
assert torch.cuda.device_count() >= 1, "need at least 1 GPU"

root = Path("${DATASET_ROOT}")
ckpt = Path("${INIT_CKPT}")
stats = Path("${NORM_STATS}")
assert root.is_dir(), f"missing dataset root: {root}"
assert (root / "meta" / "info.json").is_file() or (root / "robotwin2.0" / "meta" / "info.json").is_file(), (
    f"missing LeRobot meta under {root}"
)
assert ckpt.is_file(), f"missing init ckpt: {ckpt}"
assert stats.is_file(), f"missing dataset_stats: {stats}"
t5 = Path("${AHA_WAM_CODE_DIR}") / "checkpoints/DiffSynth-Studio/Wan-Series-Converted-Safetensors/models_t5_umt5-xxl-enc-bf16.safetensors"
vae = Path("${AHA_WAM_CODE_DIR}") / "checkpoints/DiffSynth-Studio/Wan-Series-Converted-Safetensors/Wan2.2_VAE.safetensors"
assert t5.is_file(), f"missing Wan T5: {t5}"
assert vae.is_file(), f"missing Wan VAE: {vae}"

root_ds = root if (root / "meta" / "info.json").is_file() else (root / "robotwin2.0")
videos = root_ds / "videos"
assert videos.is_dir(), f"missing videos dir: {videos}"
broken = []
ok = 0
for p in sorted(videos.iterdir()):
    if not p.is_symlink():
        continue
    if not p.exists():
        broken.append(f"{p.name} -> {p.readlink()}")
    else:
        ok += 1
print(f"video symlinks ok={ok} broken={len(broken)}")
if broken:
    raise AssertionError("broken video symlinks (mount wulann2+3+4?): " + "; ".join(broken[:5]))
assert ok > 0, "no video chunk symlinks found"
print("preflight assets: ok")
PY

TRAIN_DATA_DIR="${DATASET_ROOT}"
if [[ -f "${DATASET_ROOT}/robotwin2.0/meta/info.json" ]]; then
    TRAIN_DATA_DIR="${DATASET_ROOT}/robotwin2.0"
elif [[ -f "${DATASET_ROOT}/meta/info.json" ]]; then
    TRAIN_DATA_DIR="${DATASET_ROOT}"
else
    echo "ERROR: cannot resolve LeRobot root under ${DATASET_ROOT}"
    find "${DATASET_ROOT}" -maxdepth 3 -name info.json 2>/dev/null | head -20 || true
    exit 1
fi
echo "TRAIN_DATA_DIR=${TRAIN_DATA_DIR}"

cd "${AHA_WAM_CODE_DIR}"

# ---------- 文本 embedding（必须完整；非空≠完整）----------
# NFS 上对 ~90万 .pt 做 find|wc / du 会卡很久且无日志。有合法 .COMPLETE 则直接跳过全量计数。
log_step "text embedding cache"
mkdir -p "${TEXT_EMBEDS_DIR}"
TASKS_JSONL="${TRAIN_DATA_DIR}/meta/tasks.jsonl"
EXPECTED_EMBEDS=0
if [[ -f "${TASKS_JSONL}" ]]; then
    EXPECTED_EMBEDS="$(wc -l < "${TASKS_JSONL}" | tr -d ' ')"
fi
COMPLETE_MARK="${TEXT_EMBEDS_DIR}/.COMPLETE"
NEED_PRECOMPUTE=1
CACHE_COUNT=0

if [[ -f "${COMPLETE_MARK}" ]]; then
    MARK_N="$(tr -d '[:space:]' < "${COMPLETE_MARK}" || true)"
    if [[ "${MARK_N}" =~ ^[0-9]+$ ]] && [[ "${MARK_N}" -gt 0 ]]; then
        # Fast path: trust mark. Spot-check a few .pt exist (no full directory walk).
        SAMPLE_PT="$(find "${TEXT_EMBEDS_DIR}" -maxdepth 1 -name '*.pt' -print -quit 2>/dev/null || true)"
        if [[ -n "${SAMPLE_PT}" ]] && [[ -f "${SAMPLE_PT}" ]]; then
            if [[ "${EXPECTED_EMBEDS}" -gt 0 ]] && [[ "${MARK_N}" -lt "${EXPECTED_EMBEDS}" ]]; then
                echo "COMPLETE mark=${MARK_N} < expected=${EXPECTED_EMBEDS} -> will resume precompute"
            else
                NEED_PRECOMPUTE=0
                CACHE_COUNT="${MARK_N}"
                echo "text cache FAST skip: COMPLETE mark=${MARK_N} expected=${EXPECTED_EMBEDS} sample_pt=ok"
                echo "skip full find|wc and du on NFS (avoids multi-minute stall)"
            fi
        else
            echo "COMPLETE mark present but no .pt sample found; will recount/precompute"
        fi
    else
        echo "COMPLETE mark unreadable (${MARK_N:-empty}); will recount"
    fi
fi

if [[ "${NEED_PRECOMPUTE}" -eq 1 ]]; then
    echo "counting .pt (slow on NFS; only when cache not marked complete) ..."
    CACHE_COUNT="$(find "${TEXT_EMBEDS_DIR}" -maxdepth 1 -name '*.pt' 2>/dev/null | wc -l | tr -d ' ')"
    echo "text cache: dir=${TEXT_EMBEDS_DIR} have=${CACHE_COUNT} expected_tasks_jsonl=${EXPECTED_EMBEDS}"
    if [[ "${EXPECTED_EMBEDS}" -gt 0 ]] && [[ "${CACHE_COUNT}" -ge "${EXPECTED_EMBEDS}" ]]; then
        NEED_PRECOMPUTE=0
        echo "${EXPECTED_EMBEDS}" > "${COMPLETE_MARK}"
        echo "text cache count reached expected; wrote COMPLETE mark"
    fi
fi

if [[ "${NEED_PRECOMPUTE}" -eq 1 ]]; then
    if [[ "${MACHINE_RANK}" != "0" ]]; then
        echo "ERROR: text cache incomplete on non-rank0. Wait for rank0 COMPLETE or precompute offline."
        exit 1
    fi
    if [[ "${NUM_MACHINES}" -gt 1 ]]; then
        echo "ERROR: multi-node job refuses inline precompute (race). Ensure ${COMPLETE_MARK} exists first."
        echo "have=${CACHE_COUNT} expected=${EXPECTED_EMBEDS}"
        exit 1
    fi
    echo "cache incomplete -> resume precompute_text_embeds with torchrun (overwrite=false)"
    "${PYTHON}" -m torch.distributed.run --standalone --nproc_per_node="${NPROC_PER_NODE}" \
        scripts/precompute_text_embeds.py \
        task="${TASK_NAME}" \
        +overwrite=false \
        "data.train.dataset_dirs=[${TRAIN_DATA_DIR}]" \
        "data.val.dataset_dirs=[${TRAIN_DATA_DIR}]" \
        "data.train.text_embedding_cache_dir=${TEXT_EMBEDS_DIR}" \
        "data.val.text_embedding_cache_dir=${TEXT_EMBEDS_DIR}" \
        "data.train.pretrained_norm_stats=${NORM_STATS}" \
        "data.val.pretrained_norm_stats=${NORM_STATS}"
    echo "post-precompute recount (slow once) ..."
    CACHE_COUNT="$(find "${TEXT_EMBEDS_DIR}" -maxdepth 1 -name '*.pt' 2>/dev/null | wc -l | tr -d ' ')"
    echo "after precompute: have=${CACHE_COUNT} expected=${EXPECTED_EMBEDS}"
    if [[ "${EXPECTED_EMBEDS}" -gt 0 ]] && [[ "${CACHE_COUNT}" -lt "${EXPECTED_EMBEDS}" ]]; then
        echo "ERROR: text embed cache still incomplete (${CACHE_COUNT}/${EXPECTED_EMBEDS}). Refuse to train."
        exit 1
    fi
    if [[ "${EXPECTED_EMBEDS}" -gt 0 ]]; then
        echo "${EXPECTED_EMBEDS}" > "${COMPLETE_MARK}"
        CACHE_COUNT="${EXPECTED_EMBEDS}"
    else
        echo "${CACHE_COUNT}" > "${COMPLETE_MARK}"
    fi
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] precompute_text_embeds finished COMPLETE have=${CACHE_COUNT}"
else
    echo "skip precompute (cache complete, have≈${CACHE_COUNT})"
fi
echo "[$(date '+%Y-%m-%d %H:%M:%S')] text embeds stage done"

log_step "nvidia-smi"
nvidia-smi || true

# ---------- live progress watcher (dataset disk) ----------
PROGRESS_DIR="${OUTPUT_DIR}/progress"
mkdir -p "${PROGRESS_DIR}"
WATCHER_PY="${AHA_WAM_CODE_DIR}/infra/watch_train_progress.py"
if [[ ! -f "${WATCHER_PY}" ]]; then
    WATCHER_PY="${SCRIPT_DIR}/watch_train_progress.py"
fi
WATCHER_STOP="${PROGRESS_DIR}/WATCHER_STOP"
WATCHER_LOG="${PROGRESS_DIR}/watcher.log"
WATCHER_PID=""
rm -f "${WATCHER_STOP}"
if [[ -f "${WATCHER_PY}" ]] && [[ "${MACHINE_RANK}" == "0" ]]; then
    log_step "start live progress watcher -> ${PROGRESS_DIR}"
    echo "Open/refresh: ${PROGRESS_DIR}/progress_live.svg"
    echo "One-liner:    ${PROGRESS_DIR}/status.txt"
    nohup "${PYTHON}" "${WATCHER_PY}" \
        --log "${JOB_LOG}" \
        --log "${PROGRESS_LOG}" \
        --out-dir "${PROGRESS_DIR}" \
        --interval 15 \
        --snapshot-every 1000 \
        --title "CPP1_DITKV adapt" \
        --stop-flag "${WATCHER_STOP}" \
        >"${WATCHER_LOG}" 2>&1 &
    WATCHER_PID=$!
    echo "${WATCHER_PID}" > "${PROGRESS_DIR}/watcher.pid"
    echo "watcher pid=${WATCHER_PID}"
    echo "Viz: ${PROGRESS_DIR}/dashboard_live.png  ${PROGRESS_DIR}/progress_live.svg"
    echo "     ${PROGRESS_DIR}/status.txt  ${PROGRESS_DIR}/snapshots/"
else
    if [[ "${MACHINE_RANK}" != "0" ]]; then
        echo "watcher skipped on MACHINE_RANK=${MACHINE_RANK} (only rank0 draws plots)"
    else
        echo "WARN: watch_train_progress.py missing; skip live plot"
    fi
fi

# ---------- 训练（仅训练，不含 robotwin 评测）----------
# SKIP_V2 training: reduce CUDA allocator fragmentation
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

EXTRA_HYDRA=()
if [[ -n "${BATCH_SIZE}" ]]; then
    EXTRA_HYDRA+=("batch_size=${BATCH_SIZE}")
fi
if [[ -n "${GRAD_ACCUM}" ]]; then
    EXTRA_HYDRA+=("gradient_accumulation_steps=${GRAD_ACCUM}")
fi
if [[ -n "${NUM_WORKERS}" ]]; then
    EXTRA_HYDRA+=("num_workers=${NUM_WORKERS}")
fi
EXTRA_HYDRA+=("freeze_video_dit=true")
EXTRA_HYDRA+=("model.mot_checkpoint_mixed_attn=${MOT_CHECKPOINT_MIXED_ATTN}")
EXTRA_HYDRA+=("model.loss.lambda_video=0.0")
EXTRA_HYDRA+=("data.train.cpp1_ditkv_train=true")
EXTRA_HYDRA+=("data.val.cpp1_ditkv_train=true")
EXTRA_HYDRA+=("data.train.chunks_per_video_prefill=1")
EXTRA_HYDRA+=("data.val.chunks_per_video_prefill=1")
EXTRA_HYDRA+=("data.train.skip_phase_v2_train=false")
EXTRA_HYDRA+=("data.val.skip_phase_v2_train=false")

log_step "launch train_zero2 nproc=${NPROC_PER_NODE} nodes=${NUM_MACHINES} rank=${MACHINE_RANK} max_steps=${MAX_STEPS} (TRAIN ONLY)"
echo "FINAL OUTPUT_DIR=${OUTPUT_DIR}"
echo "PROGRESS_DIR=${PROGRESS_DIR}"
echo "EXTRA_HYDRA=${EXTRA_HYDRA[*]:-<none>}"
echo "PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF}"
echo "BUDGET: max_steps=${MAX_STEPS} timeout_s=${JOB_TIMEOUT_S} (~30h cap)"

set +e
bash scripts/train_zero2.sh "${NPROC_PER_NODE}" \
    "task=${TASK_NAME}" \
    "init_checkpoint=${INIT_CKPT}" \
    "model.action_dit_pretrained_path=null" \
    "model.skip_dit_load_from_pretrain=true" \
    "data.train.dataset_dirs=[${TRAIN_DATA_DIR}]" \
    "data.val.dataset_dirs=[${TRAIN_DATA_DIR}]" \
    "data.train.pretrained_norm_stats=${NORM_STATS}" \
    "data.val.pretrained_norm_stats=${NORM_STATS}" \
    "data.train.text_embedding_cache_dir=${TEXT_EMBEDS_DIR}" \
    "data.val.text_embedding_cache_dir=${TEXT_EMBEDS_DIR}" \
    "output_dir=${OUTPUT_DIR}" \
    "wandb.mode=offline" \
    "wandb.enabled=false" \
    "max_steps=${MAX_STEPS}" \
    "save_every=${SAVE_EVERY}" \
    "log_every=${LOG_EVERY}" \
    "${EXTRA_HYDRA[@]}"
TRAIN_RC=$?
set -e

# stop watcher
if [[ -n "${WATCHER_PID}" ]]; then
    touch "${WATCHER_STOP}"
    for _ in 1 2 3 4 5 6; do
        if ! kill -0 "${WATCHER_PID}" 2>/dev/null; then
            break
        fi
        sleep 2
    done
    kill "${WATCHER_PID}" 2>/dev/null || true
    echo "watcher stopped; see ${PROGRESS_DIR}/dashboard_live.png and progress_live.svg"
fi

log_step "train finished rc=${TRAIN_RC}"
echo "OUTPUT_DIR contents:"
ls -lah "${OUTPUT_DIR}" | head -50 || true
echo "PROGRESS_DIR=${PROGRESS_DIR}"
ls -lah "${PROGRESS_DIR}" 2>/dev/null | head -30 || true
echo "JOB_LOG=${JOB_LOG}"
echo "PROGRESS_LOG=${PROGRESS_LOG}"
if [[ -f "${PROGRESS_DIR}/status.txt" ]]; then
    echo "LATEST STATUS: $(cat "${PROGRESS_DIR}/status.txt")"
fi
if [[ "${TRAIN_RC}" -ne 0 ]]; then
    echo "ERROR: training exited with ${TRAIN_RC}"
    exit "${TRAIN_RC}"
fi
echo "[$(date '+%Y-%m-%d %H:%M:%S')] SUCCESS (train only; run eval separately)"
