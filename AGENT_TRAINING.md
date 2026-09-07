# AHA-WAM 训练任务交接（给后来的 agent）

> 本文教你：**怎么在本平台提交真正的训练/短 finetune**，脚本放哪、代码在哪、环境怎么建、权重与数据放哪、以及我们踩过的坑。  
> 评测 / ah·cpp sweep / skip_phase 请先读 [`AGENT_ONBOARDING.md`](./AGENT_ONBOARDING.md)。  
> 最后更新：2026-09-07

---

## 0. 先读这三条，避免第一天就跑偏

1. **`infra/train_mtp.sh` / `algorithm/cwr_wulan_algorithm/train_mtp.sh` 不是训练。**  
   名字带 `train_` 是平台入口历史命名，实际跑的是 **RoboTwin 评测 sweep**。  
   真正训练入口是仓库里的：
   - `scripts/train_zero1.sh` / `scripts/train_zero2.sh` → `scripts/train.py`（Accelerate + DeepSpeed）
2. **平台提交时通常不能自定义环境变量。**  
   凡是路径、SMOKE、GPU 数、输出目录，都要 **写死在入口 `.sh` 里**，并做 `/opt/huawei/dataset` ↔ `/home/ma-user/work/dataset` 自动回退。
3. **到目前为止主线工作是评测 released ckpt，不是从零训练。**  
   若要为 cpp=1/3 等部署节奏做适配，优先 **短 finetune**，不要默认开完整重训。

---

## 1. 代码与脚本应该在哪里

| 角色 | 路径 | 说明 |
|---|---|---|
| 可改代码仓库 | `dataset/cwr_dataset_wulann/AHA-WAM-anygrasp`（集群挂载常为 `/opt/huawei/dataset/cwr_dataset_wulann/AHA-WAM-anygrasp`） | 分支 **`modified-working`** |
| 官方训练 launcher | `scripts/train_zero1.sh`、`scripts/train_zero2.sh`、`scripts/train.py` | 真正的训练 |
| Hydra 配置 | `configs/train.yaml`、`configs/task/*.yaml`、`configs/model/*.yaml`、`configs/data/*.yaml` | 改 task / data / model 主要改这里 |
| **平台入口脚本（推荐维护点）** | 仓库 **`infra/`** | `train_mtp.sh`（评测）、将来真正的 `train_finetune_*.sh` 也应放这里 |
| 平台历史拷贝 | `algorithm/cwr_wulan_algorithm/train_*.sh` | 部分平台只认 `algorithm/` 下入口；改完 `infra/` 后要 **同步拷到 algorithm**，或让入口 `source`/`exec` 仓库 `infra/` |
| 评测 bridge | `experiments/robotwin/ahawam_policy/` | 评测用；训练一般不走这里 |
| baseline 只读对照 | `dataset/cwr_dataset_wulann/AHA-WAM-baseline` | worktree，勿当日常开发目录 |

GitHub（以当前 remote 为准，push 前 `git remote -v`）：

```bash
git clone -b modified-working <your-fork-url>/AHA-WAM-anygrasp.git
cd AHA-WAM-anygrasp
```

---

## 2. 模型 / 数据 / 输出放哪里

约定：大文件一律放 **dataset 盘**（探索机 ` /home/ma-user/work/dataset/...`，训练机常见 `/opt/huawei/dataset/...`，内容是同一套 turbo）。

| 资产 | 建议路径 | 备注 |
|---|---|---|
| Released 评测 ckpt | `<repo>/checkpoints/AHA-WAM-RoboTwin2.0/robotwin_ahawam.pt` + **同目录** `dataset_stats.json` | HF `SereneC/AHA-WAM-RoboTwin2.0` |
| Wan / T5 / VAE | `<repo>/checkpoints/...` 或 `DIFFSYNTH_MODEL_BASE_PATH` | 训练也要；离线设 `DIFFSYNTH_SKIP_DOWNLOAD=true` |
| ActionDiT 初始化骨干 | 自备 `.pt`，写进 `model.action_dit_pretrained_path` | 从零训前先跑 `scripts/preprocess_action_dit_backbone.py` |
| RoboTwin 训练数据 | 例如 `<DATA>/cwr_dataset_wulann/robotwin2.0`（按实际数据布局） | 在 `configs/task/*.yaml` 的 `data.train.dataset_dirs` 改成真实路径，**不要留 `path/to/...`** |
| 文本 embedding 缓存 | `text_embedding_cache_dir` | 先跑 `scripts/precompute_text_embeds.py task=...` |
| 归一化统计 | `pretrained_norm_stats` → `dataset_stats.json` | 可与 released 同文件；finetune 时通常沿用 |
| 训练输出 / ckpt | 建议 `<DATA>/cwr_dataset_wulann/aha-wam-runs/train/<run_name>/` | 用 Hydra `output_dir=` 指到 dataset，**别写满系统盘** |
| 评测 sweep 输出 | `<DATA>/cwr_dataset_wulann/aha-wam-runs/robotwin_*` | 与训练输出分目录，避免互相覆盖 |
| 离线 wheels | `<DATA>/cwr_dataset_wulann/wheels` | 集群无外网/NGC 毒源时用 |
| sapien 运行库 | `<DATA>/cwr_dataset_wulann/sapien-runtime-libs` + `nvidia-driver-libs/...` | `LD_LIBRARY_PATH` 前置 |

离线强制：

```bash
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export DIFFSYNTH_SKIP_DOWNLOAD=true
export DIFFSYNTH_MODEL_BASE_PATH=<repo>/checkpoints
export WANDB_MODE=offline   # 平台上建议 offline，除非明确配了 wandb
```

---

## 3. 环境怎么建

目标：**Python 3.10** + 镜像自带 CUDA torch（勿被 pip 覆盖）+ accelerate/deepspeed + 数据管线依赖。

### 3.1 推荐（与 README 一致）

```bash
conda create -n ahawam python=3.10 -y
conda activate ahawam
pip install -U pip setuptools wheel
pip install torch==2.7.1+cu128 torchvision==0.22.1+cu128 \
  --extra-index-url https://download.pytorch.org/whl/cu128
pip install -e .
# 确认没被盖掉：
python -c "import torch; assert torch.cuda.is_available(); print(torch.__version__)"
```

### 3.2 本平台训练机常见情况

- 镜像 **可能没有 conda / 没有预装 ahawam**。
- 入口脚本应：**华为 PyPI 镜像 + 清空 `PIP_EXTRA_INDEX_URL`**（禁 NGC），优先  
  `pip install --no-index --find-links <wheels> ...`，只补缺失包，**不要每次全量 `pip install -r requirements.txt` 还连外网**。
- 训练若暂不需要 sapien 渲染，可比评测环境更瘦；但一旦要「训完立刻同机评测」，仍按 `AGENT_ONBOARDING.md` 装齐 sapien/X11/ffmpeg/Vulkan。

### 3.3 冒烟（训练前）

```bash
python - <<'PY'
import torch
assert torch.cuda.is_available()
import accelerate, deepspeed
from ahawam.runtime import *  # 或你们实际用的 import
print("train smoke OK", torch.cuda.device_count())
PY
```

---

## 4. 真正训练怎么启（仓库内标准路径）

### 4.1 配置

常用 task（见 README）：

| task | 用途 |
|---|---|
| `robotwin_ahawam` | 标准 AHA-WAM |
| `robotwin_ahawam_offset` | **horizon-adaptive offset**（适配异步相位；和评测 cpp 最相关） |
| `robotwin_ahawam_ode` / `*_offset_ode` | ODE / Flash 蒸馏向 |

提交前把 yaml 里所有 `path/to/...` 改成 dataset 真实路径，或启动时用 Hydra override。

### 4.2 文本缓存（有数据变更时）

```bash
python scripts/precompute_text_embeds.py task=robotwin_ahawam_offset
# 多卡：
torchrun --standalone --nproc_per_node=8 scripts/precompute_text_embeds.py task=robotwin_ahawam_offset
```

### 4.3 启动

```bash
# ZeRO-1 / ZeRO-2，第一个参数 = 每机进程数（通常 = GPU 数）
bash scripts/train_zero1.sh 8 task=robotwin_ahawam_offset \
  output_dir=/opt/huawei/dataset/cwr_dataset_wulann/aha-wam-runs/train/offset_ft \
  wandb.mode=offline

bash scripts/train_zero2.sh 8 task=robotwin_ahawam_offset \
  data.train.dataset_dirs=[/opt/huawei/dataset/cwr_dataset_wulann/robotwin2.0] \
  model.action_dit_pretrained_path=/path/to/ActionDiT_....pt \
  resume=/path/to/last_ckpt_or_null
```

多机时设置（平台常注入类似变量）：`NNODES`、`NODE_RANK`、`MASTER_ADDR`、`MASTER_PORT`，并保证各机 **同一个 `RUN_ID`**（`train_zero2.sh` 里有 TCPStore 同步逻辑）。

短 finetune（例如怀疑 cpp=1/3 适配不够）建议：

- 从 released `robotwin_ahawam.pt` **resume / 加载**（按代码实际 resume 接口，勿瞎改权重文件名）
- 用 **`robotwin_ahawam_offset`**，把 offset / 刷新相关采样偏到你关心的部署节奏
- `num_epochs` 或 `max_steps` 先开小（验证 loss 下降 + 少量任务评测），再决定是否加长
- **先评测对照**：同一批任务、ah=64、cpp∈{1,2,3}，看乱象是否收敛

---

## 5. 怎么写「平台可提交」的训练入口脚本

平台只认某个入口 `.sh`（例如挂到 `algorithm/.../train_xxx.sh`）。推荐模式：

### 5.1 文件放哪

1. **权威副本**：`infra/train_<目的>.sh`（进 git，可 review）  
2. **平台入口**：`algorithm/cwr_wulan_algorithm/train_<目的>.sh`  
   - 要么与 infra **内容同步**  
   - 要么 thin wrapper：`exec bash "$AHA_WAM_CODE_DIR/infra/train_<目的>.sh"`

### 5.2 脚本必须具备的骨架

```bash
#!/usr/bin/env bash
set -euo pipefail
echo "train_xxx.sh revision: YYYY-MM-DD-desc"   # 日志里能看见版本，防提交了旧脚本

# ---- 写死配置（平台往往不能 set 环境变量）----
SMOKE=1                    # 1=冒烟 0=全量；先 1 通再改 0
NPROC_PER_NODE="${MA_NUM_GPUS:-8}"
OUTPUT_DIR="/opt/huawei/dataset/cwr_dataset_wulann/aha-wam-runs/train/xxx"

# ---- 路径自探测 ----
if [[ -d /opt/huawei/dataset/cwr_dataset_wulann/AHA-WAM-anygrasp ]]; then
  DATA=/opt/huawei/dataset
else
  DATA=/home/ma-user/work/dataset
fi
export AHA_WAM_CODE_DIR="$DATA/cwr_dataset_wulann/AHA-WAM-anygrasp"
# 若 OUTPUT_DIR 在探索机，把 /opt/huawei/dataset 替换成 $DATA

cd "$AHA_WAM_CODE_DIR"

# ---- pip：华为源 + 禁 NGC；优先 wheels ----
export PIP_INDEX_URL="http://repo.myhuaweicloud.com/repository/pypi/simple"
export PIP_TRUSTED_HOST="repo.myhuaweicloud.com"
export PIP_EXTRA_INDEX_URL=""
export PIP_CONFIG_FILE=/dev/null

# ---- 离线模型 ----
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export DIFFSYNTH_SKIP_DOWNLOAD=true
export DIFFSYNTH_MODEL_BASE_PATH="$AHA_WAM_CODE_DIR/checkpoints"
export WANDB_MODE=offline

# ---- 分布式（平台注入）----
export NNODES="${MA_NUM_HOSTS:-1}"
export NODE_RANK="${VC_TASK_INDEX:-0}"
MASTER_HOST="${VC_WORKER_HOSTS:-127.0.0.1}"
export MASTER_ADDR="${MASTER_HOST%%,*}"
export MASTER_PORT="${MASTER_PORT:-29500}"

# ---- 启动真正训练 ----
bash scripts/train_zero2.sh "$NPROC_PER_NODE" \
  task=robotwin_ahawam_offset \
  output_dir="$OUTPUT_DIR" \
  wandb.mode=offline
  # ... 其它 Hydra overrides 全部写死在这里
```

### 5.3 提交 checklist

- [ ] 入口脚本 `revision:` 字符串已改，日志能确认不是旧版  
- [ ] `SMOKE=1` 先跑通（少 step / 少卡）再全量  
- [ ] 所有 `path/to` 已替换；`output_dir` 在 dataset 盘  
- [ ] `df -h` 确认空间（ckpt + 日志很大）  
- [ ] 未在脚本里把 `stdout=PIPE` 丢给会刷屏的子进程却不读（见下节「PIPE 死锁」）  
- [ ] 训练产物路径记入 `summary` / 实验笔记，便于后续评测 `ckpt=` 指过去  

---

## 6. 和评测入口的关系（别混）

| 脚本 | 实际干什么 |
|---|---|
| `infra/train_mtp.sh` | ah/cpp **评测** sweep |
| `infra/train_skip_phase*.sh` | skip_phase **评测** |
| `scripts/train_zero*.sh` | **真正训练** |
| `experiments/robotwin/eval_robotwin_single.py` | 单任务评测 |
| `infra/run_robotwin_sweep.py` | 多卡调度评测 |

训完要验证：用新 ckpt 走评测入口，例如：

```bash
python experiments/robotwin/eval_robotwin_single.py \
  ckpt=/opt/huawei/dataset/cwr_dataset_wulann/aha-wam-runs/train/xxx/....pt \
  EVALUATION.dataset_stats_path=checkpoints/AHA-WAM-RoboTwin2.0/dataset_stats.json \
  EVALUATION.task_name=pick_dual_bottles \
  EVALUATION.eval_num_episodes=40 \
  EVALUATION.action_horizon=64 \
  EVALUATION.chunks_per_video_prefill=2
```

关于 **cpp**：`chunks_per_video_prefill` 主要是 **评测调度**；训练侧更接近的是 **offset / 异步相位分布**（`max_action_offset`、`robotwin_ahawam_offset`）。想「适应 cpp1/3」→ 改训练分布或短 finetune，而不是只在评测里扫 cpp。

---

## 7. 我们经常犯的错误（训练 + 提交相关）

1. **把 `train_mtp.sh` 当训练。** 跑了几天全是评测。看脚本是否 `accelerate launch scripts/train.py`。  
2. **平台改不了 env，却把关键路径写在「需要 export 才能用」的地方。** 全部硬编码 + 路径自探测。  
3. **提交了旧 revision。** 入口必须 `echo revision: ...`；改完没同步到 `algorithm/` 等于没改。  
4. **在线 `pip install -r requirements.txt` + NGC extra-index。** 每个包 DNS 超时，装几小时。强制华为源、清空 extra-index、用 wheels。  
5. **pip 覆盖镜像 CUDA torch。** 装完 `torch.cuda.is_available()` 变 False。  
6. **`path/to/...` 原样开训。** Hydra 配置占位符必须改掉。  
7. **输出写到容器本地盘 / 家目录。** 写满或任务结束丢失；一律 dataset。  
8. **`Popen(..., stdout=PIPE)` 又不读。** 评测 sweep 曾因此死锁；训练 launcher 若包一层同样注意，日志重定向到文件。  
9. **多卡同时改同一 symlink / 同一输出文件。** 启动前单进程准备好目录与 symlink。  
10. **评测参数看目录名不看日志。** 「baseline」目录实际可能是 ah32；永远读 `job_report` / eval 配置。  
11. **gitignore / 大文件误 push。** `aha-wam-runs/`、ckpt、wheels 不要进 git；token 不要写进 remote URL。  
12. **短 finetune 没设对照。** 至少保留 released ckpt 在相同任务、相同 ah/cpp 上的成功率，否则无法判断「适应 cpp」有没有用。  
13. **sapien/X11/ffmpeg 问题。** 纯训练可能碰不到；一接 RoboTwin 评测就爆——见 `AGENT_ONBOARDING.md`。  

---

## 8. 推荐工作流（给 agent 的默认 SOP）

1. 确认需求是 **训** 还是 **评**；评测走 onboarding，训练走本文。  
2. 在 `infra/` 写/改入口脚本 → 同步平台 `algorithm/` 入口 → `revision` 打新。  
3. `SMOKE=1` 提交（少卡、少 step）→ 看日志确认进了 `scripts/train.py`。  
4. 通了再 `SMOKE=0` 全量；产物落到 `aha-wam-runs/train/...`。  
5. 用 `eval_robotwin_single` / sweep 在目标 ah/cpp 上验收。  
6. 把 run 路径、Hydra overrides、git commit、结论写回实验笔记（或更新本文「附录」）。  

---

## 9. 相关文件速查

```text
AGENT_ONBOARDING.md          # 评测 / 环境 / 资产 / 旧坑
AGENT_TRAINING.md            # 本文：真正训练与平台提交
infra/train_mtp.sh           # 评测 sweep 入口（名字像训练）
infra/train_skip_phase*.sh   # skip_phase 评测入口
infra/run_robotwin_sweep.*   # 评测调度
scripts/train_zero1.sh       # 真·训练 ZeRO-1
scripts/train_zero2.sh       # 真·训练 ZeRO-2
scripts/train.py             # 训练主程序
configs/task/robotwin_ahawam_offset.yaml
README.md                    # 上游官方训练说明
```
