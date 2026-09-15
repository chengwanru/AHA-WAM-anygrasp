# 提交 HE=8 vs HE=32 适配（训完立刻评测）

## 这是哪个实验？

从 **released 16-chunk WAM** 各自适配到 He=8 / He=32。公平对比是 **8 vs 32**（同 init、同 FT）。Released-16 只作参考条，本 job **不训 He=16**。

| 臂 | Video \(H_p\) | 一次 Action 前向 | 执行 | 再观察 / 新 Video |
|---|---|---|---|---|
| He=8 | 64 | generate **8** | 8 | 每 8 步（cpp=1） |
| He=32 | 64 | generate **32** | 32 | 每 32 步（cpp=1） |

- **不是**「预测 16 再截断/拼接」，也不是 16×2 凑 He=32。
- Skip **关**（`VIDEO_DIT_MODE=baseline`）。Video DiT **不冻**，`lr=5e-5` cosine，`wd=0.01`，`lambda_video=1.0`。
- 训练：标准 WAM BC，只改 `action_chunk_size`；**不是** `cpp1_ditkv_train`。
- 评测：ah=64 / **cpp=1 always**；Hydra `task=robotwin_ahawam_he8|he32` 才能把 chunk 设对。

## 提交（推荐：1 个 16 卡 job）

| 项 | 值 |
|---|---|
| 算法包 | 旧：`cwr_wulan_algorithm`；**新平台：`cwr_wulan2`**（见 `docs/PLATFORM_MIGRATION.md`） |
| 入口 | **`train_finetune_he_adapt.sh`** |
| 资源 | **2×8 = 16 卡**（`MA_NUM_HOSTS=2`） |
| 分配 | `VC_TASK_INDEX=0` → **He=8**；`=1` → **He=32**（各 8 卡独立 DDP，**不是** 16 卡一张网） |
| 超时 | 平台 **≥24h**（脚本 `JOB_TIMEOUT_S=72000`≈20h；He=8 评测 Video 次数约 4× He=32） |
| 挂载 | wulann + wulann2 + wulann3 + wulann4 |
| 渲染 | 创建时打开 **graphics**；本批有 `/dev/nvidia-modeset` → NVIDIA Vulkan |

日志首行：`revision: 2026-09-15-he-adapt-8vs32-train-eval`  
开训应见：`HE_ADAPT_ARM=8` 或 `32`、`TASK_NAME=robotwin_ahawam_he8|he32`、`TOTAL` 不是 16 卡 DDP。  
`auto` 臂要求 **`MA_NUM_HOSTS=2` + `VC_TASK_INDEX∈{0,1}`**，否则直接失败（防两机同写一目录）。

### 备选：拆成两个 8 卡 job

| 入口 | 臂 |
|---|---|
| `train_finetune_he8.sh` | He=8 训+评 |
| `train_finetune_he32.sh` | He=32 训+评 |

只重评（权重已在）：`train_eval_he_adapt.sh`（16 卡 2×8）或 `train_eval_he8.sh` / `train_eval_he32.sh`。

## 写死配方

- init：`checkpoints/AHA-WAM-RoboTwin2.0/robotwin_ahawam.pt`
- `max_steps=4000`，`save_every=500`
- `batch_size=2`，`accum=8`，8 GPU → **global 128**
- `mot_checkpoint_mixed_attn=true`，`num_history_frames=6`，`num_frames=65`

## 评测（训完同 job 接着跑）

8 任务 × **30** ep，混低 / 中高 MAIN-2 NVIDIA SR（避开 `lift_pot` / `open_*` 地板任务）：

| 偏低 | MAIN-2 SR | 中高 | MAIN-2 SR |
|---|---|---|---|
| `handover_mic` | 27.5 | `stack_blocks_two` | 57.5 |
| `hanging_mug` | 37.5 | `pick_dual_bottles` | 80 |
| `place_a2b_left` | 22.5 | `place_bread_basket` | 72.5 |
| `place_object_basket` | 20 | `turn_switch` | 52.5 |

指标：SR **以及** latency（L_chunk、action_chunk_calls/ep、ms/executed step、L_prefill、prefill_calls/ep、wall_s/ep）。`EVALUATION.timing_enabled=True`。

## 产物目录（不要和 skip_v2 / cpp1 混写）

```text
$RUNS/train/he8_adapt_8x10h/          # He=8 权重
$RUNS/train/he32_adapt_8x10h/         # He=32 权重
$RUNS/robotwin/he8_adapt_always_30eps/
$RUNS/robotwin/he32_adapt_always_30eps/
```

进度：`$RUNS/train/he{8,32}_adapt_8x10h/progress/status.txt`

## 健康信号

- 训练：`task=robotwin_ahawam_he8|he32`，`action_chunk_size` 对应 8/32
- 评测：`FIXED ah=64 cpp=1`，`EVAL_HYDRA_TASK=robotwin_ahawam_he8|he32`，`modes=['baseline']`
- Vulkan：`nvidia-modeset: RDWR ok` → `Vulkan: NVIDIA ICD`
- outdir 下是 `baseline/<task>/`，**不是** `skip_phase/`
- ckpt：`.../he{8,32}_adapt_8x10h/checkpoints/weights/step_004000.pt`（或该臂最新 `step_*.pt`）

## 诚实缺口

离线训练是 clip-level Video；评测每个 He 窗口都会新 RGB + 新 Video DiT。适配的是 **Action 长度 / 重规划频率**，不是把训时 Video 也变成每 He 步一次。
