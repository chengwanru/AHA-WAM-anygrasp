# 提交：FSM skip 成功轨采集 + Success-BC 短训

> 诊断实验：粗糙 FSM skip 下 rollout，只用成功轨迹做 Action BC。  
> **不是** skip_v2 代理 FT，**不是** online RL。  
> 训练从 **AHA-WAM released** `robotwin_ahawam.pt` 起步。

## 流程

```text
1) train_collect_skip_success.sh      # 先采成功轨 → LeRobot
2) train_finetune_skip_success_bc.sh  # 4000 step Action BC + loss 曲线
3) 另交 skip_phase 评测（换新 ckpt）
```

## 1) 采集

| 项 | 值 |
|---|---|
| 入口 | **`train_collect_skip_success.sh`** |
| 资源 | 1×8 卡；超时 ≥24h；挂 wulann～wulann4 |
| ckpt | released `robotwin_ahawam.pt` |
| 默认 | smoke：`press_stapler` + `turn_switch` |

产物：`.../skip_success_collect_smoke/lerobot_success/`

## 2) 训练（写死，平台不用 set）

| 项 | 值 |
|---|---|
| 入口 | **`train_finetune_skip_success_bc.sh`** |
| revision | `2026-09-10-skip-success-bc-v2-4k` |
| 资源 | 1×8 卡 |
| 平台超时 | **≥11h**（`JOB_TIMEOUT_S=39600`） |
| init | **released** `.../AHA-WAM-RoboTwin2.0/robotwin_ahawam.pt` |
| 数据 | `.../skip_success_collect_smoke/lerobot_success`（脚本内写死） |
| max_steps | **4000** |
| save_every | **500** |
| batch / accum | **4 / 4** + `mot_checkpoint_mixed_attn` |
| Video | `freeze_video_dit=true`，`lambda_video=0` |
| skip 代理 | `skip_phase_v2_train=false` |

日志第一行必须是：`revision: 2026-09-10-skip-success-bc-v2-4k`  
并打印 `INIT_CKPT=.../robotwin_ahawam.pt`。

### 可视化（训练中自动）

```text
.../skip_success_bc_8x4k/progress/
  status.txt                 # tail -f
  progress_live.svg|png      # 实时 loss（IDE/浏览器刷新）
  snapshots/step_000500.svg  # 每 500 step 快照
  snapshots/step_001000.svg
  ...
  metrics.jsonl
```

## 防旧坑（已写进脚本）

对照 `AGENT_TRAINING.md` §7 / `SUBMIT_FINETUNE_SKIP_V2.md`：

1. 关键参数全部脚本写死（平台 set 不了）  
2. Python≥3.10 NFS `ahawam` env，不靠镜像 3.9  
3. imports 齐则 skip 重 pip（防 NFS SIGBUS）  
4. 华为 PyPI + 清空 `PIP_EXTRA_INDEX_URL`（禁 NGC）  
5. 输出在 wulann4 dataset 盘，不写容器本地  
6. `batch=4`+`accum=4`+`mot_checkpoint_mixed_attn` 防 OOM  
7. 改完同步 **algorithm/**；以日志 `revision:` 为准  
8. init 必须是 released，不是 `step_004000`  
9. 先 collect 出 `lerobot_success` 再交本训练  

## 产物

```text
.../aha-wam-runs/train/skip_success_bc_8x4k/checkpoints/weights/step_*.pt
```

## 推荐：一键流水线（采集+训练）

| 项 | 值 |
|---|---|
| 入口 | **`train_skip_success_pipeline.sh`** |
| 行为 | 先 `train_collect_skip_success.sh`，再 `train_finetune_skip_success_bc.sh` |
| 平台超时 | **≥24h**（lavapipe 采集 + 约 10h 训练） |
| 已有数据 | 若 `lerobot_success/meta/info.json` 已在，会跳过采集直接训 |

算法包须同时带上：`train_skip_success_pipeline.sh`、`train_collect_skip_success.sh`、`train_finetune_skip_success_bc.sh`。
