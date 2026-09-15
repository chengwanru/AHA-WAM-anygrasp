# 提交：CPP1_DITKV 适配训（§4.2a 路径 3）

## 这是哪个实验？

| | 本入口 | 勿选 |
|---|---|---|
| 入口 | **`train_finetune_cpp1_ditkv.sh`** | `train_finetune_skip_v2.sh` / offset / success_bc |
| 任务 | `robotwin_ahawam_cpp1_ditkv` | |
| 输出 | `.../train/cpp1_ditkv_adapt_16x30h/` | |
| 本 job | **只训练** | 不含 robotwin 评测 |
| 训完评测 | 另交：ah64/**cpp1** always（对照 released@cpp1） | |

机制：Mot **forward 真跑 Video DiT + chunk KV** + cpp=1 always 的 per-chunk obs；**Video 权重冻**，Action 主更新；`lambda_video=0`。

## 提交

| 项 | 值 |
|---|---|
| 算法包 | `cwr_wulan_algorithm`（入口脚本已同步；**代码以挂载的 AHA-WAM-anygrasp 为准**，需含 cpp1 改动） |
| 入口 | `train_finetune_cpp1_ditkv.sh` |
| 资源 | **2×8 = 16 卡**（脚本默认）；单机 16 卡也可（脚本读 `MA_NUM_HOSTS`/`MA_NUM_GPUS`） |
| 超时 | **≥30h**（脚本 `JOB_TIMEOUT_S=108000`） |
| 挂载 | wulann + wulann2 + wulann3 + wulann4 |

写死：`max_steps=10000`，`save_every=1000`，`batch_size=2`，`accum=4`，`freeze_video_dit=true`。

日志首行应有：`revision: 2026-09-11-cpp1-ditkv-16x30h-v3-hostfile`。

**多机必看**：开训后日志必须出现 `DEEPSPEED_HOSTFILE=.../deepspeed_hostfile.txt`、hostfile 两行 `slots=8`、以及 `[launch] ... num_processes=16 ... deepspeed_hostfile=...`。若出现 `Unable to find hostfile, will proceed with training with local resources only` 或 `master_addr=127.0.0.1`，说明又退化成单机 8 卡——立刻停掉重交。

## 训多久？

| | |
|---|---|
| **上限（平台）** | **30h**（硬顶，防拖垮） |
| **建议目标** | 跑满 **`max_steps=10000`**，墙钟大约 **22–28h**（16 卡、~8–11 s/step） |
| **不是** | 「必须空转到满 30h」——step 跑完就停；30h 只是超时余量 |

开训后看 `progress/status.txt` 的 `speed` / `eta`：若 ETA≪20h 可下次加长 steps；若 ETA>30h 下次降到 8000。

全局 batch ≈ `2 × 16 × 4 = 128`。

## 训练过程可视化（自动）

rank0 会起 `watch_train_progress.py`，目录：

```text
.../cpp1_ditkv_adapt_16x30h/progress/
  status.txt                 # tail -f 这一行
  progress_live.svg/png      # loss 曲线（约每 15s 刷新）
  dashboard_live.svg/png     # 多面板：loss / action / lr / grad / speed
  snapshots/step_XXXXXX.png  # 每 1000 step 存一张
  snapshots/dashboard_XXXXXX.png
  metrics.jsonl
  README.txt
```

```bash
tail -f .../progress/status.txt
# IDE/浏览器反复打开 dashboard_live.png 或 progress_live.svg
```

1. **Python≥3.10** NFS `ahawam` env，禁止镜像 3.9  
2. imports 齐则 **skip 重 pip**（防 NFS SIGBUS）  
3. **text cache 须 COMPLETE**；多机 **拒绝** job 内 precompute（防竞态）  
4. **video symlink** 坏则 preflight 失败（挂齐 wulann2+3+4）  
5. OOM：`bs=2` + `accum=4` + `mot_checkpoint_mixed_attn=true`  
6. **NCCL_CUMEM_ENABLE=0** 等（防 misaligned address）  
7. **SMOKE=1 直接拒绝**；正式输出不得含 `*smoke*`  
8. 任务名锁死 `robotwin_ahawam_cpp1_ditkv`（防误跑 skip_v2/offset）  
9. 多机：`VC_TASK_INDEX` / `MA_NUM_HOSTS`；写 `OUTPUT_DIR/deepspeed_hostfile.txt` 并传 `--deepspeed_hostfile`（否则 DS 静默变单机 8 卡）  
10. watcher **仅 rank0**  
11. 本 job **只训不评**（评测另交）

## 进度

见上文「训练过程可视化」。

## 验收（训完另交评测）

1. released @ ah64/cpp1 always vs 本 ckpt @ ah64/cpp1 always → 期望 SR 回升  
2. 抽查 cpp2：别明显崩  

ckpt 预期：`.../cpp1_ditkv_adapt_16x30h/checkpoints/weights/step_010000.pt`（以实际 save 为准）。

## 路径隔离

见仓库 `docs/EXPERIMENT_REGISTRY.md`。正式权重只认本目录 `checkpoints/`；8 卡残片在 `train/_aborted_or_partial/`。
