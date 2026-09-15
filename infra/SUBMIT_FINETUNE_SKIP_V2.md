# 提交 SKIP_V2 短 finetune（与 OFFSET 实验完全分开）

## 这是哪个实验？

| | SKIP_V2（本入口） | OFFSET（旁路，勿选） |
|---|---|---|
| 入口 | **`train_finetune_skip_v2.sh`** | `train_finetune_offset.sh` |
| 任务 | `robotwin_ahawam_skip_v2` | `robotwin_ahawam_offset` |
| 输出 | `.../train/skip_v2_ft_8x10h/` | `.../train/offset_ft_8x10h/` |
| 本 job | **只训练** | （已停） |
| 评测 | 另交 `train_skip_phase.sh` | — |

## 提交

| 项 | 值 |
|---|---|
| 算法包 | `cwr_wulan_algorithm` |
| 入口 | `train_finetune_skip_v2.sh` |
| 资源 | 1×8 卡 |
| 超时 | **10–11h**（平台填 ≥11h；脚本 `JOB_TIMEOUT_S=39600`） |
| 挂载 | wulann + wulann2 + wulann3 + wulann4 |

写死：`max_steps=4000`（**偏保守**的约 10h：慢步速也不易超时），`save_every=500`，`batch_size=4`，`accum=4`。

日志开头应有：`revision: 2026-09-08-skip-v2-ft-10h-fastcache`（有 `.COMPLETE` 时 **秒级**跳过 text cache，不再全量 find）。

## 防旧坑（脚本已写死）

1. Python≥3.10 NFS env，不靠镜像 3.9  
2. imports 齐则 skip 重 pip（防 NFS SIGBUS）  
3. text cache 须 COMPLETE（已有 ~921032 则直接跳过）  
4. video symlink 坏则 preflight 失败  
5. `batch_size=4` + `accum=4` + `mot_checkpoint_mixed_attn=true`（防 OOM）  
6. 任务是 skip_v2，不是 offset  

## 实时进度（dataset 盘）

任务跑起来后看：

```text
/opt/huawei/dataset/cwr_dataset_wulann4/aha-wam-runs/train/skip_v2_ft_8x10h/progress/
  status.txt           # 一行最新 step/loss/eta（最方便 tail）
  progress_live.svg    # 实时 loss 曲线（IDE/浏览器反复打开刷新）
  metrics.jsonl        # 历史点
  watcher.log
```

探索机：`/home/ma-user/work/dataset/cwr_dataset_wulann4/aha-wam-runs/train/skip_v2_ft_8x10h/progress/`

```bash
tail -f .../progress/status.txt
# 或打开 progress_live.svg
```

## 时长预估

- text cache：已齐 → 跳过  
- 启动：约 15–40 min  
- **`max_steps=4000`（保守）**：按 7–9 s/step ≈ **8–10 h** 纯训练；比 8000 更不容易拖过 10h  
- 平台超时 **≥11h**；开训后看 `status.txt` 的 speed/eta  
- 若明显偏快（例如 <4 s/step、ETA≪10h），下次再加长 steps 即可  
- ckpt：每 **500** step  

## 整体流程

1. **本 job**：released ckpt → SKIP_V2 短 finetune → 权重落到 `skip_v2_ft_8x10h/checkpoints/`  
2. **下一 job（评测，另交）**：`train_skip_phase.sh`，换成新 ckpt；baseline 复用已有 40ep；只补 skip_phase 臂  
3. **对比**：zero-shot skip v2（旧表）vs 训后 skip v2，看掉点是否缩小  

## 训的内容（诚实）

离线用 gripper+进度代理 + 与评测相同的 cpp=2 / max_consec=1；评测仍用完整 FSM。
