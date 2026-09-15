# 实验分区登记（路径隔离）

> 所有产物默认在 `cwr_dataset_wulann4/aha-wam-runs/`。  
> **不要**把不同实验的权重/评测写进同一叶子目录互相覆盖。  
> 更新：2026-09-15。两平台路径见 `docs/PLATFORM_MIGRATION.md`。

根路径速记：

```text
$RUNS = .../cwr_dataset_wulann4/aha-wam-runs
$CKPT = .../cwr_dataset_wulann/AHA-WAM-anygrasp/checkpoints/AHA-WAM-RoboTwin2.0
```

---

## 0. 命名约定

| 前缀 | 含义 |
|---|---|
| `$RUNS/train/<exp>/` | **训练权重**与 train 日志 |
| `$RUNS/robotwin/<exp>/` | **评测** summary / analysis |
| `$RUNS/train/_aborted_or_partial/` | 中止/误交的残片，**不当正式结果** |
| `$CKPT/robotwin_ahawam.pt` | **released baseline 权重**（只读，勿覆盖） |

---

## 1. Baseline / 正式评测表（只读参考）

| 实验 | 权重 | 结果目录 / 文档 | 说明 |
|---|---|---|---|
| **released baseline** | `$CKPT/robotwin_ahawam.pt` | MAIN-1/2 见仓库 `EXPERIMENT_RESULTS_SUMMARY.md` | ah/cpp 扫、skip vs baseline |
| MAIN-2 skip zero-shot | 同上 released | `EXPERIMENT_RESULTS_SUMMARY.md` §D；旧 NVIDIA 跑次 | ah64/cpp2 + FSM skip |
| 四组快照 | — | `docs/FOUR_WAY_EVAL_SNAPSHOT.md` | baseline / zs-skip / FT / random 汇总 |

---

## 2. 训练实验（权重隔离）

| 实验 ID | 入口 | **权重 / 输出目录** | 状态 |
|---|---|---|---|
| **baseline（released）** | — | `$CKPT/robotwin_ahawam.pt` | 冻结只读 |
| offset FT（旁路，已停） | `train_finetune_offset.sh` | `$RUNS/train/offset_ft_8x10h/` | 非主线 |
| skip_v2 proxy FT（诊断，已停） | `train_finetune_skip_v2.sh` | `$RUNS/train/skip_v2_ft_8x10h/` | 非主线 |
| success-BC smoke/full（诊断，已停） | `train_finetune_skip_success_bc.sh` | `$RUNS/train/skip_success_bc_*` | 非主线 |
| cpp1_ditkv 适配 | `train_finetune_cpp1_ditkv.sh` | `$RUNS/train/cpp1_ditkv_adapt_16x30h/` | 已训完；A/B 评测见下 |
| **HE=8 vs HE=32 适配（当前主线）** | **`train_finetune_he_adapt.sh`（2×8）** | **`$RUNS/train/he{8,32}_adapt_8x10h/`** | **待交 · 训完同 job 评测** |

### 当前主线权重落点（HE adapt，训完后）

```text
$RUNS/train/he8_adapt_8x10h/     # VC_TASK_INDEX=0，8 卡 DDP
$RUNS/train/he32_adapt_8x10h/    # VC_TASK_INDEX=1，8 卡 DDP
  checkpoints/weights/step_*.pt
  progress/status.txt
$RUNS/robotwin/he8_adapt_always_30eps/
$RUNS/robotwin/he32_adapt_always_30eps/
```

cpp1_ditkv 16 卡权重仍在 `$RUNS/train/cpp1_ditkv_adapt_16x30h/`（已完成，非本实验）。

### 误交 8 卡残片（已隔离，勿当结果）

```text
$RUNS/train/_aborted_or_partial/cpp1_ditkv_8gpu_partial_20260911/
  train_finetune_cpp1_ditkv_full_20260911_160634.log
  progress_8gpu_partial.log
```

同一父目录里可能仍留有 `..._160634.log` 副本；**以 `_aborted_or_partial` 为准，权重以 16 卡 run 的 `checkpoints/` 为准。**

---

## 3. 评测实验（结果隔离）

| 实验 | 结果目录 | 权重来源 |
|---|---|---|
| skip_v2 FT eval (lvp) | `$RUNS/robotwin/video_dit_skip_phase_40eps_v12_lvp_ft[_batch2]/` | `skip_v2_ft_8x10h/.../step_004000.pt` |
| random skip (lvp) | `$RUNS/robotwin/video_dit_random_skip_20eps_v12_lvp[_batch2]/` | released |
| success-BC eval (停) | `$RUNS/robotwin/video_dit_skip_phase_40eps_v12_lvp_success_bc*` | success-BC ckpt |
| **cpp1 A/B always（验收）** | `train_eval_cpp1_ab.sh`（2×8） | `$RUNS/robotwin/cpp1_{released,adapt}_always_40eps/` | ah64/cpp1/always；A=released B=step_010000 |
| cpp1_ditkv 训练（已完成） | — | `$RUNS/train/cpp1_ditkv_adapt_16x30h/checkpoints/weights/step_010000.pt` | 16 卡训完 rc=0 |
| **HE adapt 8 vs 32（当前）** | `train_finetune_he_adapt.sh` 链式 eval | `$RUNS/robotwin/he{8,32}_adapt_always_30eps/` | 8 任务 × 30 ep；ah64/cpp1/always；chunk=8 vs 32 |

评测时 **禁止** 把 cpp1 新权重写回 MAIN-2 旧目录或 `skip_v2_ft_*`。

---

## 4. 文档索引

| 文档 | 内容 |
|---|---|
| `docs/计划.md` | 主线方案（§4.2a/b） |
| `docs/SKIP_ADAPT_PAUSE_NOTES.md` | 旧适配线暂停 |
| `docs/FOUR_WAY_EVAL_SNAPSHOT.md` | 四组旧评测 |
| `infra/SUBMIT_FINETUNE_CPP1_DITKV.md` | cpp1_ditkv 训提交 |
| **`infra/SUBMIT_FINETUNE_HE_ADAPT.md`** | **He=8 vs 32 训+评提交** |
| **`docs/PLATFORM_MIGRATION.md`** | **旧平台 → `cwr_wulan_aha` / `cwr_wulan2` 迁移** |
| **本文** | 路径分区总表 |

---

## 5. 检查清单（新开实验前）

1. 新实验是否有 **独立叶子目录名**（含实验 id + 设定，如 `16gpu` / `lvp`）？  
2. 权重是否 **不会** 写进 `$CKPT/` 或别人的 `train/<other>/`？  
3. 评测 summary 是否 **不会** 覆写旧 `video_dit_*` 正式表？  
4. 中止 run 是否挪到 `_aborted_or_partial/`？  
