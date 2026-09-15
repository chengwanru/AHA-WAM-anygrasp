# 评测：SKIP_V2 finetune ckpt 上的 skip_phase（baseline 不重跑）

## 对比设计

| 条件 | 权重 | 是否这次重跑 |
|---|---|---|
| **baseline**（同步调度） | **未训** released `robotwin_ahawam.pt` | **否** — 用 MAIN-2 旧表 |
| skip_phase v2（zero-shot） | released | 否 — 旧表已有 |
| **skip_phase v2（ft 后）** | **`step_004000.pt`** | **是** — 本入口只跑这个 |

问的是：同一套 skip v2 规则下，**训过是否比 zero-shot skip 更接近 baseline**。

踩坑与 modeset/lavapipe/SIGSEGV 详见 [`docs_eval_cluster_pitfalls.md`](./docs_eval_cluster_pitfalls.md)。

## 提交

| 任务集 | 入口 |
|---|---|
| **batch1**（9 任务） | **`train_eval_skip_v2_ft.sh`** |
| batch2（10 任务） | `train_eval_skip_v2_ft_batch2.sh` |

- 算法包：`cwr_wulan_algorithm`（**必须同步最新**）
- 日志第一行 revision 须含：`lavapipe-v12-epidx`
- 1×8 卡 × 2 机（batch1+batch2）；平台超时 **≥48h**（`JOB_TIMEOUT_S=172800`）
- 挂载：wulann + wulann2 + wulann3 + **wulann4**
- **不要**选旧 `train_skip_phase.sh`
- 任务：MAIN-2 有 baseline 的 **19 × 40 ep**（不含 `put_bottles_dustbin`）
- ckpt：`.../skip_v2_ft_8x10h/checkpoints/weights/step_004000.pt`
- 默认 `ROBOTWIN_EVAL_VIDEO_LOG=0`（只看成功率）

## 渲染

1. 有 `/dev/nvidia-modeset` RDWR → **方案 A** NVIDIA ICD（与 MAIN-2 同）
2. 否则 → **方案 B** lavapipe：关 RT、关 shadow、`ROBOTWIN_MPLIB_STUB=1`；全量很慢
3. modeset **不能自下载**；只 export `NVIDIA_DRIVER_CAPABILITIES` 不够

可用 `train_probe_vulkan.sh` 先探 modeset。

## 输出

```text
.../cwr_dataset_wulann4/aha-wam-runs/robotwin/video_dit_skip_phase_40eps_v12_lvp_ft/
  sweep_master.log
  summary.csv
  skip_phase/<task>/
```

batch2 目录后缀 `_batch2`。

## 健康信号（提交后看这些）

- `revision: ...-lavapipe-v12-epidx`
- `modes=['skip_phase']`，ckpt 为 `step_004000.pt`
- lavapipe 时：`AHAWAM_VULKAN_MODE=lavapipe`，`SAPIEN_DISABLE_RAYTRACING=1`，`ROBOTWIN_MPLIB_STUB=1`
- 子日志：`[MplibPlanner] STUB` → `set_planner ok` → `load_actors`
- 出 `Success!` / `Fail!` 后继续有 `Success rate: x/y`（不会卡在首集 `UnboundLocalError: episode_idx`）
- **不是** ~270–450s 假 `0/40` + `return code -11`

**不要**分析 `..._v4` / `..._v9` / `..._v10` / `..._v11` 早期假零结果。
