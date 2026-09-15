# 评测：SKIP_SUCCESS_BC ckpt 上的 skip_phase（baseline 不重跑）

## 训练已完成

- 输出：`.../train/skip_success_bc_8x4k/`
- ckpt：`checkpoints/weights/step_004000.pt`（19:51 写出；`train finished rc=0`）
- 协议：released init → FSM-skip success 轨 Action BC（freeze Video DiT）

## 对比设计

| 条件 | 权重 | 是否这次重跑 |
|---|---|---|
| baseline（同步） | released | **否** — MAIN-2 |
| skip_phase（zero-shot） | released | **否** — MAIN-2 / 已有 lavapipe 表慎用 |
| skip_phase（skip_v2_ft） | `skip_v2_ft_8x10h/step_004000` | **否** — 已有 `..._v12_lvp_ft` |
| **skip_phase（success-BC）** | **`skip_success_bc_8x4k/step_004000`** | **是** — 本入口 |

问的是：同一套 FSM skip 下，**success-BC 是否比 released / skip_v2_ft 更接近 baseline**。

## 提交

| 任务集 | 入口 |
|---|---|
| **batch1**（9 任务） | **`train_eval_skip_success_bc.sh`** |
| **batch2**（10 任务） | **`train_eval_skip_success_bc_batch2.sh`** |

- 算法包：`cwr_wulan_algorithm`（同步含上述两个入口）
- 日志第一行 revision 须含：`success-bc-lavapipe-v12-epidx`
- 1×8 卡 × 2 机；平台超时 **≥48h**（`JOB_TIMEOUT_S=172800`）
- 挂载：wulann + wulann2 + wulann3 + **wulann4**
- ckpt / stats 已写死在脚本内，平台不用 set

## 输出

```text
.../robotwin/video_dit_skip_phase_40eps_v12_lvp_success_bc/
.../robotwin/video_dit_skip_phase_40eps_v12_lvp_success_bc_batch2/
```

## 健康信号

- `revision: ...-success-bc-lavapipe-v12-epidx`
- ckpt 路径含 `skip_success_bc_8x4k/.../step_004000.pt`
- `modes=['skip_phase']`，ah=64 cpp=2
- lavapipe 时：`AHAWAM_VULKAN_MODE=lavapipe`
- 不要拿 smoke 目录当正式结果
