# 评测：同预算 random_skip（监督信号来源实验）

## 要回答的问题

手写 skip v2（FSM）相对「同连续跳预算、但不看阶段」的随机跳，有没有真信息？

| 结果 | 以后 skip policy 监督 |
|---|---|
| FSM ≫ random | FSM / 阶段几何可当弱老师 |
| ≈ | 别用 FSM 当标签；改反事实价值 |
| FSM ≪ | 当前规则是负监督 |

对照用 **MAIN-2** 已有 baseline / skip_phase（40ep 表）；本次 **只跑 random_skip**。  
本次 **20 ep**；成功率与 MAIN-2 比时看相对排序/掉点。

踩坑详见 [`docs_eval_cluster_pitfalls.md`](./docs_eval_cluster_pitfalls.md)。

## 规模（16 卡）

| 项 | 值 |
|---|---|
| 任务 | MAIN-2 的 **19** 个（batch1=9 + batch2=10） |
| ep | **20** |
| 模式 | `random_skip` only |
| ckpt | **released** `robotwin_ahawam.pt` |
| ah / cpp | 64 / 2 |
| 预算 | `max_consec=1`，`RANDOM_SKIP_P=1.0` |
| 并行 | **2×8 卡** |

## 提交

| 机器 | 入口 |
|---|---|
| 8 卡 A | **`train_eval_random_skip.sh`** |
| 8 卡 B | **`train_eval_random_skip_batch2.sh`** |

- 算法包同步最新；revision 含 `lavapipe-v12-epidx`
- 挂载：wulann + wulann2 + wulann3 + **wulann4**
- 超时：**≥24h**（lavapipe；`JOB_TIMEOUT_S=86400`）
- **不要**与 ft 的 `train_eval_skip_v2_ft.sh` 搞混
- 默认关视频：`ROBOTWIN_EVAL_VIDEO_LOG=0`

## 日志应看到

- `revision: ...-random-skip-lavapipe-v12-epidx`
- `AHAWAM_VULKAN_MODE=lavapipe` 或 `nvidia`
- lavapipe：`SAPIEN_DISABLE_RAYTRACING=1`，`lights shadow=False: ok`
- 子日志：`forcing shadow=False`，开始真实 ep（非 -11）
- `modes=random_skip`，`EVAL CKPT=.../robotwin_ahawam.pt`

## 输出

```text
.../robotwin/video_dit_random_skip_20eps_v9_lvp/
```

不要用 `..._20eps_v4` 里已失败的假 `0/20` 当结果。
