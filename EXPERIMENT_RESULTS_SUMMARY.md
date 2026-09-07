# AHA-WAM RoboTwin 实验总结（全量）

> 汇总本项目在 RoboTwin 2.0 + 官方 released ckpt（eager PyTorch，A800）上的评测实验。
> **未包含论文 TensorRT / CUDA Graph / Flash 加速栈**（开源评测路径未提供）。
> 原始产物在 dataset：`aha-wam-runs/`（未进 git）。
> 日期：2026-09-07

---

## 0. 一句话结论

1. **ah/cpp sweep（20 任务 × 6 配置 × 40 ep）已全部跑完**：在 ah=64 上 cpp1/2/3 平均成功率约 **35.6% / 40.5% / 38.3%**，任务间谁好谁坏交错，**没有稳定单调趋势**；默认 **cpp=2** 均值最高。
2. **skip_phase v2（固定 ah64/cpp2）**：相对 baseline，**单次 L_chunk / L_prefill 基本不变**；真正收益是 **prefill 次数与每集 prefill 总时间下降**（约 30% 量级）。成功率有升有降，整体略损。
3. **延迟口径**：L_chunk ≈ 单次 action-chunk 推理（常态 ~450–500 ms ≈ 2 Hz，接近论文 eager ~416 ms，远低于论文优化后 41 ms）；L_prefill ≈ 单次 video prefill（常态 ~100 ms）。
4. **主线未做重训**；若怀疑 cpp1/3 不适配，见 `AGENT_TRAINING.md` 短 finetune 方案。

---

## 1. 实验版图

| 实验 | 配置 | 输出目录 | 状态 |
|---|---|---|---|
| ah/cpp sweep | 20 tasks × {ah64_cpp2, ah64_cpp1, ah64_cpp3, ah32_cpp2, ah32_cpp1, ah16_cpp1} × 40 ep | `aha-wam-runs/robotwin_ahawam_sweep_20tasks_40eps/` | **20/20 齐全** |
| skip_phase v2 | ah64/cpp2；远场可 skip、近场≤10cm 不 skip、连续 skip≤1 | `.../video_dit_skip_phase_40eps_v2/` + `_batch2/` | **完整 pair 见下表** |
| skip_phase v1 | 近场 6cm、无连续预算 → skip 过高 | `.../video_dit_skip_phase_40eps/` | **已废弃**（成功率崩） |
| 早期 batch/OVCR/complex 诊断 | 多种 | `aha-wam-runs/robotwin/batch_*`, `ovcr_*`, `complex_*` | 探索性，非正式主表 |

入口脚本（名字像 train，实际是评测）：`infra/train_mtp.sh`、`infra/train_skip_phase*.sh`。

---

## 2. Experiment 1 — ah/cpp sweep

Ckpt：`checkpoints/AHA-WAM-RoboTwin2.0/robotwin_ahawam.pt`；`demo_randomized`；40 episodes。

### 2.1 Success rate

| Task | ah=64, cpp=2 | ah=64, cpp=1 | ah=64, cpp=3 | ah=32, cpp=2 | ah=32, cpp=1 | ah=16, cpp=1 |
|---|---:|---:|---:|---:|---:|---:|
| Pick Dual Bottles | 32/40 (80.0%) | 33/40 (82.5%) | 29/40 (72.5%) | 32/40 (80.0%) | 33/40 (82.5%) | 33/40 (82.5%) |
| Pick Diverse Bottles | 20/40 (50.0%) | 23/40 (57.5%) | 18/40 (45.0%) | 20/40 (50.0%) | 23/40 (57.5%) | 23/40 (57.5%) |
| Place Bread Basket | 29/40 (72.5%) | 25/40 (62.5%) | 26/40 (65.0%) | 29/40 (72.5%) | 25/40 (62.5%) | 25/40 (62.5%) |
| Place Object Basket | 8/40 (20.0%) | 2/40 (5.0%) | 6/40 (15.0%) | 8/40 (20.0%) | 2/40 (5.0%) | 2/40 (5.0%) |
| Place Object A to B Left | 9/40 (22.5%) | 4/40 (10.0%) | 7/40 (17.5%) | 9/40 (22.5%) | 4/40 (10.0%) | 4/40 (10.0%) |
| Place Object A to B Right | 3/40 (7.5%) | 3/40 (7.5%) | 4/40 (10.0%) | 3/40 (7.5%) | 3/40 (7.5%) | 3/40 (7.5%) |
| Move Can Pot | 17/40 (42.5%) | 16/40 (40.0%) | 15/40 (37.5%) | 17/40 (42.5%) | 16/40 (40.0%) | 16/40 (40.0%) |
| Move Stapler Pad | 1/40 (2.5%) | 0/40 (0.0%) | 2/40 (5.0%) | 1/40 (2.5%) | 0/40 (0.0%) | 0/40 (0.0%) |
| Stack Blocks Two | 23/40 (57.5%) | 37/40 (92.5%) | 30/40 (75.0%) | 23/40 (57.5%) | 37/40 (92.5%) | 37/40 (92.5%) |
| Stack Bowls Three | 32/40 (80.0%) | 29/40 (72.5%) | 31/40 (77.5%) | 32/40 (80.0%) | 29/40 (72.5%) | 29/40 (72.5%) |
| Handover Block | 5/40 (12.5%) | 2/40 (5.0%) | 9/40 (22.5%) | 5/40 (12.5%) | 2/40 (5.0%) | 2/40 (5.0%) |
| Handover Microphone | 11/40 (27.5%) | 7/40 (17.5%) | 8/40 (20.0%) | 11/40 (27.5%) | 7/40 (17.5%) | 7/40 (17.5%) |
| Hanging Mug | 15/40 (37.5%) | 11/40 (27.5%) | 10/40 (25.0%) | 15/40 (37.5%) | 11/40 (27.5%) | 11/40 (27.5%) |
| Click Bell | 26/40 (65.0%) | 14/40 (35.0%) | 26/40 (65.0%) | 26/40 (65.0%) | 14/40 (35.0%) | 14/40 (35.0%) |
| Press Stapler | 40/40 (100.0%) | 37/40 (92.5%) | 40/40 (100.0%) | 40/40 (100.0%) | 37/40 (92.5%) | 37/40 (92.5%) |
| Turn Switch | 21/40 (52.5%) | 18/40 (45.0%) | 20/40 (50.0%) | 21/40 (52.5%) | 18/40 (45.0%) | 18/40 (45.0%) |
| Lift Pot | 1/40 (2.5%) | 0/40 (0.0%) | 0/40 (0.0%) | 1/40 (2.5%) | 0/40 (0.0%) | 0/40 (0.0%) |
| Beat Block Hammer | 31/40 (77.5%) | 24/40 (60.0%) | 23/40 (57.5%) | 31/40 (77.5%) | 24/40 (60.0%) | 24/40 (60.0%) |
| Open Laptop | 0/40 (0.0%) | 0/40 (0.0%) | 0/40 (0.0%) | 0/40 (0.0%) | 0/40 (0.0%) | 0/40 (0.0%) |
| Open Microwave | 0/40 (0.0%) | 0/40 (0.0%) | 2/40 (5.0%) | 0/40 (0.0%) | 0/40 (0.0%) | 0/40 (0.0%) |
| **Mean (20)** | **40.5%** | **35.6%** | **38.2%** | **40.5%** | **35.6%** | **35.6%** |

观察：同一任务上 cpp1/2/3 常互相领先；**ah=32/16 与 ah=64 在相同 cpp 上成功率高度相似**（部署调度主导差异大于 horizon 本身时会出现）。

### 2.2 L_chunk（单次 action-chunk，ms / Hz）

| Task | ah=64, cpp=2 | ah=64, cpp=1 | ah=64, cpp=3 | ah=32, cpp=2 | ah=32, cpp=1 | ah=16, cpp=1 |
|---|---:|---:|---:|---:|---:|---:|
| Pick Dual Bottles | 472.0 ms (2.12 Hz) | 446.0 ms (2.24 Hz) | 471.3 ms (2.12 Hz) | 462.5 ms (2.16 Hz) | 462.6 ms (2.16 Hz) | 451.3 ms (2.22 Hz) |
| Pick Diverse Bottles | 459.4 ms (2.18 Hz) | 469.4 ms (2.13 Hz) | 453.2 ms (2.21 Hz) | 459.3 ms (2.18 Hz) | 812.0 ms (1.23 Hz) | 822.2 ms (1.22 Hz) |
| Place Bread Basket | 446.3 ms (2.24 Hz) | 449.6 ms (2.22 Hz) | 481.3 ms (2.08 Hz) | 455.2 ms (2.20 Hz) | 819.8 ms (1.22 Hz) | 814.2 ms (1.23 Hz) |
| Place Object Basket | 443.9 ms (2.25 Hz) | 460.8 ms (2.17 Hz) | 452.2 ms (2.21 Hz) | 464.3 ms (2.15 Hz) | 461.0 ms (2.17 Hz) | 459.8 ms (2.18 Hz) |
| Place Object A to B Left | 746.5 ms (1.34 Hz) | 744.8 ms (1.34 Hz) | 449.4 ms (2.23 Hz) | 442.4 ms (2.26 Hz) | 749.5 ms (1.33 Hz) | 445.6 ms (2.24 Hz) |
| Place Object A to B Right | 745.9 ms (1.34 Hz) | 457.8 ms (2.18 Hz) | 470.8 ms (2.12 Hz) | 442.3 ms (2.26 Hz) | 471.2 ms (2.12 Hz) | 449.3 ms (2.23 Hz) |
| Move Can Pot | 765.2 ms (1.31 Hz) | 482.1 ms (2.07 Hz) | 466.9 ms (2.14 Hz) | 736.2 ms (1.36 Hz) | 456.0 ms (2.19 Hz) | 451.4 ms (2.22 Hz) |
| Move Stapler Pad | 469.3 ms (2.13 Hz) | 439.5 ms (2.28 Hz) | 486.3 ms (2.06 Hz) | 739.5 ms (1.35 Hz) | 476.7 ms (2.10 Hz) | 736.0 ms (1.36 Hz) |
| Stack Blocks Two | 457.4 ms (2.19 Hz) | 449.5 ms (2.22 Hz) | 458.0 ms (2.18 Hz) | 441.7 ms (2.26 Hz) | 449.2 ms (2.23 Hz) | 738.1 ms (1.35 Hz) |
| Stack Bowls Three | 458.0 ms (2.18 Hz) | 442.3 ms (2.26 Hz) | 745.7 ms (1.34 Hz) | 456.0 ms (2.19 Hz) | 451.3 ms (2.22 Hz) | 449.9 ms (2.22 Hz) |
| Handover Block | 742.8 ms (1.35 Hz) | 446.7 ms (2.24 Hz) | 455.2 ms (2.20 Hz) | 741.0 ms (1.35 Hz) | 448.1 ms (2.23 Hz) | 444.2 ms (2.25 Hz) |
| Handover Microphone | 468.4 ms (2.13 Hz) | 747.9 ms (1.34 Hz) | 458.5 ms (2.18 Hz) | 451.3 ms (2.22 Hz) | 450.8 ms (2.22 Hz) | 749.6 ms (1.33 Hz) |
| Hanging Mug | 453.1 ms (2.21 Hz) | 451.6 ms (2.21 Hz) | 450.2 ms (2.22 Hz) | 456.1 ms (2.19 Hz) | 745.8 ms (1.34 Hz) | 457.3 ms (2.19 Hz) |
| Click Bell | 455.7 ms (2.19 Hz) | 442.2 ms (2.26 Hz) | 444.3 ms (2.25 Hz) | 495.0 ms (2.02 Hz) | 442.2 ms (2.26 Hz) | 453.2 ms (2.21 Hz) |
| Press Stapler | 448.2 ms (2.23 Hz) | 478.8 ms (2.09 Hz) | 445.3 ms (2.25 Hz) | 451.0 ms (2.22 Hz) | 475.1 ms (2.11 Hz) | 471.4 ms (2.12 Hz) |
| Turn Switch | 481.2 ms (2.08 Hz) | 446.0 ms (2.24 Hz) | 743.1 ms (1.35 Hz) | 742.9 ms (1.35 Hz) | 491.5 ms (2.03 Hz) | 455.8 ms (2.19 Hz) |
| Lift Pot | 458.6 ms (2.18 Hz) | 747.7 ms (1.34 Hz) | 443.9 ms (2.25 Hz) | 480.8 ms (2.08 Hz) | 454.7 ms (2.20 Hz) | 452.5 ms (2.21 Hz) |
| Beat Block Hammer | 442.4 ms (2.26 Hz) | 461.9 ms (2.17 Hz) | 743.1 ms (1.35 Hz) | 438.5 ms (2.28 Hz) | 454.7 ms (2.20 Hz) | 444.3 ms (2.25 Hz) |
| Open Laptop | — | — | — | — | — | — |
| Open Microwave | 460.4 ms (2.17 Hz) | 450.7 ms (2.22 Hz) | 749.6 ms (1.33 Hz) | 759.8 ms (1.32 Hz) | 441.0 ms (2.27 Hz) | 440.0 ms (2.27 Hz) |

### 2.3 L_prefill（单次 video prefill，ms）

| Task | ah=64, cpp=2 | ah=64, cpp=1 | ah=64, cpp=3 | ah=32, cpp=2 | ah=32, cpp=1 | ah=16, cpp=1 |
|---|---:|---:|---:|---:|---:|---:|
| Pick Dual Bottles | 101.0 | 96.4 | 100.0 | 99.0 | 101.1 | 96.7 |
| Pick Diverse Bottles | 98.4 | 100.1 | 97.3 | 99.1 | 167.8 | 176.9 |
| Place Bread Basket | 96.9 | 97.4 | 103.3 | 99.4 | 172.7 | 174.1 |
| Place Object Basket | 96.9 | 99.3 | 98.8 | 100.3 | 100.0 | 98.7 |
| Place Object A to B Left | 167.2 | 165.0 | 96.8 | 95.9 | 168.3 | 96.5 |
| Place Object A to B Right | 166.7 | 98.2 | 102.3 | 98.4 | 101.6 | 97.3 |
| Move Can Pot | 168.6 | 103.1 | 100.6 | 164.7 | 98.0 | 98.6 |
| Move Stapler Pad | 101.1 | 95.8 | 103.5 | 165.0 | 102.5 | 165.4 |
| Stack Blocks Two | 98.5 | 97.9 | 99.4 | 96.1 | 97.4 | 163.6 |
| Stack Bowls Three | 101.4 | 96.2 | 166.4 | 98.5 | 97.5 | 98.2 |
| Handover Block | 165.3 | 97.6 | 98.5 | 164.9 | 96.9 | 96.7 |
| Handover Microphone | 100.8 | 167.5 | 98.6 | 98.1 | 98.4 | 166.0 |
| Hanging Mug | 97.7 | 98.3 | 98.0 | 98.3 | 166.3 | 98.3 |
| Click Bell | 97.0 | 96.1 | 95.9 | 111.1 | 95.5 | 98.8 |
| Press Stapler | 95.9 | 101.7 | 95.6 | 96.4 | 102.0 | 103.6 |
| Turn Switch | 102.2 | 96.1 | 166.6 | 167.2 | 105.4 | 98.8 |
| Lift Pot | 100.2 | 167.5 | 96.8 | 102.1 | 99.3 | 98.8 |
| Beat Block Hammer | 95.1 | 100.2 | 165.4 | 94.7 | 99.3 | 96.7 |
| Open Laptop | — | — | — | — | — | — |
| Open Microwave | 100.7 | 98.0 | 166.7 | 169.5 | 96.4 | 97.1 |

### 2.4 Prefill calls / episode（cpp 影响更明显）

| Task | ah=64, cpp=2 | ah=64, cpp=1 | ah=64, cpp=3 | ah=32, cpp=2 | ah=32, cpp=1 | ah=16, cpp=1 |
|---|---:|---:|---:|---:|---:|---:|
| Pick Dual Bottles | 5.8 | 10.2 | 4.5 | 5.8 | 10.2 | 10.2 |
| Pick Diverse Bottles | 8.5 | 14.8 | 6.3 | 8.5 | 14.8 | 14.8 |
| Place Bread Basket | 11.2 | 24.9 | 8.1 | 11.2 | 24.9 | 24.9 |
| Place Object Basket | 19.9 | 43.1 | 13.5 | 19.9 | 43.1 | 43.1 |
| Place Object A to B Left | 11.2 | 23.4 | 8.0 | 11.2 | 23.4 | 23.4 |
| Place Object A to B Right | 12.4 | 23.8 | 8.4 | 12.4 | 23.8 | 23.8 |
| Move Can Pot | 9.6 | 18.8 | 6.8 | 9.6 | 18.8 | 18.8 |
| Move Stapler Pad | 12.8 | 25.0 | 8.7 | 12.8 | 25.0 | 25.0 |
| Stack Blocks Two | 16.6 | 21.6 | 9.4 | 16.6 | 21.6 | 21.6 |
| Stack Bowls Three | 19.6 | 42.7 | 13.7 | 19.6 | 42.7 | 42.7 |
| Handover Block | 23.1 | 48.6 | 14.8 | 23.1 | 48.6 | 48.6 |
| Handover Microphone | 15.9 | 34.0 | 11.4 | 15.9 | 34.0 | 34.0 |
| Hanging Mug | 22.2 | 48.3 | 16.1 | 22.2 | 48.3 | 48.3 |
| Click Bell | 6.5 | 20.3 | 4.8 | 6.5 | 20.3 | 20.3 |
| Press Stapler | 2.8 | 7.0 | 1.9 | 2.8 | 7.0 | 7.0 |
| Turn Switch | 7.6 | 16.4 | 5.5 | 7.6 | 16.4 | 16.4 |
| Lift Pot | 12.8 | 25.0 | 9.0 | 12.8 | 25.0 | 25.0 |
| Beat Block Hammer | 6.2 | 15.0 | 5.7 | 6.2 | 15.0 | 15.0 |
| Open Laptop | — | — | — | — | — | — |
| Open Microwave | 47.0 | 94.0 | 31.1 | 47.0 | 94.0 | 94.0 |
| **Mean** | **14.3** | **29.3** | **9.9** | **14.3** | **29.3** | **29.3** |

规律：单次 L_prefill 各配置接近；**cpp 越大，每集 prefill 次数越少**。

---

## 3. Experiment 2 — skip_phase v2

固定 **ah=64, cpp=2**。只跳过 video prefill；Action DiT / OVCR 仍每 chunk 运行。

规则摘要：
- REACH / TRANSPORT：距 grasp/place **>10 cm** 可 skip；**≤10 cm** 必须 refresh
- PLACE（抓住且靠近放置点）：不 skip
- 连续 skip 最多 1 次（≈≤50%）
- 距离：左右 TCP 与任务配置物体 GT 位姿的最小欧氏距离（米）

### 3.1 Success + skip%

| Task | baseline | skip_phase | skip% |
|---|---:|---:|---:|
| Handover Microphone | 11/40 (27.5%) | 8/40 (20.0%) | 23.8% |
| Hanging Mug | 15/40 (37.5%) | 15/40 (37.5%) | 46.4% |
| Move Stapler Pad | 1/40 (2.5%) | 3/40 (7.5%) | 26.1% |
| Place Bread Basket | 29/40 (72.5%) | 23/40 (57.5%) | 39.6% |
| Place Mouse Pad | 19/40 (47.5%) | 8/40 (20.0%) | 24.2% |
| Place Object Basket | 8/40 (20.0%) | 7/40 (17.5%) | 52.0% |
| Stack Blocks Three | 7/40 (17.5%) | 4/40 (10.0%) | 14.4% |
| Stack Blocks Two | 23/40 (57.5%) | 22/40 (55.0%) | 18.8% |
| Click Bell | 26/40 (65.0%) | 29/40 (72.5%) | 38.4% |
| Pick Dual Bottles | 32/40 (80.0%) | 35/40 (87.5%) | 44.9% |
| Pick Diverse Bottles | 20/40 (50.0%) | 21/40 (52.5%) | 42.0% |
| Place Object A to B Left | 9/40 (22.5%) | 5/40 (12.5%) | 44.1% |
| Place Object A to B Right | 3/40 (7.5%) | 7/40 (17.5%) | 45.8% |
| Move Can Pot | 17/40 (42.5%) | 11/40 (27.5%) | 45.5% |
| Stack Bowls Three | 32/40 (80.0%) | 24/40 (60.0%) | 36.2% |
| Handover Block | 5/40 (12.5%) | 5/40 (12.5%) | 43.3% |
| Lift Pot | 1/40 (2.5%) | 0/40 (0.0%) | 46.3% |
| Press Stapler | 40/40 (100.0%) | 37/40 (92.5%) | 59.0% |
| Turn Switch | 21/40 (52.5%) | 21/40 (52.5%) | 26.9% |
| **Mean (19)** | **42.0%** | **37.5%** |  |

### 3.2 Prefill 次数 / 时间（加速应看这里，不是 L_chunk）

| Task | baseline L_prefill | skip L_prefill | baseline prefill/ep | skip prefill/ep | calls ↓ | baseline prefill_s/ep | skip prefill_s/ep | time ↓ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Handover Microphone | 100.2 ms | 101.1 ms | 15.9 | 13.2 | 16.9% | 1.60s | 1.34s | 16.3% |
| Hanging Mug | 101.8 ms | 104.2 ms | 22.2 | 13.0 | 41.4% | 2.26s | 1.36s | 40.0% |
| Move Stapler Pad | 100.0 ms | 104.1 ms | 12.8 | 9.6 | 25.4% | 1.28s | 0.99s | 22.3% |
| Place Bread Basket | 96.9 ms | 98.6 ms | 11.2 | 9.1 | 18.9% | 1.09s | 0.90s | 17.2% |
| Place Mouse Pad | 100.2 ms | 98.1 ms | 9.2 | 9.2 | -0.5% | 0.92s | 0.91s | 1.5% |
| Place Object Basket | 96.9 ms | 103.1 ms | 19.9 | 10.2 | 48.5% | 1.92s | 1.05s | 45.2% |
| Stack Blocks Three | 100.3 ms | 97.5 ms | 34.0 | 30.9 | 9.1% | 3.42s | 3.01s | 11.8% |
| Stack Blocks Two | 99.9 ms | 96.8 ms | 16.6 | 14.4 | 13.3% | 1.66s | 1.39s | 16.1% |
| Click Bell | 101.4 ms | 101.6 ms | 6.5 | 4.7 | 27.0% | 0.66s | 0.48s | 26.5% |
| Pick Dual Bottles | 101.0 ms | 95.9 ms | 5.8 | 3.7 | 35.8% | 0.59s | 0.36s | 39.0% |
| Pick Diverse Bottles | 98.4 ms | 101.4 ms | 8.5 | 5.6 | 33.8% | 0.84s | 0.57s | 31.9% |
| Place Object A to B Left | 167.2 ms | 99.8 ms | 11.2 | 7.2 | 35.4% | 1.88s | 0.72s | 61.5% |
| Place Object A to B Right | 166.7 ms | 98.2 ms | 12.4 | 6.9 | 44.4% | 2.07s | 0.67s | 67.4% |
| Move Can Pot | 96.1 ms | 99.3 ms | 9.6 | 6.4 | 33.6% | 0.93s | 0.63s | 31.5% |
| Stack Bowls Three | 104.1 ms | 102.6 ms | 19.6 | 17.2 | 12.2% | 2.04s | 1.77s | 13.4% |
| Handover Block | 100.9 ms | 100.1 ms | 23.1 | 13.5 | 41.4% | 2.33s | 1.35s | 41.9% |
| Lift Pot | 99.0 ms | 100.6 ms | 12.8 | 7.5 | 41.7% | 1.26s | 0.75s | 40.7% |
| Press Stapler | 98.0 ms | 100.2 ms | 2.8 | 2.3 | 17.0% | 0.28s | 0.23s | 15.1% |
| Turn Switch | 99.3 ms | 95.5 ms | 7.6 | 7.1 | 6.6% | 0.75s | 0.68s | 9.7% |
| **Mean** |  |  |  |  | **26.4%** |  |  | **28.9%** |

### 3.3 L_chunk（预期两边接近）

| Task | baseline L_chunk | skip_phase L_chunk |
|---|---:|---:|
| Handover Microphone | 462.0 ms (2.16 Hz) | 468.1 ms (2.14 Hz) |
| Hanging Mug | 476.6 ms (2.10 Hz) | 488.2 ms (2.05 Hz) |
| Move Stapler Pad | 458.7 ms (2.18 Hz) | 490.3 ms (2.04 Hz) |
| Place Bread Basket | 446.3 ms (2.24 Hz) | 453.4 ms (2.21 Hz) |
| Place Mouse Pad | 462.0 ms (2.16 Hz) | 453.8 ms (2.20 Hz) |
| Place Object Basket | 443.9 ms (2.25 Hz) | 481.6 ms (2.08 Hz) |
| Stack Blocks Three | 464.9 ms (2.15 Hz) | 447.0 ms (2.24 Hz) |
| Stack Blocks Two | 463.8 ms (2.16 Hz) | 446.3 ms (2.24 Hz) |
| Click Bell | 472.5 ms (2.12 Hz) | 479.5 ms (2.09 Hz) |
| Pick Dual Bottles | 472.0 ms (2.12 Hz) | 444.5 ms (2.25 Hz) |
| Pick Diverse Bottles | 459.4 ms (2.18 Hz) | 472.2 ms (2.12 Hz) |
| Place Object A to B Left | 746.5 ms (1.34 Hz) | 462.2 ms (2.16 Hz) |
| Place Object A to B Right | 745.9 ms (1.34 Hz) | 449.6 ms (2.22 Hz) |
| Move Can Pot | 441.7 ms (2.26 Hz) | 456.1 ms (2.19 Hz) |
| Stack Bowls Three | 488.1 ms (2.05 Hz) | 480.1 ms (2.08 Hz) |
| Handover Block | 472.0 ms (2.12 Hz) | 462.3 ms (2.16 Hz) |
| Lift Pot | 462.6 ms (2.16 Hz) | 461.5 ms (2.17 Hz) |
| Press Stapler | 461.8 ms (2.17 Hz) | 469.3 ms (2.13 Hz) |
| Turn Switch | 456.1 ms (2.19 Hz) | 442.1 ms (2.26 Hz) |

说明：skip 不加速 Action DiT；L_chunk 不变是预期现象。

### 3.4 v1（废弃）

v1 近场阈值 6 cm 且无连续 skip 预算 → skip 比例过高，成功率严重下降（例如 Place Mouse Pad 曾从 ~47.5% 掉到 ~0）。**不要用 v1 目录对比。**

v1 曾跑过的 skip_phase 任务（仅存档）：place_mouse_pad 0/40, move_stapler_pad 0/40, click_bell 11/40

---

## 4. 延迟与论文对照（避免再混口径）

| 量 | 含义 | 我们（A800 eager） | 论文 Table 8 Stage0 eager（5090D） | 论文优化后 |
|---|---|---|---|---|
| L_chunk | 一次 action-chunk 端到端 | ~450–500 ms（~2 Hz） | ~415.8 ms | 41.4 ms / Flash 17.6 ms |
| L_prefill | 一次 video prefill | ~100 ms（个别负载高 ~160+） | ~61.2 ms | 可再降（compile 等） |
| Freq | 1/L_chunk | ~2 Hz | ~2.4 Hz（eager） | ~24–57 Hz |

我们 **测了** 自己的 L_prefill / L_chunk；论文 61/41 只作对照，不是我们的数。

---

## 5. 其它探索性 run（非正式主结论）

目录均在 `aha-wam-runs/robotwin/`：`batch_analysis_*`、`complex_tasks_sweep_*`、`ovcr_improve*`、`video_dit_ablation`、`video_dit_skip_approach*`、`open_microwave_*` 等。用于诊断 OVCR / 难任务 / 早期 skip 思路，**主表以 §2–§3 为准**。

---

## 6. 遗留与建议

- ah/cpp：**现象乱、均值打平** → 先别急着分 cpp 长训；若要坚持「缺适配」假说，按 `AGENT_TRAINING.md` 做 **短 finetune + 对照评测**。
- skip_phase：报告加速请用 **prefill calls / prefill_s/ep**；不要用 L_chunk 论证 skip 变快。
- 未开源推理加速（TRT/CG/Flash）→ 无法复现论文 Table 3 的 24 Hz。
- 平台入口与环境坑见 `AGENT_ONBOARDING.md` / `AGENT_TRAINING.md`。

---

## 7. 相关路径速查

```text
dataset/cwr_wulan_aha/aha-wam-runs/robotwin_ahawam_sweep_20tasks_40eps/
dataset/cwr_wulan_aha/aha-wam-runs/robotwin/video_dit_skip_phase_40eps_v2/
dataset/cwr_wulan_aha/aha-wam-runs/robotwin/video_dit_skip_phase_40eps_v2_batch2/
AHA-WAM-anygrasp/infra/train_mtp.sh
AHA-WAM-anygrasp/infra/train_skip_phase.sh
AHA-WAM-anygrasp/AGENT_ONBOARDING.md
AHA-WAM-anygrasp/AGENT_TRAINING.md
```
