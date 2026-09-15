# 四组评测快照：baseline / zero-shot skip / skip_v2 FT / random

> 冻结日期：2026-09-11。配套说明：`SKIP_ADAPT_PAUSE_NOTES.md`。  
> **平台分裂**：baseline 与 zero-shot skip = **MAIN-2 NVIDIA**；FT 与 random = **lavapipe v12**。绝对 SR / 绝对耗时 **不可跨平台硬比**；同平台内比相对趋势。

### 设定速查

| 条件 | ckpt | 模式 | ep/task | 输出根 |
|---|---|---|---:|---|
| baseline | released | 正常 prefill ah64/cpp2 | 40 | MAIN-2（见 `EXPERIMENT_RESULTS_SUMMARY.md` §D） |
| zero-shot skip | released | FSM `skip_phase` | 40 | 同上 |
| skip_v2 FT | `skip_v2_ft_8x10h/.../step_004000.pt` | FSM `skip_phase` | 40 | `.../video_dit_skip_phase_40eps_v12_lvp_ft[_batch2]` |
| random | released | `random_skip` | **20** | `.../video_dit_random_skip_20eps_v12_lvp[_batch2]` |

`move_can_pot` 在 lavapipe 上 FT/random 均为 `returncode=1`（~240s）→ **均值默认剔除**。

---

## 1. 成功率总览（mean）

| 条件 | 平台 | Mean SR | 备注 |
|---|---|---:|---|
| baseline | MAIN-2 NVIDIA | **42.0%** | 19 task |
| zero-shot skip | MAIN-2 NVIDIA | **37.5%** | 19 task；相对 baseline **−4.5 pp** |
| skip_v2 FT + skip | lavapipe | **31.4%** | 18 task（剔 mcp）；含 mcp 则 29.7% |
| random skip | lavapipe | **27.2%** | 18 task（20ep）；含 mcp 则 25.8% |

同平台粗结论：

- MAIN-2：skip 有可跳空间，SR 小幅掉，prefill 明显省（见 §2）。
- lavapipe：FSM-FT **>** random（31.4 vs 27.2），但 **不能**据此说 FT 已超过 MAIN-2 zero-shot（37.5%）。

### 1.1 分任务 SR（%）

| Task | baseline (NV) | zs-skip (NV) | FT-skip (lvp) | random (lvp) |
|---|---:|---:|---:|---:|
| click_bell | 65.0 | 72.5 | 97.5 | 100.0 |
| handover_block | 12.5 | 12.5 | 2.5 | 5.0 |
| handover_mic | 27.5 | 20.0 | 10.0 | 5.0 |
| hanging_mug | 37.5 | 37.5 | 5.0 | 0.0 |
| lift_pot | 2.5 | 0.0 | 0.0 | 0.0 |
| move_stapler_pad | 2.5 | 7.5 | 0.0 | 0.0 |
| pick_diverse_bottles | 50.0 | 52.5 | 22.5 | 20.0 |
| pick_dual_bottles | 80.0 | 87.5 | 55.0 | 65.0 |
| place_a2b_left | 22.5 | 12.5 | 15.0 | 5.0 |
| place_a2b_right | 7.5 | 17.5 | 15.0 | 10.0 |
| place_bread_basket | 72.5 | 57.5 | 52.5 | 60.0 |
| place_mouse_pad | 47.5 | 20.0 | 37.5 | 20.0 |
| place_object_basket | 20.0 | 17.5 | 2.5 | 0.0 |
| press_stapler | 100.0 | 92.5 | 97.5 | 75.0 |
| stack_blocks_three | 17.5 | 10.0 | 5.0 | 0.0 |
| stack_blocks_two | 57.5 | 55.0 | 30.0 | 5.0 |
| stack_bowls_three | 80.0 | 60.0 | 67.5 | 60.0 |
| turn_switch | 52.5 | 52.5 | 50.0 | 60.0 |
| **Mean (18, 无 mcp)** | **41.9** | **38.1** | **31.4** | **27.2** |
| move_can_pot | 42.5 | 27.5 | ERROR | ERROR |

---

## 2. Latency / prefill（重点看省算力）

定义（与 `EXPERIMENT_RESULTS_SUMMARY.md` 一致）：

- **L_chunk**：一次 action-chunk forward 耗时（ms）
- **L_prefill**：一次 video prefill 耗时（ms）
- **calls/ep**：每局实际执行的 video prefill 次数
- **prefill_s/ep**：每局 video prefill 总时间
- **减少比例**：相对 **同平台 baseline** 的 calls↓ / prefill_s↓（任务均值再平均）

### 2.1 MAIN-2：baseline vs zero-shot skip（可报减少比例）

任务均值（19 task，由表 §D.3–D.4 再平均）：

| 指标 | baseline | zero-shot skip | 减少比例 |
|---|---:|---:|---:|
| L_chunk (ms) | ~492 | ~464 | （略降；非主信号） |
| L_prefill (ms) | ~107 | ~100 | ~flat（单次仍 ~100ms） |
| calls/ep | **13.77** | **10.09** | **≈ −26.7%**（文档任务均 −26.4%） |
| prefill_s/ep | **1.46** | **1.01** | **≈ −31.0%**（文档任务均 −28.9%） |

**Takeaway：** skip 不靠压低 L_chunk；靠 **少调用 Video DiT**。

### 2.2 lavapipe：FT vs random（无同平台 baseline → 不报 vs-baseline %）

来自各 task `job_report.json` → `latency_means`（18 task，剔 `move_can_pot`）：

| 指标 | skip_v2 FT | random |
|---|---:|---:|
| L_chunk (ms) | **519** | **515** |
| L_prefill (ms) | **110** | **110** |
| calls/ep | **10.84** | **8.66** |
| prefill_s/ep | **1.20** | **0.95** |
| skipped_prefill_ratio（决策层） | **0.37** | **0.51** |

说明：

- 单次 L_prefill / L_chunk 与 MAIN-2 量级接近，但 **sim 路径不同**，不要拿绝对值比 NVIDIA。
- random 的 `skipped_prefill_ratio≈0.5` 且 calls 更少，是 **跳得更猛/配额不同**，不是「更优调度」。
- 相对 MAIN-2 baseline calls（13.77）的粗参照（**非正式减少比**）：FT ≈ −21%，random ≈ −37%；仅作数量级，不能当论文主表。

---

## 3. 一句话冻结结论

1. **可跳空间存在**（MAIN-2：SR 42→37.5，prefill calls/time ~−26%/−29%）。  
2. **FSM ≥ random**（lavapipe：31.4% > 27.2%），规则有一点信息，但不够强。  
3. **离线 proxy skip FT 未证明能系统性救回 zero-shot 掉点**；success-BC smoke 过拟合 → 暂停。  
4. 后续先对齐「真正要做的东西」，再决定是否执行 `SKIP_ADAPT_PAUSE_NOTES.md` §3 的方案。
