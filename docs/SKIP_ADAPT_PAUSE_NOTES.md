# Skip 适配线：问题、候选方案与实验暂停说明

> **状态（2026-09-11）：实验先停在这里。**  
> 下面问题与方案先存档；若后面真正要做的方向再次撞上同类问题，再挑方案执行。  
> 真正要做的目标（学 skip + execution horizon 等）仍见 `docs/计划.md`，需另开讨论对齐。

---

## 1. 本轮已经跑过什么（诊断向，非最终产品）

| 条件 | 权重 | 调度 | 平台 / 规模 |
|---|---|---|---|
| **baseline** | released `robotwin_ahawam.pt` | 正常 prefill（ah64 / cpp2） | MAIN-2 · NVIDIA · 19 task × 40ep |
| **zero-shot skip** | 同上 released | FSM `skip_phase` v2 | MAIN-2 · NVIDIA · 19 × 40 |
| **skip_v2 FT + skip** | `skip_v2_ft_8x10h/.../step_004000.pt` | 同一套 FSM | lavapipe v12 · 19 × 40（`move_can_pot` ERROR） |
| **random skip** | released | `random_skip`（同预算意图） | lavapipe v12 · 19 × **20**ep（`move_can_pot` ERROR） |

旁路诊断（已停，不当主结论）：

- **FSM-skip success-BC**：闭环采成功轨 → Action BC（冻 Video）。smoke 仅 3 成功 ep → 过拟合，eval 近崩；**未继续 full collect**。
- 细节与入口见 `infra/SUBMIT_SKIP_SUCCESS_BC*.md`。

四组 SR / latency 汇总见同目录 **`FOUR_WAY_EVAL_SNAPSHOT.md`**；正式 MAIN-2 表仍以仓库根 `EXPERIMENT_RESULTS_SUMMARY.md` §D 为准。

---

## 2. 已暴露的问题（先记着，不必现在修）

### 2.1 训–测日程不对齐（skip_v2 离线 FT）

- 训练：`skip_phase_v2_train`，用 gripper / progress **代理**是否 skip，不是评测里的 TCP FSM。
- 目标：在「有时视觉 stale」下做 **demo Action BC**，不是直接优化 rollout SR。
- 结果：lavapipe 上 FT **没有系统性地压过**「released + 同一 FSM」（跨平台 SR 也不可硬比）；更像 **日程代理 + 分布错位**，不是「适配成功」。

### 2.2 衔接 / covariate shift

- 专家 demo 是在 **正常、成功、常刷新视觉** 下采的动作。
- 部署 / 训练若强制 skip，却仍克隆「当时 fresher 条件下的动作」，会出现 **条件变了、标签没变** → 衔接差。
- success-BC 本意是用 **FSM-skip 闭环成功轨** 缓解；但数据量与质量另一回事。

### 2.3 成功过滤 BC 的数据规模与过拟合

- 离线 demos ~**27.5k** ep；policy eval SR ~37–42% **不是** demo 成功率。
- 闭环成功条数 ≈ 尝试次数 × SR。smoke：2 task × 3 ep → **3** 成功；full 8×20=160 尝试也只期望几十条成功 ≪ 27.5k。
- smoke 训完 eval ~0%：典型 **极小成功集过拟合**，不能否定协议本身，只能说明 **不能拿 smoke 当正式训**。

### 2.4 平台不可比

- MAIN-2（NVIDIA）与 lavapipe 的 **绝对 SR / 绝对 wall latency 不能并表当同一实验**。
- lavapipe 上个别任务（如 `click_bell`）SR 会虚高/虚低；`move_can_pot` 常 ~240s ERROR，均值应 **剔除或单列**。

### 2.5 指标口径

- **L_chunk**（action chunk 一次 forward）在 skip 下应大致平坦；省时信号在 **prefill 次数 / prefill 时间**，不在 L_chunk。
- 「减少比例」应报：**calls↓、prefill_s/ep↓**（相对同平台 baseline）；决策层 `skipped_prefill_ratio` 是另一口径，勿与 calls↓ 混用。

---

## 3. 候选方案（未执行；以后需要再开）

仅当真正产品线再次遇到「适配 skip / 数据衔接 / 过拟合」时再挑：

| ID | 方案 | 想解决什么 | 代价 / 风险 |
|---|---|---|---|
| A | **混合数据**：robotwin demos + FSM-skip 成功轨（可加权） | 小成功集过拟合；保留大盘动作先验 | 需定混合比；demo 仍非 skip 分布 |
| B | **demo 上 prefill dropout / 随机 skip 日程** | 一般性「缺 prefill」鲁棒，不绑死 FSM | 与部署 FSM 不完全同分布 |
| C | **失败轨 / 近成功轨**（过滤或负样本 / 对比） | 只用成功会偏；失败也含「何时不该跳」 | 标签噪声大 |
| D | **DAgger 式**：skip 下 rollout → 专家/强策略纠错再 BC | 衔接与分布漂移 | 要可查询专家或人工；仿真贵 |
| E | **半合成**：在成功 demo 上按 FSM/随机重放 skip 日程，动作仍用 demo | 便宜造「skip 条件 + 正确动作」 | 动作在真实 skip 下未必仍对 |
| F | **full success-BC**：modeset/graphics 下大规模 FSM-skip 采成功再 BC | 直接对齐部署日程 | 贵；仍 ≪ 27.5k 时要混合 A |
| G | **学 skip gate / \(H_e\)**（见 `计划.md`） | 最终产品，不只是适配固定 FSM | 需先有可学信号与锚点扫 |

**当前约定：先不执行上表；讨论清楚真正要做的东西后再决定是否开、开哪条。**

---

## 4. 和「真正要做的东西」的关系

本轮多半是在回答：

> 固定粗糙 FSM skip 下，零样本掉多少、随机对照怎样、离线代理 FT / 小成功 BC 能不能救。

**不是**最终答案：何时调用 Video DiT、执行多长（\(H_e\)）、gate 怎么学。  
那些仍以 `docs/计划.md` 为准；本文件只冻结「适配与数据」上踩过的坑和备选药方。
