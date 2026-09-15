# FULL success-BC：先采再训（默认已改成 full，不是 smoke）

## 为什么上次只有 3 条

入口默认 `SKIP_PHASE_BATCH=smoke`（2 任务 × 3 ep），只为通管道。  
**现已改为默认 `full`。** 以后若要冒烟必须显式 `smoke`。

## 数据量对比（重要）

| | skip_v2 FT | success-BC smoke（上次） | success-BC **full**（这次） |
|---|---|---|---|
| 数据性质 | 专家 demo | FSM-skip **成功**闭环 | FSM-skip **成功**闭环 |
| episode | **≈27,500** | **3** | 尝试 **8×20=160**，留下 ≈ SR×160（估 **~50–80**） |
| frames | ≈6.1e6 | 330 | 视成功率 |

full **仍远小于** 27.5k demos——这是 success-BC 定义决定的（只要成功轨），不是又漏配了。

若你要的「所有数据」= 整份 `robotwin2.0` demo，那是 **另一类实验**（接近 skip_v2），不是本 success-BC 管线。

## 提交

| 一步到位 | 入口 |
|---|---|
| collect(full) → train | **`train_skip_success_pipeline.sh`** |

或分两步：`train_collect_skip_success.sh`（默认 full）→ 完成后 `train_finetune_skip_success_bc.sh`。

- 超时建议 **≥48h**
- 挂载 wulann~wulann4
- 训出目录：`.../train/skip_success_bc_full_8x4k/`
- 采集目录：`.../robotwin/skip_success_collect_full/`
- revision 应含：`full` / `v3-full` / `v4-full-data`

## 评测

先 **停掉** 正在跑的 smoke-ckpt 评测（`*_success_bc`），等 full 训完再评。
