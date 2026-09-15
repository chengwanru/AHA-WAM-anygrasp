# 评测：CPP1 适配验收 A/B（ah64 / cpp=1 / always）

## 测什么

| 臂 | 权重 | 设定 | 输出目录 |
|---|---|---|---|
| **A** | released `robotwin_ahawam.pt` | ah=64, **cpp=1**, always | `.../robotwin/cpp1_released_always_40eps/` |
| **B** | `cpp1_ditkv_adapt_16x30h/.../step_010000.pt` | 同上 | `.../robotwin/cpp1_adapt_always_40eps/` |

任务：MAIN-2 可比 **19 × 40 ep**（与 skip 线 batch1+batch2 一致）。  
指标：mean SR；期望 **B ≥ A**（同平台）。  
**不要** reuse 旧 ah64_cpp2 结果。

## 提交（推荐：1 个 16 卡 job 同时跑 A+B）

| 项 | 值 |
|---|---|
| 算法包 | `cwr_wulan_algorithm`（入口已同步；**代码盘 AHA-WAM 须含** `FIXED_CPP` 读 env 的 sweep） |
| 入口 | **`train_eval_cpp1_ab.sh`** |
| 资源 | **2×8 = 16 卡**（`MA_NUM_HOSTS=2`） |
| 分配 | `VC_TASK_INDEX=0` → **A released**；`=1` → **B adapt**（各 8 卡、全 19 任务） |
| 超时 | 平台 **≥96h**（脚本 `JOB_TIMEOUT_S=345600`）；有 modeset 会短很多 |
| 挂载 | wulann + wulann2 + wulann3 + wulann4 |

日志首行：`revision: 2026-09-13-cpp1-ab-always-v4-local-wheels`  
开训后应见：`FIXED ... cpp=1`、`modes=['baseline']`、两机不同 `CPP1_AB_ARM=` / `OUTPUT_DIR=`。  
依赖：`robotwin imports already ok -> skip pip`，或 `missing pkgs (N): ...`（只补缺的，不整栈重下）。  
可选：`AHAWAM_EVAL_USERBASE=<持久目录>` 跨 job 复用；`FORCE_REINSTALL_DEPS=1` 强制重装。  
`auto` 臂要求 **`MA_NUM_HOSTS=2` + `VC_TASK_INDEX∈{0,1}`**，否则直接失败（防两机都跑 A 写同目录）。

### 备选：拆成两个 8 卡 job

| 入口 | 臂 |
|---|---|
| `train_eval_cpp1_ab_released.sh` | A |
| `train_eval_cpp1_ab_adapt.sh` | B |

## 健康信号

- `cpp=${FIXED_CPP}` / sweep 日志 `FIXED ah=64 cpp=1`
- ckpt：A=`.../robotwin_ahawam.pt`；B=`.../step_010000.pt`
- `disable_baseline_reuse`；outdir 下是 `baseline/<task>/`，**不是** `skip_phase/`
- 出 `Success rate: x/y` 且 ep 推进

## 渲染

优先 NVIDIA modeset（与 MAIN-2 同）；无则 lavapipe（慢，勿当事后唯一结论）。可用 `train_probe_vulkan.sh` 先探。
