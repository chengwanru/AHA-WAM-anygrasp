# RoboTwin 集群评测踩坑录（2026-09-09）

> 本文记录 skip_v2 finetune 后评测、random_skip 对照实验在 ModelArts 上连续失败的根因与修复。  
> 正式提交说明见 [`SUBMIT_EVAL_SKIP_V2_FT.md`](./SUBMIT_EVAL_SKIP_V2_FT.md)、[`SUBMIT_EVAL_RANDOM_SKIP.md`](./SUBMIT_EVAL_RANDOM_SKIP.md)。  
> 环境总览仍以仓库根 [`AGENT_ONBOARDING.md`](../AGENT_ONBOARDING.md) 为准。

---

## 0. 一句话结论

| 优先路径 | 条件 | 说明 |
|---|---|---|
| **方案 A（MAIN-2 同款）** | 创建时注入 graphics，容器有可用 `/dev/nvidia-modeset` | NVIDIA Vulkan + RoboTwin 默认 RT；吞吐正常 |
| **方案 B（硬扛）** | 无 modeset | lavapipe + **关 RT + 关 shadow** + 建议关视频；能跑但很慢 |

`/dev/nvidia-modeset` **不能** pip / 下载 / `mknod` 冒充；必须是 nvidia-container-toolkit 在**创建容器时**按 capabilities 挂进来。入口里 `export NVIDIA_DRIVER_CAPABILITIES=...graphics...` 只改环境变量，**补不回设备节点**。

探测入口：`train_probe_vulkan.sh`（只查 modeset / Vulkan / Sapien，不跑评测）。

---

## 1. 症状 → 根因对照表

| 表面症状 | 真实原因 | 修复 |
|---|---|---|
| ~60s 全任务 `0/40`，像「跑完了」 | Sapien 找不到渲染设备；`test_render.py` 里 `exit()` 默认码 **0**，sweep 当成成功并写假结果 | `test_render` 改为 `sys.exit(1)`；sweep 不信任空 `_result_*.txt`；发现 `Render Error` 即失败 |
| `FileNotFoundError: ./task_config/demo_randomized.yml` | vendored RoboTwin 缺 `task_config/` | 从上游补回 `third_party/RoboTwin/task_config/`；入口预检缺文件直接 fail |
| 镜像 / 华为源装不上 `sapien==3.0.3` | 默认 **Python 3.9**；3.0.3 无 cp39 wheel；华为源常无该包 | 强制 `wulann4/envs/ahawam`（≥3.10），RoboTwin 栈从 **PyPI** 装 |
| `ImportError: libX11.so.6` 等 | sapien wheel 依赖系统 X11/OIDN/vulkan `.so` | `sapien-runtime-libs/` + `LD_LIBRARY_PATH`（见 onboarding） |
| `ffmpeg: command not found` | RoboTwin 调裸名 `ffmpeg`，imageio 二进制名不同 | `.../cwr_dataset_wulann/bin/ffmpeg` 软链并前置 PATH |
| 父进程 `Render Well`，子任务 ~270s `0/N`，`return code -11` | **SIGSEGV**。阶段演进见下节 | lavapipe 路径关 RT、关 shadow；预检建场景 |
| 日志 revision 是旧的（如 `nort-v6`），输出目录仍是 `..._v4` | **算法包未同步**，跑的是旧入口 | 以日志第一行 `revision:` 为准；对不上就别分析结果 |
| `ModuleNotFoundError: curobo` | 可选运动规划依赖 | **可忽略**（`planner.py` 已降级警告） |
| sapien `Failed to find Vulkan/glvnd ICD` UserWarning | 无完整 NVIDIA userspace ICD | lavapipe 模式下常见；以预检 / Render Well 为准 |

---

## 2. SIGSEGV（exit -11）三阶段

同一「~270s / 0/N」外观下，根因曾换过三代，**务必看子任务 eval 日志最后一行**：

### 2.1 开着 ray tracing（已修）

RoboTwin `_base_task.setup_scene` 默认：

```text
set_camera_shader_dir("rt")
set_ray_tracing_denoiser("oidn")
```

在 **lavapipe（CPU Vulkan）** 上会崩。  
日志特征：打印 `Policy Name` 后很快 -11；**没有** `[setup_scene] ray tracing disabled`。

修复：`AHAWAM_VULKAN_MODE=lavapipe` 或 `SAPIEN_DISABLE_RAYTRACING=1` 时跳过 RT。

### 2.2 关了 RT，仍开 shadow（已修）

日志特征：已有

```text
[setup_scene] ray tracing disabled (...)
```

然后立刻 -11，**没有** `forcing shadow=False` / `lights ok`。

修复：lavapipe / no-RT 时强制 `shadow=False`（directional / point light）。

### 2.3 关了 RT+shadow，死在 `load_robot` / URDF（已修）

日志特征：

```text
[setup_scene] lights ok
[setup_scene] done
[init_task] create_table_and_wall...
[init_task] load_robot...
```

然后 -11，**没有** `[robot] URDF load ok`。

常见原因：方案 B 下 `LD_LIBRARY_PATH` 仍含 **`nvidia-driver-libs`**（NVIDIA GLX/EGL 用户态），与 lavapipe ICD 混用，加载 Aloha 双臂 URDF 网格时 SIGSEGV。  
`export NVIDIA_DRIVER_CAPABILITIES=...graphics` **解决不了**（没有 modeset）。

修复：

- lavapipe 时从 `LD_LIBRARY_PATH` **剥掉** `nvidia-driver-libs`，并把 `/usr/lib/x86_64-linux-gnu` 置前（入口脚本 + `eval_robotwin_single.py`）
- 入口预检增加 **同一 URDF load**；预检挂则直接失败，避免 8 卡空跑 ~270s


### 2.4 URDF 过了，死在 `set_planner` / `SapienPlanningWorld`（已修）

日志特征：`[robot] URDF load ok` → `[load_robot] Robot() ok; set_planner...` → -11。

`MplibPlanner` 在 `scene is not None` 时走 `SapienPlanningWorld(scene, [robot])`，lavapipe 上易崩。  
改纯 `mplib.Planner` 后仍在构造处 SIGSEGV（v10）。  

**v11 修复**：lavapipe / `ROBOTWIN_MPLIB_STUB=1` 时用纯 Python stub（`plan_grippers` linspace + 线性 `TOPP`）。策略 qpos 评测不依赖 RRT；`take_action` 会走 stub TOPP 插值关节。  

健康日志应出现：`[MplibPlanner] STUB (no native mplib)`，然后 `set_planner ok` / `load_camera` / `load_actors` / `Success!`|`Fail!`。

### 关视频后 `UnboundLocalError: episode_idx`（v11→v12）

`ROBOTWIN_EVAL_VIDEO_LOG=0` 时原先只在开视频分支里赋值 `episode_idx`，首集 `Success!` 后写 analysis json 崩掉，sweep 记成 `0/N` ERROR。  
修复：始终 `episode_idx = TASK_ENV.test_num`。

### 2.5 仍崩时怎么定位

`_base_task` / `robot.py` 进度打印：

```text
[setup_scene] create_scene...
...
[init_task] load_robot...
[load_robot] Robot() / URDF load...
[robot] URDF load dual: ...
[robot] URDF load ok
[load_robot] done
[init_task] load_camera...
[init_task] load_actors...
```

**最后一条打印之后**即下一段嫌疑点。

---

## 3. 方案 A vs B（与 MAIN-2 的关系）

- **MAIN-2**（`EXPERIMENT_RESULTS_SUMMARY.md` 中正式表）按 onboarding 要求走 **NVIDIA graphics + modeset**；默认 RT 在 GPU Vulkan 上正常。  
  本仓库未必还留有当时的 `modeset: RDWR ok` 原始日志；结论依据是文档要求 + 真实成功率表，而非本盘探针铁证。
- **2026-09 乌兰 ModelArts 训练任务**实测：原始能力常为 `compute,utility`（或带 `video`），**无** `/dev/nvidia-modeset`；`mknod` 后 RDWR 仍失败。  
  → 不是租户/规格选错，是**创建通道默认不注入 graphics 设备**。
- 租户 / 团队 / A800 8 卡 /「训练任务」入口脚本本身通常没问题；表单里若看不到 graphics / modeset 相关项，加 env 也往往无效，需问平台能否在创建时注入。

---

## 4. 只关心成功率时

- 设 `ROBOTWIN_EVAL_VIDEO_LOG=0`（ft / random 入口已默认）。  
- **省下的**：ffmpeg 编码与写盘。  
- **省不掉的**：策略仍要 head/wrist RGB → 每步仍 lavapipe 渲相机。  
- 关视频是有用优化，**不是**数量级加速；数量级靠方案 A。

---

## 5. 当前入口与 revision 约定

| 实验 | 入口 | 期望 revision 含 |
|---|---|---|
| ft skip_phase（`step_004000.pt`） | `train_eval_skip_v2_ft.sh` (+ `_batch2`) | `lavapipe-v12-epidx` |
| random_skip（released ckpt） | `train_eval_random_skip.sh` (+ `_batch2`) | `lavapipe-v12-epidx` |
| 仅探测 Vulkan | `train_probe_vulkan.sh` | `probe-v1` |

 lavapipe 硬扛超时建议：ft **≥48h**（`JOB_TIMEOUT_S=172800`），random **≥24h**。  
输出目录用带 `v8_noshadow` 的新路径，**不要**复用曾写入假 `0/N` 的 `..._v4` / 早期失败目录（除非已确认 sweep 会重跑未完成项且你清楚副作用）。

挂载：**wulann + wulann2 + wulann3 + wulann4**（ahawam Python≥3.10 在 wulann4）。

---

## 6. 健康日志检查清单

父日志：

- [ ] `revision: ...-lavapipe-v8-noshadow-preflight`（或更新的约定串）
- [ ] `Using python: .../envs/ahawam/...`（≥3.10）
- [ ] `AHAWAM_VULKAN_MODE=lavapipe` 或 `nvidia`
- [ ] lavapipe 时：`SAPIEN_DISABLE_RAYTRACING=1`
- [ ] `lights shadow=False: ok` / `sapien.Scene: ok (lavapipe-safe preflight)`
- [ ] `asset ok: .../task_config/demo_randomized.yml`

子任务日志：

- [ ] `Render Well`
- [ ] `eval_video_log=False`（若关了视频）
- [ ] `[setup_scene] ray tracing disabled`（lavapipe）
- [ ] `[setup_scene] forcing shadow=False...` → `lights ok` → `done`
- [ ] `[init_task] load_actors...` 之后开始出现 Success/Fail / ep 进度  
- [ ] **不是** ~270s 后 `return code -11` 且无任何 ep

探测任务：

- [ ] `PROBE_RESULT modeset=1 ...` → 可走方案 A  
- [ ] `modeset=0 ... lavapipe` → 只能方案 B  

---

## 7. 相关代码位置（勿在旧拷贝上改）

| 内容 | 路径 |
|---|---|
| 关 RT / 关 shadow / 进度日志 | `third_party/RoboTwin/envs/_base_task.py` → `setup_scene` |
| 渲染失败非零退出 | `third_party/RoboTwin/script/test_render.py` |
| 关视频 env | `third_party/RoboTwin/script/eval_policy.py` ← `ROBOTWIN_EVAL_VIDEO_LOG` |
| 子进程 Vulkan / 强制 no-RT | `experiments/robotwin/eval_robotwin_single.py` |
| sweep / 假结果防护 | `experiments/robotwin/run_skip_phase_sweep.py` |
| 平台入口 | `infra/train_eval_skip_v2_ft.sh`、`train_eval_random_skip.sh`、`train_probe_vulkan.sh`（算法包同步同名文件） |

算法包目录里的 `.sh` 是提交入口；**RoboTwin / eval 逻辑以数据集挂载的 `AHA-WAM-anygrasp` 为准**。只更新算法包、不更新盘上代码时，关 shadow 等补丁不会生效——但本次补丁已落在 `cwr_dataset_wulann/AHA-WAM-anygrasp`。
