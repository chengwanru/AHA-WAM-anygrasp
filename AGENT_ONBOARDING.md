# AHA-WAM RoboTwin 任务交接文档(给后来的 agent)

> 本文档记录 AHA-WAM-anygrasp 在 RoboTwin 2.0 上做评测/调参 sweep 的完整工作方式:
> 目录约定、环境搭建、模型资产、任务提交流程、以及我们踩过的所有坑。
> 最后更新:2026-09-07
>
> **真正训练 / 短 finetune / 平台训练入口怎么写** → 见 [`AGENT_TRAINING.md`](./AGENT_TRAINING.md)。  
> **全部实验结果总结（ah/cpp + skip_phase 主表）** → 见 [`EXPERIMENT_RESULTS_SUMMARY.md`](./EXPERIMENT_RESULTS_SUMMARY.md)。

---

## 0. 一句话背景

我们在 **官方发布的 AHA-WAM RoboTwin checkpoint** 上做评测和 action_horizon / chunks_per_video_prefill
参数 sweep。主线 **尚未做过完整重训**——平台上叫 `train_mtp.sh` 的入口实际是评测 sweep(历史命名,别被骗去当训练)。
若要适配 cpp=1/3 等部署节奏,按 [`AGENT_TRAINING.md`](./AGENT_TRAINING.md) 做短 finetune,不要默认从零开训。
任何"权重丢了怎么办"的问题,答案都是:从 HuggingFace 重新下载 released ckpt,或从 `aha-wam-runs/train/` 找你们自己的 finetune 产物。

---

## 1. 代码位置

| 内容 | 位置 |
|---|---|
| 代码仓库 | 以 `git remote -v` 为准(常见 fork 在个人账号下),分支 **`modified-working`**(不是 main!) |
| 集群/评测基础设施 | 仓库内 `infra/`(`train_mtp.sh`、`run_robotwin_sweep.{sh,py}`、`train_skip_phase*.sh`、`docs_skip_phase.md`) |
| 真正训练说明 | [`AGENT_TRAINING.md`](./AGENT_TRAINING.md) + `scripts/train_zero{1,2}.sh` |
| 评测入口 | `experiments/robotwin/eval_robotwin_single.py`(Hydra 配置) |
| Policy 桥接代码 | `experiments/robotwin/ahawam_policy/`(deploy_policy.py 等) |
| `third_party/RoboTwin/policy/ahawam_policy` | **symlink**,指向仓库内 `experiments/robotwin/ahawam_policy`,**必须用相对路径**(`../../../experiments/robotwin/ahawam_policy`),绝对路径换个机器就断 |

```bash
git clone -b modified-working https://github.com/chengwanru/AHA-WAM-anygrasp.git
```

## 2. 模型和资产放哪里(全部公开可下载,别迁移,重下)

约定根目录 = 仓库根(`AHA-WAM-anygrasp/`)。`third_party/RoboTwin/checkpoints` 是指向仓库内
`checkpoints/` 的 symlink,不要重复拷贝。

| 资产 | 路径 | 大小 | 来源 |
|---|---|---|---|
| **AHA-WAM 评测 ckpt** | `checkpoints/AHA-WAM-RoboTwin2.0/robotwin_ahawam.pt` | 14G | `huggingface-cli download SereneC/AHA-WAM-RoboTwin2.0 --local-dir checkpoints/AHA-WAM-RoboTwin2.0` |
| 归一化统计 | **同目录** `dataset_stats.json` | 87K | 同上。**代码按 `ckpt.with_name("dataset_stats.json")` 查找,必须和 .pt 同目录** |
| T5 文本编码器 | `checkpoints/DiffSynth-Studio/Wan-Series-Converted-Safetensors/models_t5_umt5-xxl-enc-bf16.safetensors` | 11G | HF `DiffSynth-Studio/Wan-Series-Converted-Safetensors`;首次运行也会经 ModelScope 自动下载 |
| Wan2.2 VAE | 同目录 `Wan2.2_VAE.safetensors` | 1.4G | 同上 |
| umt5 tokenizer | `checkpoints/Wan-AI/Wan2.1-T2V-1.3B/google/umt5-xxl/` | 21M | HF `Wan-AI/Wan2.1-T2V-1.3B` 里的 tokenizer 文件 |
| RoboTwin 仿真资产 | `third_party/RoboTwin/assets/` | 16G | `cd third_party/RoboTwin/assets && python _download.py`(HF dataset `TianxingChen/RoboTwin2.0`),下完解压 background_texture / embodiments / objects 三个 zip |

下载完之后切离线,否则跑任务时会卡在联网检查:

```bash
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 DIFFSYNTH_SKIP_DOWNLOAD=true WANDB_MODE=offline
```

也可以用 `export DIFFSYNTH_MODEL_BASE_PATH=<别的缓存目录>` 把 Wan 组件放到仓库外,但相对布局要保持一致。

## 3. 环境怎么建

**Python 3.10**(conda env,例如叫 `ahawam`)。torch 用 cu128 官方版,**装依赖时绝不能让 pip 覆盖镜像里的 CUDA torch**(先 `import torch; torch.cuda.is_available()` 验证)。

```bash
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu128
# RoboTwin 评测栈(pin 死这些版本,升级会炸):
pip install "sapien==3.0.3" "open3d==0.18.0" "mplib==0.2.1" "numpy==1.26.4" \
    "opencv-python==4.8.1.78" "h5py==3.16.0" transforms3d lxml toppra gymnasium \
    trimesh yourdfpy imageio imageio-ffmpeg hydra-core
```

**sapien 系统库(最大的坑)**:sapien wheel 能装上,但 import 需要系统里有
`libX11.so.6 / libxcb.so.1 / libXau.so.6 / libXdmcp.so.6 / libOpenImageDenoise.so.2 / libvulkan.so.1`。
训练/评测容器通常缺 X11 系。把这些 .so 收集到一个目录(我们叫 `sapien-runtime-libs/`),然后:

```bash
export LD_LIBRARY_PATH=<sapien-runtime-libs 绝对路径>:$LD_LIBRARY_PATH
# 验证:ldd <python site-packages>/sapien/pysapien.cpython-310-*.so | grep "not found"  # 必须为空
```

**Vulkan/渲染**:容器需要 `NVIDIA_DRIVER_CAPABILITIES` 含 `graphics` 且能访问 `/dev/nvidia-modeset`,
否则 sapien 初始化报 `VK_ERROR_INCOMPATIBLE_DRIVER`。docker 启动示例:
`docker run --gpus all -e NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics,display,video ...`
(有 graphics 能力时 nvidia-container-toolkit 会自动注入 nvidia-modeset)。
sapien 自带 vulkan loader,装好后可显式指:
`export SAPIEN_VULKAN_LIBRARY_PATH=<sapien>/vulkan_library/libvulkan.so.1.3.224 VK_ICD_FILENAMES=<sapien>/vulkan_library/nvidia_icd.json`

**ffmpeg**:RoboTwin 的 `eval_policy.py` 调的是裸命令 `ffmpeg`。imageio-ffmpeg 装的二进制叫
`ffmpeg-linux-x86_64-v7.0.2` 之类,必须软链成 PATH 上的 `ffmpeg`:
`ln -s <imageio_ffmpeg 二进制> <某 bin 目录>/ffmpeg && export PATH=<该目录>:$PATH`

**冒烟检查(每次建新环境必跑,别省)**:

```bash
python - <<'PY'
import torch; assert torch.cuda.is_available()
import sapien, open3d, mplib, gymnasium, h5py, cv2
from pathlib import Path
import sapien.render
r = sapien.render.SapienRenderer()   # 渲染不通在这里就炸
scene = sapien.Scene()
print("smoke OK")
PY
which ffmpeg && ffmpeg -version
```

## 4. 怎么提交任务(sweep 流程)

入口是 `infra/train_mtp.sh`,流程:**先 SMOKE=1(1 episode、4 卡)确认全链路通,再 SMOKE=0 全量(40 eps/task、8 卡)**。

```bash
# 单发评测(调试单个任务/配置):
python experiments/robotwin/eval_robotwin_single.py \
    ckpt=checkpoints/AHA-WAM-RoboTwin2.0/robotwin_ahawam.pt \
    EVALUATION.task_name=click_alarmclock EVALUATION.task_config=demo_randomized \
    EVALUATION.eval_num_episodes=5 EVALUATION.action_horizon=64 \
    EVALUATION.chunks_per_video_prefill=2 gpu_id=0
```

关键参数含义:`action_horizon`(=ah,一次出多少个 action)、`action_chunk_size=16`(内部 chunk 大小,固定)、
`chunks_per_video_prefill`(=cpp,一次 video prefill 的 KV cache 供几个 chunk 用)、`num_inference_steps=10`(每 chunk 扩散步数)。

时序 sanity check(ah=64, cpp=2, A100 级 GPU):video prefill 单次前向 ~70ms;action denoise 每 chunk
~550ms(10 步 × ~55ms/步)。**video DIT 只占总耗时 6-12%,action denoise 是大头**——这是单次前向 vs
10 步扩散循环的量级差,不是测错了。

跑 sweep 前:
1. 平台 pip 必须强制可用镜像、清掉 `PIP_EXTRA_INDEX_URL`(NGC 源超时会把安装拖死)
2. 多卡启动**之前**先建好 policy symlink(`ln -sfn <repo>/experiments/robotwin/ahawam_policy <repo>/third_party/RoboTwin/policy/ahawam_policy`),避免并发 unlink 竞态
3. 留够磁盘:sweep 输出含视频,共享盘可能只剩 1-2T
4. timing 分析排除 episode 1(warmup 污染均值)

## 5. 我们犯过的错误(按吸取教训的价值排序)

1. **把非 baseline 当 baseline**。目录名叫 baseline 就信了,实际跑的是 ah=32 cpp=2,推导出"video DIT 只占 8%"后被质疑才发现。**永远从日志/配置里确认实际参数,不看目录名**。
2. **在集群上 pip install 全量 requirements.txt**。NGC 源每个包都超时,装了几个小时。正解:离线 wheels 目录 `--no-index --find-links`,或只装缺失的包,并强制可用镜像 + `PIP_EXTRA_INDEX_URL=""`。
3. **sapien `ImportError: libX11.so.6`**。wheel 装好了但容器缺 X11 系统库。用 `ldd pysapien*.so` 定位缺失项,收集 .so 挂 `LD_LIBRARY_PATH`。**这是集群上最后一个 blocker,之前所有"环境装好了"的判断都是错的**。
4. **Python 版本错配**。系统默认 python3.9,sapien 只有 cp310 wheel。认准 Python 3.10 环境,别在 3.9 上浪费一小时。
5. **容器无 graphics 能力**导致 `VK_ERROR_INCOMPATIBLE_DRIVER`:`NVIDIA_DRIVER_CAPABILITIES=compute,utility` 不够,要加 `graphics`(自动带 /dev/nvidia-modeset)。container 重启不够,通常要重建。
6. **多卡并发 symlink 竞态**:8 卡同时 `ln -sfn` 会互踩。启动前由单进程先建好。
7. **ffmpeg 不在 PATH**:RoboTwin 调裸 `ffmpeg`,imageio-ffmpeg 的名字对不上,评测写视频时炸。
8. **gitignore 打错字**(`ahawam-runs/` vs `aha-wam-runs/`),结果垃圾文件被 stage。改完 .gitignore 用 `git check-ignore <路径>` 验证。
9. **推 GitHub 前没查权限**。token 属于 B 账号、repo 在 A 账号下,403。没有写权限就 fork 到自己账号再 push。
10. **把 token 写进 remote URL 持久化**。正解:临时 URL push 完立刻 `git remote set-url` 改回;token 用完 revoke。
11. **episode 1 混入 timing 均值**。warmup episode 比其他慢好几倍,算均值必须剔除。
12. **磁盘写满共享盘**。600T 盘 100% 时是大家一起满;跑 40eps×20task 前先 `df -h`。
13. **以为要迁移 28G 权重**。全是公开资产,新环境重新下载即可;唯一"不可再生"的误判是 released ckpt——它也是官方公开的。真正的教训:**先确认每个文件的来源,再决定迁移策略**。

## 6. 老平台遗留物(仅供考古,不要依赖)

- 旧评测结果:`<repo>/aha-wam-runs/`(sweep summary)和 `evaluate_results/`(逐 episode 输出),已被 gitignore,未迁移
- 旧 sweep 运行目录:`~/work/algorithm/cwr_wulan2/robotwin_ahawam*/`(2026-08-19 debug 输出)
- `cwr_wulan1/AHA-WAM-anygrasp-transfer/`(28G 拷贝)= 纯冗余,官方源都能重下
- 监控脚本 `/tmp/monitor_experiments.py`、timing 脚本 `/tmp/run_timing_pair.sh` 属会话临时文件
