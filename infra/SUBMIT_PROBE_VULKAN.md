# 集群探针：训练任务到底有没有 NVIDIA Vulkan / modeset

> 探索机 notebook **不能**代表集群训练容器。请用本入口交一个短 ModelArts 任务。

## 提交（照做）

| 项 | 值 |
|---|---|
| 算法包 | `cwr_wulan_algorithm`（**重新上传**含本脚本） |
| 入口 | **`train_probe_vulkan.sh`** |
| 资源 | 1 机 × **1~2 卡**即可（8 卡也行） |
| 超时 | **30 min** |
| 挂载 | **wulann + wulann4** |
| 日志 revision | `probe-v3-cluster` |

不跑评测，只写诊断文件。

## 结果看哪里（探索机）

```bash
# 最新一次
cat /home/ma-user/work/dataset/cwr_dataset_wulann4/aha-wam-runs/robotwin/vulkan_probe_LATEST/CONCLUSION.md

# 或
ls -ltd /home/ma-user/work/dataset/cwr_dataset_wulann4/aha-wam-runs/robotwin/vulkan_probe_*
```

目录内：

- `CONCLUSION.md` — 人话结论  
- `forensics.json` — 完整证据  
- `nvidia-smi.txt` — 卡是否可见  

日志关键一行：

```text
PROBE_RESULT modeset=0|1 nvidia_vulkan=0|1 cgroup_195_254=0|1 env=cluster plan=lavapipe|nvidia
```

## 怎么判

| 结果 | 含义 |
|---|---|
| `modeset=1` / `nvidia_vulkan=1` | 集群**可以**走 NVIDIA Vulkan 无头渲 → 评测应改回方案 A，会快很多 |
| `modeset=0` 且 `cgroup_195_254=0` | 集群和 notebook 一样只有 compute → **没有** NVIDIA Vulkan，只能 lavapipe；找平台创建时注入 graphics |

## 给平台（若 modeset=0）

> 训练容器有 `/dev/nvidia0`（compute），但缺少 `/dev/nvidia-modeset`（`c 195:254`），cgroup `devices.list` 也无此项。  
> 请在创建任务时通过 nvidia-container-toolkit 注入 `graphics`（例如 `NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics`），挂上 modeset。  
> 任务内事后 `export` / `mknod` 无效。完整证据见共享盘 `vulkan_probe_*/forensics.json`。
