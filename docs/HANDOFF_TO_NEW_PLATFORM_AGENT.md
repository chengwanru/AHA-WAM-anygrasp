# 给新平台 Cursor Agent 的交接（请整份当任务执行）

你是**新平台**上的 coding agent。对面（旧平台）已经把 AHA-WAM / He-adapt 实验和迁移约定写进 git 仓库。用户会把本文件整份贴给你。请**按下面做完**，不要另起一套路径或另写一套训练脚本。

更细的对照表在仓库：`docs/PLATFORM_MIGRATION.md`。和本文冲突时以仓库里那份为准，但先把下面清单做完。

---

## 你所在平台的路径（必须用这些，不要用旧名）

| 角色 | 路径 |
|---|---|
| 代码 git 仓库 | `dataset/cwr_wulan_aha/AHA-WAM-anygrasp` |
| 算法包（集群提交入口，**不是 git 源**） | `algorithm/cwr_wulan2` |
| 探索机前缀 | `/home/ma-user/work/` |
| 训练机前缀 | `/opt/huawei/` |
| 新下载 / 新权重只能放 | `dataset/cwr_wulan_aha/` 下面 |
| 训练+评测产出 | `dataset/cwr_wulan_aha/aha-wam-runs/` |

旧平台对应关系（不要在新平台再去找这些当代码根）：

- 旧仓库：`dataset/cwr_dataset_wulann/AHA-WAM-anygrasp`
- 旧算法包：`algorithm/cwr_wulan_algorithm`
- 旧产出：`dataset/cwr_dataset_wulann4/aha-wam-runs/`

两平台之间**不能搬 NFS、不能传 14G 权重**。只能 **本 repo git push/pull**。大文件在你这边重新下载。

提交训练任务的方式和旧平台一样：算法包里选 `.sh` 当入口，挂 dataset。

---

## 先做：拿到仓库并 sync 算法包

1. 确认仓库在：

   `/home/ma-user/work/dataset/cwr_wulan_aha/AHA-WAM-anygrasp`

   若还没有：按用户给的 remote **clone 或 pull** 到这个路径（不要 clone 到 `cwr_dataset_wulann`）。

2. 在仓库根目录执行：

```bash
cd /home/ma-user/work/dataset/cwr_wulan_aha/AHA-WAM-anygrasp
git pull
bash infra/sync_algorithm_package.sh
```

这会把入口脚本 copy 到 `/home/ma-user/work/algorithm/cwr_wulan2/` **根目录**（文件名保持不变，不要放进子目录）。

3. 立刻检查：

```bash
test -f /home/ma-user/work/algorithm/cwr_wulan2/train_finetune_he_adapt.sh
test -f /home/ma-user/work/algorithm/cwr_wulan2/cluster_paths.sh
test -f configs/task/robotwin_ahawam_he8.yaml
test -f configs/task/robotwin_ahawam_he32.yaml
```

（后两个在仓库里，cwd 需是仓库根。）

---

## 再做：大文件只能你这边下（不要等 git）

`.gitignore` 已经挡住 `*.pt` / `*.safetensors` / `/checkpoints/`。

### 1) 初始权重（He-adapt 的 init）

Hugging Face：`SereneC/AHA-WAM-RoboTwin2.0`  
文件：`robotwin_ahawam.pt`（约 14G）+ `dataset_stats.json`

优先下到仓库内（脚本默认会找这里）：

```bash
cd /home/ma-user/work/dataset/cwr_wulan_aha/AHA-WAM-anygrasp
huggingface-cli download SereneC/AHA-WAM-RoboTwin2.0 \
  robotwin_ahawam.pt dataset_stats.json \
  --local-dir checkpoints/AHA-WAM-RoboTwin2.0
```

若不想放进 git 工作树，下到：

```text
/home/ma-user/work/dataset/cwr_wulan_aha/checkpoints/AHA-WAM-RoboTwin2.0/
```

`infra/cluster_paths.sh` 会按这两个位置依次找。仓库里已经有一份小的 `infra/assets/robotwin_dataset_stats.json`（87KB），stats 丢了也能用。

### 2) Wan 组件（T5 / VAE / tokenizer）

按仓库 `README.md` 的 Model Assets，放到：

```text
checkpoints/DiffSynth-Studio/Wan-Series-Converted-Safetensors/
checkpoints/Wan-AI/Wan2.1-T2V-1.3B/google/umt5-xxl/
```

训练脚本会 `assert` 这些文件存在。

### 3) RoboTwin2.0 训练数据（LeRobot）

若新平台已经挂了旧数据盘 `cwr_dataset_wulann2/robotwin2.0`，`cluster_paths.sh` 会直接用。  
没有的话，下到 `cwr_wulan_aha/robotwin2.0/`（或 `cwr_wulan_aha/data/robotwin2.0/`）。公开包：`yuanty/robotwin2.0-fastwam`。

### 4) Python 3.10 env / wheels / sapien / nvidia GL

评测需要 sapien 3.0.3 + Vulkan。有 `/dev/nvidia-modeset` 时走 NVIDIA ICD。

建议：

- env：`dataset/cwr_wulan_aha/envs/ahawam/`
- wheels：`dataset/cwr_wulan_aha/wheels/`
- sapien-runtime-libs、nvidia-driver-libs：同样放在 `cwr_wulan_aha/` 下（与旧平台 `cwr_dataset_wulann/` 里同名目录对应）

缺这些时训练也许能起、评测会挂。

---

## 当前要跑的实验（不要改方案）

**He=8 vs He=32 adapt**：从 released 16-chunk WAM 短训到 chunk=8 和 chunk=32，**不训 He=16**。Skip 关，Video DiT 不冻，`lr=5e-5`，4000 step，8 卡/臂。

提交（用户去点平台，你负责把入口和权重准备好）：

| 项 | 值 |
|---|---|
| 算法包 | **`cwr_wulan2`** |
| 入口 | **`train_finetune_he_adapt.sh`** |
| 资源 | **2×8 = 16 卡** |
| 分配 | `VC_TASK_INDEX=0` → He=8；`=1` → He=32（各 8 卡独立 DDP，不是 16 卡一张网） |
| 超时 | 平台 **≥24h** |
| 挂载 | 至少 `cwr_wulan_aha`；数据若在别的 dataset 盘也要挂 |
| 渲染 | 创建时 graphics；有 nvidia-modeset |

训完**同一 job 接着评测**：8 任务 × 30 ep，ah=64 / cpp=1 / always。任务列表写死在 `train_eval_he_adapt.sh`，不要改成全 19 任务或 smoke。

产出应在：

```text
.../cwr_wulan_aha/aha-wam-runs/train/he8_adapt_8x10h/
.../cwr_wulan_aha/aha-wam-runs/train/he32_adapt_8x10h/
.../cwr_wulan_aha/aha-wam-runs/robotwin/he8_adapt_always_30eps/
.../cwr_wulan_aha/aha-wam-runs/robotwin/he32_adapt_always_30eps/
```

开训日志必须有：`cluster_paths: platform=wulan_aha`，以及 `AHA_WAM_CODE_DIR=.../cwr_wulan_aha/AHA-WAM-anygrasp`。若出现 `platform=wulann` 或路径仍是 `cwr_dataset_wulann/AHA-WAM-anygrasp`，说明仓库放错地方或旧目录残留抢了探测（探测**优先** `cwr_wulan_aha`）。

细节：`infra/SUBMIT_FINETUNE_HE_ADAPT.md`。

---

## 和旧平台 agent 怎么分工

- **旧平台**：改代码、yaml、脚本 → 只 commit/push **git 仓库**（AHA-WAM-anygrasp）。不要把 14G pt 推进 git。
- **你（新平台）**：pull → `bash infra/sync_algorithm_package.sh` → 下大文件 → 帮用户核对提交项。
- 日常改完代码：旧平台 push，你 pull + **再 sync 一次**。只 pull 不同步，集群仍跑 `cwr_wulan2` 里的旧 `.sh`。

Hydra yaml、`src/`、`experiments/` **不用** copy 进算法包；job 会 `cd` 到 git 仓库再训。算法包只要入口 `.sh` + `cluster_paths.sh` + `watch_train_progress.py`。

---

## 不要做的事

- 不要在新平台把仓库放到 `cwr_dataset_wulann/`。
- 不要默认 smoke（2 任务 × 3 episode）。这是正式 He-adapt。
- 不要改 He=8/32 配方、任务列表、30 ep、cpp=1。
- 不要 16 卡当一张 DDP 网；必须两机各 8 卡。
- 不要把 `algorithm/cwr_wulan2` 当成代码源去大改；改完还是要回到 git 仓库，否则旧平台同步不到。
- 不要问用户「要不要用旧路径」；路径已经定了。缺文件就下载或报缺哪一项。

---

## 做完后回用户这几句

1. 仓库路径、`git log -1 --oneline`  
2. `sync_algorithm_package.sh` 是否成功、`cwr_wulan2` 里是否有 `train_finetune_he_adapt.sh`  
3. `robotwin_ahawam.pt` 实际路径和大小  
4. Wan / robotwin2.0 / python3.10 还缺什么  
5. 提交页应填：算法包 `cwr_wulan2`，入口 `train_finetune_he_adapt.sh`，2×8，≥24h  

先把仓库和算法包对齐，再下权重；权重没到不要假装可以交正式训练。
