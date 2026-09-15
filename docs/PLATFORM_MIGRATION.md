# 两平台迁移指南（git 传小文件，大文件对面重下）

> 更新：2026-09-15。  
> 源仓库：本目录 `AHA-WAM-anygrasp`。  
> **两台机器之间不能搬 NFS / 不能搬 14G 权重**；只能 **本 repo 的 push / pull**。  
> 提交训练的方式和现在一样：算法包里选入口脚本，挂 dataset。  
> 给对面 agent 整份粘贴：`docs/HANDOFF_TO_NEW_PLATFORM_AGENT.md`。

---

## 1. 两套路径怎么对应

| 角色 | 旧平台（现在这台） | 新平台 |
|---|---|---|
| 代码 git 仓库 | `dataset/cwr_dataset_wulann/AHA-WAM-anygrasp` | **`dataset/cwr_wulan_aha/AHA-WAM-anygrasp`** |
| 算法包（集群入口，**不是 git 源**） | `algorithm/cwr_wulan_algorithm` | **`algorithm/cwr_wulan2`** |
| 探索机前缀 | `/home/ma-user/work/` | 同样（用法一样） |
| 训练机前缀 | `/opt/huawei/` | 同样 |
| 新权重 / 新下载落点 | `cwr_dataset_wulann` / `wulann4` | **只往 `dataset/cwr_wulan_aha/` 下** |
| 训练产物 | `cwr_dataset_wulann4/aha-wam-runs/` | `cwr_wulan_aha/aha-wam-runs/` |

集群 job 真正 `cd` 去跑 Python 的是 **dataset 上的 git 仓库**；算法包只是平台「选哪个 `.sh` 当入口」。所以：

1. **改代码 / 改 Hydra / 改评测** → 只改 git 仓库，push。  
2. **对面 pull** 之后，必须再把入口脚本 **copy 到算法包根目录**（见 §3）。  
3. **`.pt` / 数据集 / wheels / env** → 对面重新下载，不要指望 git。

He-adapt 入口已 `source cluster_paths.sh`：只要新平台上存在 `cwr_wulan_aha/AHA-WAM-anygrasp`，就会走新路径，不必再手改 `cwr_dataset_wulann`。

**注意：** 若同一台机器上 **两个仓库目录都在**，探测 **优先 `cwr_wulan_aha`**。旧平台不要随手 clone 一份 aha，否则会误走新路径。

---

## 2. 什么能 push，什么必须对面重下

`.gitignore` 已经挡住权重和数据（`/checkpoints/`、`*.pt`、`*.safetensors`、wheels 等）。

### 走 git（小文件）

| 路径 | 说明 |
|---|---|
| `src/` `configs/` `scripts/` `experiments/` | 训练 / 评测代码 |
| `configs/task/robotwin_ahawam_he8.yaml` `he32.yaml` | He-adapt Hydra |
| `infra/*.sh` `infra/*.md` `infra/cluster_paths.sh` | **入口脚本的源** |
| `infra/sync_algorithm_package.sh` | pull 后一键 copy 到算法包 |
| `infra/assets/robotwin_dataset_stats.json` | **87KB** 归一化统计（不要 14G 的 pt） |
| `infra/watch_train_progress.py` | 训练进度 watcher |
| `docs/PLATFORM_MIGRATION.md`（本文） | 迁移说明 |
| `docs/EXPERIMENT_REGISTRY.md` | 实验目录登记 |

### 不能走 git（对面重下，放到 `cwr_wulan_aha/`）

| 东西 | 建议落点（新平台） | 怎么来 |
|---|---|---|
| released 权重 `robotwin_ahawam.pt`（~14G） | `cwr_wulan_aha/AHA-WAM-anygrasp/checkpoints/AHA-WAM-RoboTwin2.0/` **或** `cwr_wulan_aha/checkpoints/AHA-WAM-RoboTwin2.0/` | Hugging Face [`SereneC/AHA-WAM-RoboTwin2.0`](https://huggingface.co/SereneC/AHA-WAM-RoboTwin2.0) |
| Wan T5 / VAE / tokenizer | 仓库 `checkpoints/DiffSynth-Studio/...` 与 `checkpoints/Wan-AI/...` | ModelScope / HF，见仓库 README「Model Assets」 |
| RoboTwin2.0 LeRobot 数据 | `cwr_wulan_aha/robotwin2.0/` 或仍用对面已有的 `cwr_dataset_wulann2/robotwin2.0` | [`yuanty/robotwin2.0-fastwam`](https://huggingface.co/datasets/yuanty/robotwin2.0-fastwam) 或旧盘若已挂载可直接用 |
| text embedding cache | `cwr_wulan_aha/text_embeds_cache/robotwin/` | 训前脚本会生成；有旧 cache 的盘可挂上复用 |
| Python env | `cwr_wulan_aha/envs/ahawam/` | conda python 3.10 + 旧环境同配方 |
| sapien / nvidia GL / wheels | `cwr_wulan_aha/wheels`、`sapien-runtime-libs`、`nvidia-driver-libs/` | 对面按旧平台同样方式准备；评测需要 |

`dataset_stats.json` 已放进 repo（`infra/assets/robotwin_dataset_stats.json`）。14G 的 `robotwin_ahawam.pt` **必须**对面 `huggingface-cli download`。

```bash
# 在新平台、仓库根目录：
huggingface-cli download SereneC/AHA-WAM-RoboTwin2.0 \
  robotwin_ahawam.pt dataset_stats.json \
  --local-dir checkpoints/AHA-WAM-RoboTwin2.0
```

若权重不想放进 git 工作树，下到：

```bash
huggingface-cli download SereneC/AHA-WAM-RoboTwin2.0 \
  robotwin_ahawam.pt dataset_stats.json \
  --local-dir /home/ma-user/work/dataset/cwr_wulan_aha/checkpoints/AHA-WAM-RoboTwin2.0
```

`cluster_paths.sh` 会按上面两个位置依次找 `robotwin_ahawam.pt`。

---

## 3. Pull 之后：脚本要 copy 到哪里

**不要**指望平台去跑 `dataset/.../infra/train_finetune_he_adapt.sh`。  
提交页选的是 **算法包根目录** 里的同名文件。

```text
# 新平台
git pull   # 在 dataset/cwr_wulan_aha/AHA-WAM-anygrasp
bash infra/sync_algorithm_package.sh
# → 写入 algorithm/cwr_wulan2/
```

旧平台同样命令会写入 `algorithm/cwr_wulan_algorithm`。强制指定：

```bash
ALG_PKG=/home/ma-user/work/algorithm/cwr_wulan2 bash infra/sync_algorithm_package.sh
```

### 必 copy 清单（同步脚本默认这些）

从 **`infra/` → 算法包根目录**（保持文件名，不要放进子文件夹）：

| repo 里 | 算法包里 | 用途 |
|---|---|---|
| `infra/cluster_paths.sh` | `cluster_paths.sh` | 双平台路径 |
| `infra/watch_train_progress.py` | `watch_train_progress.py` | 训练曲线 |
| `infra/train_finetune_he_adapt.sh` | **提交入口** | 16 卡 He=8 vs 32 训+评 |
| `infra/train_finetune_he8.sh` | 同名 | 单 8 卡 He=8 |
| `infra/train_finetune_he32.sh` | 同名 | 单 8 卡 He=32 |
| `infra/train_eval_he_adapt.sh` | 同名 | 只评测 |
| `infra/train_eval_he8.sh` `train_eval_he32.sh` | 同名 | 单臂评测 |
| `infra/SUBMIT_FINETUNE_HE_ADAPT.md` | 同名 | 提交说明 |
| `docs/EXPERIMENT_REGISTRY.md` | `EXPERIMENT_REGISTRY.md` | 目录登记 |
| `infra/assets/robotwin_dataset_stats.json` | `infra/assets/robotwin_dataset_stats.json` | 小统计文件 |

Hydra yaml、`src/`、`experiments/` **不用** copy 到算法包；job 会 `cd` 到 `AHA_WAM_CODE_DIR`（git 仓库）再 `train_zero2.sh`。

旧 cpp1 / skip 入口默认不同步。若还要：`SYNC_OPTIONAL=1 bash infra/sync_algorithm_package.sh`。

手 copy 也可以：

```bash
REPO=/home/ma-user/work/dataset/cwr_wulan_aha/AHA-WAM-anygrasp
ALG=/home/ma-user/work/algorithm/cwr_wulan2
cp -f "$REPO"/infra/{cluster_paths.sh,watch_train_progress.py,train_finetune_he*.sh,train_eval_he*.sh,SUBMIT_FINETUNE_HE_ADAPT.md} "$ALG"/
chmod +x "$ALG"/*.sh
```

---

## 4. 推荐操作顺序

### 旧平台（现在）

1. 确认 He-adapt 脚本、yaml、本文都在 git 仓库里（不要只改算法包）。  
2. `git add` / `commit` / `push`（**不要** add `checkpoints/*.pt`）。  
3. 算法包若要在旧平台继续交任务，这里再跑一次 `bash infra/sync_algorithm_package.sh`。

### 新平台

1. 把本仓库 clone 或 pull 到 **`dataset/cwr_wulan_aha/AHA-WAM-anygrasp`**（不要放到 `cwr_dataset_wulann`）。  
2. `bash infra/sync_algorithm_package.sh` → `algorithm/cwr_wulan2`。  
3. 下载 14G 初始权重（§2）。Wan 组件按 README 放到仓库 `checkpoints/`。  
4. 准备 LeRobot `robotwin2.0`（重下或挂已有 dataset 盘）。  
5. 提交页：算法包选 **`cwr_wulan2`**，入口 **`train_finetune_he_adapt.sh`**，2×8，挂上 `cwr_wulan_aha`（以及数据盘）。  
6. 日志应出现：`cluster_paths: platform=wulan_aha` 和 `AHA_WAM_CODE_DIR=.../cwr_wulan_aha/AHA-WAM-anygrasp`。

---

## 5. 提交时两边差什么

| 项 | 旧 | 新 |
|---|---|---|
| 算法包名 | `cwr_wulan_algorithm` | **`cwr_wulan2`** |
| 入口文件名 | `train_finetune_he_adapt.sh` | **相同** |
| 资源 | 2×8，`VC_TASK_INDEX` 0→He8 / 1→He32 | 相同 |
| 代码目录 | `.../cwr_dataset_wulann/AHA-WAM-anygrasp` | `.../cwr_wulan_aha/AHA-WAM-anygrasp` |
| 产出 | `.../cwr_dataset_wulann4/aha-wam-runs/train/he*_adapt_8x10h/` | `.../cwr_wulan_aha/aha-wam-runs/train/he*_adapt_8x10h/` |

更细的实验设定见 `infra/SUBMIT_FINETUNE_HE_ADAPT.md`。

---

## 6. 日常改完代码怎么传

```text
旧探索机  改 AHA-WAM-anygrasp  →  git push
新探索机  git pull  →  bash infra/sync_algorithm_package.sh  →  再提交
```

只 pull **不** sync：集群仍跑算法包里的旧 `.sh`，但会 `cd` 到新代码——入口和仓库可能不一致。  
只 sync **不** pull：入口是新的，仓库还是旧的。两边都要做。

---

## 7. 健康检查（新平台 pull + sync 之后）

```bash
# 仓库在 aha 下
test -d /home/ma-user/work/dataset/cwr_wulan_aha/AHA-WAM-anygrasp/configs/task/robotwin_ahawam_he8.yaml

# 算法包有入口
test -f /home/ma-user/work/algorithm/cwr_wulan2/train_finetune_he_adapt.sh
test -f /home/ma-user/work/algorithm/cwr_wulan2/cluster_paths.sh

# 权重在（二选一）
ls -lh /home/ma-user/work/dataset/cwr_wulan_aha/AHA-WAM-anygrasp/checkpoints/AHA-WAM-RoboTwin2.0/robotwin_ahawam.pt
# 或
ls -lh /home/ma-user/work/dataset/cwr_wulan_aha/checkpoints/AHA-WAM-RoboTwin2.0/robotwin_ahawam.pt
```

开训日志关键字：`revision: 2026-09-15-he-adapt-8vs32-train-eval`、`platform=wulan_aha`、`HE_ADAPT_ARM=8` 或 `32`。
