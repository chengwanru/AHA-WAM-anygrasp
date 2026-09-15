# OFFSET EXPERIMENT（旁路，已与 SKIP_V2 分开）

本目录下与 **OFFSET** 相关的入口/配置：

| 文件 | 角色 |
|---|---|
| `algorithm/.../train_finetune_offset.sh` | OFFSET 训练入口 |
| `algorithm/.../SUBMIT_FINETUNE_OFFSET.md` | OFFSET 提交说明 |
| `configs/task/robotwin_ahawam_offset.yaml` | OFFSET Hydra task（`max_action_offset>0`） |
| 输出 | `.../aha-wam-runs/train/offset_ft_*` |

**主线现为 SKIP_V2**：见 `train_finetune_skip_v2.sh` / `SUBMIT_FINETUNE_SKIP_V2.md` / `robotwin_ahawam_skip_v2.yaml`。

两实验互斥：同一 run 不得同时开 `action_offset` 与 `skip_phase_v2`。
