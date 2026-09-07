# skip_phase vs baseline

入口：`train_skip_phase.sh`（不要用 `train_mtp.sh`）。

- 固定 ah64 / cpp2，40 ep
- 输出：`.../robotwin/video_dit_skip_phase_40eps/{baseline,skip_phase}/<task>/`
- 自动复用已有 `ah64_cpp2` 的 40ep baseline（目前 place_bread_basket、place_object_basket）
- 单 job 超时默认 18h（`JOB_TIMEOUT_S=64800`）

## 落盘分析
- `analysis/episode*_analysis.json`：timing + **latency**（含 Hz）+ skip_phase_summary + step_log
- `job_report.json` / `.txt` / `episodes_summary.csv`：job 级聚合
- `summary.json` / `summary.csv`：sweep 总表
