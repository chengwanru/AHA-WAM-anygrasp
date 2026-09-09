# skip_phase v2 artifacts (slim)

Pushed metrics for **baseline** and **skip_phase** (v2 batch1 + batch2).

## Size note (original on dataset disk)

| Tree | Total | mp4 videos | analysis json | slim (this folder) |
|---|---:|---:|---:|---:|
| `video_dit_skip_phase_40eps_v2` | ~1.8G | ~1.4G | ~495M | included |
| `video_dit_skip_phase_40eps_v2_batch2` | ~1.0G | ~714M | ~300M | included |
| **Together** | **~2.8G** | **~2.1G** | **~800M** | **see `du` below** |

**Not pushed (too large for git):** episode `.mp4`, full `analysis/episode*_analysis.json`, bulky logs.

**Pushed:** `summary.json/csv`, per-task `job_report.*`, `episodes_summary.csv`, `_result_random.txt`, `metrics_compact.json`, and `comparison_table.csv`.

Settings: ah=64, cpp=2, 40 eps, near=0.10m, max consecutive skips=1. See `EXPERIMENT_RESULTS_SUMMARY.md` §D.

Source path: `/home/ma-user/work/dataset/cwr_wulan_aha/aha-wam-runs/robotwin/video_dit_skip_phase_40eps_v2*`
