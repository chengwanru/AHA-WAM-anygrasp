# AHA-WAM RoboTwin — Experiment Settings & Results (Complete)

Date: 2026-09-07

This doc lists **every experiment directory we ran**, with **settings first**, then **results**.
Raw artifacts live on dataset disk under `aha-wam-runs/` (not in git).
Code/entrypoints: fork `chengwanru/AHA-WAM-anygrasp` branch `modified-working` / `main`.

---

## A. Shared settings (all formal evals unless noted)

| Item | Value |
|---|---|
| Model / ckpt | Official released `checkpoints/AHA-WAM-RoboTwin2.0/robotwin_ahawam.pt` + same-dir `dataset_stats.json` |
| Code | AHA-WAM-anygrasp (`modified-working`), RoboTwin via `third_party/RoboTwin` |
| Benchmark | RoboTwin 2.0 |
| Task config | `demo_randomized` |
| Hardware | NVIDIA **A800** (cluster); eager PyTorch (no TensorRT / CUDA Graph / Flash deploy) |
| Action chunk size | 16 (model fixed) |
| Denoising steps | `num_inference_steps=10` |
| Timing | `EVALUATION.timing_enabled=True`, per-episode `analysis/episode*_analysis.json` |
| Latency metrics | **L_chunk** = one action-chunk forward (`action_chunk_s_per_call`); **Freq** = 1000/L_chunk; **L_prefill** = one video prefill (`prefill_s_per_call`) |
| Not used | Paper Table-3 optimized stack (TRT+CG+Flash) — not open-sourced in eval path |

Platform entry scripts are named `train_*.sh` but **run evaluation**, not training.

---

## B. Inventory — all experiment directories

| Directory under `aha-wam-runs/` | Role | Notes |
|---|---|---|
| `robotwin_ahawam_sweep_20tasks_40eps` | **MAIN** | ah/cpp full sweep 20×6×40 (yes) |
| `robotwin_ahawam_sweep_40tasks_40eps` | **EARLY/PARTIAL** | earlier wider task list; superseded by 20-task formal sweep (yes) |
| `robotwin/video_dit_skip_phase_40eps_v2` | **MAIN** | skip_phase v2 batch1 (yes) |
| `robotwin/video_dit_skip_phase_40eps_v2_batch2` | **MAIN** | skip_phase v2 batch2 (yes) |
| `robotwin/video_dit_skip_phase_40eps` | **DEPRECATED** | skip_phase v1 (too aggressive) (yes) |
| `robotwin/video_dit_skip_phase_smoke_v2` | **SMOKE** | local/cluster smoke for v2 (yes) |
| `robotwin/video_dit_skip_approach` | **PILOT** | early skip-on-approach idea, 5ep (yes) |
| `robotwin/video_dit_skip_approach_batch2` | **PILOT** | skip_approach batch2, 5ep (yes) |
| `robotwin/video_dit_ablation` | **PILOT** | baseline vs skip_approach, 5ep (yes) |
| `robotwin/video_dit_skip_all_check` | **PILOT** | skip ALL prefills sanity check (yes) |
| `robotwin/video_dit_freq_analysis` | **PILOT** | cpp frequency analysis (yes) |
| `robotwin/ovcr_improve` | **PILOT** | OVCR baseline/off/oracle, 5ep (yes) |
| `robotwin/ovcr改进` | **PILOT** | duplicate/alias of ovcr_improve (yes) |
| `robotwin/ovcr_improve_collided_20260903_102650` | **FAILED** | collided run (yes) |
| `robotwin/ovcr改进_failed_hydra_20260903_095133` | **FAILED** | hydra failure (yes) |
| `robotwin/complex_tasks_sweep_gpu0` | **PILOT** | early complex-task ah/cpp, fewer tasks (yes) |
| `robotwin/complex_tasks_sweep_gpu1` | **PILOT** | early complex-task ah/cpp (yes) |
| `robotwin/param_sweep_place_phone_stand` | **PILOT** | place_phone_stand 5ep×5 cfg (yes) |
| `robotwin/param_sweep_missing_tasks` | **PILOT** | missing-task patch runs (yes) |
| `robotwin/batch_analysis_pilot` | **DIAG** | early batch analysis logs (yes) |
| `robotwin/batch_analysis_pilot_v2` | **DIAG** | batch analysis v2 (yes) |
| `robotwin/batch_analysis_round2` | **DIAG** | batch analysis round2 (yes) |
| `robotwin/batch_analysis_clean` | **DIAG** | cleaned batch analysis (yes) |
| `robotwin/batch_analysis_combined` | **DIAG** | combined analysis dumps (yes) |
| `robotwin/batch_analysis_turn_switch_extended` | **DIAG** | turn_switch extended logs (yes) |
| `robotwin/open_microwave_clean_diagnostic` | **DIAG** | open_microwave debug (yes) |
| `robotwin/open_microwave_tcp_diagnostic` | **DIAG** | open_microwave TCP debug (yes) |
| `robotwin/consolidated_table` | **META** | table drafts (yes) |

**MAIN** = formal conclusions. **PILOT/DIAG** = exploratory (small N or superseded). **DEPRECATED/FAILED** = do not use for claims.

---

## C. MAIN-1: ah / cpp parameter sweep

### C.1 Settings

| Item | Value |
|---|---|
| Entrypoint | `infra/train_mtp.sh` → `infra/run_robotwin_sweep.{sh,py}` |
| Output | `aha-wam-runs/robotwin_ahawam_sweep_20tasks_40eps/` |
| Episodes / job | **40** |
| GPUs | typically 8× A800, one job per GPU |
| Configs (6) | `(ah,cpp) ∈ {(64,2),(64,1),(64,3),(32,2),(32,1),(16,1)}` |
| Tasks (20) | pick_dual_bottles, pick_diverse_bottles, place_bread_basket, place_object_basket, place_a2b_left, place_a2b_right, move_can_pot, move_stapler_pad, stack_blocks_two, stack_bowls_three, handover_block, handover_mic, hanging_mug, click_bell, press_stapler, turn_switch, lift_pot, beat_block_hammer, open_laptop, open_microwave |
| Total jobs | 20 × 6 = **120** (all complete) |
| VIDEO_DIT_MODE | baseline (normal prefill schedule; no skip_phase) |
| Meaning of ah | `action_horizon`: actions covered before planner horizon rolls |
| Meaning of cpp | `chunks_per_video_prefill`: how many action chunks reuse one video prefill |

### C.2 Results — success rate

| Task | ah=64, cpp=2 | ah=64, cpp=1 | ah=64, cpp=3 | ah=32, cpp=2 | ah=32, cpp=1 | ah=16, cpp=1 |
|---|---:|---:|---:|---:|---:|---:|
| Pick Dual Bottles | 32/40 (80.0%) | 33/40 (82.5%) | 29/40 (72.5%) | 32/40 (80.0%) | 33/40 (82.5%) | 33/40 (82.5%) |
| Pick Diverse Bottles | 20/40 (50.0%) | 23/40 (57.5%) | 18/40 (45.0%) | 20/40 (50.0%) | 23/40 (57.5%) | 23/40 (57.5%) |
| Place Bread Basket | 29/40 (72.5%) | 25/40 (62.5%) | 26/40 (65.0%) | 29/40 (72.5%) | 25/40 (62.5%) | 25/40 (62.5%) |
| Place Object Basket | 8/40 (20.0%) | 2/40 (5.0%) | 6/40 (15.0%) | 8/40 (20.0%) | 2/40 (5.0%) | 2/40 (5.0%) |
| Place Object A to B Left | 9/40 (22.5%) | 4/40 (10.0%) | 7/40 (17.5%) | 9/40 (22.5%) | 4/40 (10.0%) | 4/40 (10.0%) |
| Place Object A to B Right | 3/40 (7.5%) | 3/40 (7.5%) | 4/40 (10.0%) | 3/40 (7.5%) | 3/40 (7.5%) | 3/40 (7.5%) |
| Move Can Pot | 17/40 (42.5%) | 16/40 (40.0%) | 15/40 (37.5%) | 17/40 (42.5%) | 16/40 (40.0%) | 16/40 (40.0%) |
| Move Stapler Pad | 1/40 (2.5%) | 0/40 (0.0%) | 2/40 (5.0%) | 1/40 (2.5%) | 0/40 (0.0%) | 0/40 (0.0%) |
| Stack Blocks Two | 23/40 (57.5%) | 37/40 (92.5%) | 30/40 (75.0%) | 23/40 (57.5%) | 37/40 (92.5%) | 37/40 (92.5%) |
| Stack Bowls Three | 32/40 (80.0%) | 29/40 (72.5%) | 31/40 (77.5%) | 32/40 (80.0%) | 29/40 (72.5%) | 29/40 (72.5%) |
| Handover Block | 5/40 (12.5%) | 2/40 (5.0%) | 9/40 (22.5%) | 5/40 (12.5%) | 2/40 (5.0%) | 2/40 (5.0%) |
| Handover Microphone | 11/40 (27.5%) | 7/40 (17.5%) | 8/40 (20.0%) | 11/40 (27.5%) | 7/40 (17.5%) | 7/40 (17.5%) |
| Hanging Mug | 15/40 (37.5%) | 11/40 (27.5%) | 10/40 (25.0%) | 15/40 (37.5%) | 11/40 (27.5%) | 11/40 (27.5%) |
| Click Bell | 26/40 (65.0%) | 14/40 (35.0%) | 26/40 (65.0%) | 26/40 (65.0%) | 14/40 (35.0%) | 14/40 (35.0%) |
| Press Stapler | 40/40 (100.0%) | 37/40 (92.5%) | 40/40 (100.0%) | 40/40 (100.0%) | 37/40 (92.5%) | 37/40 (92.5%) |
| Turn Switch | 21/40 (52.5%) | 18/40 (45.0%) | 20/40 (50.0%) | 21/40 (52.5%) | 18/40 (45.0%) | 18/40 (45.0%) |
| Lift Pot | 1/40 (2.5%) | 0/40 (0.0%) | 0/40 (0.0%) | 1/40 (2.5%) | 0/40 (0.0%) | 0/40 (0.0%) |
| Beat Block Hammer | 31/40 (77.5%) | 24/40 (60.0%) | 23/40 (57.5%) | 31/40 (77.5%) | 24/40 (60.0%) | 24/40 (60.0%) |
| Open Laptop | 0/40 (0.0%) | 0/40 (0.0%) | 0/40 (0.0%) | 0/40 (0.0%) | 0/40 (0.0%) | 0/40 (0.0%) |
| Open Microwave | 0/40 (0.0%) | 0/40 (0.0%) | 2/40 (5.0%) | 0/40 (0.0%) | 0/40 (0.0%) | 0/40 (0.0%) |
| **Mean (20)** | **40.5%** | **35.6%** | **38.2%** | **40.5%** | **35.6%** | **35.6%** |

**Takeaway:** on ah=64, mean success cpp1/2/3 ≈ **35.6% / 40.5% / 38.3%** — flat, no monotonic cpp trend; **cpp=2** highest mean.

### C.3 Results — L_chunk (ms / Hz)

| Task | ah=64, cpp=2 | ah=64, cpp=1 | ah=64, cpp=3 | ah=32, cpp=2 | ah=32, cpp=1 | ah=16, cpp=1 |
|---|---:|---:|---:|---:|---:|---:|
| Pick Dual Bottles | 472.0 ms (2.12 Hz) | 446.0 ms (2.24 Hz) | 471.3 ms (2.12 Hz) | 462.5 ms (2.16 Hz) | 462.6 ms (2.16 Hz) | 451.3 ms (2.22 Hz) |
| Pick Diverse Bottles | 459.4 ms (2.18 Hz) | 469.4 ms (2.13 Hz) | 453.2 ms (2.21 Hz) | 459.3 ms (2.18 Hz) | 812.0 ms (1.23 Hz) | 822.2 ms (1.22 Hz) |
| Place Bread Basket | 446.3 ms (2.24 Hz) | 449.6 ms (2.22 Hz) | 481.3 ms (2.08 Hz) | 455.2 ms (2.20 Hz) | 819.8 ms (1.22 Hz) | 814.2 ms (1.23 Hz) |
| Place Object Basket | 443.9 ms (2.25 Hz) | 460.8 ms (2.17 Hz) | 452.2 ms (2.21 Hz) | 464.3 ms (2.15 Hz) | 461.0 ms (2.17 Hz) | 459.8 ms (2.18 Hz) |
| Place Object A to B Left | 746.5 ms (1.34 Hz) | 744.8 ms (1.34 Hz) | 449.4 ms (2.23 Hz) | 442.4 ms (2.26 Hz) | 749.5 ms (1.33 Hz) | 445.6 ms (2.24 Hz) |
| Place Object A to B Right | 745.9 ms (1.34 Hz) | 457.8 ms (2.18 Hz) | 470.8 ms (2.12 Hz) | 442.3 ms (2.26 Hz) | 471.2 ms (2.12 Hz) | 449.3 ms (2.23 Hz) |
| Move Can Pot | 765.2 ms (1.31 Hz) | 482.1 ms (2.07 Hz) | 466.9 ms (2.14 Hz) | 736.2 ms (1.36 Hz) | 456.0 ms (2.19 Hz) | 451.4 ms (2.22 Hz) |
| Move Stapler Pad | 469.3 ms (2.13 Hz) | 439.5 ms (2.28 Hz) | 486.3 ms (2.06 Hz) | 739.5 ms (1.35 Hz) | 476.7 ms (2.10 Hz) | 736.0 ms (1.36 Hz) |
| Stack Blocks Two | 457.4 ms (2.19 Hz) | 449.5 ms (2.22 Hz) | 458.0 ms (2.18 Hz) | 441.7 ms (2.26 Hz) | 449.2 ms (2.23 Hz) | 738.1 ms (1.35 Hz) |
| Stack Bowls Three | 458.0 ms (2.18 Hz) | 442.3 ms (2.26 Hz) | 745.7 ms (1.34 Hz) | 456.0 ms (2.19 Hz) | 451.3 ms (2.22 Hz) | 449.9 ms (2.22 Hz) |
| Handover Block | 742.8 ms (1.35 Hz) | 446.7 ms (2.24 Hz) | 455.2 ms (2.20 Hz) | 741.0 ms (1.35 Hz) | 448.1 ms (2.23 Hz) | 444.2 ms (2.25 Hz) |
| Handover Microphone | 468.4 ms (2.13 Hz) | 747.9 ms (1.34 Hz) | 458.5 ms (2.18 Hz) | 451.3 ms (2.22 Hz) | 450.8 ms (2.22 Hz) | 749.6 ms (1.33 Hz) |
| Hanging Mug | 453.1 ms (2.21 Hz) | 451.6 ms (2.21 Hz) | 450.2 ms (2.22 Hz) | 456.1 ms (2.19 Hz) | 745.8 ms (1.34 Hz) | 457.3 ms (2.19 Hz) |
| Click Bell | 455.7 ms (2.19 Hz) | 442.2 ms (2.26 Hz) | 444.3 ms (2.25 Hz) | 495.0 ms (2.02 Hz) | 442.2 ms (2.26 Hz) | 453.2 ms (2.21 Hz) |
| Press Stapler | 448.2 ms (2.23 Hz) | 478.8 ms (2.09 Hz) | 445.3 ms (2.25 Hz) | 451.0 ms (2.22 Hz) | 475.1 ms (2.11 Hz) | 471.4 ms (2.12 Hz) |
| Turn Switch | 481.2 ms (2.08 Hz) | 446.0 ms (2.24 Hz) | 743.1 ms (1.35 Hz) | 742.9 ms (1.35 Hz) | 491.5 ms (2.03 Hz) | 455.8 ms (2.19 Hz) |
| Lift Pot | 458.6 ms (2.18 Hz) | 747.7 ms (1.34 Hz) | 443.9 ms (2.25 Hz) | 480.8 ms (2.08 Hz) | 454.7 ms (2.20 Hz) | 452.5 ms (2.21 Hz) |
| Beat Block Hammer | 442.4 ms (2.26 Hz) | 461.9 ms (2.17 Hz) | 743.1 ms (1.35 Hz) | 438.5 ms (2.28 Hz) | 454.7 ms (2.20 Hz) | 444.3 ms (2.25 Hz) |
| Open Laptop | — | — | — | — | — | — |
| Open Microwave | 460.4 ms (2.17 Hz) | 450.7 ms (2.22 Hz) | 749.6 ms (1.33 Hz) | 759.8 ms (1.32 Hz) | 441.0 ms (2.27 Hz) | 440.0 ms (2.27 Hz) |

### C.4 Results — L_prefill (ms)

| Task | ah=64, cpp=2 | ah=64, cpp=1 | ah=64, cpp=3 | ah=32, cpp=2 | ah=32, cpp=1 | ah=16, cpp=1 |
|---|---:|---:|---:|---:|---:|---:|
| Pick Dual Bottles | 101.0 | 96.4 | 100.0 | 99.0 | 101.1 | 96.7 |
| Pick Diverse Bottles | 98.4 | 100.1 | 97.3 | 99.1 | 167.8 | 176.9 |
| Place Bread Basket | 96.9 | 97.4 | 103.3 | 99.4 | 172.7 | 174.1 |
| Place Object Basket | 96.9 | 99.3 | 98.8 | 100.3 | 100.0 | 98.7 |
| Place Object A to B Left | 167.2 | 165.0 | 96.8 | 95.9 | 168.3 | 96.5 |
| Place Object A to B Right | 166.7 | 98.2 | 102.3 | 98.4 | 101.6 | 97.3 |
| Move Can Pot | 168.6 | 103.1 | 100.6 | 164.7 | 98.0 | 98.6 |
| Move Stapler Pad | 101.1 | 95.8 | 103.5 | 165.0 | 102.5 | 165.4 |
| Stack Blocks Two | 98.5 | 97.9 | 99.4 | 96.1 | 97.4 | 163.6 |
| Stack Bowls Three | 101.4 | 96.2 | 166.4 | 98.5 | 97.5 | 98.2 |
| Handover Block | 165.3 | 97.6 | 98.5 | 164.9 | 96.9 | 96.7 |
| Handover Microphone | 100.8 | 167.5 | 98.6 | 98.1 | 98.4 | 166.0 |
| Hanging Mug | 97.7 | 98.3 | 98.0 | 98.3 | 166.3 | 98.3 |
| Click Bell | 97.0 | 96.1 | 95.9 | 111.1 | 95.5 | 98.8 |
| Press Stapler | 95.9 | 101.7 | 95.6 | 96.4 | 102.0 | 103.6 |
| Turn Switch | 102.2 | 96.1 | 166.6 | 167.2 | 105.4 | 98.8 |
| Lift Pot | 100.2 | 167.5 | 96.8 | 102.1 | 99.3 | 98.8 |
| Beat Block Hammer | 95.1 | 100.2 | 165.4 | 94.7 | 99.3 | 96.7 |
| Open Laptop | — | — | — | — | — | — |
| Open Microwave | 100.7 | 98.0 | 166.7 | 169.5 | 96.4 | 97.1 |

### C.5 Results — prefill calls / episode

| Task | ah=64, cpp=2 | ah=64, cpp=1 | ah=64, cpp=3 | ah=32, cpp=2 | ah=32, cpp=1 | ah=16, cpp=1 |
|---|---:|---:|---:|---:|---:|---:|
| Pick Dual Bottles | 5.8 | 10.2 | 4.5 | 5.8 | 10.2 | 10.2 |
| Pick Diverse Bottles | 8.5 | 14.8 | 6.3 | 8.5 | 14.8 | 14.8 |
| Place Bread Basket | 11.2 | 24.9 | 8.1 | 11.2 | 24.9 | 24.9 |
| Place Object Basket | 19.9 | 43.1 | 13.5 | 19.9 | 43.1 | 43.1 |
| Place Object A to B Left | 11.2 | 23.4 | 8.0 | 11.2 | 23.4 | 23.4 |
| Place Object A to B Right | 12.4 | 23.8 | 8.4 | 12.4 | 23.8 | 23.8 |
| Move Can Pot | 9.6 | 18.8 | 6.8 | 9.6 | 18.8 | 18.8 |
| Move Stapler Pad | 12.8 | 25.0 | 8.7 | 12.8 | 25.0 | 25.0 |
| Stack Blocks Two | 16.6 | 21.6 | 9.4 | 16.6 | 21.6 | 21.6 |
| Stack Bowls Three | 19.6 | 42.7 | 13.7 | 19.6 | 42.7 | 42.7 |
| Handover Block | 23.1 | 48.6 | 14.8 | 23.1 | 48.6 | 48.6 |
| Handover Microphone | 15.9 | 34.0 | 11.4 | 15.9 | 34.0 | 34.0 |
| Hanging Mug | 22.2 | 48.3 | 16.1 | 22.2 | 48.3 | 48.3 |
| Click Bell | 6.5 | 20.3 | 4.8 | 6.5 | 20.3 | 20.3 |
| Press Stapler | 2.8 | 7.0 | 1.9 | 2.8 | 7.0 | 7.0 |
| Turn Switch | 7.6 | 16.4 | 5.5 | 7.6 | 16.4 | 16.4 |
| Lift Pot | 12.8 | 25.0 | 9.0 | 12.8 | 25.0 | 25.0 |
| Beat Block Hammer | 6.2 | 15.0 | 5.7 | 6.2 | 15.0 | 15.0 |
| Open Laptop | — | — | — | — | — | — |
| Open Microwave | 47.0 | 94.0 | 31.1 | 47.0 | 94.0 | 94.0 |
| **Mean** | **14.3** | **29.3** | **9.9** | **14.3** | **29.3** | **29.3** |

**Takeaway:** unit L_prefill ~100 ms across configs; **cpp mainly changes call count** (cpp=3 fewer refreshes).

---

## D. MAIN-2: skip_phase v2 (baseline vs skip)

### D.1 Settings

| Item | Value |
|---|---|
| Entrypoint | `infra/train_skip_phase.sh` / `train_skip_phase_batch2.sh` (NOT `train_mtp.sh`) |
| Output batch1 | `.../robotwin/video_dit_skip_phase_40eps_v2/` |
| Output batch2 | `.../robotwin/video_dit_skip_phase_40eps_v2_batch2/` |
| Episodes | **40** |
| Fixed ah / cpp | **ah=64, cpp=2** (locked; not swept) |
| Modes | `baseline` vs `VIDEO_DIT_MODE=skip_phase` |
| What is skipped | **video prefill only**; Action DiT + OVCR still every chunk |
| Near threshold | **0.10 m (10 cm)** TCP↔target |
| Distance | min Euclidean over left/right TCP × grasp/place GT actor poses |
| Max consecutive skips | **1** (≈ ≤50% when always eligible) |
| OVCR | `OVCR_DIAG_MODE=baseline` required when skipping |
| Holding latch | open→close gripper + TCP↔grasp ≤8 cm; drop if open or object >15 cm |
| Phases | REACH (not holding): skip if far from grasp; TRANSPORT (holding, far from place): skip if far; PLACE (holding+near place): never skip |
| Contact tasks | click_bell / press_stapler / turn_switch: far/near vs contact target only |
| Batch1 tasks (10) | handover_mic, hanging_mug, move_stapler_pad, place_bread_basket, place_mouse_pad, place_object_basket, put_bottles_dustbin, stack_blocks_three, stack_blocks_two, click_bell |
| Batch2 tasks (10) | pick_dual_bottles, pick_diverse_bottles, place_a2b_left, place_a2b_right, move_can_pot, stack_bowls_three, handover_block, lift_pot, press_stapler, turn_switch |
| Baseline reuse | if present, reuse ah64_cpp2 from MAIN-1 |

### D.2 Results — success + skip%

| Task | baseline | skip_phase | skip% |
|---|---:|---:|---:|
| Handover Microphone | 11/40 (27.5%) | 8/40 (20.0%) | 23.8% |
| Hanging Mug | 15/40 (37.5%) | 15/40 (37.5%) | 46.4% |
| Move Stapler Pad | 1/40 (2.5%) | 3/40 (7.5%) | 26.1% |
| Place Bread Basket | 29/40 (72.5%) | 23/40 (57.5%) | 39.6% |
| Place Mouse Pad | 19/40 (47.5%) | 8/40 (20.0%) | 24.2% |
| Place Object Basket | 8/40 (20.0%) | 7/40 (17.5%) | 52.0% |
| Stack Blocks Three | 7/40 (17.5%) | 4/40 (10.0%) | 14.4% |
| Stack Blocks Two | 23/40 (57.5%) | 22/40 (55.0%) | 18.8% |
| Click Bell | 26/40 (65.0%) | 29/40 (72.5%) | 38.4% |
| Pick Dual Bottles | 32/40 (80.0%) | 35/40 (87.5%) | 44.9% |
| Pick Diverse Bottles | 20/40 (50.0%) | 21/40 (52.5%) | 42.0% |
| Place Object A to B Left | 9/40 (22.5%) | 5/40 (12.5%) | 44.1% |
| Place Object A to B Right | 3/40 (7.5%) | 7/40 (17.5%) | 45.8% |
| Move Can Pot | 17/40 (42.5%) | 11/40 (27.5%) | 45.5% |
| Stack Bowls Three | 32/40 (80.0%) | 24/40 (60.0%) | 36.2% |
| Handover Block | 5/40 (12.5%) | 5/40 (12.5%) | 43.3% |
| Lift Pot | 1/40 (2.5%) | 0/40 (0.0%) | 46.3% |
| Press Stapler | 40/40 (100.0%) | 37/40 (92.5%) | 59.0% |
| Turn Switch | 21/40 (52.5%) | 21/40 (52.5%) | 26.9% |
| **Mean (19)** | **42.0%** | **37.5%** |  |

### D.3 Results — prefill savings (this is the speedup signal)

| Task | base L_prefill | skip L_prefill | base calls/ep | skip calls/ep | calls↓ | base prefill_s/ep | skip prefill_s/ep | time↓ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Handover Microphone | 100.2 | 101.1 | 15.9 | 13.2 | 16.9% | 1.60s | 1.34s | 16.3% |
| Hanging Mug | 101.8 | 104.2 | 22.2 | 13.0 | 41.4% | 2.26s | 1.36s | 40.0% |
| Move Stapler Pad | 100.0 | 104.1 | 12.8 | 9.6 | 25.4% | 1.28s | 0.99s | 22.3% |
| Place Bread Basket | 96.9 | 98.6 | 11.2 | 9.1 | 18.9% | 1.09s | 0.90s | 17.2% |
| Place Mouse Pad | 100.2 | 98.1 | 9.2 | 9.2 | -0.5% | 0.92s | 0.91s | 1.5% |
| Place Object Basket | 96.9 | 103.1 | 19.9 | 10.2 | 48.5% | 1.92s | 1.05s | 45.2% |
| Stack Blocks Three | 100.3 | 97.5 | 34.0 | 30.9 | 9.1% | 3.42s | 3.01s | 11.8% |
| Stack Blocks Two | 99.9 | 96.8 | 16.6 | 14.4 | 13.3% | 1.66s | 1.39s | 16.1% |
| Click Bell | 101.4 | 101.6 | 6.5 | 4.7 | 27.0% | 0.66s | 0.48s | 26.5% |
| Pick Dual Bottles | 101.0 | 95.9 | 5.8 | 3.7 | 35.8% | 0.59s | 0.36s | 39.0% |
| Pick Diverse Bottles | 98.4 | 101.4 | 8.5 | 5.6 | 33.8% | 0.84s | 0.57s | 31.9% |
| Place Object A to B Left | 167.2 | 99.8 | 11.2 | 7.2 | 35.4% | 1.88s | 0.72s | 61.5% |
| Place Object A to B Right | 166.7 | 98.2 | 12.4 | 6.9 | 44.4% | 2.07s | 0.67s | 67.4% |
| Move Can Pot | 96.1 | 99.3 | 9.6 | 6.4 | 33.6% | 0.93s | 0.63s | 31.5% |
| Stack Bowls Three | 104.1 | 102.6 | 19.6 | 17.2 | 12.2% | 2.04s | 1.77s | 13.4% |
| Handover Block | 100.9 | 100.1 | 23.1 | 13.5 | 41.4% | 2.33s | 1.35s | 41.9% |
| Lift Pot | 99.0 | 100.6 | 12.8 | 7.5 | 41.7% | 1.26s | 0.75s | 40.7% |
| Press Stapler | 98.0 | 100.2 | 2.8 | 2.3 | 17.0% | 0.28s | 0.23s | 15.1% |
| Turn Switch | 99.3 | 95.5 | 7.6 | 7.1 | 6.6% | 0.75s | 0.68s | 9.7% |
| **Mean** |  |  |  |  | **26.4%** |  |  | **28.9%** |

### D.4 Results — L_chunk (should stay ~flat)

| Task | baseline L_chunk | skip_phase L_chunk |
|---|---:|---:|
| Handover Microphone | 462.0 ms (2.16 Hz) | 468.1 ms (2.14 Hz) |
| Hanging Mug | 476.6 ms (2.10 Hz) | 488.2 ms (2.05 Hz) |
| Move Stapler Pad | 458.7 ms (2.18 Hz) | 490.3 ms (2.04 Hz) |
| Place Bread Basket | 446.3 ms (2.24 Hz) | 453.4 ms (2.21 Hz) |
| Place Mouse Pad | 462.0 ms (2.16 Hz) | 453.8 ms (2.20 Hz) |
| Place Object Basket | 443.9 ms (2.25 Hz) | 481.6 ms (2.08 Hz) |
| Stack Blocks Three | 464.9 ms (2.15 Hz) | 447.0 ms (2.24 Hz) |
| Stack Blocks Two | 463.8 ms (2.16 Hz) | 446.3 ms (2.24 Hz) |
| Click Bell | 472.5 ms (2.12 Hz) | 479.5 ms (2.09 Hz) |
| Pick Dual Bottles | 472.0 ms (2.12 Hz) | 444.5 ms (2.25 Hz) |
| Pick Diverse Bottles | 459.4 ms (2.18 Hz) | 472.2 ms (2.12 Hz) |
| Place Object A to B Left | 746.5 ms (1.34 Hz) | 462.2 ms (2.16 Hz) |
| Place Object A to B Right | 745.9 ms (1.34 Hz) | 449.6 ms (2.22 Hz) |
| Move Can Pot | 441.7 ms (2.26 Hz) | 456.1 ms (2.19 Hz) |
| Stack Bowls Three | 488.1 ms (2.05 Hz) | 480.1 ms (2.08 Hz) |
| Handover Block | 472.0 ms (2.12 Hz) | 462.3 ms (2.16 Hz) |
| Lift Pot | 462.6 ms (2.16 Hz) | 461.5 ms (2.17 Hz) |
| Press Stapler | 461.8 ms (2.17 Hz) | 469.3 ms (2.13 Hz) |
| Turn Switch | 456.1 ms (2.19 Hz) | 442.1 ms (2.26 Hz) |

**Takeaway:** skip does not reduce L_chunk (action path unchanged). Report **prefill calls / prefill_s/ep**.

---

## E. DEPRECATED: skip_phase v1

### E.1 Settings

| Item | Value |
|---|---|
| Output | `robotwin/video_dit_skip_phase_40eps/` |
| ah/cpp | 64 / 2 |
| Near thresh | **0.06 m** |
| Consecutive skip budget | **none** (could skip almost every eligible step) |

### E.2 Results (partial; do not use)

| Task | mode | success |
|---|---|---|
| place_bread_basket | baseline | 29/40 (72.5%) |
| place_object_basket | baseline | 8/40 (20.0%) |
| click_bell | baseline | 26/40 (65.0%) |
| place_mouse_pad | baseline | 19/40 (47.5%) |
| click_bell | skip_phase | 11/40 (27.5%) |
| place_mouse_pad | skip_phase | 0/40 (0.0%) |
| move_stapler_pad | baseline | 1/40 (2.5%) |
| move_stapler_pad | skip_phase | 0/40 (0.0%) |

Success collapsed when skip%~90% (e.g. Place Mouse Pad ~47.5%→0%). Replaced by v2.

---

## F. PILOT / DIAG experiments (settings + compact results)

These are **not** the primary claim set (often 5 episodes). Included so the log of what we ran is complete.

### F.1 OVCR diag (`ovcr_improve` / `ovcr改进`)

**Settings:** ah=64, cpp=2, **5 ep**, modes `baseline` / `off` / `oracle` (`OVCR_DIAG_MODE`). Tasks: move_can_pot, place_fan, place_phone_stand, stack_bowls_two/three, turn_switch.

| Task | baseline | off | oracle |
|---|---:|---:|---:|
| move_can_pot | 2/5 | 1/5 | 1/5 |
| place_fan | 5/5 | 5/5 | 0/5 |
| place_phone_stand | 0/5 | 0/5 | 0/5 |
| stack_bowls_three | 3/5 | 4/5 | 1/5 |
| stack_bowls_two | 5/5 | 5/5 | 3/5 |
| turn_switch | 4/5 | 4/5 | 3/5 |

Failed siblings: `ovcr_improve_collided_*`, `ovcr改进_failed_hydra_*`.

### F.2 Early `skip_approach` / ablation (5 ep)

**Settings:** precursor to skip_phase; skip prefills during approach heuristically. Dirs: `video_dit_ablation`, `video_dit_skip_approach`, `video_dit_skip_approach_batch2`.

**`robotwin/video_dit_ablation/summary.json`**

| Task | baseline | skip_approach |
|---|---:|---:|
| open_laptop | 4/5 | 4/5 |
| pick_diverse_bottles | 5/5 | 5/5 |
| pick_dual_bottles | 0/5 | 5/5 |
| place_fan | 5/5 | 3/5 |
| stack_blocks_three | 2/5 | 2/5 |
| stack_blocks_two | 2/5 | 2/5 |
| stack_bowls_three | 5/5 | 5/5 |

**`robotwin/video_dit_skip_approach_batch2/summary.json`**

| Task | baseline | skip_approach |
|---|---:|---:|
| handover_mic | 1/5 | 2/5 |
| hanging_mug | 2/5 | 3/5 |
| move_stapler_pad | 0/5 | 0/5 |
| place_bread_basket | 4/5 | 4/5 |
| place_mouse_pad | 2/5 | 0/5 |
| place_object_basket | 0/5 | 1/5 |
| put_bottles_dustbin | 1/5 | 1/5 |
| stack_blocks_three | 2/5 | 0/5 |
| stack_blocks_two | 2/5 | 4/5 |

### F.3 `skip_all` check

**Settings:** skip every video prefill (stress test). Dir: `video_dit_skip_all_check`.

- stack_bowls_three baseline: 3/5
- stack_bowls_three skip_all: 0/5

### F.4 Early complex-task ah/cpp pilots

**Settings:** subset of tasks × ah/cpp (no ah64_cpp3 yet), dirs `complex_tasks_sweep_gpu0/1`. Superseded by MAIN-1.

**`robotwin/complex_tasks_sweep_gpu0/summary.json`** (15 jobs)

| Task | tag | success |
|---|---|---|
| open_laptop | ah64_cpp2 | 4/5 |
| open_laptop | ah64_cpp1 | 4/5 |
| open_laptop | ah32_cpp2 | 5/5 |
| open_laptop | ah32_cpp1 | 4/5 |
| open_laptop | ah16_cpp1 | 4/5 |
| pick_diverse_bottles | ah64_cpp2 | 5/5 |
| pick_diverse_bottles | ah64_cpp1 | 5/5 |
| pick_diverse_bottles | ah32_cpp2 | 5/5 |
| pick_diverse_bottles | ah32_cpp1 | 5/5 |
| pick_diverse_bottles | ah16_cpp1 | 5/5 |
| pick_dual_bottles | ah64_cpp2 | 5/5 |
| pick_dual_bottles | ah64_cpp1 | 5/5 |
| pick_dual_bottles | ah32_cpp2 | 4/5 |
| pick_dual_bottles | ah32_cpp1 | 5/5 |
| pick_dual_bottles | ah16_cpp1 | 5/5 |

**`robotwin/complex_tasks_sweep_gpu1/summary.json`** (15 jobs)

| Task | tag | success |
|---|---|---|
| stack_bowls_two | ah64_cpp2 | 4/5 |
| stack_bowls_two | ah64_cpp1 | 5/5 |
| stack_bowls_two | ah32_cpp2 | 5/5 |
| stack_bowls_two | ah32_cpp1 | 5/5 |
| stack_bowls_two | ah16_cpp1 | 5/5 |
| stack_bowls_three | ah64_cpp2 | 2/5 |
| stack_bowls_three | ah64_cpp1 | 4/5 |
| stack_bowls_three | ah32_cpp2 | 4/5 |
| stack_bowls_three | ah32_cpp1 | 5/5 |
| stack_bowls_three | ah16_cpp1 | 5/5 |
| place_fan | ah64_cpp2 | 4/5 |
| place_fan | ah64_cpp1 | 1/5 |
| place_fan | ah32_cpp2 | 4/5 |
| place_fan | ah32_cpp1 | 1/5 |
| place_fan | ah16_cpp1 | 1/5 |

### F.5 `param_sweep_place_phone_stand` (5 ep)

**Settings:** single task place_phone_stand × 5 ah/cpp configs, 5 episodes.

| Config | success |
|---|---|
| ah64_cpp2 | 4/5 |
| ah64_cpp1 | 2/5 |
| ah32_cpp2 | 4/5 |
| ah32_cpp1 | 2/5 |
| ah16_cpp1 | 2/5 |

### F.6 Batch analysis / open_microwave diagnostics

**Settings:** logging / failure analysis dumps, not formal success tables.

| Dir | Purpose |
|---|---|
| `robotwin/batch_analysis_pilot` | early multi-task analysis |
| `robotwin/batch_analysis_pilot_v2` | expanded pilot |
| `robotwin/batch_analysis_round2` | round 2 |
| `robotwin/batch_analysis_clean` | cleaned |
| `robotwin/batch_analysis_combined` | combined dumps |
| `robotwin/batch_analysis_turn_switch_extended` | turn_switch extended |
| `robotwin/open_microwave_clean_diagnostic` | microwave debug |
| `robotwin/open_microwave_tcp_diagnostic` | TCP/grasp debug |
| `robotwin/param_sweep_missing_tasks` | gap fills |
| `robotwin/video_dit_freq_analysis` | prefill frequency probe |
| `robotwin/consolidated_table` | table scratch |

### F.7 Early `robotwin_ahawam_sweep_40tasks_40eps`

**Settings:** earlier attempt at broader task list (5 configs, no ah64_cpp3 in some layouts). **Formal MAIN-1 is the 20-task × 6-config sweep**; treat 40-task tree as historical/partial.

---

## G. Latency reference (ours vs paper)

| Metric | Our A800 eager (measured) | Paper eager 5090D | Paper optimized |
|---|---|---|---|
| L_chunk | ~450–500 ms (~2 Hz) | ~415.8 ms | 41.4 ms / Flash 17.6 ms |
| L_prefill | ~100 ms typical | ~61.2 ms | lower with compile/TRT |

Numbers in MAIN tables are **our measurements**, not paper copies.

---

## H. Pointers

- Agent eval onboarding: `AGENT_ONBOARDING.md`
- Real training submit guide: `AGENT_TRAINING.md`
- skip_phase notes: `infra/docs_skip_phase.md`
- Entry: `infra/train_mtp.sh`, `infra/train_skip_phase.sh`

