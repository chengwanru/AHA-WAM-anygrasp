#!/usr/bin/env python3
"""baseline vs skip_phase sweep (fixed ah64 / cpp2, 40 eps).

Does NOT touch the ah/cpp parameter sweep. Always uses:
  action_horizon=64, chunks_per_video_prefill=2
  VIDEO_DIT_MODE = baseline | skip_phase

Output (separate from robotwin_ahawam_sweep_*):
  .../aha-wam-runs/robotwin/video_dit_skip_phase_40eps/{baseline,skip_phase}/<task>/
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(
    os.environ.get(
        "AHA_WAM_CODE_DIR",
        "/opt/huawei/dataset/cwr_dataset_wulann/AHA-WAM-anygrasp"
        if Path("/opt/huawei/dataset/cwr_dataset_wulann/AHA-WAM-anygrasp").exists()
        else "/home/ma-user/work/dataset/cwr_dataset_wulann/AHA-WAM-anygrasp",
    )
).resolve()
EVAL_ENTRY = PROJECT_ROOT / "experiments" / "robotwin" / "eval_robotwin_single.py"
PYTHON = os.environ.get("PYTHON", sys.executable)

# batch1: original 9 batch2-style tasks + click_bell (contact).
DEFAULT_TASKS_BATCH1 = [
    "handover_mic",
    "hanging_mug",
    "move_stapler_pad",
    "place_bread_basket",
    "place_mouse_pad",
    "place_object_basket",
    "put_bottles_dustbin",
    "stack_blocks_three",
    "stack_blocks_two",
    "click_bell",
]

# batch2: 10 tasks from ah/cpp sweep, different types from batch1.
DEFAULT_TASKS_BATCH2 = [
    "pick_dual_bottles",
    "pick_diverse_bottles",
    "place_a2b_left",
    "place_a2b_right",
    "move_can_pot",
    "stack_bowls_three",
    "handover_block",
    "lift_pot",
    "press_stapler",
    "turn_switch",
]

DEFAULT_TASKS = DEFAULT_TASKS_BATCH1

# Locked to match prior skip_approach / baseline comparison (NOT the ah/cpp sweep).
FIXED_ACTION_HORIZON = 64
FIXED_CPP = 2
DEFAULT_MODES = ["baseline", "skip_phase"]

# Approximate wall hours @ ah64_cpp2 × 40ep (from batch2×8 + measured).
# Used only for shortest-first scheduling so early complete task pairs show up ASAP.
TASK_ETA_HOURS = {
    # batch1
    "place_bread_basket": 3.0,
    "place_mouse_pad": 3.2,
    "click_bell": 3.3,
    "move_stapler_pad": 3.6,
    "stack_blocks_two": 5.0,
    "handover_mic": 4.8,
    "hanging_mug": 5.5,
    "place_object_basket": 6.5,
    "stack_blocks_three": 9.0,
    "put_bottles_dustbin": 12.5,
    # batch2
    "pick_dual_bottles": 2.5,
    "pick_diverse_bottles": 2.8,
    "place_a2b_left": 2.8,
    "place_a2b_right": 2.8,
    "move_can_pot": 3.5,
    "stack_bowls_three": 6.0,
    "handover_block": 4.5,
    "lift_pot": 3.0,
    "press_stapler": 2.5,
    "turn_switch": 2.5,
}

DEFAULT_OUTPUT = Path(
    os.environ.get(
        "SKIP_PHASE_OUTPUT_DIR",
        "/opt/huawei/dataset/cwr_dataset_wulann/aha-wam-runs/robotwin/video_dit_skip_phase_40eps_v2"
        if Path("/opt/huawei/dataset").exists()
        else "/home/ma-user/work/dataset/cwr_dataset_wulann/aha-wam-runs/robotwin/video_dit_skip_phase_40eps_v2",
    )
)

# Prior ah64_cpp2 × 40ep runs (default VIDEO_DIT=baseline) that can be reused.
# Prefer both platform and explore mounts — /opt may exist but lack run dirs.
BASELINE_REUSE_ROOTS = []
for _base in (
    Path("/opt/huawei/dataset/cwr_dataset_wulann"),
    Path("/home/ma-user/work/dataset/cwr_dataset_wulann"),
):
    for _sweep in (
        "aha-wam-runs/robotwin_ahawam_sweep_20tasks_40eps/ah64_cpp2",
        "aha-wam-runs/robotwin_ahawam_sweep_40tasks_40eps/ah64_cpp2",
    ):
        root = _base / _sweep
        if root.is_dir() and root not in BASELINE_REUSE_ROOTS:
            BASELINE_REUSE_ROOTS.append(root)


def log(msg: str, master_log: Path) -> None:
    line = time.strftime("%Y-%m-%d %H:%M:%S") + " " + msg
    print(line, flush=True)
    with open(master_log, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def job_complete(task_dir: Path, num_episodes: int) -> bool:
    an = task_dir / "analysis"
    if an.is_dir() and len(list(an.glob("episode*_analysis.json"))) >= int(num_episodes):
        return True
    return (task_dir / "_result_random.txt").exists()


def find_reusable_baseline(task: str, num_episodes: int):
    for root in BASELINE_REUSE_ROOTS:
        cand = root / task
        if job_complete(cand, num_episodes):
            return cand
    return None


def link_or_copy_baseline(src: Path, dst: Path) -> None:
    """Materialize prior baseline into skip_phase output tree for unified analysis."""
    import shutil

    dst.mkdir(parents=True, exist_ok=True)
    # Prefer symlinks for large analysis/video trees; fall back to copy.
    for name in ("analysis", "_result_random.txt", "job_report.json", "job_report.txt", "episodes_summary.csv"):
        s = src / name
        d = dst / name
        if not s.exists():
            continue
        if d.exists() or d.is_symlink():
            continue
        try:
            os.symlink(s, d)
        except OSError:
            if s.is_dir():
                shutil.copytree(s, d, dirs_exist_ok=True)
            else:
                shutil.copy2(s, d)
    # Also link a few eval logs if present
    for p in src.glob("eval_*.log"):
        d = dst / p.name
        if d.exists() or d.is_symlink():
            continue
        try:
            os.symlink(p, d)
        except OSError:
            pass
    meta = {
        "reused_from": str(src),
        "action_horizon": FIXED_ACTION_HORIZON,
        "chunks_per_video_prefill": FIXED_CPP,
        "video_dit_mode": "baseline",
        "note": "Reused ah64_cpp2 40ep run (default baseline video DiT).",
    }
    (dst / "reuse_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")


def parse_rate(task_dir: Path, num_episodes: int) -> tuple[float, int]:
    rf = task_dir / "_result_random.txt"
    if rf.exists():
        try:
            rate = float(rf.read_text(encoding="utf-8").strip().splitlines()[-1].strip())
            return rate, int(round(rate * num_episodes))
        except Exception:
            pass
    files = list((task_dir / "analysis").glob("episode*_analysis.json")) if (task_dir / "analysis").is_dir() else []
    if not files:
        return 0.0, 0
    succ = 0
    for f in files:
        try:
            if json.loads(f.read_text(encoding="utf-8")).get("success"):
                succ += 1
        except Exception:
            pass
    return (succ / len(files)), succ


def _mean(xs):
    xs = [x for x in xs if isinstance(x, (int, float))]
    return (sum(xs) / len(xs)) if xs else None


def write_job_report(task_dir: Path, meta: dict) -> dict:
    """Aggregate analysis/*.json into job_report.json with latency / phase stats."""
    an = task_dir / "analysis"
    episodes = []
    if an.is_dir():
        for f in sorted(an.glob("episode*_analysis.json")):
            try:
                episodes.append(json.loads(f.read_text(encoding="utf-8")))
            except Exception:
                continue

    latency_keys = [
        "infer_s",
        "sim_s",
        "prefill_s",
        "action_chunk_s",
        "infer_per_action_s",
        "infer_s_per_call",
        "prefill_s_per_call",
        "action_chunk_s_per_call",
        "action_hz_policy_only",
        "action_hz_infer_plus_sim",
        "chunk_hz",
        "skipped_prefills",
        "prefill_decisions",
        "skipped_prefill_ratio",
        "infer_calls",
        "prefill_calls",
    ]
    lat_means = {}
    for k in latency_keys:
        vals = []
        for ep in episodes:
            lat = ep.get("latency") or {}
            timing = ep.get("timing") or {}
            v = lat.get(k)
            if v is None:
                v = timing.get(k)
            if isinstance(v, (int, float)):
                vals.append(float(v))
        if vals:
            lat_means[k] = {
                "mean": _mean(vals),
                "min": min(vals),
                "max": max(vals),
                "n": len(vals),
            }

    phase_totals = {}
    skip_true = 0
    skip_n = 0
    for ep in episodes:
        sp = ep.get("skip_phase_summary") or {}
        for ph, c in (sp.get("phase_step_counts") or {}).items():
            phase_totals[ph] = phase_totals.get(ph, 0) + int(c)
        skip_true += int(sp.get("skip_true") or 0)
        skip_n += int(sp.get("skip_decisions") or 0)
        # Also scan step_log if summary missing
        if not sp:
            for s in ep.get("step_log") or []:
                info = s.get("skip_phase") if isinstance(s, dict) else None
                if not isinstance(info, dict):
                    continue
                ph = str(info.get("phase", "UNKNOWN"))
                phase_totals[ph] = phase_totals.get(ph, 0) + 1
                if "skip" in info:
                    skip_n += 1
                    if info.get("skip"):
                        skip_true += 1

    succ = sum(1 for e in episodes if e.get("success"))
    report = {
        **meta,
        "num_analysis_files": len(episodes),
        "num_success": succ,
        "success_rate": (succ / len(episodes)) if episodes else meta.get("success_rate", 0.0),
        "latency_means": lat_means,
        "skip_phase_aggregate": {
            "phase_step_counts": phase_totals,
            "skip_decisions": skip_n,
            "skip_true": skip_true,
            "skip_ratio": (skip_true / skip_n) if skip_n else None,
        },
        "episodes_brief": [
            {
                "episode_idx": e.get("episode_idx"),
                "seed": e.get("seed"),
                "success": e.get("success"),
                "num_steps": e.get("num_steps"),
                "infer_per_action_s": (e.get("latency") or {}).get("infer_per_action_s")
                or ((e.get("timing") or {}).get("infer_s", 0) / max(e.get("num_steps") or 1, 1)),
                "skipped_prefill_ratio": (e.get("latency") or {}).get("skipped_prefill_ratio")
                or (e.get("timing") or {}).get("skipped_prefill_ratio"),
                "video_dit_mode": e.get("video_dit_mode"),
            }
            for e in episodes
        ],
    }
    (task_dir / "job_report.json").write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")

    # Human-readable companion
    lines = [
        f"=== JOB REPORT {meta.get('mode')}/{meta.get('task_name')} ===",
        f"success: {succ}/{len(episodes)} ({100 * report['success_rate']:.1f}%)",
        f"ah={meta.get('action_horizon')} cpp={meta.get('chunks_per_video_prefill')} mode={meta.get('mode')}",
    ]
    if lat_means.get("infer_per_action_s"):
        lines.append(
            f"infer_per_action_s mean={lat_means['infer_per_action_s']['mean']*1000:.2f}ms"
        )
    if lat_means.get("infer_s_per_call"):
        lines.append(f"infer_s_per_call mean={lat_means['infer_s_per_call']['mean']*1000:.1f}ms")
    if lat_means.get("action_hz_policy_only"):
        lines.append(f"action_hz_policy_only mean={lat_means['action_hz_policy_only']['mean']:.2f}")
    if lat_means.get("skipped_prefill_ratio"):
        lines.append(
            f"skipped_prefill_ratio mean={lat_means['skipped_prefill_ratio']['mean']:.3f}"
        )
    lines.append(f"skip_phase phases={phase_totals} skip_ratio={report['skip_phase_aggregate']['skip_ratio']}")
    (task_dir / "job_report.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")

    # CSV of per-episode latency
    csv_path = task_dir / "episodes_summary.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "episode_idx",
                "seed",
                "success",
                "num_steps",
                "infer_s",
                "sim_s",
                "prefill_s",
                "infer_per_action_s",
                "infer_s_per_call",
                "action_hz_policy_only",
                "action_hz_infer_plus_sim",
                "skipped_prefill_ratio",
            ],
        )
        w.writeheader()
        for e in episodes:
            lat = e.get("latency") or {}
            t = e.get("timing") or {}
            w.writerow(
                {
                    "episode_idx": e.get("episode_idx"),
                    "seed": e.get("seed"),
                    "success": e.get("success"),
                    "num_steps": e.get("num_steps"),
                    "infer_s": lat.get("infer_s", t.get("infer_s")),
                    "sim_s": lat.get("sim_s", t.get("sim_s")),
                    "prefill_s": lat.get("prefill_s", t.get("prefill_s")),
                    "infer_per_action_s": lat.get("infer_per_action_s"),
                    "infer_s_per_call": lat.get("infer_s_per_call", t.get("infer_s_per_call")),
                    "action_hz_policy_only": lat.get("action_hz_policy_only"),
                    "action_hz_infer_plus_sim": lat.get("action_hz_infer_plus_sim"),
                    "skipped_prefill_ratio": lat.get(
                        "skipped_prefill_ratio", t.get("skipped_prefill_ratio")
                    ),
                }
            )
    return report


def gpu_busy(gpu_id: int) -> bool:
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader"],
            text=True,
        )
    except Exception:
        return False
    # Conservative: if any compute app on this index via env of PIDs
    for pid_dir in Path("/proc").glob("[0-9]*"):
        try:
            cmdline = (pid_dir / "cmdline").read_bytes()
            if b"eval_robotwin_single.py" not in cmdline and b"eval_policy.py" not in cmdline:
                continue
            environ = (pid_dir / "environ").read_bytes()
            for item in environ.split(b"\0"):
                if item.startswith(b"CUDA_VISIBLE_DEVICES="):
                    val = item.split(b"=", 1)[1].decode("utf-8", errors="ignore")
                    if val == str(gpu_id):
                        return True
        except Exception:
            continue
    return False


def write_summary(results: list, output_dir: Path) -> None:
    (output_dir / "summary.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    with open(output_dir / "summary.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "task_name",
                "mode",
                "action_horizon",
                "chunks_per_video_prefill",
                "num_episodes",
                "num_success",
                "success_rate",
                "wall_time",
                "returncode",
                "skipped_existing",
                "gpu",
            ],
        )
        w.writeheader()
        for r in results:
            w.writerow({k: r.get(k) for k in w.fieldnames})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", nargs="+", default=DEFAULT_TASKS)
    parser.add_argument("--modes", nargs="+", default=DEFAULT_MODES)
    parser.add_argument("--num_episodes", type=int, default=40)
    parser.add_argument("--num_gpus", type=int, default=int(os.environ.get("MA_NUM_GPUS", "8")))
    parser.add_argument("--output_dir", type=str, default=str(DEFAULT_OUTPUT))
    parser.add_argument("--timeout_s", type=int, default=int(os.environ.get("JOB_TIMEOUT_S", "64800")))
    parser.add_argument("--kill_signal_timeout_s", type=int, default=60)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    master_log = output_dir / "sweep_master.log"

    cfg = {
        "tasks": list(args.tasks),
        "modes": list(args.modes),
        "num_episodes": args.num_episodes,
        "action_horizon": FIXED_ACTION_HORIZON,
        "chunks_per_video_prefill": FIXED_CPP,
        "num_gpus": args.num_gpus,
        "baseline_reuse_roots": [str(p) for p in BASELINE_REUSE_ROOTS],
        "skip_phase_policy": "v2",
        "skip_phase_batch": os.environ.get("SKIP_PHASE_BATCH", "batch1"),
        "skip_phase_near_thresh_m": float(os.environ.get("SKIP_PHASE_NEAR_THRESH_M", "0.10")),
        "skip_phase_max_consecutive_skips": int(
            os.environ.get("SKIP_PHASE_MAX_CONSECUTIVE_SKIPS", "1")
        ),
        "ovcr_diag_mode": os.environ.get("OVCR_DIAG_MODE", "baseline"),
        "note": "ah/cpp locked; v2=far eligible + consec skip<=1 + OVCR baseline; reuses ah64_cpp2 baselines when available",
    }
    (output_dir / "sweep_config.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")

    log("=" * 80, master_log)
    log("Starting skip_phase vs baseline sweep", master_log)
    log(f"Python: {PYTHON}", master_log)
    log(f"PROJECT_ROOT: {PROJECT_ROOT}", master_log)
    log(
        f"FIXED ah={FIXED_ACTION_HORIZON} cpp={FIXED_CPP} | episodes={args.num_episodes} | "
        f"tasks={len(args.tasks)} modes={args.modes} | "
        f"skip_policy=v2 near={os.environ.get('SKIP_PHASE_NEAR_THRESH_M', '0.10')} "
        f"max_consec={os.environ.get('SKIP_PHASE_MAX_CONSECUTIVE_SKIPS', '1')} "
        f"OVCR={os.environ.get('OVCR_DIAG_MODE', 'baseline')}",
        master_log,
    )
    log(f"Output: {output_dir}", master_log)
    log("=" * 80, master_log)

    results = []
    remaining = []
    for mode in args.modes:
        for task in args.tasks:
            task_dir = output_dir / mode / task
            reused_from = None
            if mode == "baseline" and not job_complete(task_dir, args.num_episodes):
                src = find_reusable_baseline(task, args.num_episodes)
                if src is not None:
                    link_or_copy_baseline(src, task_dir)
                    reused_from = str(src)
                    log(f"[REUSE] baseline/{task} <- {src}", master_log)

            if job_complete(task_dir, args.num_episodes):
                rate, succ = parse_rate(task_dir, args.num_episodes)
                meta = {
                    "task_name": task,
                    "mode": mode,
                    "action_horizon": FIXED_ACTION_HORIZON,
                    "chunks_per_video_prefill": FIXED_CPP,
                    "num_episodes": args.num_episodes,
                    "num_success": succ,
                    "success_rate": rate,
                    "wall_time": 0.0,
                    "returncode": 0,
                    "skipped_existing": True,
                    "reused_from": reused_from,
                    "gpu": -1,
                }
                try:
                    write_job_report(task_dir, meta)
                except Exception as e:
                    log(f"[WARN] job_report failed for {mode}/{task}: {e}", master_log)
                results.append(meta)
                log(
                    f"[SKIP] {mode}/{task}: already complete "
                    f"{succ}/{args.num_episodes} ({100 * rate:.1f}%)"
                    + (f" (reused)" if reused_from else ""),
                    master_log,
                )
            else:
                remaining.append((mode, task))

    # Shortest-first + prefer finishing pairs when baseline already done.
    # Goal: by tomorrow morning, several full (baseline, skip_phase) task pairs exist.
    done_keys = {(r["mode"], r["task_name"]) for r in results}

    def _sched_key(item):
        mode, task = item
        eta = float(TASK_ETA_HOURS.get(task, 6.0))
        # If the other mode is already complete, boost this job (finish the pair).
        other = "skip_phase" if mode == "baseline" else "baseline"
        pair_boost = -100.0 if (other, task) in done_keys else 0.0
        # Slightly prefer baseline before skip_phase for a fresh task (stable compare).
        mode_order = 0 if mode == "baseline" else 1
        return (eta + pair_boost, eta, mode_order, task)

    remaining.sort(key=_sched_key)
    log(
        "Schedule (shortest-first; pair-complete boosted): "
        + ", ".join(f"{m}/{t}" for m, t in remaining[:12])
        + (" ..." if len(remaining) > 12 else ""),
        master_log,
    )

    write_summary(results, output_dir)
    log(f"Queue: {len(remaining)} to run, {len(results)} skipped/reused", master_log)

    running = {}

    def finalize(gpu_id, proc, mode, task, start, log_file, log_fh, rc_override=None):
        if log_fh is not None and not log_fh.closed:
            try:
                log_fh.flush()
                log_fh.close()
            except Exception:
                pass
        try:
            proc.wait(timeout=max(1, args.kill_signal_timeout_s))
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
        wall = time.perf_counter() - start
        task_dir = output_dir / mode / task
        rate, succ = parse_rate(task_dir, args.num_episodes)
        rc = rc_override if rc_override is not None else proc.returncode
        result = {
            "task_name": task,
            "mode": mode,
            "action_horizon": FIXED_ACTION_HORIZON,
            "chunks_per_video_prefill": FIXED_CPP,
            "num_episodes": args.num_episodes,
            "num_success": succ,
            "success_rate": rate,
            "wall_time": wall,
            "returncode": rc,
            "skipped_existing": False,
            "gpu": gpu_id,
            "log_file": str(log_file),
        }
        try:
            report = write_job_report(task_dir, result)
            lat = report.get("latency_means") or {}
            ipa = (lat.get("infer_per_action_s") or {}).get("mean")
            spr = (lat.get("skipped_prefill_ratio") or {}).get("mean")
            if ipa is not None:
                result["infer_per_action_s_mean"] = ipa
            if spr is not None:
                result["skipped_prefill_ratio_mean"] = spr
        except Exception as e:
            log(f"[WARN] job_report failed for {mode}/{task}: {e}", master_log)
        results.append(result)
        write_summary(results, output_dir)
        tag = "DONE" if rc == 0 else ("TIMEOUT" if rc_override == -9 else "ERROR")
        extra = ""
        if result.get("infer_per_action_s_mean") is not None:
            extra += f" | infer/action={result['infer_per_action_s_mean']*1000:.1f}ms"
        if result.get("skipped_prefill_ratio_mean") is not None:
            extra += f" | skip_prefill={result['skipped_prefill_ratio_mean']*100:.1f}%"
        log(
            f"[{tag}] {mode}/{task} GPU{gpu_id}: {succ}/{args.num_episodes} "
            f"({100 * rate:.1f}%) | {wall:.1f}s{extra}",
            master_log,
        )

    while remaining or running:
        finished = []
        for gpu_id, (proc, mode, task, start, log_file, log_fh) in list(running.items()):
            ret = proc.poll()
            elapsed = time.perf_counter() - start
            if ret is None and elapsed > args.timeout_s:
                log(f"[TIMEOUT] {mode}/{task} GPU{gpu_id} after {elapsed:.1f}s", master_log)
                proc.terminate()
                try:
                    proc.wait(timeout=args.kill_signal_timeout_s)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait()
                finished.append(gpu_id)
                finalize(gpu_id, proc, mode, task, start, log_file, log_fh, rc_override=-9)
                continue
            if ret is not None:
                finished.append(gpu_id)
                finalize(gpu_id, proc, mode, task, start, log_file, log_fh)
        for gpu_id in finished:
            del running[gpu_id]

        for gpu_id in range(args.num_gpus):
            if gpu_id in running or not remaining:
                continue
            if gpu_busy(gpu_id):
                continue
            mode, task = remaining.pop(0)
            task_dir = output_dir / mode / task
            task_dir.mkdir(parents=True, exist_ok=True)
            log_file = task_dir / f"{task}_{mode}_gpu{gpu_id}.log"

            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
            env["GPU_NUMBER"] = str(gpu_id + 1)
            env["SKIP_EXPERT_CHECK"] = "1"
            env["WANDB_MODE"] = "offline"
            env["VIDEO_DIT_MODE"] = mode  # baseline | skip_phase
            env["SKIP_PHASE_TASK_NAME"] = task
            env["SKIP_PHASE_NEAR_THRESH_M"] = os.environ.get("SKIP_PHASE_NEAR_THRESH_M", "0.10")
            env["SKIP_PHASE_MAX_CONSECUTIVE_SKIPS"] = os.environ.get(
                "SKIP_PHASE_MAX_CONSECUTIVE_SKIPS", "1"
            )
            env["OVCR_DIAG_MODE"] = os.environ.get("OVCR_DIAG_MODE", "baseline")
            src = str(PROJECT_ROOT / "src")
            env["PYTHONPATH"] = (
                f"{src}{os.pathsep}{env['PYTHONPATH']}" if env.get("PYTHONPATH") else src
            )
            py_path = Path(PYTHON).resolve()
            if any(tok in str(py_path) for tok in ("/envs/", "miniconda", "anaconda", "conda")):
                conda_bin = py_path.parent
                conda_lib = conda_bin.parent / "lib"
                env["PATH"] = f"{conda_bin}{os.pathsep}{env.get('PATH', '')}"
                if conda_lib.is_dir():
                    env["LD_LIBRARY_PATH"] = (
                        f"{conda_lib}{os.pathsep}{env.get('LD_LIBRARY_PATH', '')}"
                    )

            cmd = [
                PYTHON,
                "-u",
                str(EVAL_ENTRY),
                f"EVALUATION.task_name={task}",
                f"EVALUATION.eval_num_episodes={args.num_episodes}",
                "EVALUATION.timing_enabled=True",
                "EVALUATION.detailed_analysis=True",
                f"EVALUATION.action_horizon={FIXED_ACTION_HORIZON}",
                f"EVALUATION.chunks_per_video_prefill={FIXED_CPP}",
                "EVALUATION.task_config=demo_randomized",
                f"EVALUATION.output_dir={task_dir}",
            ]
            log(f"[LAUNCH] {mode}/{task} on GPU{gpu_id} (ah64_cpp2)", master_log)
            log_fh = open(log_file, "w", encoding="utf-8", buffering=1)
            proc = subprocess.Popen(
                cmd,
                cwd=str(PROJECT_ROOT),
                env=env,
                stdout=log_fh,
                stderr=subprocess.STDOUT,
                text=True,
            )
            running[gpu_id] = (proc, mode, task, time.perf_counter(), log_file, log_fh)

        time.sleep(30)

    write_summary(results, output_dir)
    log("All jobs finished.", master_log)


if __name__ == "__main__":
    main()
