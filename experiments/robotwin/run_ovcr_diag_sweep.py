#!/usr/bin/env python3
"""OVCR diagnostic probe sweep (no OVCR algorithm changes).

Compares three inference-path modes under fixed ah64 / cpp2:
  - baseline: normal OVCR
  - off: stale planner KV, OVCR disabled
  - oracle: fresh video prefill every action chunk, OVCR disabled

Results land under aha-wam-runs/robotwin/ovcr改进/.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import signal
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
EVAL_ENTRY = PROJECT_ROOT / "experiments" / "robotwin" / "eval_robotwin_single.py"
DEFAULT_OUTPUT = Path(
    "/home/ma-user/work/dataset/cwr_wulan_aha/aha-wam-runs/robotwin/ovcr改进"
)

# Estimated wall minutes for 5 episodes @ ah64_cpp2 (from prior sweeps).
TASK_ETA_MIN = {
    "place_fan": 15,
    "place_phone_stand": 15,
    "move_can_pot": 25,
    "stack_bowls_two": 35,
    "turn_switch": 30,
    "stack_blocks_two": 40,
    "open_laptop": 45,
    "hanging_mug": 45,
    "stack_bowls_three": 71,
    "move_stapler_pad": 30,
}

DEFAULT_TASKS = [
    "place_fan",
    "place_phone_stand",
    "move_can_pot",
    "stack_bowls_two",
    "turn_switch",
    "stack_blocks_two",
    "open_laptop",
    "hanging_mug",
    "move_stapler_pad",
    "stack_bowls_three",
]

DEFAULT_MODES = ["baseline", "off", "oracle"]


def build_env(gpu_id: int, mode: str, log_path: Path) -> dict:
    env = os.environ.copy()
    conda_bin = Path(sys.executable).parent
    conda_lib = conda_bin.parent / "lib"
    env["PATH"] = f"{conda_bin}{os.pathsep}{env.get('PATH', '')}"
    env["LD_LIBRARY_PATH"] = f"{conda_lib}{os.pathsep}{env.get('LD_LIBRARY_PATH', '')}"
    # Prefer extracted NVIDIA user-space libs used by prior RoboTwin evals.
    nvidia_lib = Path(
        "/home/ma-user/work/dataset/cwr_wulan_aha/nvidia-driver-libs/nvidia-535.183.01"
    )
    if nvidia_lib.is_dir():
        env["LD_LIBRARY_PATH"] = f"{nvidia_lib}{os.pathsep}{env['LD_LIBRARY_PATH']}"
    src_path = str(PROJECT_ROOT / "src")
    env["PYTHONPATH"] = (
        f"{src_path}{os.pathsep}{env['PYTHONPATH']}"
        if env.get("PYTHONPATH")
        else src_path
    )
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    env["OVCR_DIAG_MODE"] = mode
    env["OVCR_DIAG_LOG"] = "1"
    env["OVCR_DIAG_LOG_PATH"] = str(log_path)
    env["VIDEO_DIT_MODE"] = "baseline"
    return env


def preflight_cuda() -> None:
    """Fail fast if this interpreter cannot see a CUDA GPU."""
    code = (
        "import torch; "
        "ok=torch.cuda.is_available(); "
        "print(f'torch={torch.__version__} cuda={ok} n={torch.cuda.device_count()}'); "
        "raise SystemExit(0 if ok else 2)"
    )
    env = os.environ.copy()
    src_path = str(PROJECT_ROOT / "src")
    env["PYTHONPATH"] = (
        f"{src_path}{os.pathsep}{env['PYTHONPATH']}"
        if env.get("PYTHONPATH")
        else src_path
    )
    nvidia_lib = Path(
        "/home/ma-user/work/dataset/cwr_wulan_aha/nvidia-driver-libs/nvidia-535.183.01"
    )
    if nvidia_lib.is_dir():
        env["LD_LIBRARY_PATH"] = (
            f"{nvidia_lib}{os.pathsep}{env.get('LD_LIBRARY_PATH', '')}"
        )
    proc = subprocess.run(
        [sys.executable, "-c", code],
        env=env,
        capture_output=True,
        text=True,
    )
    print(proc.stdout.strip() or proc.stderr.strip(), flush=True)
    if proc.returncode != 0:
        raise RuntimeError(
            "CUDA is not available in the current Python interpreter "
            f"({sys.executable}). Install a CUDA build of torch before launching "
            "the OVCR diagnostic sweep."
        )


def parse_result_file(result_file: Path, num_episodes: int):
    try:
        text = result_file.read_text(encoding="utf-8").strip()
        last_line = text.splitlines()[-1].strip().strip("[]")
        rates = [float(x) for x in last_line.split() if x]
        success_rate = float(sum(rates) / len(rates)) if rates else 0.0
        num_success = int(round(success_rate * num_episodes))
        return num_success, success_rate
    except Exception as exc:  # noqa: BLE001
        print(f"[WARN] Failed to parse {result_file}: {exc}")
        return 0, 0.0


def run_one(job: dict) -> dict:
    task_name = job["task_name"]
    mode = job["mode"]
    gpu_id = int(job["gpu_id"])
    action_horizon = int(job["action_horizon"])
    cpp = int(job["cpp"])
    num_episodes = int(job["num_episodes"])
    timeout_seconds = int(job["timeout_seconds"])
    output_root = Path(job["output_root"])

    tag = f"ah{action_horizon}_cpp{cpp}_{mode}"
    task_output_dir = output_root / tag / task_name
    task_output_dir.mkdir(parents=True, exist_ok=True)
    log_path = task_output_dir / "ovcr_diag.jsonl"
    run_log = task_output_dir / f"{task_name}_{tag}_gpu{gpu_id}.log"

    cmd = [
        sys.executable,
        "-u",
        str(EVAL_ENTRY),
        f"EVALUATION.task_name={task_name}",
        f"EVALUATION.eval_num_episodes={num_episodes}",
        "EVALUATION.timing_enabled=True",
        "EVALUATION.detailed_analysis=True",
        f"EVALUATION.action_horizon={action_horizon}",
        f"EVALUATION.chunks_per_video_prefill={cpp}",
        "EVALUATION.task_config=demo_randomized",
        f"EVALUATION.output_dir={task_output_dir}",
        f"gpu_id={gpu_id}",
    ]

    print(f"[GPU{gpu_id}] START {task_name} | {tag}", flush=True)
    wall_t0 = time.time()
    env = build_env(gpu_id=gpu_id, mode=mode, log_path=log_path)
    proc = None
    stdout = ""
    return_code = -999
    try:
        proc = subprocess.Popen(
            cmd,
            cwd=str(PROJECT_ROOT),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        try:
            stdout, _ = proc.communicate(timeout=timeout_seconds)
            return_code = int(proc.returncode)
        except subprocess.TimeoutExpired:
            print(f"[GPU{gpu_id}] TIMEOUT {task_name} {tag}", flush=True)
            proc.send_signal(signal.SIGTERM)
            try:
                proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
            stdout = proc.stdout.read() if proc.stdout else ""
            return_code = -1
    except Exception as exc:  # noqa: BLE001
        stdout = f"launch failed: {exc}"
        return_code = -2
    finally:
        run_log.write_text(stdout or "", encoding="utf-8")

    wall_time = time.time() - wall_t0
    result_file = task_output_dir / task_name / "_result_random.txt"
    # eval_robotwin_single nests under run_ts; search.
    matches = list(task_output_dir.rglob("_result_random.txt"))
    if matches:
        # Prefer newest.
        result_file = max(matches, key=lambda p: p.stat().st_mtime)
    num_success, success_rate = parse_result_file(result_file, num_episodes)

    row = {
        "task_name": task_name,
        "mode": mode,
        "tag": tag,
        "action_horizon": action_horizon,
        "chunks_per_video_prefill": cpp,
        "gpu": gpu_id,
        "wall_time": wall_time,
        "returncode": return_code,
        "num_episodes": num_episodes,
        "num_success": num_success,
        "success_rate": success_rate,
        "ovcr_diag_log": str(log_path),
        "run_log": str(run_log),
    }
    print(
        f"[GPU{gpu_id}] DONE {task_name} | {tag} | sr={success_rate:.0%} "
        f"({num_success}/{num_episodes}) | {wall_time/60:.1f}min rc={return_code}",
        flush=True,
    )
    return row


def estimate_budget(tasks, modes, hours_per_gpu: float, num_gpus: int):
    per_mode = sum(TASK_ETA_MIN.get(t, 40) for t in tasks)
    total_min = per_mode * len(modes)
    capacity_min = hours_per_gpu * 60 * num_gpus
    return {
        "tasks": tasks,
        "modes": modes,
        "jobs": len(tasks) * len(modes),
        "eta_gpu_minutes": total_min,
        "eta_hours_wall_2gpu": total_min / max(num_gpus, 1) / 60.0,
        "capacity_gpu_minutes": capacity_min,
        "fits": total_min <= capacity_min * 0.95,
    }


def write_summary(output_root: Path, rows: list[dict]) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    summary_json = output_root / "summary.json"
    summary_csv = output_root / "summary_running.csv"
    summary_json.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    fields = [
        "task_name",
        "mode",
        "tag",
        "action_horizon",
        "chunks_per_video_prefill",
        "gpu",
        "num_episodes",
        "num_success",
        "success_rate",
        "wall_time",
        "returncode",
    ]
    with summary_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    # Compact pivot: task x mode success rate
    modes = sorted({r["mode"] for r in rows})
    tasks = sorted({r["task_name"] for r in rows})
    lines = ["| task | " + " | ".join(modes) + " |", "| --- | " + " | ".join(["---"] * len(modes)) + " |"]
    for task in tasks:
        cells = [task]
        for mode in modes:
            hit = [r for r in rows if r["task_name"] == task and r["mode"] == mode]
            if not hit:
                cells.append("n/a")
            else:
                r = hit[-1]
                cells.append(f"{r['success_rate']*100:.0f}% ({r['num_success']}/{r['num_episodes']})")
        lines.append("| " + " | ".join(cells) + " |")
    (output_root / "comparison.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="OVCR diagnostic probe sweep")
    parser.add_argument("--gpus", default="0,1")
    parser.add_argument("--tasks", nargs="+", default=DEFAULT_TASKS)
    parser.add_argument("--modes", nargs="+", default=DEFAULT_MODES)
    parser.add_argument("--action-horizon", type=int, default=64)
    parser.add_argument("--cpp", type=int, default=2)
    parser.add_argument("--num-episodes", type=int, default=5)
    parser.add_argument("--hours-per-gpu", type=float, default=10.0)
    parser.add_argument("--timeout-seconds", type=int, default=9000)
    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_OUTPUT))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    gpu_ids = [int(x) for x in str(args.gpus).split(",") if x.strip() != ""]
    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    budget = estimate_budget(args.tasks, args.modes, args.hours_per_gpu, len(gpu_ids))
    (output_root / "budget_estimate.json").write_text(
        json.dumps(budget, indent=2), encoding="utf-8"
    )
    print(json.dumps(budget, indent=2), flush=True)
    if not budget["fits"]:
        print(
            "[WARN] Estimated jobs exceed 95% of 2x10h capacity; "
            "launcher will still run in queue order and can be stopped early.",
            flush=True,
        )

    if not args.dry_run:
        preflight_cuda()

    jobs = []
    # Round-robin GPU assignment keeps both cards busy.
    for idx, (task, mode) in enumerate(
        (t, m) for t in args.tasks for m in args.modes
    ):
        jobs.append(
            {
                "task_name": task,
                "mode": mode,
                "gpu_id": gpu_ids[idx % len(gpu_ids)],
                "action_horizon": args.action_horizon,
                "cpp": args.cpp,
                "num_episodes": args.num_episodes,
                "timeout_seconds": args.timeout_seconds,
                "output_root": str(output_root),
            }
        )
    (output_root / "job_queue.json").write_text(json.dumps(jobs, indent=2), encoding="utf-8")

    readme = output_root / "README.md"
    if not readme.exists():
        readme.write_text(
            "\n".join(
                [
                    "# OVCR 改进 — 探究实验（非算法改动）",
                    "",
                    "固定 `action_horizon=64`, `chunks_per_video_prefill=2`，对比：",
                    "",
                    "- `baseline`: 正常 OVCR",
                    "- `off`: 关闭 OVCR（复用旧 first-frame KV）",
                    "- `oracle`: 每个 action chunk 强制重新 video prefill，且关闭 OVCR",
                    "",
                    "目的：判断当前 OVCR 在 cpp=2 下贡献多少，以及“新鲜 encode”上界有多高。",
                    "每个 run 另写 `ovcr_diag.jsonl`（gate / ΔKV 相对范数）。",
                    "",
                ]
            ),
            encoding="utf-8",
        )

    if args.dry_run:
        print(f"[DRY-RUN] {len(jobs)} jobs queued under {output_root}")
        return

    rows: list[dict] = []
    by_gpu = {gid: [j for j in jobs if j["gpu_id"] == gid] for gid in gpu_ids}

    with ProcessPoolExecutor(max_workers=len(gpu_ids)) as ex:
        futs = {
            ex.submit(_run_serial_jobs, by_gpu[gid], output_root): gid for gid in gpu_ids
        }
        for fut in as_completed(futs):
            gid = futs[fut]
            part = fut.result()
            rows.extend(part)
            write_summary(output_root, rows)
            print(f"[MERGE] GPU{gid} finished {len(part)} jobs", flush=True)

    write_summary(output_root, rows)
    print(f"[DONE] wrote summary to {output_root}", flush=True)


def _run_serial_jobs(gpu_jobs: list[dict], output_root: Path) -> list[dict]:
    """Run jobs one-by-one on a single GPU; flush partial CSV after each."""
    rows = []
    partial_path = output_root / f"summary_gpu{gpu_jobs[0]['gpu_id'] if gpu_jobs else 'x'}.json"
    for job in gpu_jobs:
        row = run_one(job)
        rows.append(row)
        partial_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")
        # Also update global running CSV best-effort (may race; final merge overwrites).
        try:
            existing = []
            summary_json = output_root / "summary.json"
            if summary_json.exists():
                existing = json.loads(summary_json.read_text(encoding="utf-8"))
            # Replace same tag/task if present.
            key = (row["task_name"], row["mode"])
            existing = [r for r in existing if (r["task_name"], r["mode"]) != key]
            existing.append(row)
            write_summary(output_root, existing)
        except Exception as exc:  # noqa: BLE001
            print(f"[WARN] partial summary update failed: {exc}", flush=True)
    return rows


if __name__ == "__main__":
    # ProcessPool needs top-level target.
    main()
