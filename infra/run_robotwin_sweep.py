#!/usr/bin/env python3
"""Large-scale RoboTwin AHA-WAM ah/cpp sweep across multiple GPUs.

20 tasks x 6 configs x 40 episodes.
Distributes work across available GPUs and writes detailed logs for latency
and failure analysis (stdout + analysis/*.json + job_report.json).
"""
import argparse
import csv
import glob
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

# AHA-WAM code root: prefer platform mount path, fall back to local dataset path.
PROJECT_ROOT = Path(
    os.environ.get(
        "AHA_WAM_CODE_DIR",
        "/opt/huawei/dataset/cwr_dataset_wulann/AHA-WAM-anygrasp"
        if Path("/opt/huawei/dataset/cwr_dataset_wulann/AHA-WAM-anygrasp").exists()
        else "/home/ma-user/work/dataset/cwr_dataset_wulann/AHA-WAM-anygrasp",
    )
)
EVAL_ENTRY = PROJECT_ROOT / "experiments" / "robotwin" / "eval_robotwin_single.py"
PYTHON = os.environ.get("PYTHON", sys.executable)

# 20 diverse RoboTwin tasks (cover pick/place/move/stack/handover/open/misc).
TASKS = [
    "pick_dual_bottles",
    "pick_diverse_bottles",
    "place_bread_basket",
    "place_object_basket",
    "place_a2b_left",
    "place_a2b_right",
    "move_can_pot",
    "move_stapler_pad",
    "stack_blocks_two",
    "stack_bowls_three",
    "handover_block",
    "handover_mic",
    "hanging_mug",
    "lift_pot",
    "open_laptop",
    "open_microwave",
    "press_stapler",
    "click_bell",
    "beat_block_hammer",
    "turn_switch",
]

# 5 baseline ah/cpp configs + ah64_cpp3.
# tag format: ah{action_horizon}_cpp{chunks_per_video_prefill}
CONFIGS = [
    (64, 2),
    (64, 1),
    (64, 3),  # extra: more chunks per video prefill at ah=64
    (32, 2),
    (32, 1),
    (16, 1),
]


def log(msg, log_path=None):
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}"
    print(line)
    if log_path is not None:
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(line + "\n")


def gpu_busy(gpu_id):
    """Check if a GPU is already running eval_robotwin_single.py."""
    for cmdline_path in glob.glob("/proc/[0-9]*/cmdline"):
        try:
            with open(cmdline_path, "rb") as f:
                data = f.read()
            if not data:
                continue
            cmd = data.replace(b"\0", b" ").decode("utf-8", errors="ignore")
            if "eval_robotwin_single.py" not in cmd:
                continue
            env_path = cmdline_path.replace("/cmdline", "/environ")
            env = {}
            try:
                with open(env_path, "rb") as f:
                    for item in f.read().split(b"\0"):
                        if b"=" in item:
                            k, v = item.split(b"=", 1)
                            env[k.decode("utf-8", errors="ignore")] = v.decode(
                                "utf-8", errors="ignore"
                            )
            except Exception:
                pass
            if env.get("CUDA_VISIBLE_DEVICES", "") == str(gpu_id):
                return True
        except Exception:
            continue
    return False


def run_config(task_name, action_horizon, cpp, num_episodes, gpu_id, output_dir, master_log):
    """Run one (task, action_horizon, cpp) combo on a specific GPU."""
    tag = f"ah{action_horizon}_cpp{cpp}"
    task_output_dir = output_dir / tag / task_name
    task_output_dir.mkdir(parents=True, exist_ok=True)
    log_file = task_output_dir / f"{task_name}_{tag}_gpu{gpu_id}.log"

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    env["GPU_NUMBER"] = str(gpu_id + 1)
    env["SKIP_EXPERT_CHECK"] = "1"
    env["WANDB_MODE"] = "offline"
    # Only prepend conda env lib/bin when PYTHON is inside a real conda env.
    # System /usr/bin/python3 would otherwise put /usr/lib first and can shadow
    # dataset sapien-runtime-libs (libX11 / libstdc++).
    py_path = Path(PYTHON).resolve()
    py_str = str(py_path)
    if any(tok in py_str for tok in ("/envs/", "miniconda", "anaconda", "conda")):
        conda_bin = py_path.parent
        conda_lib = conda_bin.parent / "lib"
        env["PATH"] = f"{conda_bin}{os.pathsep}{env.get('PATH', '')}"
        if conda_lib.is_dir():
            env["LD_LIBRARY_PATH"] = f"{conda_lib}{os.pathsep}{env.get('LD_LIBRARY_PATH', '')}"

    cmd = [
        PYTHON,
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
    ]

    log(f"[LAUNCH] {task_name} {tag} on GPU{gpu_id}", master_log)
    t0 = time.perf_counter()
    with open(log_file, "w", encoding="utf-8") as log_fh:
        proc = subprocess.run(
            cmd,
            cwd=str(PROJECT_ROOT),
            env=env,
            text=True,
            stdout=log_fh,
            stderr=subprocess.STDOUT,
        )
    wall_time = time.perf_counter() - t0
    try:
        stdout = log_file.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        stdout = ""

    result = {
        "task_name": task_name,
        "tag": tag,
        "action_horizon": action_horizon,
        "chunks_per_video_prefill": cpp,
        "gpu": gpu_id,
        "wall_time": wall_time,
        "returncode": proc.returncode,
        "num_episodes": num_episodes,
        "num_success": 0,
        "success_rate": 0.0,
        "log_file": str(log_file),
    }

    if proc.returncode != 0:
        log(f"[ERROR] {task_name} {tag} GPU{gpu_id} failed after {wall_time:.1f}s", master_log)
        log(stdout[-2000:], master_log)
        return result

    # Parse success rate from this job's unique output dir first.
    result_file = task_output_dir / "_result_random.txt"
    if not result_file.exists():
        result_dirs = sorted(
            (PROJECT_ROOT / "evaluate_results" / "robotwin" / "robotwin_ahawam").rglob(
                f"{task_name}/_result_random.txt"
            ),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        result_file = result_dirs[0] if result_dirs else None
    if result_file is not None and result_file.exists():
        try:
            text = result_file.read_text(encoding="utf-8").strip()
            last_line = text.splitlines()[-1].strip().strip("[]")
            rates = [float(x) for x in last_line.split() if x]
            result["success_rate"] = float(sum(rates) / len(rates)) if rates else 0.0
            result["num_success"] = int(round(result["success_rate"] * result["num_episodes"]))
        except Exception as e:
            log(f"[WARN] parse error for {task_name} {tag}: {e}", master_log)

    # Extract average latency from the log for quick summary.
    timing = extract_latency(proc.stdout)
    result.update(timing)

    log(
        f"[DONE] {task_name} {tag} GPU{gpu_id}: "
        f"{result['num_success']}/{result['num_episodes']} "
        f"({result['success_rate']*100:.1f}%) | "
        f"infer/action={result.get('infer_per_action_s', 0)*1000:.1f}ms | "
        f"{wall_time:.1f}s",
        master_log,
    )
    return result


def extract_latency(stdout):
    """Extract last Timing line values from eval log."""
    import re

    timing = {
        "infer_per_action_s": 0.0,
        "sim_per_action_s": 0.0,
        "prefill_avg_s": 0.0,
        "action_chunk_avg_s": 0.0,
        "take_action_cnt": 0,
        "infer_calls": 0,
        "prefill_calls": 0,
        "action_chunk_calls": 0,
    }
    lines = [ln for ln in stdout.splitlines() if "Timing |" in ln]
    if not lines:
        return timing
    last = lines[-1]
    for key in timing:
        m = re.search(rf"{key}=([\d.]+)", last)
        if m:
            timing[key] = float(m.group(1))
    return timing


def parse_success_progress(stdout: str):
    """Parse last 'Success rate: X/Y => Z%' from eval stdout (works for partial/timeout)."""
    import re

    matches = re.findall(
        r"Success rate:.*?(\d+)\s*/\s*(\d+)\s*=>\s*([\d.]+)\s*%",
        stdout,
        flags=re.IGNORECASE,
    )
    if not matches:
        return None
    succ, total, pct = matches[-1]
    return {
        "num_success": int(succ),
        "episodes_done": int(total),
        "success_rate": float(pct) / 100.0,
    }


def parse_result_file(result_file: Path, num_episodes: int):
    if result_file is None or not result_file.exists():
        return None
    try:
        text = result_file.read_text(encoding="utf-8").strip()
        last_line = text.splitlines()[-1].strip().strip("[]")
        rates = [float(x) for x in last_line.split() if x]
        if not rates:
            return None
        rate = float(sum(rates) / len(rates))
        return {
            "success_rate": rate,
            "num_success": int(round(rate * num_episodes)),
        }
    except Exception:
        return None


def job_already_complete(task_output_dir: Path, num_episodes: int) -> bool:
    """True if this (tag,task) dir already has a full eval (resume/skip)."""
    analysis_dir = task_output_dir / "analysis"
    if analysis_dir.is_dir():
        n = len(list(analysis_dir.glob("episode*_analysis.json")))
        if n >= int(num_episodes):
            return True
    parsed = parse_result_file(task_output_dir / "_result_random.txt", num_episodes)
    return parsed is not None


def aggregate_episode_analysis(task_output_dir: Path, meta: dict):
    """Load analysis/*.json and write job_report.json + episodes_summary.csv."""
    analysis_dir = task_output_dir / "analysis"
    episodes = []
    if analysis_dir.is_dir():
        for json_path in sorted(analysis_dir.glob("episode*_analysis.json")):
            try:
                ep = json.loads(json_path.read_text(encoding="utf-8"))
                episodes.append(ep)
            except Exception:
                continue

    report = {
        **meta,
        "num_analysis_files": len(episodes),
        "episodes": [],
        "failures": [],
        "timing_means": {},
    }

    timing_sums = {}
    timing_counts = {}
    for ep in episodes:
        timing = ep.get("timing") or {}
        entry = {
            "episode_idx": ep.get("episode_idx"),
            "seed": ep.get("seed"),
            "success": ep.get("success"),
            "num_steps": ep.get("num_steps"),
            "analysis_file": f"analysis/episode{ep.get('episode_idx')}_analysis.json",
            "timing": {
                k: timing.get(k)
                for k in (
                    "infer_s",
                    "sim_s",
                    "prefill_s",
                    "action_chunk_s",
                    "infer_calls",
                    "prefill_calls",
                    "take_action_cnt",
                )
                if k in timing or True
            },
        }
        # keep only present timing keys
        entry["timing"] = {k: v for k, v in entry["timing"].items() if v is not None}
        report["episodes"].append(entry)
        if not ep.get("success"):
            report["failures"].append(entry)
        for k, v in timing.items():
            try:
                fv = float(v)
            except (TypeError, ValueError):
                continue
            timing_sums[k] = timing_sums.get(k, 0.0) + fv
            timing_counts[k] = timing_counts.get(k, 0) + 1

    for k, s in timing_sums.items():
        n = max(timing_counts.get(k, 1), 1)
        report["timing_means"][k] = s / n

    report_path = task_output_dir / "job_report.json"
    report_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")

    csv_path = task_output_dir / "episodes_summary.csv"
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "episode_idx", "seed", "success", "num_steps",
            "infer_s", "sim_s", "prefill_s", "action_chunk_s", "analysis_file",
        ])
        for e in report["episodes"]:
            t = e.get("timing") or {}
            writer.writerow([
                e.get("episode_idx"),
                e.get("seed"),
                e.get("success"),
                e.get("num_steps"),
                t.get("infer_s", ""),
                t.get("sim_s", ""),
                t.get("prefill_s", ""),
                t.get("action_chunk_s", ""),
                e.get("analysis_file"),
            ])

    # Human-readable appendix for the per-job log
    lines = [
        "",
        "=" * 72,
        f"JOB REPORT {meta.get('tag')} / {meta.get('task_name')}",
        f"success={meta.get('num_success')}/{meta.get('episodes_done') or meta.get('num_episodes')} "
        f"rate={float(meta.get('success_rate', 0))*100:.1f}% "
        f"timed_out={meta.get('timed_out')} wall_s={meta.get('wall_time')}",
        f"analysis_files={len(episodes)} failures={len(report['failures'])}",
        f"job_report={report_path}",
        f"episodes_summary={csv_path}",
        "-" * 72,
    ]
    for e in report["episodes"]:
        lines.append(
            f"ep{e.get('episode_idx')}: success={e.get('success')} steps={e.get('num_steps')} "
            f"seed={e.get('seed')} infer_s={e.get('timing', {}).get('infer_s', '')} "
            f"sim_s={e.get('timing', {}).get('sim_s', '')}"
        )
    if report["failures"]:
        lines.append("-" * 72)
        lines.append("FAILURE episodes (see analysis/*.json step_log / task_state_history):")
        for e in report["failures"]:
            lines.append(
                f"  FAIL ep{e.get('episode_idx')} seed={e.get('seed')} "
                f"steps={e.get('num_steps')} file={e.get('analysis_file')}"
            )
    lines.append("=" * 72)
    return "\n".join(lines) + "\n", report


def main():
    parser = argparse.ArgumentParser(description="RoboTwin AHA-WAM large-scale ah/cpp sweep")
    parser.add_argument("--num_episodes", type=int, default=40)
    parser.add_argument("--num_gpus", type=int, default=8)
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/opt/huawei/dataset/cwr_dataset_wulann/aha-wam-runs/robotwin_ahawam_sweep_20tasks_40eps",
    )
    parser.add_argument("--tasks", nargs="+", default=TASKS)
    parser.add_argument(
        "--timeout_s",
        type=int,
        default=43200,
        help="Per-job wall-clock timeout in seconds (default: 12h; 40eps needs >6h).",
    )
    parser.add_argument(
        "--kill_signal_timeout_s",
        type=int,
        default=60,
        help="Seconds to wait after SIGTERM before SIGKILL.",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    master_log = output_dir / "sweep_master.log"

    # Persist sweep config for analysis
    (output_dir / "sweep_config.json").write_text(
        json.dumps(
            {
                "tasks": list(args.tasks),
                "configs": [{"action_horizon": ah, "chunks_per_video_prefill": cpp} for ah, cpp in CONFIGS],
                "num_episodes": args.num_episodes,
                "num_gpus": args.num_gpus,
                "timeout_s": args.timeout_s,
                "total_jobs": len(args.tasks) * len(CONFIGS),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    log("=" * 80, master_log)
    log(f"Starting large-scale sweep", master_log)
    log(f"Python executable: {PYTHON}", master_log)
    log(f"Tasks: {len(args.tasks)} | Configs: {len(CONFIGS)} | Episodes: {args.num_episodes}", master_log)
    log(f"Configs: {CONFIGS}", master_log)
    log(f"Total jobs: {len(args.tasks) * len(CONFIGS)} | GPUs: {args.num_gpus}", master_log)
    log(f"Per-job timeout: {args.timeout_s}s", master_log)
    log(f"Output dir: {output_dir}", master_log)
    log("=" * 80, master_log)

    results = []
    running = {}
    all_jobs = [(t, ah, cpp) for t in args.tasks for ah, cpp in CONFIGS]
    remaining = []
    for task, ah, cpp in all_jobs:
        tag = f"ah{ah}_cpp{cpp}"
        task_output_dir = output_dir / tag / task
        if job_already_complete(task_output_dir, args.num_episodes):
            parsed = parse_result_file(task_output_dir / "_result_random.txt", args.num_episodes) or {}
            result = {
                "task_name": task,
                "tag": tag,
                "action_horizon": ah,
                "chunks_per_video_prefill": cpp,
                "gpu": -1,
                "wall_time": 0.0,
                "returncode": 0,
                "num_episodes": args.num_episodes,
                "num_success": parsed.get("num_success", 0),
                "success_rate": parsed.get("success_rate", 0.0),
                "episodes_done": args.num_episodes,
                "timed_out": False,
                "skipped_existing": True,
                "log_file": "",
                "task_output_dir": str(task_output_dir),
            }
            report_text, report = aggregate_episode_analysis(task_output_dir, result)
            result["num_analysis_files"] = report.get("num_analysis_files", 0)
            result["num_failures_logged"] = len(report.get("failures", []))
            result["job_report"] = str(task_output_dir / "job_report.json")
            if result["num_success"] == 0 and report.get("episodes"):
                succ = sum(1 for e in report["episodes"] if e.get("success"))
                done = len(report["episodes"])
                result["num_success"] = succ
                result["episodes_done"] = done
                result["success_rate"] = (succ / done) if done else 0.0
            (task_output_dir / "job_report.txt").write_text(report_text, encoding="utf-8")
            results.append(result)
            log(
                f"[SKIP] {task} {tag}: already complete "
                f"{result['num_success']}/{result['num_episodes']} "
                f"({result['success_rate']*100:.1f}%)",
                master_log,
            )
        else:
            remaining.append((task, ah, cpp))
    if results:
        write_summary(results, output_dir)
    log(
        f"Queue: {len(remaining)} to run, {len(results)} skipped (already complete)",
        master_log,
    )

    def finalize_job(gpu_id, proc, task, ah, cpp, start_time, log_file, log_fh=None, returncode_override=None):
        """Wait for process, read its log file, build result dict, append to results."""
        if log_fh is not None and not log_fh.closed:
            try:
                log_fh.flush()
            except Exception:
                pass
        # Drain/wait without using PIPE (stdout already redirected to log_file).
        try:
            proc.wait(timeout=max(1, int(args.kill_signal_timeout_s)))
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
        if log_fh is not None and not log_fh.closed:
            try:
                log_fh.close()
            except Exception:
                pass

        wall_time = time.perf_counter() - start_time
        try:
            stdout = log_file.read_text(encoding="utf-8", errors="ignore") if log_file.exists() else ""
        except Exception:
            stdout = ""

        tag = f"ah{ah}_cpp{cpp}"
        task_output_dir = output_dir / tag / task
        result = {
            "task_name": task,
            "tag": tag,
            "action_horizon": ah,
            "chunks_per_video_prefill": cpp,
            "gpu": gpu_id,
            "wall_time": wall_time,
            "returncode": returncode_override if returncode_override is not None else proc.returncode,
            "num_episodes": args.num_episodes,
            "num_success": 0,
            "success_rate": 0.0,
            "episodes_done": 0,
            "timed_out": returncode_override == -9,
            "log_file": str(log_file),
            "task_output_dir": str(task_output_dir),
        }

        # Prefer this job's result file; fall back to last Success rate line in stdout.
        parsed = parse_result_file(task_output_dir / "_result_random.txt", args.num_episodes)
        progress = parse_success_progress(stdout)
        if parsed is not None:
            result["success_rate"] = parsed["success_rate"]
            result["num_success"] = parsed["num_success"]
            if progress is not None:
                result["episodes_done"] = progress["episodes_done"]
                if result["timed_out"]:
                    result["num_success"] = progress["num_success"]
                    result["success_rate"] = (
                        progress["num_success"] / progress["episodes_done"]
                        if progress["episodes_done"] > 0
                        else 0.0
                    )
        elif progress is not None:
            result["num_success"] = progress["num_success"]
            result["episodes_done"] = progress["episodes_done"]
            result["success_rate"] = progress["success_rate"]
        # If analysis already has full 40 eps but process hung on pipe, treat as done.
        if result["episodes_done"] == 0:
            n_an = len(list((task_output_dir / "analysis").glob("episode*_analysis.json"))) if (task_output_dir / "analysis").exists() else 0
            if n_an > 0:
                result["episodes_done"] = n_an

        result.update(extract_latency(stdout))

        report_text, report = aggregate_episode_analysis(task_output_dir, result)
        result["num_analysis_files"] = report.get("num_analysis_files", 0)
        result["num_failures_logged"] = len(report.get("failures", []))
        result["job_report"] = str(task_output_dir / "job_report.json")
        if result["num_success"] == 0 and report.get("episodes"):
            succ = sum(1 for e in report["episodes"] if e.get("success"))
            done = len(report["episodes"])
            result["num_success"] = succ
            result["episodes_done"] = done
            result["success_rate"] = (succ / done) if done else 0.0

        # Append structured report for offline analysis (keep existing eval stdout).
        try:
            with open(log_file, "a", encoding="utf-8") as f:
                f.write(report_text)
        except Exception:
            pass
        (task_output_dir / "job_report.txt").write_text(report_text, encoding="utf-8")

        if result["returncode"] == 0:
            log(
                f"[DONE] {task} {tag} GPU{gpu_id}: "
                f"{result['num_success']}/{result['num_episodes']} "
                f"({result['success_rate']*100:.1f}%) | "
                f"infer/action={result.get('infer_per_action_s', 0)*1000:.1f}ms | "
                f"analysis={result['num_analysis_files']} | "
                f"{wall_time:.1f}s",
                master_log,
            )
        elif result["timed_out"]:
            log(
                f"[TIMEOUT-PARTIAL] {task} {tag} GPU{gpu_id}: "
                f"{result['num_success']}/{result.get('episodes_done', 0)} done "
                f"(target {result['num_episodes']}, rate {result['success_rate']*100:.1f}%) "
                f"analysis={result['num_analysis_files']} after {wall_time:.1f}s",
                master_log,
            )
        else:
            log(f"[ERROR] {task} {tag} GPU{gpu_id} failed after {wall_time:.1f}s", master_log)
            log(stdout[-2000:], master_log)
        results.append(result)
        write_summary(results, output_dir)

    while remaining or running:
        finished = []
        for gpu_id, (proc, task, ah, cpp, start_time, log_file, log_fh) in list(running.items()):
            ret = proc.poll()
            elapsed = time.perf_counter() - start_time

            # Timeout handling: terminate stuck jobs so one hung eval does not
            # block the GPU forever.
            if ret is None and elapsed > args.timeout_s:
                log(
                    f"[TIMEOUT] {task} {f'ah{ah}_cpp{cpp}'} GPU{gpu_id} "
                    f"after {elapsed:.1f}s; terminating...",
                    master_log,
                )
                proc.terminate()
                try:
                    proc.wait(timeout=args.kill_signal_timeout_s)
                except subprocess.TimeoutExpired:
                    log(f"[TIMEOUT] {task} did not terminate, sending SIGKILL", master_log)
                    proc.kill()
                    proc.wait()
                finished.append(gpu_id)
                finalize_job(
                    gpu_id, proc, task, ah, cpp, start_time, log_file, log_fh, returncode_override=-9
                )
                continue

            if ret is not None:
                finished.append(gpu_id)
                finalize_job(gpu_id, proc, task, ah, cpp, start_time, log_file, log_fh)

        for gpu_id in finished:
            del running[gpu_id]

        for gpu_id in range(args.num_gpus):
            if gpu_id in running or not remaining:
                continue
            if gpu_busy(gpu_id):
                continue
            task, ah, cpp = remaining.pop(0)
            task_output_dir = output_dir / f"ah{ah}_cpp{cpp}" / task
            task_output_dir.mkdir(parents=True, exist_ok=True)
            log_file = task_output_dir / f"{task}_ah{ah}_cpp{cpp}_gpu{gpu_id}.log"

            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
            env["GPU_NUMBER"] = str(gpu_id + 1)
            env["SKIP_EXPERT_CHECK"] = "1"
            env["WANDB_MODE"] = "offline"
            py_path = Path(PYTHON).resolve()
            py_str = str(py_path)
            if any(tok in py_str for tok in ("/envs/", "miniconda", "anaconda", "conda")):
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
                f"EVALUATION.action_horizon={ah}",
                f"EVALUATION.chunks_per_video_prefill={cpp}",
                "EVALUATION.task_config=demo_randomized",
                f"EVALUATION.output_dir={task_output_dir}",
            ]

            log(f"[LAUNCH] {task} ah{ah}_cpp{cpp} on GPU{gpu_id}", master_log)
            # CRITICAL: redirect stdout to file. Using PIPE without a reader
            # deadlocks after ~few MB of Timing/analysis logs (jobs finish on
            # disk but parent never sees exit → no next wave).
            log_fh = open(log_file, "w", encoding="utf-8", buffering=1)
            proc = subprocess.Popen(
                cmd,
                cwd=str(PROJECT_ROOT),
                env=env,
                stdout=log_fh,
                stderr=subprocess.STDOUT,
                text=True,
            )
            running[gpu_id] = (proc, task, ah, cpp, time.perf_counter(), log_file, log_fh)

        time.sleep(30)

    write_summary(results, output_dir)
    log("=" * 80, master_log)
    log(f"All {len(results)} jobs finished. Summary: {output_dir / 'summary.json'}", master_log)
    log("=" * 80, master_log)


def write_summary(results, output_dir):
    summary_json = output_dir / "summary.json"
    summary_json.write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")

    summary_csv = output_dir / "summary.csv"
    with open(summary_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "task_name", "tag", "action_horizon", "chunks_per_video_prefill",
            "num_episodes", "episodes_done", "num_success", "success_rate", "wall_time",
            "returncode", "timed_out", "infer_per_action_s", "sim_per_action_s",
            "prefill_avg_s", "action_chunk_avg_s", "take_action_cnt",
            "infer_calls", "prefill_calls", "action_chunk_calls", "log_file",
        ])
        for r in results:
            writer.writerow([
                r.get("task_name"),
                r.get("tag"),
                r.get("action_horizon"),
                r.get("chunks_per_video_prefill"),
                r.get("num_episodes"),
                r.get("episodes_done", ""),
                r.get("num_success"),
                f"{r.get('success_rate', 0):.4f}",
                f"{r.get('wall_time', 0):.2f}",
                r.get("returncode"),
                r.get("timed_out", False),
                f"{r.get('infer_per_action_s', 0):.6f}",
                f"{r.get('sim_per_action_s', 0):.6f}",
                f"{r.get('prefill_avg_s', 0):.6f}",
                f"{r.get('action_chunk_avg_s', 0):.6f}",
                r.get("take_action_cnt", 0),
                r.get("infer_calls", 0),
                r.get("prefill_calls", 0),
                r.get("action_chunk_calls", 0),
                r.get("log_file", ""),
            ])


if __name__ == "__main__":
    main()
