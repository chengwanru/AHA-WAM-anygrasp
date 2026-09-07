"""
Systematic chunks_per_video_prefill sweep.

Fixes action_horizon and only varies chunks_per_video_prefill (cpp) to isolate
the effect of video DiT calling frequency on latency and task success.
"""

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
EVAL_ENTRY = PROJECT_ROOT / "experiments" / "robotwin" / "eval_robotwin_single.py"


def build_env():
    env = os.environ.copy()
    conda_bin = Path(sys.executable).parent
    conda_lib = conda_bin.parent / "lib"
    env["PATH"] = f"{conda_bin}{os.pathsep}{env.get('PATH', '')}"
    env["LD_LIBRARY_PATH"] = f"{conda_lib}{os.pathsep}{env.get('LD_LIBRARY_PATH', '')}"
    return env


def parse_result_file(result_file: Path, num_episodes: int):
    """Parse the _result_random.txt written by eval_policy."""
    try:
        text = result_file.read_text(encoding="utf-8").strip()
        last_line = text.splitlines()[-1].strip().strip("[]")
        rates = [float(x) for x in last_line.split() if x]
        success_rate = float(sum(rates) / len(rates)) if rates else 0.0
        num_success = int(round(success_rate * num_episodes))
        return num_success, success_rate
    except Exception as e:
        print(f"[WARN] Failed to parse result file {result_file}: {e}")
        return 0, 0.0


def load_episode_logs(analysis_dir: Path, run_start_time: float):
    """Load analysis json files written during this run."""
    episodes = []
    if not analysis_dir.exists():
        return episodes
    for json_path in sorted(analysis_dir.glob("episode*_analysis.json")):
        try:
            if json_path.stat().st_mtime < run_start_time:
                continue
            with open(json_path, "r", encoding="utf-8") as f:
                ep = json.load(f)
            episodes.append(ep)
        except Exception as e:
            print(f"[WARN] Failed to load {json_path}: {e}")
    return episodes


def run_config(
    task_name: str,
    action_horizon: int,
    cpp: int,
    num_episodes: int,
    output_dir: Path,
    timeout_seconds: int = 2400,
):
    """Run one (task, action_horizon, cpp) combo with timeout and error handling."""
    tag = f"ah{action_horizon}_cpp{cpp}"
    task_output_dir = output_dir / tag / task_name
    task_output_dir.mkdir(parents=True, exist_ok=True)

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
    ]

    print(f"\n{'='*80}")
    print(f"Running {task_name} | {tag} | episodes={num_episodes}")
    print(f"{'='*80}")

    wall_t0 = time.time()
    t0 = time.perf_counter()
    env = build_env()
    proc = None
    stdout = ""
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
            return_code = proc.returncode
        except subprocess.TimeoutExpired:
            print(f"[TIMEOUT] {task_name} {tag} exceeded {timeout_seconds}s, killing...")
            proc.send_signal(signal.SIGTERM)
            try:
                proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
            stdout = proc.stdout.read() if proc.stdout else ""
            return_code = -1
    except Exception as e:
        print(f"[ERROR] {task_name} {tag} launch failed: {e}")
        return_code = -1
    finally:
        if proc is not None and proc.poll() is None:
            try:
                proc.kill()
                proc.wait(timeout=10)
            except Exception:
                pass

    wall_time = time.perf_counter() - t0

    log_file = task_output_dir / f"{task_name}_{tag}.log"
    with open(log_file, "w", encoding="utf-8") as f:
        f.write(stdout)

    result = {
        "task_name": task_name,
        "tag": tag,
        "action_horizon": action_horizon,
        "chunks_per_video_prefill": cpp,
        "wall_time": wall_time,
        "num_episodes": num_episodes,
        "num_success": 0,
        "success_rate": 0.0,
        "return_code": return_code,
        "analysis_dir": None,
        "episode_logs": [],
    }

    if return_code != 0:
        print(f"[ERROR] {task_name} {tag} failed after {wall_time:.1f}s")
        print(stdout[-1500:])
        return result

    # Find latest result dir for this task
    result_dirs = sorted(
        (PROJECT_ROOT / "evaluate_results" / "robotwin" / "robotwin_ahawam").rglob(
            f"{task_name}/_result_random.txt"
        ),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )

    if result_dirs:
        result_file = result_dirs[0]
        analysis_dir = result_file.parent / "analysis"
        result["analysis_dir"] = str(analysis_dir)
        result["num_success"], result["success_rate"] = parse_result_file(
            result_file, num_episodes
        )
        result["episode_logs"] = load_episode_logs(analysis_dir, wall_t0)
        actual = len(result["episode_logs"])
        if actual > 0:
            result["num_episodes"] = actual
            result["num_success"] = int(
                sum(1 for ep in result["episode_logs"] if ep.get("success", False))
            )
            result["success_rate"] = result["num_success"] / actual

    print(
        f"[DONE] {task_name} {tag}: {result['num_success']}/{result['num_episodes']} "
        f"({result['success_rate']*100:.1f}%) | {wall_time:.1f}s"
    )
    return result


def main():
    parser = argparse.ArgumentParser(
        description="Systematic chunks_per_video_prefill sweep for AHA-WAM"
    )
    parser.add_argument("--tasks", nargs="+", required=True, help="Tasks to evaluate")
    parser.add_argument("--action_horizon", type=int, default=128,
                        help="Fixed action_horizon (default 128 so cpp can go up to 8)")
    parser.add_argument("--cpp_values", nargs="+", type=int, default=[1, 2, 4, 8],
                        help="chunks_per_video_prefill values to test")
    parser.add_argument("--num_episodes", type=int, default=10,
                        help="Episodes per config")
    parser.add_argument("--timeout", type=int, default=2400,
                        help="Per-config timeout in seconds")
    parser.add_argument(
        "--output_dir",
        type=str,
        default=str(PROJECT_ROOT / "aha-wam-runs" / "robotwin" / "cpp_sweep_analysis"),
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_results = []
    for task_name in args.tasks:
        for cpp in args.cpp_values:
            if cpp > args.action_horizon // 16:
                print(
                    f"[SKIP] {task_name} ah={args.action_horizon} cpp={cpp}: "
                    f"cpp > num_chunks={args.action_horizon // 16}"
                )
                continue
            result = run_config(
                task_name=task_name,
                action_horizon=args.action_horizon,
                cpp=cpp,
                num_episodes=args.num_episodes,
                output_dir=output_dir,
                timeout_seconds=args.timeout,
            )
            all_results.append(result)
            summary_path = output_dir / "summary.json"
            with open(summary_path, "w", encoding="utf-8") as f:
                json.dump(all_results, f, indent=2, default=str)

    # Write CSV
    csv_path = output_dir / "summary.csv"
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write(
            "task_name,tag,action_horizon,chunks_per_video_prefill,num_episodes,"
            "num_success,success_rate,wall_time,return_code\n"
        )
        for r in all_results:
            f.write(
                f"{r['task_name']},{r['tag']},{r['action_horizon']},"
                f"{r['chunks_per_video_prefill']},{r['num_episodes']},"
                f"{r['num_success']},{r['success_rate']:.4f},"
                f"{r['wall_time']:.2f},{r['return_code']}\n"
            )

    print(f"\n{'='*80}")
    print(f"CPP sweep finished. Summary: {summary_path} and {csv_path}")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
