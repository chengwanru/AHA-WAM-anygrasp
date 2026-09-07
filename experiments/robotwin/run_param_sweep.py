"""
Parameter sweep for AHA-WAM RoboTwin evaluation.

Tests the effect of action_horizon and chunks_per_video_prefill on failure-prone tasks.
"""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
EVAL_ENTRY = PROJECT_ROOT / "experiments" / "robotwin" / "eval_robotwin_single.py"


def run_config(task_name: str, action_horizon: int, cpp: int, num_episodes: int, output_dir: Path):
    """Run eval_robotwin_single.py for one (task, action_horizon, cpp) combo."""
    tag = f"ah{action_horizon}_cpp{cpp}"
    task_output_dir = output_dir / tag / task_name
    task_output_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        "-u",
        str(EVAL_ENTRY),
        f"EVALUATION.task_name={task_name}",
        f"EVALUATION.eval_num_episodes={num_episodes}",
        f"EVALUATION.timing_enabled=True",
        f"EVALUATION.detailed_analysis=True",
        f"EVALUATION.action_horizon={action_horizon}",
        f"EVALUATION.chunks_per_video_prefill={cpp}",
        f"EVALUATION.task_config=demo_randomized",
        f"EVALUATION.output_dir={task_output_dir}",
    ]

    print(f"\n{'='*80}")
    print(f"Running {task_name} | {tag} | episodes={num_episodes}")
    print(f"{'='*80}")

    wall_t0 = time.time()
    t0 = time.perf_counter()
    env = os.environ.copy()
    conda_bin = Path(sys.executable).parent
    conda_lib = conda_bin.parent / "lib"
    env["PATH"] = f"{conda_bin}{os.pathsep}{env.get('PATH', '')}"
    env["LD_LIBRARY_PATH"] = f"{conda_lib}{os.pathsep}{env.get('LD_LIBRARY_PATH', '')}"
    proc = subprocess.run(
        cmd,
        cwd=str(PROJECT_ROOT),
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    wall_time = time.perf_counter() - t0

    log_file = task_output_dir / f"{task_name}_{tag}.log"
    with open(log_file, "w", encoding="utf-8") as f:
        f.write(proc.stdout)

    if proc.returncode != 0:
        print(f"[ERROR] {task_name} {tag} failed after {wall_time:.1f}s")
        print(proc.stdout[-2000:])
        return None

    # Find latest result dir
    result_dirs = sorted(
        (PROJECT_ROOT / "evaluate_results" / "robotwin" / "robotwin_ahawam").rglob(
            f"{task_name}/_result_random.txt"
        ),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )

    result = {
        "task_name": task_name,
        "tag": tag,
        "action_horizon": action_horizon,
        "chunks_per_video_prefill": cpp,
        "wall_time": wall_time,
        "num_episodes": num_episodes,
        "num_success": 0,
        "success_rate": 0.0,
        "analysis_dir": None,
        "episode_logs": [],
    }

    if result_dirs:
        result_file = result_dirs[0]
        analysis_dir = result_file.parent / "analysis"
        result["analysis_dir"] = str(analysis_dir)

        # Parse the result file for the per-run success rate.
        try:
            text = result_file.read_text(encoding="utf-8").strip()
            last_line = text.splitlines()[-1].strip().strip("[]")
            rates = [float(x) for x in last_line.split() if x]
            result["success_rate"] = float(sum(rates) / len(rates)) if rates else 0.0
            result["num_success"] = int(round(result["success_rate"] * result["num_episodes"]))
        except Exception as e:
            print(f"[WARN] Failed to parse result file {result_file}: {e}")

        # Load only analysis files written during this run to avoid mixing
        # with episodes from previous executions.
        if analysis_dir.exists():
            for json_path in sorted(analysis_dir.glob("episode*_analysis.json")):
                if json_path.stat().st_mtime < wall_t0:
                    continue
                try:
                    with open(json_path, "r", encoding="utf-8") as f:
                        ep = json.load(f)
                    result["episode_logs"].append(ep)
                except Exception as e:
                    print(f"[WARN] Failed to load {json_path}: {e}")
            # If we got episode logs, use the actual count.
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
    parser = argparse.ArgumentParser(description="AHA-WAM parameter sweep")
    parser.add_argument("--tasks", nargs="+", required=True, help="Tasks to evaluate")
    parser.add_argument("--num_episodes", type=int, default=5)
    parser.add_argument(
        "--output_dir",
        type=str,
        default=str(
            PROJECT_ROOT / "aha-wam-runs" / "robotwin" / "param_sweep_analysis"
        ),
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Configs to sweep. Baseline: ah=64, cpp=2
    configs = [
        (64, 2),  # baseline
        (64, 1),  # more frequent video prefill
        (32, 2),  # shorter action horizon
        (32, 1),  # both
        (16, 2),  # very short action horizon
        (16, 1),  # very short + frequent prefill
    ]

    all_results = []
    for task_name in args.tasks:
        for ah, cpp in configs:
            # Skip invalid: cpp cannot exceed num_chunks = ah // 16
            if cpp > ah // 16:
                print(f"[SKIP] {task_name} ah={ah} cpp={cpp}: cpp > num_chunks")
                continue
            result = run_config(
                task_name=task_name,
                action_horizon=ah,
                cpp=cpp,
                num_episodes=args.num_episodes,
                output_dir=output_dir,
            )
            if result is not None:
                all_results.append(result)
                summary_path = output_dir / "summary.json"
                with open(summary_path, "w", encoding="utf-8") as f:
                    json.dump(all_results, f, indent=2, default=str)

    # Write CSV
    csv_path = output_dir / "summary.csv"
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("task_name,tag,action_horizon,chunks_per_video_prefill,num_episodes,num_success,success_rate,wall_time\n")
        for r in all_results:
            f.write(
                f"{r['task_name']},{r['tag']},{r['action_horizon']},{r['chunks_per_video_prefill']},"
                f"{r['num_episodes']},{r['num_success']},{r['success_rate']:.4f},{r['wall_time']:.2f}\n"
            )

    print(f"\n{'='*80}")
    print(f"Param sweep finished. Summary: {summary_path} and {csv_path}")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
