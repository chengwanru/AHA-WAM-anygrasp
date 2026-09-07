"""
Batch RoboTwin evaluation for analysis.

Runs a list of tasks through experiments/robotwin/eval_robotwin_single.py
with timing and detailed_analysis enabled, then aggregates per-task
success rates, timing statistics, and per-episode step logs into a summary.

Example:
    python experiments/robotwin/batch_eval_for_analysis.py \
        --tasks click_alarmclock open_microwave place_can_basket handover_block \
        --num_episodes 5 \
        --output_dir /home/ma-user/work/dataset/cwr_dataset_wulann/aha-wam-runs/robotwin/batch_analysis
"""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
EVAL_ENTRY = PROJECT_ROOT / "experiments" / "robotwin" / "eval_robotwin_single.py"


def run_single_task(
    task_name: str,
    num_episodes: int,
    output_dir: Path,
    num_inference_steps: int | None = None,
    task_config: str | None = None,
) -> dict[str, Any]:
    """Run eval_robotwin_single.py for one task and return aggregated results."""
    print(f"\n{'='*60}")
    print(f"Running task: {task_name} | episodes: {num_episodes} | config: {task_config}")
    print(f"{'='*60}\n")

    task_output_dir = output_dir / task_name
    task_output_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        "-u",
        str(EVAL_ENTRY),
        f"EVALUATION.task_name={task_name}",
        f"EVALUATION.eval_num_episodes={num_episodes}",
        f"EVALUATION.timing_enabled=True",
        f"EVALUATION.detailed_analysis=True",
        f"EVALUATION.output_dir={task_output_dir}",
    ]
    if num_inference_steps is not None:
        cmd.append(f"EVALUATION.num_inference_steps={num_inference_steps}")
    if task_config is not None:
        cmd.append(f"EVALUATION.task_config={task_config}")

    t0 = time.perf_counter()
    env = os.environ.copy()
    conda_bin = Path(sys.executable).parent
    env["PATH"] = f"{conda_bin}{os.pathsep}{env.get('PATH', '')}"
    proc = subprocess.run(
        cmd,
        cwd=str(PROJECT_ROOT),
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    wall_time = time.perf_counter() - t0

    # Write full log
    log_file = task_output_dir / f"{task_name}_batch.log"
    with open(log_file, "w", encoding="utf-8") as f:
        f.write(proc.stdout)

    if proc.returncode != 0:
        print(f"[ERROR] Task {task_name} failed after {wall_time:.1f}s")
        print(proc.stdout[-2000:])
        return {
            "task_name": task_name,
            "success": False,
            "error": "subprocess failed",
            "wall_time": wall_time,
            "num_episodes": 0,
            "num_success": 0,
            "success_rate": 0.0,
        }

    # Find the latest run directory under evaluate_results/robotwin/...
    # eval_robotwin_single.py creates evaluate_results/robotwin/<ckpt_tag>/<run_ts>/<task_name>/
    result_dirs = sorted(
        (PROJECT_ROOT / "evaluate_results" / "robotwin").rglob(f"{task_name}/_result_random.txt"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )

    result = {
        "task_name": task_name,
        "success": True,
        "wall_time": wall_time,
        "num_episodes": num_episodes,
        "num_success": 0,
        "success_rate": 0.0,
        "result_file": None,
        "analysis_dir": None,
        "timing": {},
        "episode_logs": [],
    }

    if result_dirs:
        result_file = result_dirs[0]
        result["result_file"] = str(result_file)
        run_dir = result_file.parent
        analysis_dir = run_dir / "analysis"
        result["analysis_dir"] = str(analysis_dir)

        # Collect per-episode analysis JSONs and derive success counts from them
        # (more reliable than the mean success-rate line in _result_random.txt)
        if result["analysis_dir"] and Path(result["analysis_dir"]).exists():
            for json_path in sorted(Path(result["analysis_dir"]).glob("episode*_analysis.json")):
                try:
                    with open(json_path, "r", encoding="utf-8") as f:
                        ep = json.load(f)
                    result["episode_logs"].append(ep)
                except Exception as e:
                    print(f"[WARN] Failed to load {json_path}: {e}")

        result["num_success"] = int(sum(1 for ep in result["episode_logs"] if ep.get("success", False)))
        actual_episodes = len(result["episode_logs"])
        if actual_episodes > 0:
            result["num_episodes"] = actual_episodes
            result["success_rate"] = result["num_success"] / actual_episodes
        else:
            # Fallback: parse the mean success rate from _result_random.txt
            try:
                text = result_file.read_text(encoding="utf-8").strip()
                last_line = text.splitlines()[-1].strip().strip("[]")
                rates = [float(x) for x in last_line.split() if x]
                result["success_rate"] = float(np.mean(rates)) if rates else 0.0
                result["num_success"] = int(round(result["success_rate"] * result["num_episodes"]))
            except Exception as e:
                print(f"[WARN] Failed to parse result file {result_file}: {e}")

        # Aggregate timing across episodes
        if result["episode_logs"]:
            timing_keys = list(result["episode_logs"][0].get("timing", {}).keys())
            for key in timing_keys:
                values = [ep["timing"][key] for ep in result["episode_logs"] if key in ep.get("timing", {})]
                if values:
                    result["timing"][key] = {
                        "mean": float(np.mean(values)),
                        "std": float(np.std(values)),
                        "min": float(np.min(values)),
                        "max": float(np.max(values)),
                        "sum": float(np.sum(values)),
                    }

    print(
        f"[DONE] {task_name}: success {result['num_success']}/{result['num_episodes']} "
        f"({result['success_rate']*100:.1f}%) | wall {wall_time:.1f}s"
    )
    return result


def main():
    parser = argparse.ArgumentParser(description="Batch RoboTwin eval for analysis")
    parser.add_argument(
        "--tasks",
        nargs="+",
        required=True,
        help="List of RoboTwin task names to evaluate",
    )
    parser.add_argument(
        "--num_episodes",
        type=int,
        default=5,
        help="Number of episodes per task",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=str(
            PROJECT_ROOT / "evaluate_results" / "robotwin" / "batch_analysis"
        ),
        help="Directory to store batch summary and per-task logs",
    )
    parser.add_argument(
        "--num_inference_steps",
        type=int,
        default=None,
        help="Override diffusion inference steps",
    )
    parser.add_argument(
        "--task_config",
        type=str,
        default=None,
        help="RoboTwin task config to use (e.g. demo_clean or demo_randomized)",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_results = []
    for task_name in args.tasks:
        result = run_single_task(
            task_name=task_name,
            num_episodes=args.num_episodes,
            output_dir=output_dir,
            num_inference_steps=args.num_inference_steps,
            task_config=args.task_config,
        )
        all_results.append(result)

        # Save incremental summary after each task
        summary_path = output_dir / "summary.json"
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(all_results, f, indent=2, default=str)

    # Also write a CSV-style summary
    csv_path = output_dir / "summary.csv"
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("task_name,num_episodes,num_success,success_rate,wall_time,sim_per_action_mean,infer_per_action_mean\n")
        for r in all_results:
            timing = r.get("timing", {})
            sim_per_action = timing.get("sim_s", {}).get("mean", 0.0) / max(r.get("num_episodes", 1), 1)
            infer_per_action = timing.get("infer_s", {}).get("mean", 0.0) / max(r.get("num_episodes", 1), 1)
            # Adjust: timing sums are per-episode, so mean per episode divided by num_steps? Keep simple.
            f.write(
                f"{r['task_name']},{r['num_episodes']},{r['num_success']},"
                f"{r['success_rate']:.4f},{r['wall_time']:.2f},"
                f"{sim_per_action:.4f},{infer_per_action:.4f}\n"
            )

    print(f"\n{'='*60}")
    print(f"Batch eval finished. Summary saved to {summary_path} and {csv_path}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
