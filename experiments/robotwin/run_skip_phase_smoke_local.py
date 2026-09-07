#!/usr/bin/env python3
"""Local 2-GPU smoke: baseline vs skip_phase (budgeted) on one task.

Uses ahawam conda env. Does not touch the cluster sweep job.
"""
from __future__ import annotations

import json
import os
import subprocess
import time
from pathlib import Path

PROJECT_ROOT = Path("/home/ma-user/work/dataset/cwr_dataset_wulann/AHA-WAM-anygrasp")
EVAL_ENTRY = PROJECT_ROOT / "experiments/robotwin/eval_robotwin_single.py"
OUTPUT_BASE = Path(
    "/home/ma-user/work/dataset/cwr_dataset_wulann/aha-wam-runs/robotwin/video_dit_skip_phase_smoke_v2"
)
LOG = Path("/tmp/run_skip_phase_smoke_local.log")
PYTHON = "/opt/huawei/miniconda/envs/python39/envs/ahawam/bin/python"
AHAWAM_LIB = "/opt/huawei/miniconda/envs/python39/envs/ahawam/lib"
DATA = "/home/ma-user/work/dataset/cwr_dataset_wulann"

TASK = os.environ.get("SMOKE_TASK", "place_mouse_pad")
NUM_EPISODES = int(os.environ.get("SMOKE_NUM_EPISODES", "5"))
ACTION_HORIZON = 64
CPP = 2


def log(msg: str) -> None:
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}"
    print(line, flush=True)
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def build_env(gpu_id: int, mode: str) -> dict:
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    env["GPU_NUMBER"] = str(gpu_id + 1)
    env["SKIP_EXPERT_CHECK"] = "1"
    env["VIDEO_DIT_MODE"] = mode
    env["OVCR_DIAG_MODE"] = "baseline"
    env["SKIP_PHASE_TASK_NAME"] = TASK
    env["SKIP_PHASE_NEAR_THRESH_M"] = os.environ.get("SKIP_PHASE_NEAR_THRESH_M", "0.10")
    env["SKIP_PHASE_MAX_CONSECUTIVE_SKIPS"] = os.environ.get(
        "SKIP_PHASE_MAX_CONSECUTIVE_SKIPS", "1"
    )
    env["DIFFSYNTH_SKIP_DOWNLOAD"] = "true"
    env["DIFFSYNTH_MODEL_BASE_PATH"] = str(PROJECT_ROOT / "checkpoints")
    env["HF_HUB_OFFLINE"] = "1"
    env["TRANSFORMERS_OFFLINE"] = "1"
    env["PYOPENGL_PLATFORM"] = "egl"

    path_parts = [
        str(Path(PYTHON).parent),
        f"{DATA}/bin",
        env.get("PATH", ""),
    ]
    env["PATH"] = ":".join(p for p in path_parts if p)

    ld = [
        f"{DATA}/nvidia-driver-libs/nvidia-535.183.01",
        f"{DATA}/sapien-runtime-libs",
        AHAWAM_LIB,
        env.get("LD_LIBRARY_PATH", ""),
    ]
    env["LD_LIBRARY_PATH"] = ":".join(p for p in ld if p)
    env["PYTHONPATH"] = f"{PROJECT_ROOT}/src:{PROJECT_ROOT}:{env.get('PYTHONPATH', '')}"
    return env


def launch(mode: str, gpu_id: int) -> subprocess.Popen:
    out = OUTPUT_BASE / mode / TASK
    out.mkdir(parents=True, exist_ok=True)
    log_file = out / f"{TASK}_{mode}_gpu{gpu_id}.log"
    cmd = [
        PYTHON,
        "-u",
        str(EVAL_ENTRY),
        f"EVALUATION.task_name={TASK}",
        f"EVALUATION.eval_num_episodes={NUM_EPISODES}",
        "EVALUATION.timing_enabled=True",
        "EVALUATION.detailed_analysis=True",
        f"EVALUATION.action_horizon={ACTION_HORIZON}",
        f"EVALUATION.chunks_per_video_prefill={CPP}",
        "EVALUATION.task_config=demo_randomized",
        f"EVALUATION.output_dir={out}",
    ]
    env = build_env(gpu_id, mode)
    log(f"[LAUNCH] {mode}/{TASK} on GPU{gpu_id} eps={NUM_EPISODES} near=0.10 max_consec=1 OVCR=baseline")
    log(f"  log -> {log_file}")
    fout = open(log_file, "w", encoding="utf-8")
    proc = subprocess.Popen(
        cmd,
        cwd=str(PROJECT_ROOT),
        env=env,
        stdout=fout,
        stderr=subprocess.STDOUT,
    )
    proc._smoke_log = fout  # type: ignore[attr-defined]
    proc._smoke_meta = {"mode": mode, "gpu": gpu_id, "out": str(out), "t0": time.time()}  # type: ignore
    return proc


def summarize_job(out_dir: Path, mode: str) -> dict:
    analysis = out_dir / "analysis"
    eps = sorted(analysis.glob("episode*_analysis.json")) if analysis.is_dir() else []
    n = len(eps)
    succ = 0
    skip_ratios = []
    eligible_ratios = []
    for p in eps:
        d = json.loads(p.read_text())
        succ += int(bool(d.get("success")))
        sm = d.get("skip_phase_summary") or {}
        if "skip_ratio_among_decisions" in sm:
            skip_ratios.append(float(sm["skip_ratio_among_decisions"]))
        # fallback from timing
        timing = d.get("timing") or {}
        decisions = float(timing.get("prefill_decisions") or 0)
        skipped = float(timing.get("skipped_prefills") or 0)
        if decisions > 0:
            skip_ratios.append(skipped / decisions)
        # eligible from step_log if present
        elig = 0
        skips = 0
        decisions_sl = 0
        for step in d.get("step_log") or []:
            sp = step.get("skip_phase")
            if not isinstance(sp, dict):
                continue
            if "eligible" in sp or "skip" in sp:
                decisions_sl += 1
                elig += int(bool(sp.get("eligible")))
                skips += int(bool(sp.get("skip")))
        if decisions_sl:
            eligible_ratios.append(elig / decisions_sl)
    return {
        "mode": mode,
        "episodes": n,
        "success": succ,
        "success_rate": (succ / n) if n else None,
        "skip_ratio_mean": (sum(skip_ratios) / len(skip_ratios)) if skip_ratios else None,
        "eligible_ratio_mean": (sum(eligible_ratios) / len(eligible_ratios)) if eligible_ratios else None,
        "output_dir": str(out_dir),
    }


def main() -> None:
    if LOG.exists():
        LOG.unlink()
    OUTPUT_BASE.mkdir(parents=True, exist_ok=True)
    log("=" * 72)
    log(f"skip_phase smoke v2 | task={TASK} eps={NUM_EPISODES} ah={ACTION_HORIZON} cpp={CPP}")
    log("policy: far REACH/TRANSPORT eligible; near never; consec skip<=1; OVCR=baseline")
    log("=" * 72)

    # Preflight CUDA on both GPUs
    for gid in (0, 1):
        env = build_env(gid, "baseline")
        r = subprocess.run(
            [
                PYTHON,
                "-c",
                "import torch; assert torch.cuda.is_available(); "
                "print(torch.__version__, torch.cuda.device_count(), torch.cuda.get_device_name(0))",
            ],
            env=env,
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            text=True,
        )
        if r.returncode != 0:
            log(f"GPU{gid} preflight FAILED: {r.stderr}")
            raise SystemExit(1)
        log(f"GPU{gid} preflight ok: {r.stdout.strip()}")

    procs = [
        launch("baseline", 0),
        launch("skip_phase", 1),
    ]
    results = []
    try:
        while procs:
            for proc in list(procs):
                rc = proc.poll()
                if rc is None:
                    continue
                meta = proc._smoke_meta  # type: ignore[attr-defined]
                proc._smoke_log.close()  # type: ignore[attr-defined]
                wall = time.time() - meta["t0"]
                out = Path(meta["out"])
                summary = summarize_job(out, meta["mode"])
                summary["returncode"] = rc
                summary["wall_s"] = wall
                results.append(summary)
                log(
                    f"[DONE] {meta['mode']}/{TASK} GPU{meta['gpu']} rc={rc} "
                    f"{summary['success']}/{summary['episodes']} "
                    f"skip={summary['skip_ratio_mean']} "
                    f"elig={summary['eligible_ratio_mean']} "
                    f"| {wall:.1f}s"
                )
                procs.remove(proc)
            if procs:
                time.sleep(5)
    finally:
        for proc in procs:
            proc.terminate()

    summary_path = OUTPUT_BASE / "smoke_summary.json"
    summary_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    log(f"Wrote {summary_path}")
    for r in results:
        log(f"  {r}")


if __name__ == "__main__":
    main()
