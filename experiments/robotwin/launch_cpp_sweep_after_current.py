#!/usr/bin/env python3
"""
Wait for the current parameter sweep to finish, then launch the systematic
cpp sweep. Designed to run unattended overnight.
"""

import os
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CPP_SWEEP = PROJECT_ROOT / "experiments" / "robotwin" / "run_cpp_sweep.py"


def build_env():
    env = os.environ.copy()
    conda_bin = Path(sys.executable).parent
    conda_lib = conda_bin.parent / "lib"
    env["PATH"] = f"{conda_bin}{os.pathsep}{env.get('PATH', '')}"
    env["LD_LIBRARY_PATH"] = f"{conda_lib}{os.pathsep}{env.get('LD_LIBRARY_PATH', '')}"
    return env


def sweep_is_running():
    ret = subprocess.run(
        ["pgrep", "-f", "run_param_sweep.py"],
        capture_output=True,
    )
    return ret.returncode == 0


def main():
    log_file = Path("/tmp/cpp_sweep_launcher.log")
    log_file.write_text(
        f"Launcher started at {time.strftime('%Y-%m-%d %H:%M:%S')}\n",
        encoding="utf-8",
    )

    print("Waiting for current parameter sweep to finish...")
    while sweep_is_running():
        time.sleep(60)

    # Give filesystem a moment to settle
    time.sleep(30)

    msg = f"Current sweep finished at {time.strftime('%Y-%m-%d %H:%M:%S')}. Launching cpp sweep.\n"
    print(msg.strip())
    log_file.write_text(msg, encoding="utf-8")

    env = build_env()
    cmd = [
        sys.executable,
        "-u",
        str(CPP_SWEEP),
        "--tasks",
        "open_microwave",
        "move_stapler_pad",
        "move_can_pot",
        "place_bread_basket",
        "place_shoe",
        "put_bottles_dustbin",
        "place_object_basket",
        "turn_switch",
        "--action_horizon",
        "64",
        "--cpp_values",
        "1",
        "2",
        "4",
        "--num_episodes",
        "5",
        "--timeout",
        "2400",
    ]

    with open(log_file, "a", encoding="utf-8") as f:
        f.write(f"Running: {' '.join(cmd)}\n")
        proc = subprocess.Popen(
            cmd,
            cwd=str(PROJECT_ROOT),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        for line in proc.stdout:
            f.write(line)
            f.flush()
        proc.wait()
        f.write(f"Cpp sweep finished with return code {proc.returncode}\n")


if __name__ == "__main__":
    main()
