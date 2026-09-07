"""
RobotWin single-task evaluation entrypoint (Hydra).

Features:
- Read `configs/sim_robotwin.yaml`.
- Check or create the symlink:
  `RoboTwin/policy/ahawam -> experiments/robotwin/ahawam`.
- Forward config overrides to the official RoboTwin entrypoint
  `script/eval_policy.py` and save logs.

Common arguments:
- `ckpt`: path to the AHAWAM checkpoint (required).
- `EVALUATION.task_name`: task name to evaluate (required).
- `gpu_id`: sets `CUDA_VISIBLE_DEVICES`.

Examples:
1) Minimal run
   python experiments/robotwin/eval_robotwin_single.py \
     ckpt=/path/to/ckpt.pt \
     EVALUATION.task_name=click_alarmclock

2) Run with more evaluation overrides
   python experiments/robotwin/eval_robotwin_single.py \
     ckpt=/path/to/ckpt.pt \
     EVALUATION.task_name=click_alarmclock \
     EVALUATION.task_config=demo_randomized \
     EVALUATION.num_inference_steps=4 \
     gpu_id=0
"""

import importlib
import os
import subprocess
import sys
import sysconfig
from datetime import datetime
from pathlib import Path
from typing import Any

import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

PROJECT_ROOT = Path(__file__).resolve().parents[2]
POLICY_NAME = "ahawam_policy"


# Candidate roots used to auto-detect cluster vs local mounts when a hardcoded
# path does not exist in the current environment.
_CANDIDATE_DATASET_ROOTS = [
    Path("/home/ma-user/work/dataset"),
    Path("/opt/huawei/dataset"),
]


def _resolve_path(path_str: str, *, base: Path) -> Path:
    path = Path(os.path.expanduser(os.path.expandvars(str(path_str))))
    if not path.is_absolute():
        path = (base / path).resolve()
    return path.resolve()


def _resolve_optional_path(path_value: Any, *, base: Path) -> Path | None:
    if path_value is None:
        return None
    text = str(path_value).strip()
    if text == "" or text.lower() in {"none", "null"}:
        return None
    return _resolve_path(text, base=base)


def _resolve_path_with_fallbacks(path_str: str, *, base: Path) -> Path:
    """Resolve a path, falling back to alternate dataset roots if missing."""
    path = _resolve_path(path_str, base=base)
    if path.exists():
        return path

    # If the path starts with a known dataset root, try swapping it for other
    # candidate roots. This lets the same config work on local workstation and
    # cluster mounts without manual edits.
    for root in _CANDIDATE_DATASET_ROOTS:
        if str(path).startswith(str(root)):
            relative = path.relative_to(root)
            for other_root in _CANDIDATE_DATASET_ROOTS:
                if other_root == root:
                    continue
                candidate = (other_root / relative).resolve()
                if candidate.exists():
                    return candidate
            break
    return path


def _resolve_dataset_stats_path(cfg: DictConfig, ckpt_path: Path) -> Path:
    explicit = _resolve_optional_path(cfg.EVALUATION.dataset_stats_path, base=PROJECT_ROOT)
    candidates: list[Path] = []
    if explicit is not None:
        candidates.append(explicit)

    for parent in list(ckpt_path.parents)[:4]:
        candidates.append((parent / "dataset_stats.json").resolve())

    seen: set[Path] = set()
    for path in candidates:
        resolved = path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        if resolved.exists():
            return resolved

    raise FileNotFoundError(
        "Failed to locate dataset_stats.json. Tried explicit "
        "EVALUATION.dataset_stats_path and checkpoint parent directories. "
        "Please pass EVALUATION.dataset_stats_path=/path/to/dataset_stats.json."
    )


def _resolve_ckpt_tag(ckpt_path: Path) -> str:
    parts = ckpt_path.resolve().parts
    if "runs" in parts:
        runs_idx = parts.index("runs")
        if runs_idx + 2 >= len(parts):
            raise ValueError(
                f"`ckpt` under runs must follow .../runs/<task>/<date_dir>/..., got: {ckpt_path}"
            )
        task_name = parts[runs_idx + 1]
        date_dir = parts[runs_idx + 2]
        if task_name == "" or date_dir == "":
            raise ValueError(
                f"`ckpt` under runs must follow .../runs/<task>/<date_dir>/..., got: {ckpt_path}"
            )
        return f"{task_name}_{date_dir}"
    return ckpt_path.stem


def _ensure_policy_symlink(robotwin_root: Path, policy_source_dir: Path) -> Path:
    """Create RoboTwin policy symlink; safe under multi-GPU concurrent launches."""
    import time

    policy_root = robotwin_root / "policy"
    if not policy_root.is_dir():
        raise FileNotFoundError(f"RoboTwin policy directory not found: {policy_root}")

    policy_target = policy_root / POLICY_NAME
    source_resolved = policy_source_dir.resolve()

    def _points_to_source() -> bool:
        try:
            if not policy_target.is_symlink() and not policy_target.exists():
                return False
            return policy_target.resolve() == source_resolved
        except FileNotFoundError:
            return False

    last_err: Exception | None = None
    for _ in range(10):
        if _points_to_source():
            return policy_target

        # Remove stale / broken link (another process may win the race).
        try:
            if policy_target.is_symlink() or policy_target.exists():
                # If it's a real directory (not symlink), do not delete.
                if policy_target.exists() and not policy_target.is_symlink():
                    raise RuntimeError(
                        f"Path already exists and is not a symlink: {policy_target}. "
                        "Please handle it manually to avoid overriding existing policy files."
                    )
                policy_target.unlink(missing_ok=True)
        except RuntimeError:
            raise
        except OSError as e:
            last_err = e

        try:
            policy_target.symlink_to(source_resolved, target_is_directory=True)
        except FileExistsError as e:
            last_err = e
        except OSError as e:
            last_err = e

        if _points_to_source():
            return policy_target
        time.sleep(0.05)

    raise RuntimeError(
        f"Failed to ensure policy symlink {policy_target} -> {source_resolved}: {last_err}"
    )


def _format_override_value(value: Any) -> str:
    if isinstance(value, bool):
        return "True" if value else "False"
    if value is None:
        return "None"
    if isinstance(value, (int, float)):
        return str(value)
    return repr(str(value))


def _append_override(overrides: list[str], key: str, value: Any, *, skip_none: bool = True) -> None:
    if skip_none and value is None:
        return
    overrides.extend([f"--{key}", _format_override_value(value)])


@hydra.main(version_base="1.3", config_path="../../configs", config_name="sim_robotwin.yaml")
def main(cfg: DictConfig):
    if cfg.ckpt is None:
        raise ValueError("`ckpt` must not be None.")
    if cfg.EVALUATION.task_name is None:
        raise ValueError("`EVALUATION.task_name` must not be None.")

    ckpt_path = _resolve_path_with_fallbacks(str(cfg.ckpt), base=PROJECT_ROOT)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    ckpt_tag = _resolve_ckpt_tag(ckpt_path)

    robotwin_root = _resolve_path_with_fallbacks(str(cfg.EVALUATION.robotwin_root), base=PROJECT_ROOT)
    if not robotwin_root.exists():
        raise FileNotFoundError(f"RoboTwin root not found: {robotwin_root}")

    policy_source_dir = (PROJECT_ROOT / "experiments" / "robotwin" / POLICY_NAME).resolve()
    if not policy_source_dir.is_dir():
        raise FileNotFoundError(f"Policy source directory not found: {policy_source_dir}")

    _ensure_policy_symlink(robotwin_root=robotwin_root, policy_source_dir=policy_source_dir)

    output_dir = _resolve_path(str(cfg.EVALUATION.output_dir), base=PROJECT_ROOT)
    if str(output_dir).strip() == "" or output_dir.name == "":
        raise ValueError(f"Invalid EVALUATION.output_dir: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    # CRITICAL: eval artifacts (videos / _result_*.txt) must be unique per sweep
    # job. Using only output_dir.name (== task_name) caused all ah*_cpp* jobs for
    # the same task to race on episode0.mp4 and corrupt each other's results.
    run_output_dir = output_dir
    log_file = run_output_dir / (
        f"eval_{str(cfg.EVALUATION.task_name)}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    )
    robotwin_eval_base = output_dir

    sim_cfg_path = (PROJECT_ROOT / "configs" / "sim_robotwin.yaml").resolve()
    sim_task = HydraConfig.get().runtime.choices.get("task")

    dataset_stats_path = _resolve_dataset_stats_path(cfg, ckpt_path)

    overrides: list[str] = []
    _append_override(overrides, "task_name", cfg.EVALUATION.task_name)
    _append_override(overrides, "task_config", cfg.EVALUATION.task_config)
    _append_override(overrides, "ckpt_setting", str(ckpt_path))
    _append_override(overrides, "seed", cfg.seed)
    _append_override(overrides, "policy_name", cfg.EVALUATION.policy_name)
    _append_override(overrides, "instruction_type", cfg.EVALUATION.instruction_type)
    _append_override(overrides, "eval_num_episodes", cfg.EVALUATION.eval_num_episodes)

    _append_override(overrides, "sim_cfg_path", str(sim_cfg_path))
    _append_override(overrides, "sim_task", sim_task)
    _append_override(overrides, "eval_output_dir", str(robotwin_eval_base))
    _append_override(overrides, "mixed_precision", cfg.mixed_precision)
    _append_override(overrides, "device", cfg.EVALUATION.device)
    _append_override(overrides, "dataset_stats_path", str(dataset_stats_path))
    _append_override(overrides, "action_horizon", cfg.EVALUATION.action_horizon)
    _append_override(
        overrides,
        "chunks_per_video_prefill",
        cfg.EVALUATION.chunks_per_video_prefill,
    )
    _append_override(overrides, "num_inference_steps", cfg.EVALUATION.num_inference_steps)
    _append_override(overrides, "sigma_shift", cfg.EVALUATION.sigma_shift)
    _append_override(overrides, "text_cfg_scale", cfg.EVALUATION.text_cfg_scale)
    _append_override(overrides, "negative_prompt", cfg.EVALUATION.negative_prompt)
    _append_override(overrides, "rand_device", cfg.EVALUATION.rand_device)
    _append_override(overrides, "tiled", cfg.EVALUATION.tiled)
    _append_override(overrides, "timing_enabled", cfg.EVALUATION.timing_enabled)
    _append_override(overrides, "detailed_analysis", cfg.EVALUATION.detailed_analysis)

    print(f"eval_output_dir (unique): {robotwin_eval_base}")
    print(f"eval log: {log_file}")

    cmd = [
        sys.executable,
        "-u",
        "script/eval_policy.py",
        "--config",
        f"policy/{POLICY_NAME}/deploy_policy.yml",
        "--overrides",
        *overrides,
    ]

    env = os.environ.copy()
    # NOTE: Do NOT set CUDA_VISIBLE_DEVICES here. Sapien's Vulkan renderer
    # requires the Vulkan device to be visible to CUDA, and pinning the
    # visible CUDA devices breaks that mapping in this container.
    if cfg.get("gpu_id") is not None and str(cfg.gpu_id).strip() != "":
        pass  # reserved for future per-process GPU pinning if compatible
    env["PYTHONUNBUFFERED"] = "1"

    # RoboTwin calls bare "ffmpeg"; ensure dataset/.../bin is on PATH for workers.
    ffmpeg_bin_dir = _resolve_path_with_fallbacks(
        "/home/ma-user/work/dataset/cwr_dataset_wulann/bin",
        base=PROJECT_ROOT,
    )
    if (ffmpeg_bin_dir / "ffmpeg").exists():
        env["PATH"] = f"{ffmpeg_bin_dir}{os.pathsep}{env.get('PATH', '')}"
    else:
        try:
            import imageio_ffmpeg

            ff = Path(imageio_ffmpeg.get_ffmpeg_exe())
            if ff.exists():
                env["PATH"] = f"{ff.parent}{os.pathsep}{env.get('PATH', '')}"
                # Also expose as "ffmpeg" via temp symlink if needed
                if ff.name != "ffmpeg":
                    alias_dir = Path(os.environ.get("PYTHONUSERBASE", "/tmp/ahawam_user")) / "bin"
                    alias_dir.mkdir(parents=True, exist_ok=True)
                    alias = alias_dir / "ffmpeg"
                    if not alias.exists():
                        try:
                            alias.symlink_to(ff)
                        except OSError:
                            pass
                    env["PATH"] = f"{alias_dir}{os.pathsep}{env.get('PATH', '')}"
        except Exception:
            pass

    # Headless rendering setup. The container lacks host NVIDIA GL/EGL
    # libraries, so we use the extracted 535.183.01 driver user-space libs
    # together with the Vulkan loader/ICD files shipped by sapien.
    conda_prefix = Path(sys.executable).parent.parent
    conda_lib = conda_prefix / "lib"
    # Prefer the installed sapien package location (works with --user installs).
    try:
        import sapien as _sapien_pkg

        sapien_vulkan_dir = Path(_sapien_pkg.__file__).resolve().parent / "vulkan_library"
    except Exception:
        sapien_vulkan_dir = Path(sysconfig.get_path("purelib")) / "sapien" / "vulkan_library"

    nvidia_driver_dir = _resolve_path_with_fallbacks(
        "/home/ma-user/work/dataset/cwr_dataset_wulann/nvidia-driver-libs/nvidia-535.183.01",
        base=PROJECT_ROOT,
    )
    sapien_libs_dir = _resolve_path_with_fallbacks(
        "/home/ma-user/work/dataset/cwr_dataset_wulann/sapien-runtime-libs",
        base=PROJECT_ROOT,
    )
    sapien_vulkan_lib = sapien_vulkan_dir / "libvulkan.so.1.3.224"
    sapien_nvidia_icd = sapien_vulkan_dir / "nvidia_icd.json"
    sapien_nvidia_egl = sapien_vulkan_dir / "10_nvidia.json"
    lavapipe_icd = conda_prefix / "share" / "vulkan" / "icd.d" / "lvp_icd.x86_64.json"

    if (
        sapien_vulkan_lib.exists()
        and sapien_nvidia_icd.exists()
        and (nvidia_driver_dir / "libGLX_nvidia.so.0").exists()
    ):
        env["SAPIEN_VULKAN_LIBRARY_PATH"] = str(sapien_vulkan_lib)
        env["VK_ICD_FILENAMES"] = str(sapien_nvidia_icd)
        env["__EGL_VENDOR_LIBRARY_FILENAMES"] = str(sapien_nvidia_egl)
        ld_parts = [str(nvidia_driver_dir)]
        if sapien_libs_dir.is_dir():
            ld_parts.append(str(sapien_libs_dir))
        if conda_lib.is_dir() and any(
            tok in str(conda_prefix) for tok in ("/envs/", "miniconda", "anaconda", "conda")
        ):
            ld_parts.append(str(conda_lib))
        if env.get("LD_LIBRARY_PATH"):
            ld_parts.append(env["LD_LIBRARY_PATH"])
        env["LD_LIBRARY_PATH"] = os.pathsep.join(ld_parts)
    else:
        # Fallback: software Vulkan (lavapipe) without ray tracing.
        env["SAPIEN_VULKAN_LIBRARY_PATH"] = str(
            sapien_vulkan_lib if sapien_vulkan_lib.exists() else conda_lib / "libvulkan.so.1"
        )
        if lavapipe_icd.exists():
            env["VK_ICD_FILENAMES"] = str(lavapipe_icd)
        if sapien_nvidia_egl.exists():
            env["__EGL_VENDOR_LIBRARY_FILENAMES"] = str(sapien_nvidia_egl)
        env["LD_LIBRARY_PATH"] = f"{conda_lib}{os.pathsep}{env.get('LD_LIBRARY_PATH', '')}"

    with open(log_file, "w", encoding="utf-8") as log_f:
        process = subprocess.Popen(
            cmd,
            cwd=str(robotwin_root),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            log_f.write(line)
            log_f.flush()
        return_code = process.wait()

    if return_code != 0:
        raise RuntimeError(f"RoboTwin evaluation failed with return code {return_code}. Log: {log_file}")

    print(f"Evaluation finished successfully. Log saved to: {log_file}")
    OmegaConf.save(
        config=cfg,
        f=str(run_output_dir / f"eval_config_{str(cfg.EVALUATION.task_name)}.yaml"),
    )


if __name__ == "__main__":
    main()
