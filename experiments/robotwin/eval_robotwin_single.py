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

    # Headless rendering setup.
    # Prefer parent-provided ICD when valid; if modeset is unusable / AHAWAM_VULKAN_MODE=lavapipe,
    # force software lavapipe (nvidia GLX files may exist but still cannot render).
    conda_prefix = Path(sys.executable).parent.parent
    conda_lib = conda_prefix / "lib"
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
    vulkan_icd_dir = _resolve_path_with_fallbacks(
        "/home/ma-user/work/dataset/cwr_dataset_wulann/vulkan-icd",
        base=PROJECT_ROOT,
    )
    sapien_vulkan_lib = sapien_vulkan_dir / "libvulkan.so.1.3.224"
    lavapipe_icd_candidates = [
        vulkan_icd_dir / "lvp_icd_runtime.json",
        vulkan_icd_dir / "lvp_icd.x86_64.json",
        Path("/usr/share/vulkan/icd.d/lvp_icd.x86_64.json"),
        Path("/usr/share/vulkan/icd.d/lvp_icd.json"),
        conda_prefix / "share" / "vulkan" / "icd.d" / "lvp_icd.x86_64.json",
    ]
    lavapipe_so_candidates = [
        vulkan_icd_dir / "libvulkan_lvp.so",
        Path("/usr/lib/x86_64-linux-gnu/libvulkan_lvp.so"),
    ]

    env["NVIDIA_DRIVER_CAPABILITIES"] = env.get(
        "NVIDIA_DRIVER_CAPABILITIES", "compute,utility,graphics,display,video"
    )
    env["PYOPENGL_PLATFORM"] = env.get("PYOPENGL_PLATFORM", "egl")

    def _modeset_ok() -> bool:
        p = Path("/dev/nvidia-modeset")
        if not p.exists():
            return False
        try:
            fd = os.open(str(p), os.O_RDWR)
            os.close(fd)
            return True
        except OSError:
            return False

    def _force_lavapipe(reason: str) -> None:
        lvp_so = next((p for p in lavapipe_so_candidates if p.is_file()), None)
        icd_path = None
        if lvp_so is not None:
            icd_dir = Path(os.environ.get("PYTHONUSERBASE", "/tmp/ahawam_user")) / "vulkan_icd"
            icd_dir.mkdir(parents=True, exist_ok=True)
            icd_path = icd_dir / "lvp_icd_abs.json"
            icd_path.write_text(
                "{\n"
                '  "file_format_version": "1.0.0",\n'
                '  "ICD": {\n'
                f'    "library_path": "{lvp_so.resolve()}",\n'
                '    "api_version": "1.1.255"\n'
                "  }\n"
                "}\n",
                encoding="utf-8",
            )
            env["LD_LIBRARY_PATH"] = (
                f"{lvp_so.parent}{os.pathsep}{env.get('LD_LIBRARY_PATH', '')}"
            )
        else:
            for c in lavapipe_icd_candidates:
                if c.is_file():
                    icd_path = c
                    break
        if icd_path is None:
            raise RuntimeError(
                f"lavapipe requested ({reason}) but no ICD/libvulkan_lvp.so found. "
                f"Expected under {vulkan_icd_dir} or /usr/share/vulkan/icd.d/"
            )
        env["VK_ICD_FILENAMES"] = str(icd_path)
        env.pop("__EGL_VENDOR_LIBRARY_FILENAMES", None)
        if Path("/usr/lib/x86_64-linux-gnu/libvulkan.so.1").is_file():
            env["SAPIEN_VULKAN_LIBRARY_PATH"] = "/usr/lib/x86_64-linux-gnu/libvulkan.so.1"
        elif sapien_vulkan_lib.exists():
            env["SAPIEN_VULKAN_LIBRARY_PATH"] = str(sapien_vulkan_lib)
        env["AHAWAM_VULKAN_MODE"] = "lavapipe"
        env["SAPIEN_DISABLE_RAYTRACING"] = "1"
        env["ROBOTWIN_MPLIB_NO_SAPIEN_WORLD"] = "1"
        # Native mplib.Planner also SIGSEGV under lavapipe; stub TOPP/grippers.
        env["ROBOTWIN_MPLIB_STUB"] = "1"
        print(f"Vulkan: lavapipe ({reason}) ICD={icd_path} SO={lvp_so}")

    # Always put nvidia user-space driver libs first (before conda lib) unless lavapipe-only.
    ld_parts = []
    force_lvp = str(env.get("AHAWAM_VULKAN_MODE", "")).strip().lower() == "lavapipe"
    modeset_ok = _modeset_ok()
    if not force_lvp and not modeset_ok:
        force_lvp = True

    def _strip_nvidia_driver_libs(path: str) -> str:
        """NVIDIA GLX/EGL userspace + missing modeset => SIGSEGV on complex URDF under lavapipe."""
        keep = []
        for part in (path or "").split(os.pathsep):
            if not part:
                continue
            if "nvidia-driver-libs" in part:
                continue
            keep.append(part)
        return os.pathsep.join(keep)

    if not force_lvp and nvidia_driver_dir.is_dir():
        ld_parts.append(str(nvidia_driver_dir))
    if sapien_libs_dir.is_dir():
        ld_parts.append(str(sapien_libs_dir))
    if conda_lib.is_dir() and any(
        tok in str(conda_prefix) for tok in ("/envs/", "miniconda", "anaconda", "conda")
    ):
        ld_parts.append(str(conda_lib))
    if env.get("LD_LIBRARY_PATH"):
        inherited = env["LD_LIBRARY_PATH"]
        if force_lvp:
            inherited = _strip_nvidia_driver_libs(inherited)
        ld_parts.append(inherited)
    if force_lvp:
        # Mesa / system Vulkan first
        sys_lib = "/usr/lib/x86_64-linux-gnu"
        ld_parts = [sys_lib] + [p for p in ld_parts if p != sys_lib]
    if ld_parts:
        # de-dupe preserve order
        seen = set()
        ordered = []
        for p in ld_parts:
            for part in p.split(os.pathsep):
                if part and part not in seen:
                    seen.add(part)
                    ordered.append(part)
        env["LD_LIBRARY_PATH"] = os.pathsep.join(ordered)
        if force_lvp:
            print(
                "lavapipe LD_LIBRARY_PATH: stripped nvidia-driver-libs; "
                f"head={env['LD_LIBRARY_PATH'][:180]}..."
            )

    parent_vk = str(env.get("VK_ICD_FILENAMES", "") or "").strip()
    glx = nvidia_driver_dir / "libGLX_nvidia.so.0"
    egl = nvidia_driver_dir / "libEGL_nvidia.so.0"

    if force_lvp:
        _force_lavapipe(
            "AHAWAM_VULKAN_MODE=lavapipe"
            if str(env.get("AHAWAM_VULKAN_MODE", "")).strip().lower() == "lavapipe"
            else "nvidia-modeset unusable"
        )
    elif parent_vk and Path(parent_vk.split(":")[0]).exists():
        # Keep parent abs ICD (from train_eval_*.sh), but if it points at nvidia while
        # modeset is bad we already branched above.
        if sapien_vulkan_lib.exists() and not env.get("SAPIEN_VULKAN_LIBRARY_PATH"):
            env["SAPIEN_VULKAN_LIBRARY_PATH"] = str(sapien_vulkan_lib)
        print(f"Vulkan: reuse parent VK_ICD_FILENAMES={parent_vk}")
    elif glx.exists() and egl.exists() and modeset_ok:
        icd_dir = Path(os.environ.get("PYTHONUSERBASE", "/tmp/ahawam_user")) / "vulkan_icd"
        icd_dir.mkdir(parents=True, exist_ok=True)
        vk_icd = icd_dir / "nvidia_icd_abs.json"
        egl_icd = icd_dir / "10_nvidia_abs.json"
        vk_icd.write_text(
            "{\n"
            '  "file_format_version": "1.0.0",\n'
            '  "ICD": {\n'
            f'    "library_path": "{glx}",\n'
            '    "api_version": "1.3.242"\n'
            "  }\n"
            "}\n",
            encoding="utf-8",
        )
        egl_icd.write_text(
            "{\n"
            '  "file_format_version": "1.0.0",\n'
            '  "ICD": {\n'
            f'    "library_path": "{egl}"\n'
            "  }\n"
            "}\n",
            encoding="utf-8",
        )
        env["VK_ICD_FILENAMES"] = str(vk_icd)
        env["__EGL_VENDOR_LIBRARY_FILENAMES"] = str(egl_icd)
        if sapien_vulkan_lib.exists():
            env["SAPIEN_VULKAN_LIBRARY_PATH"] = str(sapien_vulkan_lib)
        env["AHAWAM_VULKAN_MODE"] = "nvidia"
        print(f"Vulkan: NVIDIA abs ICD @ {vk_icd}")
    else:
        _force_lavapipe("fallback")

    print(f"AHAWAM_VULKAN_MODE={env.get('AHAWAM_VULKAN_MODE')}")
    print(f"VK_ICD_FILENAMES={env.get('VK_ICD_FILENAMES')}")
    print(f"__EGL_VENDOR_LIBRARY_FILENAMES={env.get('__EGL_VENDOR_LIBRARY_FILENAMES')}")
    print(f"SAPIEN_VULKAN_LIBRARY_PATH={env.get('SAPIEN_VULKAN_LIBRARY_PATH')}")
    print(f"NVIDIA_DRIVER_CAPABILITIES={env.get('NVIDIA_DRIVER_CAPABILITIES')}")

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

    # RoboTwin test_render historically called exit() with code 0 on render failure.
    log_text = log_file.read_text(encoding="utf-8", errors="replace") if log_file.exists() else ""
    if "Render Error" in log_text or "failed to find a rendering device" in log_text:
        raise RuntimeError(
            f"RoboTwin render failed (Vulkan/EGL). Log: {log_file}\n"
            "Set AHAWAM_VULKAN_MODE=lavapipe with cwr_dataset_wulann/vulkan-icd/, "
            "or create the job with NVIDIA_DRIVER_CAPABILITIES including graphics + /dev/nvidia-modeset."
        )
    if return_code != 0:
        raise RuntimeError(f"RoboTwin evaluation failed with return code {return_code}. Log: {log_file}")

    print(f"Evaluation finished successfully. Log saved to: {log_file}")
    OmegaConf.save(
        config=cfg,
        f=str(run_output_dir / f"eval_config_{str(cfg.EVALUATION.task_name)}.yaml"),
    )


if __name__ == "__main__":
    main()
