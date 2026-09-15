"""Record RoboTwin policy rollouts for success-filtered BC.

Enabled when env ROLLOUT_RECORD_DIR is set. Each episode is written under:
  {ROLLOUT_RECORD_DIR}/{task_name}/ep_{NNNNNN}/
    meta.json
    states.npy   # [T, 14] float32
    actions.npy  # [T, 14] float32
    cam_high/frame_XXXXXX.jpg
    cam_left_wrist/frame_XXXXXX.jpg
    cam_right_wrist/frame_XXXXXX.jpg

Only episodes with success=True are kept on disk (failures deleted unless
ROLLOUT_KEEP_FAILURES=1).
"""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

try:
    from PIL import Image
except ImportError:  # pragma: no cover
    Image = None  # type: ignore


def _env_flag(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "y", "on"}


def _as_uint8_rgb(img: Any) -> np.ndarray:
    arr = np.asarray(img)
    if arr.dtype != np.uint8:
        if np.issubdtype(arr.dtype, np.floating):
            max_v = float(np.nanmax(arr)) if arr.size else 1.0
            if max_v <= 1.5:
                arr = np.clip(arr * 255.0, 0, 255).astype(np.uint8)
            else:
                arr = np.clip(arr, 0, 255).astype(np.uint8)
        else:
            arr = np.clip(arr, 0, 255).astype(np.uint8)
    if arr.ndim != 3 or arr.shape[-1] != 3:
        raise ValueError(f"expected HxWx3 RGB, got shape {arr.shape}")
    return np.ascontiguousarray(arr)


class RolloutRecorder:
    def __init__(self, root: str | Path, *, task_name: str = "unknown") -> None:
        self.root = Path(root)
        self.task_name = str(task_name or "unknown").strip() or "unknown"
        self.keep_failures = _env_flag("ROLLOUT_KEEP_FAILURES", False)
        self._ep_dir: Optional[Path] = None
        self._states: list[np.ndarray] = []
        self._actions: list[np.ndarray] = []
        self._instruction: str = ""
        self._seed: Optional[int] = None
        self._episode_idx: int = 0
        self.root.mkdir(parents=True, exist_ok=True)
        self._task_dir = self.root / self.task_name
        self._task_dir.mkdir(parents=True, exist_ok=True)
        self._counter_path = self._task_dir / "_episode_counter.txt"
        if not self._counter_path.exists():
            self._counter_path.write_text("0\n", encoding="utf-8")

    def _next_episode_id(self) -> int:
        try:
            cur = int(self._counter_path.read_text(encoding="utf-8").strip() or "0")
        except Exception:
            cur = 0
        nxt = cur + 1
        self._counter_path.write_text(f"{nxt}\n", encoding="utf-8")
        return cur

    def start_episode(
        self,
        *,
        instruction: str = "",
        seed: Optional[int] = None,
        episode_idx: Optional[int] = None,
    ) -> None:
        self.finish_episode(success=False, discarded=True)
        eid = int(episode_idx) if episode_idx is not None else self._next_episode_id()
        self._episode_idx = eid
        self._ep_dir = self._task_dir / f"ep_{eid:06d}"
        if self._ep_dir.exists():
            shutil.rmtree(self._ep_dir)
        self._ep_dir.mkdir(parents=True, exist_ok=True)
        for cam in ("cam_high", "cam_left_wrist", "cam_right_wrist"):
            (self._ep_dir / cam).mkdir(parents=True, exist_ok=True)
        self._states = []
        self._actions = []
        self._instruction = str(instruction or "")
        self._seed = seed

    def record_step(self, observation: Dict[str, Any], action: np.ndarray) -> None:
        if self._ep_dir is None:
            return
        if Image is None:
            raise RuntimeError("Pillow is required for rollout recording")
        obs_data = observation.get("observation", observation)
        joint = observation.get("joint_action") or obs_data.get("joint_action") or {}
        vector = joint.get("vector")
        if vector is None:
            raise ValueError("observation missing joint_action.vector")
        state = np.asarray(vector, dtype=np.float32).reshape(-1)
        act = np.asarray(action, dtype=np.float32).reshape(-1)
        if state.shape[0] != 14 or act.shape[0] != 14:
            raise ValueError(f"expected 14-d state/action, got {state.shape}/{act.shape}")

        head = _as_uint8_rgb(obs_data["head_camera"]["rgb"])
        left = _as_uint8_rgb(obs_data["left_camera"]["rgb"])
        right = _as_uint8_rgb(obs_data["right_camera"]["rgb"])
        t = len(self._states)
        Image.fromarray(head).save(self._ep_dir / "cam_high" / f"frame_{t:06d}.jpg", quality=90)
        Image.fromarray(left).save(self._ep_dir / "cam_left_wrist" / f"frame_{t:06d}.jpg", quality=90)
        Image.fromarray(right).save(
            self._ep_dir / "cam_right_wrist" / f"frame_{t:06d}.jpg", quality=90
        )
        self._states.append(state.copy())
        self._actions.append(act.copy())

    def finish_episode(
        self,
        *,
        success: bool,
        discarded: bool = False,
        extra_meta: Optional[Dict[str, Any]] = None,
    ) -> Optional[Path]:
        if self._ep_dir is None:
            return None
        ep_dir = self._ep_dir
        n = len(self._states)
        meta = {
            "task_name": self.task_name,
            "instruction": self._instruction,
            "seed": self._seed,
            "episode_idx": self._episode_idx,
            "success": bool(success),
            "num_steps": int(n),
            "discarded": bool(discarded),
        }
        if extra_meta:
            meta.update(extra_meta)

        keep = (not discarded) and (bool(success) or self.keep_failures) and n > 0
        if keep:
            np.save(ep_dir / "states.npy", np.stack(self._states, axis=0))
            np.save(ep_dir / "actions.npy", np.stack(self._actions, axis=0))
            (ep_dir / "meta.json").write_text(
                json.dumps(meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
            )
            marker = "SUCCESS" if success else "FAIL"
            print(
                f"[RolloutRecorder] saved {marker} {ep_dir} steps={n} "
                f"task={self.task_name}",
                flush=True,
            )
        else:
            shutil.rmtree(ep_dir, ignore_errors=True)
            if not discarded:
                print(
                    f"[RolloutRecorder] dropped ep_{self._episode_idx:06d} "
                    f"success={success} steps={n} task={self.task_name}",
                    flush=True,
                )
            ep_dir = None

        self._ep_dir = None
        self._states = []
        self._actions = []
        self._instruction = ""
        self._seed = None
        return ep_dir


def maybe_make_recorder(task_name: str = "") -> Optional[RolloutRecorder]:
    root = os.environ.get("ROLLOUT_RECORD_DIR", "").strip()
    if not root:
        return None
    tn = (
        task_name
        or os.environ.get("SKIP_PHASE_TASK_NAME", "")
        or os.environ.get("ROBOTWIN_TASK_NAME", "")
        or "unknown"
    )
    return RolloutRecorder(root, task_name=tn)
