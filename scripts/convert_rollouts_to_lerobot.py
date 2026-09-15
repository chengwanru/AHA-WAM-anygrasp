#!/usr/bin/env python3
"""Convert success-filtered raw rollouts → LeRobot v2.1 dataset for AHA-WAM BC.

Input layout (from rollout_recorder.py):
  raw_root/{task}/ep_XXXXXX/{meta.json,states.npy,actions.npy,cam_*/*.jpg}

Output:
  lerobot_root/{meta,data,videos}/

Usage:
  PYTHONPATH=src python scripts/convert_rollouts_to_lerobot.py \\
    --raw_root .../skip_success_raw \\
    --out_root .../robotwin_skip_success_lerobot \\
    --success_only
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_ROOT))
sys.path.insert(0, str(PROJECT_ROOT))

from ahawam.datasets.lerobot.lerobot.lerobot_dataset import LeRobotDataset  # noqa: E402


CAM_KEYS = (
    "observation.images.cam_high",
    "observation.images.cam_left_wrist",
    "observation.images.cam_right_wrist",
)
RAW_CAM_DIRS = ("cam_high", "cam_left_wrist", "cam_right_wrist")
TARGET_HW = (480, 640)  # H, W — match robotwin2.0 meta


def _features() -> dict:
    names = [
        [
            "left_waist",
            "left_shoulder",
            "left_elbow",
            "left_forearm_roll",
            "left_wrist_angle",
            "left_wrist_rotate",
            "left_gripper",
            "right_waist",
            "right_shoulder",
            "right_elbow",
            "right_forearm_roll",
            "right_wrist_angle",
            "right_wrist_rotate",
            "right_gripper",
        ]
    ]
    feats = {
        "observation.state": {"dtype": "float32", "shape": (14,), "names": names},
        "action": {"dtype": "float32", "shape": (14,), "names": names},
    }
    for key in CAM_KEYS:
        feats[key] = {
            "dtype": "video",
            "shape": (TARGET_HW[0], TARGET_HW[1], 3),
            "names": ["height", "width", "rgb"],
        }
    return feats


def _resize_rgb(path: Path) -> np.ndarray:
    img = Image.open(path).convert("RGB")
    if img.size != (TARGET_HW[1], TARGET_HW[0]):
        img = img.resize((TARGET_HW[1], TARGET_HW[0]), Image.BILINEAR)
    return np.asarray(img, dtype=np.uint8)


def _iter_episodes(raw_root: Path, *, success_only: bool):
    eps = sorted(raw_root.glob("*/ep_*"))
    for ep_dir in eps:
        meta_path = ep_dir / "meta.json"
        if not meta_path.exists():
            continue
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if success_only and not bool(meta.get("success", False)):
            continue
        states = np.load(ep_dir / "states.npy")
        actions = np.load(ep_dir / "actions.npy")
        if states.shape[0] != actions.shape[0] or states.shape[0] == 0:
            print(f"[skip] bad length {ep_dir}: {states.shape} {actions.shape}")
            continue
        yield ep_dir, meta, states, actions


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw_root", type=Path, required=True)
    ap.add_argument("--out_root", type=Path, required=True)
    ap.add_argument("--repo_id", type=str, default="local/robotwin_skip_success")
    ap.add_argument("--fps", type=int, default=10, help="policy-step fps for timestamps")
    ap.add_argument("--success_only", action="store_true", default=True)
    ap.add_argument("--keep_failures", action="store_true", default=False)
    ap.add_argument("--min_steps", type=int, default=16)
    ap.add_argument("--video_codec", type=str, default="h264")
    args = ap.parse_args()

    success_only = bool(args.success_only) and not bool(args.keep_failures)
    raw_root = args.raw_root.resolve()
    out_root = args.out_root.resolve()
    if out_root.exists():
        # LeRobotDataset.create() uses mkdir(exist_ok=False); empty dirs from
        # parent mkdir -p must be removed, non-empty dirs must be moved aside.
        remaining = [p for p in out_root.iterdir()]
        if remaining:
            bak = out_root.with_name(
                out_root.name + f".bak_{time.strftime('%Y%m%d_%H%M%S')}"
            )
            print(f"[warn] out_root non-empty → move to {bak}")
            out_root.rename(bak)
        else:
            out_root.rmdir()
    out_root.parent.mkdir(parents=True, exist_ok=True)

    episodes = list(_iter_episodes(raw_root, success_only=success_only))
    if not episodes:
        raise SystemExit(f"no episodes under {raw_root} (success_only={success_only})")

    ds = LeRobotDataset.create(
        repo_id=args.repo_id,
        fps=int(args.fps),
        features=_features(),
        root=str(out_root),
        robot_type="aloha",
        use_videos=True,
        video_codec=args.video_codec,  # type: ignore[arg-type]
        is_compute_episode_stats_image=False,
    )

    kept = 0
    for ep_dir, meta, states, actions in episodes:
        n = int(states.shape[0])
        if n < int(args.min_steps):
            print(f"[skip] too short ({n}<{args.min_steps}): {ep_dir}")
            continue
        instruction = str(meta.get("instruction") or meta.get("task_name") or "robot task")
        task_tuple = [instruction, instruction, "success", "success"]
        for t in range(n):
            cams = {}
            for cam_key, raw_dir in zip(CAM_KEYS, RAW_CAM_DIRS):
                frame_path = ep_dir / raw_dir / f"frame_{t:06d}.jpg"
                if not frame_path.exists():
                    raise FileNotFoundError(frame_path)
                cams[cam_key] = _resize_rgb(frame_path)
            frame = {
                "observation.state": states[t].astype(np.float32),
                "action": actions[t].astype(np.float32),
                **cams,
            }
            # Persist jpegs ourselves (upstream _save_image is commented out).
            for cam_key, arr in cams.items():
                img_path = ds._get_image_file_path(
                    episode_index=int(ds.episode_buffer["episode_index"]),
                    image_key=cam_key,
                    frame_index=int(ds.episode_buffer["size"]),
                )
                img_path.parent.mkdir(parents=True, exist_ok=True)
                Image.fromarray(arr).save(img_path, quality=90)
            ds.add_frame(frame, task=task_tuple)
        ds.save_episode()
        kept += 1
        print(f"[ok] episode {kept} from {ep_dir} steps={n} task={meta.get('task_name')}")

    print(f"DONE kept={kept} out={out_root}")


if __name__ == "__main__":
    main()
