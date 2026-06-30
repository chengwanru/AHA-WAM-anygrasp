from __future__ import annotations

import hashlib
import os
import random
import sys
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset


MAX_GETITEM_ATTEMPT = 5
DEFAULT_PROMPT = "A video recorded from a robot's point of view executing the following instruction: {task}"


@dataclass(frozen=True)
class _ClipRef:
    source_index: int
    episode_id: int
    clip_index: int


def _ensure_marmalade_importable(marmalade_src_path: str | None) -> None:
    if marmalade_src_path:
        path = Path(marmalade_src_path).expanduser().resolve()
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
        return

    workspace_src = Path(__file__).resolve().parents[4] / "marmalade" / "src"
    if workspace_src.exists() and str(workspace_src) not in sys.path:
        sys.path.insert(0, str(workspace_src))


def _expand_dataset_dirs(dataset_dirs: Sequence[str]) -> list[str]:
    expanded: list[str] = []
    for raw in dataset_dirs:
        text = str(raw)
        matches = sorted(str(p) for p in Path().glob(text) if p.exists()) if not Path(text).is_absolute() else []
        if not matches:
            import glob

            matches = sorted(glob.glob(text))
        expanded.extend(matches or [text])
    deduped = list(dict.fromkeys(expanded))
    if not deduped:
        raise ValueError("`dataset_dirs` resolved to an empty dataset list.")
    return deduped


def _load_state_normalizer(lrds):
    from marmalade.data.action_normalize import StateNormalize

    raw = lrds.meta.stats.get("observation.state") if lrds.meta.stats else None
    if raw is None or "min" not in raw or "max" not in raw:
        return None
    lo = np.asarray(raw["min"], dtype=np.float32).reshape(-1)
    hi = np.asarray(raw["max"], dtype=np.float32).reshape(-1)
    if lo.shape != hi.shape or lo.ndim != 1:
        return None
    return StateNormalize(lo, hi)


def _load_action_normalizer(action_stats_path: str | None, action_norm_modes: dict[str, str] | None):
    if not action_stats_path:
        return None
    from marmalade.data.action_normalize import Normalize, load_part_stats

    default_modes = {
        "left_arm_joint": "min_max",
        "right_arm_joint": "min_max",
        "left_arm_pose": "min_max",
        "right_arm_pose": "min_max",
        "camera_joint": "min_max",
        "camera_pose": "min_max",
        "base_vel": "min_max",
        "left_hand": "min_max",
        "right_hand": "min_max",
    }
    if action_norm_modes is not None:
        default_modes.update(dict(action_norm_modes))
    return Normalize(load_part_stats(action_stats_path), modes=default_modes)


class AnyGraspV2LeRobotDataset(Dataset):
    """Map-style AnyGrasp v2 LeRobot v3 adapter for AHA-WAM.

    The adapter reuses Marmalade's LeRobot v3 episode loading, torchcodec video
    decode, shared64 joint action parsing, action normalization, and state
    normalization. It emits AHA-WAM training samples:

    - video: [3, 9, 256, 448]
    - action: [64, 64]
    - action_dim_mask: [64, 64], True for valid shared64 dimensions
    - proprio: [64, 33]
    - context/context_mask from AHA-WAM text embedding cache
    """

    def __init__(
        self,
        dataset_dirs: Sequence[str],
        *,
        marmalade_src_path: str | None = None,
        main_camera_key: str = "observation.images.top",
        action_stats_path: str | None = None,
        action_norm_modes: dict[str, str] | None = None,
        text_embedding_cache_dir: str | None = None,
        context_len: int = 128,
        prompt_field: str = "expand_task",
        video_size: Sequence[int] = (256, 448),
        video_window_frames: int = 65,
        video_sampled_frames: Sequence[int] | None = None,
        video_stride_frames: int | None = 1,
        video_fps: int = 30,
        action_rate_hz: float = 30.0,
        action_horizon: int = 64,
        action_history_seconds: float = 0.0,
        action_window_alignment: str = "frame_intervals",
        state_context_steps: int | None = None,
        video_backend: str = "pyav",
        pretrained_norm_stats: str | None = None,
        val_set_proportion: float = 0.01,
        is_training_set: bool = True,
        seed: int = 42,
        decode_device: str | None = "cpu",
        max_padding_retry: int = 5,
        episode_cache_size: int = 32,
    ) -> None:
        del pretrained_norm_stats
        _ensure_marmalade_importable(marmalade_src_path)

        from ledataset.datasets.lerobot_dataset import LeRobotDataset
        from marmalade.data.providers.lerobot import (
            LeRobotEpisodicDataset,
            local_lerobot_episode_indices,
        )
        from marmalade.data.video_utils import count_valid_video_clips

        self.dataset_dirs = _expand_dataset_dirs(dataset_dirs)
        self.main_camera_key = str(main_camera_key)
        self.text_embedding_cache_dir = text_embedding_cache_dir
        self.context_len = int(context_len)
        self.prompt_field = str(prompt_field)
        self.video_size = tuple(int(x) for x in video_size)
        self.video_window_frames = int(video_window_frames)
        self.video_sampled_frames = (
            tuple(range(0, self.video_window_frames, 8))
            if video_sampled_frames is None
            else tuple(int(x) for x in video_sampled_frames)
        )
        self.video_stride_frames = None if video_stride_frames is None else int(video_stride_frames)
        self.video_fps = int(video_fps)
        self.action_rate_hz = float(action_rate_hz)
        self.action_horizon = int(action_horizon)
        self.action_history_seconds = float(action_history_seconds)
        self.action_window_alignment = str(action_window_alignment)
        self.state_context_steps = int(state_context_steps or action_horizon)
        self.max_padding_retry = int(max_padding_retry)
        self._episode_proprio_cache_maxsize = max(int(episode_cache_size), 0)
        self._episode_proprio_cache: OrderedDict[tuple[int, int], tuple[Any, Any, Any, Any]] = OrderedDict()
        self._text_context_cache: OrderedDict[str, tuple[torch.Tensor, torch.Tensor]] = OrderedDict()
        self._text_context_cache_maxsize = 128

        if len(self.video_sampled_frames) != 9:
            raise ValueError(
                f"AnyGrasp AHA-WAM adapter expects 9 sampled video frames, got {len(self.video_sampled_frames)}."
            )
        if self.action_horizon != 64:
            raise ValueError(f"AnyGrasp shared64 AHA-WAM adapter expects action_horizon=64, got {action_horizon}.")

        rng = np.random.default_rng(int(seed))
        action_normalizer = _load_action_normalizer(action_stats_path, action_norm_modes)

        self._providers = []
        self._index: list[_ClipRef] = []
        for source_index, ds_dir in enumerate(self.dataset_dirs):
            root = Path(ds_dir).expanduser().resolve()
            repo_id = root.name
            lrds = LeRobotDataset(
                repo_id,
                root=str(root),
                video_backend=video_backend,
                episodes=local_lerobot_episode_indices(root),
            )
            episode_ids = [
                int(episode.get("episode_index", row_idx))
                for row_idx, episode in enumerate(lrds.meta.episodes)
            ]
            if val_set_proportion > 1e-6:
                shuffled = np.asarray(episode_ids, dtype=np.int64)
                rng.shuffle(shuffled)
                split_idx = int(len(shuffled) * (1.0 - float(val_set_proportion)))
                selected = shuffled[:split_idx] if is_training_set else shuffled[split_idx:]
                episode_ids = [int(x) for x in selected.tolist()]

            provider = LeRobotEpisodicDataset(
                lrds,
                main_camera_key=self.main_camera_key,
                shuffle=False,
                seed=seed,
                video_window_frames=self.video_window_frames,
                video_stride_frames=self.video_stride_frames,
                video_sampled_frames=self.video_sampled_frames,
                video_fps=self.video_fps,
                action_rate_hz=self.action_rate_hz,
                action_history_seconds=self.action_history_seconds,
                action_window_steps=self.action_horizon,
                action_window_alignment=self.action_window_alignment,
                state_context_steps=self.state_context_steps,
                video_resolution=self.video_size,
                action_space="shared64",
                action_part_mode="fourier",
                hand_mode="joint",
                normalizer=action_normalizer,
                state_normalizer=_load_state_normalizer(lrds),
                episode_indices=np.asarray(episode_ids, dtype=np.int64),
                rank=0,
                world_size=1,
                decode_device=decode_device,
            )
            self._providers.append(provider)

            for episode_id in episode_ids:
                episode = provider._episode_metadata(episode_id)
                start_ts = float(episode[f"videos/{self.main_camera_key}/from_timestamp"])
                end_ts = float(episode[f"videos/{self.main_camera_key}/to_timestamp"])
                n_clips = count_valid_video_clips(
                    start_ts=start_ts,
                    end_ts=end_ts,
                    fps=self.video_fps,
                    num_frames_per_clip=self.video_window_frames,
                    video_stride_frames=self.video_stride_frames,
                )
                self._index.extend(
                    _ClipRef(source_index=source_index, episode_id=int(episode_id), clip_index=i)
                    for i in range(n_clips)
                )

        if not self._index:
            raise RuntimeError("AnyGraspV2LeRobotDataset built zero valid clips.")

    def __len__(self) -> int:
        return len(self._index)

    def _get_cached_text_context(self, prompt: str) -> tuple[torch.Tensor, torch.Tensor]:
        if self.text_embedding_cache_dir is None:
            raise ValueError("text_embedding_cache_dir is not set.")
        hashed = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
        if hashed in self._text_context_cache:
            self._text_context_cache.move_to_end(hashed)
            context, context_mask = self._text_context_cache[hashed]
            return context.clone(), context_mask.clone()

        cache_path = os.path.join(
            self.text_embedding_cache_dir,
            f"{hashed}.t5_len{self.context_len}.wan22ti2v5b.pt",
        )
        if not os.path.exists(cache_path):
            raise FileNotFoundError(
                f"Missing text embedding cache: {cache_path}. "
                "Run scripts/precompute_text_embeds.py first."
            )
        payload = torch.load(cache_path, map_location="cpu")
        context = payload["context"]
        context_mask = payload["mask"].bool()
        if context.ndim != 2 or context.shape[0] != self.context_len:
            raise ValueError(f"Cached context shape mismatch in {cache_path}: {tuple(context.shape)}")
        if context_mask.ndim != 1 or context_mask.shape[0] != self.context_len:
            raise ValueError(f"Cached context mask shape mismatch in {cache_path}: {tuple(context_mask.shape)}")

        self._text_context_cache[hashed] = (context, context_mask)
        if len(self._text_context_cache) > self._text_context_cache_maxsize:
            self._text_context_cache.popitem(last=False)
        return context.clone(), context_mask.clone()

    def _episode_task(self, provider, episode_id: int) -> str:
        episode = provider._episode_metadata(episode_id)
        value = episode.get(self.prompt_field) if isinstance(episode, dict) else None
        if value is not None:
            if isinstance(value, (list, tuple)) and value:
                return str(value[0])
            text = str(value).strip()
            if text:
                return text
        return str(provider._resolve_episode_task(episode, episode_id))

    def _load_episode_proprio(self, source_index: int, episode_id: int):
        key = (int(source_index), int(episode_id))
        if key in self._episode_proprio_cache:
            self._episode_proprio_cache.move_to_end(key)
            return self._episode_proprio_cache[key]
        provider = self._providers[source_index]
        timestamps, actions, states, action_fields = provider._load_episode_proprio(episode_id)
        if provider.state_normalizer is not None:
            states = np.asarray(provider.state_normalizer(states), dtype=np.float32)
        value = (timestamps, actions, states, action_fields)
        if self._episode_proprio_cache_maxsize > 0:
            self._episode_proprio_cache[key] = value
            if len(self._episode_proprio_cache) > self._episode_proprio_cache_maxsize:
                self._episode_proprio_cache.popitem(last=False)
        return value

    def _getitem_once(self, index: int) -> dict[str, Any]:
        from marmalade.data.providers.lerobot import _clip_frame_is_pad
        from marmalade.data.shared_action_space import ACTION_PART_KEYS, attach_actions_and_mask
        from marmalade.data.utils import slice_action_window

        ref = self._index[int(index)]
        provider = self._providers[ref.source_index]

        all_video = provider._load_episode_video(ref.episode_id)
        clip = all_video[ref.clip_index]
        frames = provider.transforms(clip.data)
        if frames.ndim != 4 or frames.shape[0] != 9 or frames.shape[1] != 3:
            raise ValueError(f"Expected frames [9,3,H,W], got {tuple(frames.shape)}")
        video = frames.permute(1, 0, 2, 3).contiguous()

        clip_start_ts = float(clip.pts_seconds[0].item())
        window_start_ts = clip_start_ts - self.action_history_seconds
        timestamps, all_actions, all_states, all_action_fields = self._load_episode_proprio(
            ref.source_index,
            ref.episode_id,
        )

        state_values, _ = slice_action_window(
            timestamps,
            all_states,
            clip_start_ts,
            self.state_context_steps,
        )

        chunk_values: dict[str, Any] = {
            "states": torch.from_numpy(state_values),
            "task": self._episode_task(provider, ref.episode_id),
            "episode_id": ref.episode_id,
            "clip_index": ref.clip_index,
            "is_last": ref.clip_index == len(all_video) - 1,
        }

        action_is_pad = None
        if all_actions is not None:
            sliced, action_is_pad = provider._slice_action_field(
                timestamps,
                all_actions,
                "actions",
                clip_start_ts,
                window_start_ts,
                self.action_horizon,
            )
            chunk_values["actions"] = torch.from_numpy(sliced)

        for key in ACTION_PART_KEYS:
            values = all_action_fields[key]
            if values is None:
                chunk_values[key] = None
                continue
            sliced, part_pad = provider._slice_action_field(
                timestamps,
                values,
                key,
                clip_start_ts,
                window_start_ts,
                self.action_horizon,
            )
            chunk_values[key] = torch.from_numpy(sliced)
            if action_is_pad is None:
                action_is_pad = part_pad

        if action_is_pad is None:
            raise ValueError("Failed to build action padding mask for AnyGrasp sample.")
        attach_actions_and_mask(chunk_values, arm_mode="joint")

        image_is_pad = _clip_frame_is_pad(clip)
        if image_is_pad is None:
            image_is_pad = torch.zeros(video.shape[1], dtype=torch.bool)
        else:
            image_is_pad = image_is_pad.to(dtype=torch.bool, device="cpu")

        instruction = DEFAULT_PROMPT.format(task=chunk_values["task"])
        context, context_mask = self._get_cached_text_context(instruction)
        context[~context_mask] = 0.0
        context_mask = torch.ones_like(context_mask)

        action = chunk_values["actions"].to(torch.float32)
        action_dim_mask = chunk_values["action_dim_mask"].to(torch.bool)
        proprio = chunk_values["states"].to(torch.float32)
        if action.shape != (self.action_horizon, 64):
            raise ValueError(f"Expected action [64,64], got {tuple(action.shape)}")
        if action_dim_mask.shape != action.shape:
            raise ValueError(
                f"Expected action_dim_mask {tuple(action.shape)}, got {tuple(action_dim_mask.shape)}"
            )
        if proprio.shape != (self.state_context_steps, 33):
            raise ValueError(f"Expected proprio [{self.state_context_steps},33], got {tuple(proprio.shape)}")

        return {
            "video": video,
            "action": action,
            "action_is_pad": torch.as_tensor(action_is_pad, dtype=torch.bool),
            "action_dim_mask": action_dim_mask,
            "proprio": proprio,
            "proprio_is_pad": torch.zeros(self.state_context_steps, dtype=torch.bool),
            "prompt": instruction,
            "context": context,
            "context_mask": context_mask,
            "image_is_pad": image_is_pad,
            "_sample_idx": int(index),
            "episode_id": int(ref.episode_id),
            "clip_index": int(ref.clip_index),
        }

    def __getitem__(self, index: int) -> dict[str, Any]:
        last_error: Exception | None = None
        sample_index = int(index)
        for _ in range(max(self.max_padding_retry, 1)):
            try:
                return self._getitem_once(sample_index)
            except Exception as exc:  # noqa: BLE001 - keep dataset robust under bad clips.
                last_error = exc
                sample_index = random.randrange(len(self._index))
        assert last_error is not None
        raise last_error
