"""Train-time skip_phase v2 schedule helpers (SKIP_V2 experiment).

Eval skip_v2 (deploy_policy + SkipPhaseFSM) needs sim GT TCP↔object distances.
Offline demos do not have those poses, so training uses a *proxy* eligibility:

- REACH (not holding): eligible to skip (approx far REACH; no near-field GT)
- TRANSPORT (holding, not late): eligible
- PLACE-like (holding + late episode progress, or late window): never eligible
- Budget: max_consecutive_skips (default 1) on cpp windows, same as eval

Within one training window, for each action chunk we pick which chunk-start
frame supplies OVCR/visual obs: the last *non-skipped* prefill window start.
Actions stay time-aligned (unlike the OFFSET experiment).
"""

from __future__ import annotations

from typing import List, Sequence, Tuple

import numpy as np


def gripper_holding(
    proprio_t: np.ndarray,
    *,
    closed_threshold: float = 0.3,
) -> bool:
    """RoboTwin 14-d proprio: dims 6 and 13 are left/right gripper in ~[0,1]."""
    vec = np.asarray(proprio_t, dtype=np.float64).reshape(-1)
    if vec.size < 14:
        return False
    return float(min(vec[6], vec[13])) < float(closed_threshold)


def proxy_eligible(
    *,
    holding: bool,
    episode_progress: float,
    place_progress_thresh: float = 0.85,
) -> bool:
    """Return True if this prefill decision may skip (before consecutive budget)."""
    progress = float(np.clip(episode_progress, 0.0, 1.0))
    if progress >= float(place_progress_thresh):
        return False
    if holding:
        # TRANSPORT-like: still skip-eligible until place_progress.
        return True
    # REACH-like: eligible (near-field cannot be enforced without GT distance).
    return True


def apply_consecutive_skip_budget(
    eligible: Sequence[bool],
    *,
    max_consecutive_skips: int = 1,
) -> List[bool]:
    """Mirror deploy_policy: at most N consecutive skips when eligible."""
    max_n = max(int(max_consecutive_skips), 0)
    out: List[bool] = []
    consec = 0
    for elig in eligible:
        if bool(elig) and consec < max_n:
            out.append(True)
            consec += 1
        else:
            out.append(False)
            consec = 0
    return out


def chunk_obs_source_indices(
    *,
    num_chunks: int,
    chunks_per_video_prefill: int,
    skip_prefill_per_window: Sequence[bool],
) -> List[int]:
    """For each action chunk, which chunk-start index provides visual obs.

    Prefill windows are [0, cpp), [cpp, 2*cpp), ... Matching eval cpp cadence.
    If a window skips, chunks in that window reuse the previous refresh chunk index.
    """
    n = int(num_chunks)
    cpp = int(chunks_per_video_prefill)
    if n <= 0:
        raise ValueError(f"num_chunks must be positive, got {n}")
    if cpp <= 0:
        raise ValueError(f"chunks_per_video_prefill must be positive, got {cpp}")
    num_windows = (n + cpp - 1) // cpp
    if len(skip_prefill_per_window) != num_windows:
        raise ValueError(
            f"skip_prefill_per_window length {len(skip_prefill_per_window)} "
            f"!= num_windows {num_windows}"
        )

    sources = [0] * n
    last_refresh = 0
    for w in range(num_windows):
        start = w * cpp
        if w == 0 or not bool(skip_prefill_per_window[w]):
            last_refresh = start
        for c in range(start, min(start + cpp, n)):
            sources[c] = last_refresh
    return sources


def build_skip_v2_chunk_sources_from_proprio(
    proprio: np.ndarray,
    *,
    action_horizon: int,
    action_chunk_size: int,
    chunks_per_video_prefill: int = 2,
    max_consecutive_skips: int = 1,
    episode_progress: float = 0.5,
    place_progress_thresh: float = 0.85,
    closed_threshold: float = 0.3,
) -> Tuple[List[int], List[bool], List[bool]]:
    """Build per-chunk obs source indices + eligibility/skip logs for one sample.

    Returns
    -------
    sources : list[int]
        Length num_chunks; index into chunk-start list for visual frame.
    eligible : list[bool]
        Per prefill window, before budget.
    skipped : list[bool]
        Per prefill window, after budget (True = reuse previous KV/obs).
    """
    ah = int(action_horizon)
    cs = int(action_chunk_size)
    cpp = int(chunks_per_video_prefill)
    if ah % cs != 0:
        raise ValueError(f"action_horizon={ah} not divisible by action_chunk_size={cs}")
    num_chunks = ah // cs
    prop = np.asarray(proprio, dtype=np.float64)
    if prop.ndim != 2 or prop.shape[0] < ah:
        raise ValueError(
            f"`proprio` must be [T,D] with T>=action_horizon={ah}, got {prop.shape}"
        )

    num_windows = (num_chunks + cpp - 1) // cpp
    eligible: List[bool] = []
    for w in range(num_windows):
        chunk_idx = w * cpp
        t = chunk_idx * cs
        holding = gripper_holding(prop[t], closed_threshold=closed_threshold)
        # Slightly advance progress within the window for PLACE gating.
        local_progress = float(episode_progress)
        if num_windows > 1:
            local_progress = min(
                1.0,
                float(episode_progress) + 0.05 * (w / max(num_windows - 1, 1)),
            )
        eligible.append(
            proxy_eligible(
                holding=holding,
                episode_progress=local_progress,
                place_progress_thresh=place_progress_thresh,
            )
        )

    # First window always refreshes (match deploy: need an initial prefill).
    # Clear eligibility *before* budget so the consecutive-skip slot is not consumed.
    if eligible:
        eligible[0] = False
    skipped = apply_consecutive_skip_budget(
        eligible, max_consecutive_skips=max_consecutive_skips
    )
    sources = chunk_obs_source_indices(
        num_chunks=num_chunks,
        chunks_per_video_prefill=cpp,
        skip_prefill_per_window=skipped,
    )
    return sources, eligible, skipped
