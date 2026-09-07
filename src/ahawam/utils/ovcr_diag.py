"""OVCR diagnostic helpers for probe experiments (not OVCR algorithm changes).

Modes (env ``OVCR_DIAG_MODE``):
  - baseline: normal OVCR update
  - off: skip OVCR; reuse first-frame planner KV as-is
  - oracle: force fresh video prefill every action chunk (deploy-side);
    OVCR is left off so the upper bound is "fresh encode alone"
"""

from __future__ import annotations

import json
import os
import threading
from pathlib import Path
from typing import Any, Optional

import torch

VALID_MODES = ("baseline", "off", "oracle")


def get_ovcr_diag_mode(default: str = "baseline") -> str:
    mode = str(os.environ.get("OVCR_DIAG_MODE", default)).strip().lower()
    if mode not in VALID_MODES:
        raise ValueError(
            f"Unsupported OVCR_DIAG_MODE={mode!r}. Expected one of {VALID_MODES}."
        )
    return mode


def ovcr_diag_logging_enabled() -> bool:
    # Default OFF so shared/cluster jobs are unaffected unless explicitly enabled.
    raw = str(os.environ.get("OVCR_DIAG_LOG", "0")).strip().lower()
    return raw in {"1", "true", "yes", "on"}


def resolve_ovcr_diag_log_path() -> Optional[Path]:
    raw = os.environ.get("OVCR_DIAG_LOG_PATH")
    if raw is None or not str(raw).strip():
        return None
    path = Path(str(raw)).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


_log_lock = threading.Lock()


def append_ovcr_diag_record(path: Optional[Path], record: dict[str, Any]) -> None:
    if path is None:
        return
    line = json.dumps(record, ensure_ascii=False)
    with _log_lock:
        with path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")


def summarize_ovcr_update(
    *,
    layer_idx: int,
    base_k: torch.Tensor,
    base_v: torch.Tensor,
    updated_k: torch.Tensor,
    updated_v: torch.Tensor,
    delta_k: Optional[torch.Tensor],
    delta_v: Optional[torch.Tensor],
    gate: Optional[torch.Tensor],
) -> dict[str, float]:
    """Compute lightweight norms for one layer's OVCR update."""

    def _rel(delta: torch.Tensor, base: torch.Tensor) -> float:
        base_n = float(base.detach().float().norm().item()) + 1e-8
        return float(delta.detach().float().norm().item()) / base_n

    out: dict[str, float] = {
        "layer": float(layer_idx),
        "k_rel_change": _rel(updated_k - base_k, base_k),
        "v_rel_change": _rel(updated_v - base_v, base_v),
    }
    if delta_k is not None:
        out["delta_k_rel"] = _rel(delta_k, base_k)
    if delta_v is not None:
        out["delta_v_rel"] = _rel(delta_v, base_v)
    if gate is not None:
        out["gate"] = float(gate.detach().float().item())
    return out
