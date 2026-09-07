"""Phase-aware video-prefill skip schedule for VIDEO_DIT_MODE=skip_phase.

Only decides whether to skip a *video prefill* refresh. Action-chunk inference
always runs. When the decision is "do not skip", behavior matches baseline.

Holding / drop (pick-place)
---------------------------
Do NOT treat "gripper < 0.3" alone as holding (unsafe if init is partially closed).

Enter HOLDING only when ALL are true:
  1) gripper looks closed (below closed_threshold)
  2) GT attach: TCP within attach_thresh of a grasp object
  3) gripper was seen open earlier this episode (open→close), unless already latched

Leave HOLDING (→ REACH / retry) when ANY is true:
  - gripper opens above open_threshold, OR
  - object lost: min TCP↔grasp_object distance > drop_thresh

Retry after drop is classified as REACH again (far skip / near no-skip).

Intentional place release (near place + open gripper) also returns to REACH for
multi-object tasks; skip rules are the same as a fresh approach.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

# task_name -> actor attribute names on the RoboTwin env instance.
TASK_PHASE_TARGETS: Dict[str, Dict[str, Any]] = {
    "handover_mic": {
        "template": "pick_place",
        "grasp_attrs": ["microphone"],
        "place_attrs": ["handover_middle_pose"],
    },
    "hanging_mug": {
        "template": "pick_place",
        "grasp_attrs": ["mug"],
        "place_attrs": ["rack"],
    },
    "move_stapler_pad": {
        "template": "pick_place",
        "grasp_attrs": ["stapler"],
        "place_attrs": ["pad"],
    },
    "place_bread_basket": {
        "template": "pick_place",
        "grasp_attrs": ["bread"],
        "place_attrs": ["breadbasket"],
    },
    "place_mouse_pad": {
        "template": "pick_place",
        "grasp_attrs": ["mouse"],
        "place_attrs": ["target"],
    },
    "place_object_basket": {
        "template": "pick_place",
        "grasp_attrs": ["object"],
        "place_attrs": ["basket"],
    },
    "put_bottles_dustbin": {
        "template": "pick_place",
        "grasp_attrs": ["bottles"],
        "place_attrs": ["dustbin"],
    },
    "stack_blocks_three": {
        "template": "pick_place",
        "grasp_attrs": ["block1", "block2", "block3"],
        "place_attrs": ["block1_target_pose", "block1", "block2"],
    },
    "stack_blocks_two": {
        "template": "pick_place",
        "grasp_attrs": ["block1", "block2"],
        "place_attrs": ["block1_target_pose", "block1"],
    },
    "click_bell": {
        "template": "contact",
        "grasp_attrs": ["bell"],
        "place_attrs": [],
    },
    # batch2 — diverse types (no overlap with batch1 skip_phase tasks)
    "pick_dual_bottles": {
        "template": "pick_place",
        "grasp_attrs": ["bottle1", "bottle2"],
        "place_attrs": ["left_target_pose", "right_target_pose"],
    },
    "pick_diverse_bottles": {
        "template": "pick_place",
        "grasp_attrs": ["bottle1", "bottle2"],
        "place_attrs": ["left_target_pose", "right_target_pose"],
    },
    "place_a2b_left": {
        "template": "pick_place",
        "grasp_attrs": ["object"],
        "place_attrs": ["target_object"],
    },
    "place_a2b_right": {
        "template": "pick_place",
        "grasp_attrs": ["object"],
        "place_attrs": ["target_object"],
    },
    "move_can_pot": {
        "template": "pick_place",
        "grasp_attrs": ["can"],
        "place_attrs": ["pot", "target_pose"],
    },
    "stack_bowls_three": {
        "template": "pick_place",
        "grasp_attrs": ["bowl1", "bowl2", "bowl3"],
        "place_attrs": ["bowl1_target_pose", "bowl1", "bowl2"],
    },
    "handover_block": {
        "template": "pick_place",
        "grasp_attrs": ["box"],
        "place_attrs": ["target_box", "block_middle_pose"],
    },
    "lift_pot": {
        "template": "pick_place",
        "grasp_attrs": ["pot"],
        "place_attrs": ["pot"],
    },
    "press_stapler": {
        "template": "contact",
        "grasp_attrs": ["stapler"],
        "place_attrs": [],
    },
    "turn_switch": {
        "template": "contact",
        "grasp_attrs": ["switch"],
        "place_attrs": [],
    },
}


def _as_pos(obj: Any) -> Optional[np.ndarray]:
    if obj is None:
        return None
    if isinstance(obj, np.ndarray):
        arr = np.asarray(obj, dtype=np.float64).reshape(-1)
        return arr[:3] if arr.size >= 3 else None
    if isinstance(obj, (list, tuple)):
        if len(obj) >= 3 and all(isinstance(x, (int, float, np.floating)) for x in obj[:3]):
            return np.asarray(obj[:3], dtype=np.float64)
        return None
    if hasattr(obj, "get_pose"):
        try:
            return np.asarray(obj.get_pose().p, dtype=np.float64).reshape(3)
        except Exception:
            return None
    if hasattr(obj, "p"):
        try:
            return np.asarray(obj.p, dtype=np.float64).reshape(3)
        except Exception:
            return None
    return None


def _collect_positions(env: Any, attrs: Sequence[str]) -> List[np.ndarray]:
    positions: List[np.ndarray] = []
    for name in attrs:
        if not hasattr(env, name):
            continue
        obj = getattr(env, name)
        if obj is None:
            continue
        if isinstance(obj, (list, tuple)):
            for item in obj:
                p = _as_pos(item)
                if p is not None:
                    positions.append(p)
            p = _as_pos(obj)
            if p is not None:
                positions.append(p)
        else:
            p = _as_pos(obj)
            if p is not None:
                positions.append(p)
    return positions


def _tcp_positions(task_env: Any) -> List[np.ndarray]:
    robot = getattr(task_env, "robot", None)
    if robot is None:
        return []
    out: List[np.ndarray] = []
    for getter in ("get_left_tcp_pose", "get_right_tcp_pose"):
        fn = getattr(robot, getter, None)
        if not callable(fn):
            continue
        try:
            pose = fn()
            p = _as_pos(pose)
            if p is not None:
                out.append(p)
        except Exception:
            continue
    return out


def _min_distance(tcps: Sequence[np.ndarray], targets: Sequence[np.ndarray]) -> Optional[float]:
    if not tcps or not targets:
        return None
    best = None
    for tcp in tcps:
        for tgt in targets:
            d = float(np.linalg.norm(tcp - tgt))
            if best is None or d < best:
                best = d
    return best


def _gripper_vals(task_env: Any, observation: Optional[Dict[str, Any]]) -> Tuple[float, float]:
    left, right = 1.0, 1.0
    robot = getattr(task_env, "robot", None) if task_env is not None else None
    if robot is not None:
        try:
            left = float(robot.get_left_gripper_val())
            right = float(robot.get_right_gripper_val())
            return left, right
        except Exception:
            pass
    if observation is not None:
        obs = observation.get("observation", observation)
        joint = obs.get("joint_action", {}) if isinstance(obs, dict) else {}
        vec = joint.get("vector")
        if vec is not None and len(vec) >= 14:
            left = float(vec[6])
            right = float(vec[13])
    return left, right


class SkipPhaseFSM:
    """Phase-aware *eligibility* for skipping video prefill.

    Plain class (not dataclass) so importlib/exec_module preflight is safe.

    Eligibility (this class)
    -----------------------
    - REACH / REACH_RETRY / contact: eligible iff TCP↔grasp is farther than
      ``near_thresh_m`` (near field always refreshes).
    - TRANSPORT (holding, not near place): eligible iff TCP↔place farther than
      ``near_thresh_m``.
    - PLACE (holding + near place): never eligible.

    Actual skip rate is capped in deploy_policy via consecutive-skip budget
    (default: at most 1 skipped prefill in a row ⇒ ~≤50% when always eligible).
    Skipped prefills still run OVCR on action chunks (OVCR_DIAG_MODE=baseline).
    """

    def __init__(
        self,
        task_name: str,
        near_thresh_m: float = 0.10,
        closed_threshold: float = 0.3,
        open_threshold: float = 0.7,
        attach_thresh_m: float = 0.08,
        drop_thresh_m: float = 0.15,
    ) -> None:
        self.task_name = task_name
        self.near_thresh_m = float(near_thresh_m)
        # Hysteresis on gripper (RoboTwin vals: ~1 open, ~0 closed).
        self.closed_threshold = float(closed_threshold)
        self.open_threshold = float(open_threshold)
        # GT attach / drop (meters).
        self.attach_thresh_m = float(attach_thresh_m)
        self.drop_thresh_m = float(drop_thresh_m)
        self.phase = "REACH"
        self.holding = False
        self.seen_open = False
        self.last_info: Dict[str, Any] = {}
        self._reload_task_cfg()

    def _reload_task_cfg(self) -> None:
        self.cfg = TASK_PHASE_TARGETS.get(
            self.task_name,
            {"template": "pick_place", "grasp_attrs": [], "place_attrs": []},
        )
        self.template = str(self.cfg.get("template", "pick_place"))

    def set_task_name(self, task_name: str) -> None:
        self.task_name = str(task_name).strip()
        self._reload_task_cfg()

    def reset(self) -> None:
        self.phase = "REACH"
        self.holding = False
        self.seen_open = False
        self.last_info = {}

    def _update_holding_latch(
        self,
        *,
        left_g: float,
        right_g: float,
        dist_grasp: Optional[float],
        near_place: bool,
    ) -> str:
        """Update sticky holding; return reason tag for logging."""
        g_min = min(left_g, right_g)
        gripper_closed = g_min < self.closed_threshold
        gripper_open = g_min > self.open_threshold
        attached = dist_grasp is not None and dist_grasp <= self.attach_thresh_m
        lost_object = dist_grasp is not None and dist_grasp > self.drop_thresh_m

        if gripper_open:
            self.seen_open = True

        if not self.holding:
            # Require open→close style grasp + object actually at the hand.
            if self.seen_open and gripper_closed and attached:
                self.holding = True
                return "grasp_attach"
            return "not_holding"
        # already holding
        if gripper_open:
            self.holding = False
            self.seen_open = True
            return "release_open" if near_place else "drop_or_release_open"
        if lost_object:
            self.holding = False
            # Object left the hand while fingers still closed → treat as drop; need re-open
            # before next attach, but allow REACH immediately.
            return "drop_object_far"
        return "holding_ok"

    def should_skip_prefill(
        self,
        task_env: Any,
        observation: Optional[Dict[str, Any]],
    ) -> bool:
        """Return True if this prefill decision is *eligible* to skip.

        deploy_policy applies the consecutive-skip budget on top of this flag.
        ``last_info['eligible']`` is set here; ``last_info['skip']`` is the
        eligibility bit (budget may later force skip=False in deploy).
        """
        if task_env is None:
            self.last_info = {
                "phase": self.phase,
                "reason": "no_task_env",
                "eligible": False,
                "skip": False,
            }
            return False

        left_g, right_g = _gripper_vals(task_env, observation)
        tcps = _tcp_positions(task_env)
        grasp_pos = _collect_positions(task_env, self.cfg.get("grasp_attrs") or [])
        place_pos = _collect_positions(task_env, self.cfg.get("place_attrs") or [])
        dist_grasp = _min_distance(tcps, grasp_pos)
        dist_place = _min_distance(tcps, place_pos)

        near_grasp = dist_grasp is not None and dist_grasp <= self.near_thresh_m
        near_place = dist_place is not None and dist_place <= self.near_thresh_m

        if self.template == "contact":
            self.phase = "REACH"
            self.holding = False
            if dist_grasp is None:
                eligible = False
                reason = "no_grasp_dist_fallback_baseline"
            else:
                eligible = not near_grasp
                reason = "far_contact" if eligible else "near_contact"
            self.last_info = {
                "phase": self.phase,
                "template": self.template,
                "holding": False,
                "dist_grasp": dist_grasp,
                "dist_place": dist_place,
                "near_grasp": near_grasp,
                "near_place": near_place,
                "left_gripper": left_g,
                "right_gripper": right_g,
                "eligible": eligible,
                "skip": eligible,
                "reason": reason,
            }
            return eligible

        hold_reason = self._update_holding_latch(
            left_g=left_g,
            right_g=right_g,
            dist_grasp=dist_grasp,
            near_place=near_place,
        )

        if not self.holding:
            # REACH + retry-after-drop share the same schedule.
            self.phase = "REACH"
            if "drop" in hold_reason:
                self.phase = "REACH_RETRY"
            if dist_grasp is None:
                eligible = False
                reason = f"{hold_reason}|reach_no_dist_fallback_baseline"
            else:
                eligible = not near_grasp
                reason = f"{hold_reason}|{'reach_far' if eligible else 'reach_near'}"
        else:
            if near_place:
                self.phase = "PLACE"
                eligible = False
                reason = f"{hold_reason}|place_near"
            else:
                self.phase = "TRANSPORT"
                if dist_place is None:
                    eligible = False
                    reason = f"{hold_reason}|transport_no_place_fallback_baseline"
                else:
                    eligible = not near_place
                    reason = f"{hold_reason}|{'transport_far' if eligible else 'transport_near'}"

        self.last_info = {
            "phase": self.phase,
            "template": self.template,
            "holding": self.holding,
            "seen_open": self.seen_open,
            "dist_grasp": dist_grasp,
            "dist_place": dist_place,
            "near_grasp": near_grasp,
            "near_place": near_place,
            "left_gripper": left_g,
            "right_gripper": right_g,
            "eligible": eligible,
            "skip": eligible,
            "reason": reason,
            "hold_reason": hold_reason,
        }
        return eligible
