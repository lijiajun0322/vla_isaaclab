"""Observation, reward and termination terms for bimanual bowl pivoting.

The bowl starts upside down on the table; the task is to turn it upright. The
main signal is object-centric: the angle between the bowl's upright axis and
world +Z (pi when upside down, 0 when upright). Rim yaw is irrelevant, the bowl
is rotationally symmetric.

The reset event (``reset_from_pregrasp_table``) records the bowl's start pose on
the env as ``grasp_object_start_pos``; "ran away" is measured from there.
"""

from __future__ import annotations

import math

import torch

from isaaclab.utils.math import quat_apply, quat_conjugate, subtract_frame_transforms

from vla_isaaclab.envs.common import object_up_axis
from vla_isaaclab.envs.ycb_grasp.mdp import grasp_rl


SIDES = ("left", "right")
PALM_BODIES = {side: f"{side}_hand_palm_link" for side in SIDES}
# Palm plus the three distal links: the points that measure how close a hand is to the bowl.
HAND_KEYPOINT_BODIES = {
    side: tuple(f"{side}_hand_{link}_link" for link in ("palm", "thumb_2", "index_1", "middle_1")) for side in SIDES
}
HAND_LINK_SHORT_NAMES = ("palm", "thumb_0", "thumb_1", "thumb_2", "index_0", "index_1", "middle_0", "middle_1")
# Hand/bowl contact sensors, one per link (see BowlPivotSceneCfg). The left
# names are the grasp-lift scene's.
HAND_CONTACT_SENSORS = {
    "left": tuple(f"object_contact_{short}" for short in HAND_LINK_SHORT_NAMES),
    "right": tuple(f"object_contact_right_{short}" for short in HAND_LINK_SHORT_NAMES),
}
ALL_HAND_CONTACT_SENSORS = HAND_CONTACT_SENSORS["left"] + HAND_CONTACT_SENSORS["right"]
CONTACT_THRESHOLD_N = 0.5
# Bowl collider bounding radius about its collider center (lip radius 81 mm).
BOWL_RADIUS_M = 0.081


def _bowl(env):
    return env.scene["object"]


def _robot(env):
    return env.scene["robot"]


def _body_ids(env, names: tuple[str, ...]) -> list[int]:
    cache = env.__dict__.setdefault("_bowl_pivot_body_ids", {})
    if names not in cache:
        cache[names], _ = _robot(env).find_bodies(list(names), preserve_order=True)
    return cache[names]


def _start_pos(env) -> torch.Tensor:
    start = getattr(env, "grasp_object_start_pos", None)
    return _bowl(env).data.root_pos_w if start is None else start


def bowl_up(env) -> torch.Tensor:
    """World direction of the bowl's upright axis (opening direction), (num_envs, 3)."""
    return object_up_axis(env.cfg.object_spec, _bowl(env).data.root_quat_w)


def bowl_tilt(env) -> torch.Tensor:
    """Angle (rad) between the bowl's upright axis and world +Z: pi upside down, 0 upright."""
    return torch.acos(bowl_up(env)[:, 2].clamp(-1.0, 1.0))


def bowl_center(env) -> torch.Tensor:
    """World position of the bowl collider's center (the root is the bowl's foot)."""
    bowl = _bowl(env)
    center = bowl.data.root_pos_w.new_tensor(env.cfg.object_spec.box_center_m).expand_as(bowl.data.root_pos_w)
    return bowl.data.root_pos_w + quat_apply(bowl.data.root_quat_w, center)


def bowl_lowest_height(env) -> torch.Tensor:
    """Height of the bowl's lowest point above the table top, (num_envs,)."""
    return grasp_rl.object_lowest_height(env)


def hand_contact_forces(env, side: str) -> torch.Tensor:
    return grasp_rl.contact_forces(env, HAND_CONTACT_SENSORS[side])


def hand_touches_bowl(env, side: str) -> torch.Tensor:
    return hand_contact_forces(env, side).amax(dim=-1) > CONTACT_THRESHOLD_N


def hand_bowl_gap(env, side: str) -> torch.Tensor:
    """Closest hand keypoint distance to the bowl's bounding sphere (0 inside), (num_envs,)."""
    points = _robot(env).data.body_pos_w[:, _body_ids(env, HAND_KEYPOINT_BODIES[side])]
    distance = torch.linalg.vector_norm(points - bowl_center(env).unsqueeze(1), dim=-1).amin(dim=-1)
    return (distance - BOWL_RADIUS_M).clamp(min=0.0)


# -- observations -------------------------------------------------------------


def _to_base(env, pos_w: torch.Tensor, quat_w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    robot = _robot(env)
    return subtract_frame_transforms(robot.data.root_pos_w, robot.data.root_quat_w, pos_w, quat_w)


def palm_poses_b(env) -> torch.Tensor:
    """Left and right palm pose (pos, quat wxyz) in the robot base frame, (num_envs, 14)."""
    robot = _robot(env)
    poses = []
    for side in SIDES:
        palm = _body_ids(env, (PALM_BODIES[side],))[0]
        poses.extend(_to_base(env, robot.data.body_pos_w[:, palm], robot.data.body_quat_w[:, palm]))
    return torch.cat(poses, dim=-1)


def bowl_pose_b(env) -> torch.Tensor:
    """Bowl root pose in the robot base frame, (num_envs, 7)."""
    bowl = _bowl(env)
    return torch.cat(_to_base(env, bowl.data.root_pos_w, bowl.data.root_quat_w), dim=-1)


def bowl_up_b(env) -> torch.Tensor:
    """Bowl upright axis in the robot base frame, plus its tilt from world +Z, (num_envs, 4)."""
    up = quat_apply(quat_conjugate(_robot(env).data.root_quat_w), bowl_up(env))
    return torch.cat((up, bowl_tilt(env).unsqueeze(-1)), dim=-1)


def bowl_in_palms(env) -> torch.Tensor:
    """Bowl collider center in each palm's frame, (num_envs, 6)."""
    robot = _robot(env)
    center = bowl_center(env)
    local = []
    for side in SIDES:
        palm = _body_ids(env, (PALM_BODIES[side],))[0]
        pos, _ = subtract_frame_transforms(robot.data.body_pos_w[:, palm], robot.data.body_quat_w[:, palm], center)
        local.append(pos)
    return torch.cat(local, dim=-1)


def bowl_displacement(env) -> torch.Tensor:
    """Bowl root position minus its start position, in the robot base frame, (num_envs, 3)."""
    return quat_apply(quat_conjugate(_robot(env).data.root_quat_w), _bowl(env).data.root_pos_w - _start_pos(env))


def hands_contact_force(env, max_force: float = 10.0) -> torch.Tensor:
    """Per-link hand/bowl contact force of both hands scaled to [0, 1], (num_envs, 16)."""
    return grasp_rl.contact_forces(env, ALL_HAND_CONTACT_SENSORS).clamp(max=max_force) / max_force


# -- rewards ------------------------------------------------------------------


def upright(env) -> torch.Tensor:
    """Main, object-centric reward: 1 - tilt / pi; 0 upside down, 1 upright.

    Linear in the angle, so it has a gradient at the upside-down start (a
    cosine-based reward is flat there).
    """
    return 1.0 - bowl_tilt(env) / math.pi


def rim_on_table(env, max_height_m: float = 0.01) -> torch.Tensor:
    """The bowl's lowest point within max_height_m of the table: it pivots on its rim, not in the air."""
    return bowl_lowest_height(env) < max_height_m


def tilt_progress(env) -> torch.Tensor:
    """Pays only when the bowl turns further toward upright than ever before this episode, with its rim on the table.

    The grasp-lift progress reward on tilt / pi (0..1): max(best - error, 0),
    then best = min(best, error), so rocking the bowl back and forth earns
    nothing. The best also advances while the bowl is in the air but that
    turn is not paid, so a toss does not collect it on landing either. Needs
    grasp_rl.reset_grasp_progress as a reset event.
    """
    return grasp_rl._progress_on(env, "bowl_tilt", bowl_tilt(env) / math.pi) * rim_on_table(env).float()


# -- gentle turning: no flight, soft landing, no fast spin -----------------------

OBJECT_TABLE_SENSOR = "object_table_contact"


def free_flight(env, min_height_m: float = 0.01) -> torch.Tensor:
    """1 while the bowl is off the table (lowest point above min_height_m) and neither hand touches it."""
    untouched = ~(hand_touches_bowl(env, "left") | hand_touches_bowl(env, "right"))
    return ((bowl_lowest_height(env) > min_height_m) & untouched).float()


def table_impact_force(env) -> torch.Tensor:
    """Largest bowl/table contact force over the last control step (N); 0 on an episode's first step."""
    return grasp_rl.contact_forces(env, (OBJECT_TABLE_SENSOR,))[:, 0]


def hard_landing(env, threshold_n: float = 3.0, scale_n: float = 20.0) -> torch.Tensor:
    """(bowl/table force - threshold_n) / scale_n in [0, 1]; the bowl alone weighs 1.4 N."""
    return ((table_impact_force(env) - threshold_n) / scale_n).clamp(0.0, 1.0)


def fast_spin(env, threshold_radps: float = 4.0, scale_radps: float = 4.0) -> torch.Tensor:
    """(bowl angular speed - threshold) / scale, 0 below the threshold (a toss spins at ~10 rad/s)."""
    speed = torch.linalg.vector_norm(_bowl(env).data.root_ang_vel_w, dim=-1)
    return ((speed - threshold_radps) / scale_radps).clamp(min=0.0)


_GENTLE_ATTR = "bowl_pivot_gentle"


def _gentle(env) -> dict[str, torch.Tensor]:
    state = env.__dict__.get(_GENTLE_ATTR)
    if state is None:
        state = {"flew": torch.zeros(env.num_envs, dtype=torch.bool, device=env.device),
                 "peak_impact": torch.zeros(env.num_envs, device=env.device)}
        env.__dict__[_GENTLE_ATTR] = state
    return state


def reset_gentle(env, env_ids: torch.Tensor | None):
    """Reset event: clear the episode's free-flight flag and peak landing force."""
    ids = slice(None) if env_ids is None else env_ids
    state = _gentle(env)
    state["flew"][ids] = False
    state["peak_impact"][ids] = 0.0


def gentle_so_far(env, max_impact_n: float = 10.0) -> torch.Tensor:
    """No free flight and no bowl/table force above max_impact_n so far this episode (updated once per step)."""
    cache = env.__dict__.setdefault("_bowl_pivot_gentle_cache", {})
    step = int(env.common_step_counter)
    if cache.get("step") != step:
        state = _gentle(env)
        state["flew"] |= free_flight(env) > 0
        state["peak_impact"] = torch.maximum(state["peak_impact"], table_impact_force(env))
        cache.update(step=step, value=~state["flew"] & (state["peak_impact"] <= max_impact_n))
    return cache["value"]


def gentle_success(env) -> torch.Tensor:
    """1 on the success step if the bowl was turned without flying or a hard landing."""
    return (env.termination_manager.get_term("success") & gentle_so_far(env)).float()


def hands_near_bowl(env, scale_m: float = 0.05) -> torch.Tensor:
    """Mean over both hands of 1 - tanh(gap / scale_m); 1 when both hands reach the bowl."""
    gaps = torch.stack([hand_bowl_gap(env, side) for side in SIDES], dim=-1)
    return (1.0 - torch.tanh(gaps / scale_m)).mean(dim=-1)


def hands_contact(env) -> torch.Tensor:
    """0.5 per hand with any link on the bowl."""
    return 0.5 * sum(hand_touches_bowl(env, side).float() for side in SIDES)


# -- success: upright and at rest on the table for a while ---------------------

_STREAK_ATTR = "bowl_pivot_upright_streak"


def _streak(env) -> torch.Tensor:
    streak = getattr(env, _STREAK_ATTR, None)
    if streak is None:
        streak = torch.zeros(env.num_envs, device=env.device)
        setattr(env, _STREAK_ATTR, streak)
    return streak


def reset_upright_streak(env, env_ids: torch.Tensor | None):
    """Reset event: clear the stable-upright step count."""
    _streak(env)[slice(None) if env_ids is None else env_ids] = 0.0


def upright_and_still(env, max_tilt_deg: float = 15.0, max_height_m: float = 0.02,
                      max_speed_mps: float = 0.05, max_angular_speed_radps: float = 0.5) -> torch.Tensor:
    """Bowl upright within max_tilt_deg, its foot within max_height_m of the table top, and nearly at rest."""
    bowl = _bowl(env)
    resting = bowl.data.root_pos_w[:, 2] - env.cfg.object_spec.support_height_m < max_height_m
    still = (torch.linalg.vector_norm(bowl.data.root_lin_vel_w, dim=-1) < max_speed_mps) & (
        torch.linalg.vector_norm(bowl.data.root_ang_vel_w, dim=-1) < max_angular_speed_radps
    )
    return (bowl_tilt(env) < math.radians(max_tilt_deg)) & resting & still


def pivot_success(env, hold_steps: int = 10, max_tilt_deg: float = 15.0, max_height_m: float = 0.02,
                  max_speed_mps: float = 0.05, max_angular_speed_radps: float = 0.5) -> torch.Tensor:
    """Termination: upright_and_still for hold_steps control steps in a row (counted once per step)."""
    cache = env.__dict__.setdefault("_bowl_pivot_success_cache", {})
    step = int(env.common_step_counter)
    if cache.get("step") != step:
        streak = _streak(env)
        now = upright_and_still(env, max_tilt_deg, max_height_m, max_speed_mps, max_angular_speed_radps)
        streak[:] = torch.where(now, streak + 1.0, torch.zeros_like(streak))
        cache.update(step=step, value=streak >= hold_steps)
    return cache["value"]


# -- failure terminations -----------------------------------------------------


def bowl_ran_away(env, max_xy_m: float = 0.15, max_rise_m: float = 0.20) -> torch.Tensor:
    """Termination: bowl slid or was pushed more than max_xy_m, or thrown more than max_rise_m up."""
    offset = _bowl(env).data.root_pos_w - _start_pos(env)
    return (torch.linalg.vector_norm(offset[:, :2], dim=-1) > max_xy_m) | (offset[:, 2] > max_rise_m)


def bowl_spinning(env, max_angular_speed_radps: float = 20.0) -> torch.Tensor:
    """Termination: bowl spinning implausibly fast (a sign of a solver blow-up, not pivoting)."""
    return torch.linalg.vector_norm(_bowl(env).data.root_ang_vel_w, dim=-1) > max_angular_speed_radps


__all__ = [
    "bowl_displacement",
    "bowl_in_palms",
    "bowl_lowest_height",
    "bowl_pose_b",
    "bowl_ran_away",
    "bowl_spinning",
    "bowl_tilt",
    "bowl_up_b",
    "hands_contact",
    "hands_contact_force",
    "hands_near_bowl",
    "fast_spin",
    "free_flight",
    "gentle_success",
    "hard_landing",
    "reset_gentle",
    "rim_on_table",
    "palm_poses_b",
    "pivot_success",
    "reset_upright_streak",
    "tilt_progress",
    "upright",
    "upright_and_still",
]
