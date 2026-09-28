"""Observation, reward and termination terms for closed-loop grasp-and-lift RL.

The episode starts at a pregrasp state (see ``vla_isaaclab.rl.pregrasp_table``),
which records the box's start pose on the env as ``grasp_box_start_pos`` and
``grasp_box_start_up``. Lift is measured from that start height.
"""

from __future__ import annotations

import math

import torch

from isaaclab.assets import Articulation, RigidObject
from isaaclab.utils.math import quat_apply, quat_conjugate, quat_mul, subtract_frame_transforms


# Sugar-box collider half extents in its root frame: x width, y height, z thickness.
BOX_HALF_EXTENTS_M = (0.0463, 0.0881, 0.0226)
PALM_BODY = "left_hand_palm_link"
FINGERTIP_BODIES = ("left_hand_thumb_2_link", "left_hand_index_1_link", "left_hand_middle_1_link")
# Hand/box contact sensors (one per link; see YCBSugarBoxStateSceneCfg).
THUMB_CONTACT_SENSORS = ("box_contact_thumb_1", "box_contact_thumb_2")
FINGER_CONTACT_SENSORS = (
    "box_contact_index_0",
    "box_contact_index_1",
    "box_contact_middle_0",
    "box_contact_middle_1",
)
HAND_CONTACT_SENSORS = ("box_contact_palm", "box_contact_thumb_0", *THUMB_CONTACT_SENSORS, *FINGER_CONTACT_SENSORS)
CONTACT_THRESHOLD_N = 0.5


def _box(env) -> RigidObject:
    return env.scene["object"]


def _robot(env) -> Articulation:
    return env.scene["robot"]


def _body_ids(env, names) -> list[int]:
    cache = env.__dict__.setdefault("_grasp_rl_body_ids", {})
    if names not in cache:
        ids, _ = _robot(env).find_bodies(list(names), preserve_order=True)
        cache[names] = ids
    return cache[names]


def _start_pos(env) -> torch.Tensor:
    start = getattr(env, "grasp_box_start_pos", None)
    return _box(env).data.root_pos_w if start is None else start


def _up_axis(quat: torch.Tensor) -> torch.Tensor:
    # The box height is its local +Y axis.
    y = torch.zeros(quat.shape[0], 3, device=quat.device)
    y[:, 1] = 1.0
    return quat_apply(quat, y)


def contact_forces(env, sensors=HAND_CONTACT_SENSORS) -> torch.Tensor:
    """Hand-link/box contact force magnitudes, (num_envs, len(sensors))."""
    return torch.stack(
        [
            torch.linalg.vector_norm(env.scene.sensors[name].data.force_matrix_w, dim=-1).amax(dim=(1, 2))
            for name in sensors
        ],
        dim=-1,
    )


def grasp_flag(env) -> torch.Tensor:
    """Thumb on the box together with index or middle finger."""
    thumb = contact_forces(env, THUMB_CONTACT_SENSORS).amax(dim=-1) > CONTACT_THRESHOLD_N
    finger = contact_forces(env, FINGER_CONTACT_SENSORS).amax(dim=-1) > CONTACT_THRESHOLD_N
    return thumb & finger


def lift_height(env) -> torch.Tensor:
    return _box(env).data.root_pos_w[:, 2] - _start_pos(env)[:, 2]


def box_tilt(env) -> torch.Tensor:
    """Angle (rad) of the box's height axis from its start orientation."""
    start_up = getattr(env, "grasp_box_start_up", None)
    up = _up_axis(_box(env).data.root_quat_w)
    if start_up is None:
        start_up = torch.zeros_like(up)
        start_up[:, 2] = 1.0
    return torch.acos((up * start_up).sum(-1).clamp(-1.0, 1.0))


def _max_lift(env) -> torch.Tensor:
    value = getattr(env, "grasp_max_lift", None)
    if value is None or value.shape[0] != env.num_envs:
        value = torch.zeros(env.num_envs, device=env.device)
        env.grasp_max_lift = value
    return value


def fingertips_in_box(env) -> torch.Tensor:
    """Fingertip link positions in the box frame, (num_envs, 3, 3)."""
    box = _box(env)
    tips = _robot(env).data.body_pos_w[:, _body_ids(env, FINGERTIP_BODIES)]
    count = tips.shape[1]
    quat = quat_conjugate(box.data.root_quat_w).unsqueeze(1).expand(-1, count, -1).reshape(-1, 4)
    local = quat_apply(quat, (tips - box.data.root_pos_w.unsqueeze(1)).reshape(-1, 3))
    return local.reshape(env.num_envs, count, 3)


# -- observations -------------------------------------------------------------


def palm_pose_b(env) -> torch.Tensor:
    robot = _robot(env)
    palm = _body_ids(env, (PALM_BODY,))[0]
    pos, quat = subtract_frame_transforms(
        robot.data.root_pos_w, robot.data.root_quat_w, robot.data.body_pos_w[:, palm], robot.data.body_quat_w[:, palm]
    )
    return torch.cat((pos, quat), dim=-1)


def box_pose_in_palm(env) -> torch.Tensor:
    robot = _robot(env)
    box = _box(env)
    palm = _body_ids(env, (PALM_BODY,))[0]
    pos, quat = subtract_frame_transforms(
        robot.data.body_pos_w[:, palm], robot.data.body_quat_w[:, palm], box.data.root_pos_w, box.data.root_quat_w
    )
    return torch.cat((pos, quat), dim=-1)


def box_velocity_b(env) -> torch.Tensor:
    robot = _robot(env)
    box = _box(env)
    inv = quat_conjugate(robot.data.root_quat_w)
    return torch.cat((quat_apply(inv, box.data.root_lin_vel_w), quat_apply(inv, box.data.root_ang_vel_w)), dim=-1)


def fingertips_in_box_obs(env) -> torch.Tensor:
    return fingertips_in_box(env).reshape(env.num_envs, -1)


def lift_obs(env) -> torch.Tensor:
    return torch.stack((lift_height(env), box_tilt(env)), dim=-1)


def hand_contact_binary(env) -> torch.Tensor:
    return (contact_forces(env) > CONTACT_THRESHOLD_N).float()


def hand_contact_force(env, max_force: float = 20.0) -> torch.Tensor:
    return contact_forces(env).clamp(max=max_force) / max_force


# -- rewards --------------------------------------------------------------------


def fingertip_reach(env, std: float = 0.02) -> torch.Tensor:
    """Shaping toward contact: fingertip-link distance to the box surface."""
    tips = fingertips_in_box(env)
    half = tips.new_tensor(BOX_HALF_EXTENTS_M)
    outside = (tips.abs() - half).clamp(min=0.0)
    distance = torch.linalg.vector_norm(outside, dim=-1)
    return (1.0 - torch.tanh(distance / std)).mean(dim=-1)


def grasp_contact(env) -> torch.Tensor:
    return grasp_flag(env).float()


def lift(env, target_m: float = 0.08) -> torch.Tensor:
    """Lift fraction toward target_m, only while the box is actually grasped."""
    return grasp_flag(env).float() * (lift_height(env).clamp(min=0.0, max=target_m) / target_m)


def stable_hold(env, lifted_m: float = 0.02, lin_std: float = 0.05, ang_std: float = 1.0) -> torch.Tensor:
    """While lifted and grasped: the box moves with the palm and does not spin."""
    box = _box(env)
    palm = _body_ids(env, (PALM_BODY,))[0]
    relative = torch.linalg.vector_norm(box.data.root_lin_vel_w - _robot(env).data.body_lin_vel_w[:, palm], dim=-1)
    spin = torch.linalg.vector_norm(box.data.root_ang_vel_w - _robot(env).data.body_ang_vel_w[:, palm], dim=-1)
    active = (lift_height(env) > lifted_m) & grasp_flag(env)
    return active.float() * torch.exp(-relative / lin_std - spin / ang_std)


def hold_success(env, height_m: float = 0.06, max_tilt_rad: float = 0.35) -> torch.Tensor:
    """Binary: lifted to height_m, grasped and upright-ish. Its episode sum is the success proxy."""
    return ((lift_height(env) > height_m) & grasp_flag(env) & (box_tilt(env) < max_tilt_rad)).float()


def tilt_penalty(env) -> torch.Tensor:
    return box_tilt(env)


# -- terminations ----------------------------------------------------------------


def box_dropped(env, lifted_m: float = 0.03, dropped_m: float = 0.01) -> torch.Tensor:
    """Once lifted above lifted_m, the box fell back below dropped_m."""
    height = lift_height(env)
    peak = _max_lift(env)
    peak[:] = torch.maximum(peak, height)
    return (peak > lifted_m) & (height < dropped_m)


def box_tipped(env, max_tilt_rad: float = 0.52) -> torch.Tensor:
    return box_tilt(env) > max_tilt_rad


def box_pushed(env, max_xy_m: float = 0.05) -> torch.Tensor:
    """The box slid away along the table instead of being lifted."""
    displacement = torch.linalg.vector_norm(_box(env).data.root_pos_w[:, :2] - _start_pos(env)[:, :2], dim=-1)
    return displacement > max_xy_m



# -- v2: progress-based rewards (DextrAH-G / DexPBT style) ---------------------
#
# A progress reward only pays when the episode beats its own best so far, so
# hovering or oscillating earns nothing. ``reset_grasp_progress`` clears the
# per-env bests; each term then starts from its value on the first step.

_PROGRESS_ATTR = "grasp_progress"


def _progress(env, key: str, width: int = 0) -> torch.Tensor:
    buffers = env.__dict__.setdefault(_PROGRESS_ATTR, {})
    if key not in buffers:
        shape = (env.num_envs, width) if width else (env.num_envs,)
        buffers[key] = torch.full(shape, float("nan"), device=env.device)
    return buffers[key]


def reset_grasp_progress(env, env_ids: torch.Tensor | None):
    """Reset event: forget the per-env bests of every progress reward."""
    for value in env.__dict__.get(_PROGRESS_ATTR, {}).values():
        value[slice(None) if env_ids is None else env_ids] = float("nan")


def fingertip_surface_distance(env) -> torch.Tensor:
    """Fingertip-link distance to the box surface (0 inside), (num_envs, 3)."""
    tips = fingertips_in_box(env)
    outside = (tips.abs() - tips.new_tensor(BOX_HALF_EXTENTS_M)).clamp(min=0.0)
    return torch.linalg.vector_norm(outside, dim=-1)


def reach_progress(env) -> torch.Tensor:
    """Metres by which the fingertips got closer to the box than ever before, summed over fingers."""
    distance = fingertip_surface_distance(env)
    best = _progress(env, "reach", distance.shape[1])
    previous = torch.where(best.isnan(), distance, best)
    best[:] = torch.minimum(previous, distance)
    return (previous - distance).clamp(min=0.0).sum(dim=-1)


def thumb_finger_contact(env, thumb_only: float = 0.2) -> torch.Tensor:
    """1 with thumb and index/middle on the box, thumb_only with the thumb alone."""
    thumb = contact_forces(env, THUMB_CONTACT_SENSORS).amax(dim=-1) > CONTACT_THRESHOLD_N
    finger = contact_forces(env, FINGER_CONTACT_SENSORS).amax(dim=-1) > CONTACT_THRESHOLD_N
    return (thumb & finger).float() + thumb_only * (thumb & ~finger).float()


def lift_progress(env, max_m: float = 0.10) -> torch.Tensor:
    """Metres of new lift height (capped at max_m) gained while grasped."""
    height = lift_height(env).clamp(min=0.0, max=max_m)
    best = _progress(env, "lift")
    previous = torch.where(best.isnan(), torch.zeros_like(height), best)
    grasped = grasp_flag(env)
    best[:] = torch.where(grasped, torch.maximum(previous, height), previous)
    return grasped.float() * (height - previous).clamp(min=0.0)


def lift_bonus(env, height_m: float = 0.05) -> torch.Tensor:
    """One-time 1 the first step the grasped box passes height_m."""
    paid = _progress(env, "lift_bonus")
    reached = grasp_flag(env) & (lift_height(env) > height_m)
    first = reached & paid.isnan()
    paid[first] = 1.0
    return first.float()


def hand_contact_force_clipped(env, max_force: float = 10.0) -> torch.Tensor:
    """Per-link contact force scaled to [0, 1]; the actor sees how hard it presses."""
    return contact_forces(env).clamp(max=max_force) / max_force


def lifted_hold(env, max_m: float = 0.10, zero_tilt_rad: float = math.radians(45.0)) -> torch.Tensor:
    """Per step while grasped: lift height as a fraction of max_m (DexPBT's r_pick),
    scaled down linearly as the box tilts, reaching 0 at zero_tilt_rad."""
    upright = (1.0 - box_tilt(env) / zero_tilt_rad).clamp(min=0.0)
    return grasp_flag(env).float() * upright * lift_height(env).clamp(min=0.0, max=max_m) / max_m


def arm_action_l2(env, term: str = "arm") -> torch.Tensor:
    """Squared palm action only; the relative hand must keep squeezing to hold the box."""
    dim = env.action_manager.get_term(term).action_dim
    return env.action_manager.action[:, :dim].square().sum(dim=-1)


# -- v2 success (DextrAH-G style: hold the goal state briefly, then end the episode) --


def grasp_success(env, height_m: float = 0.10, max_tilt_rad: float = math.radians(20.0),
                  hold_steps: int = 15) -> torch.Tensor:
    """Termination: grasped, lifted height_m and tilted under max_tilt_rad for hold_steps in a row."""
    streak = _progress(env, "success_streak")
    streak[streak.isnan()] = 0.0
    holding = grasp_flag(env) & (lift_height(env) > height_m) & (box_tilt(env) < max_tilt_rad)
    streak[:] = torch.where(holding, streak + 1.0, torch.zeros_like(streak))
    return streak >= hold_steps


def success_bonus(env, term: str = "success") -> torch.Tensor:
    """On the success step: the fraction of the episode left, so faster success pays more."""
    left = 1.0 - env.episode_length_buf.float() / env.max_episode_length
    return env.termination_manager.get_term(term).float() * left.clamp(min=0.0)


def tilt_when_lifted(env, lifted_m: float = 0.02) -> torch.Tensor:
    """Box tilt (rad), only once it is off the table; nudging it on the table is part of grasping."""
    return (lift_height(env) > lifted_m).float() * box_tilt(env)


# -- v2: DextrAH-G reward (arXiv 2407.02274, FGP training) -----------------------
#
#   to-object  w=5     progress on ||fingertips - x_obj||
#   lift       w=50    progress on (z_lifted - z_obj), until lifted
#   lifted     w=50    once, the first time the object is lifted
#   to-goal    w=1000  progress on ||x_goal - x_obj||, only while lifted
#   reached    w=40    every step within d_success of the goal
#   success    w=100   reached for T_success, times the time left; ends the episode

GOAL_ATTR = "grasp_goal_pos"


def sample_lift_goal(env, env_ids: torch.Tensor | None, xy_range_m: float = 0.05,
                     z_range_m: tuple[float, float] = (0.20, 0.30)):
    """Reset event: a goal point above the box's start position (world frame)."""
    goal = _goal(env)
    ids = torch.arange(env.num_envs, device=env.device) if env_ids is None else env_ids
    offset = torch.empty(len(ids), 3, device=env.device)
    offset[:, :2].uniform_(-xy_range_m, xy_range_m)
    offset[:, 2].uniform_(*z_range_m)
    goal[ids] = _start_pos(env)[ids] + offset


def _goal(env) -> torch.Tensor:
    """Goal position; observation shapes are probed before the first reset samples one."""
    goal = getattr(env, GOAL_ATTR, None)
    if goal is None:
        goal = _start_pos(env).clone()
        goal[:, 2] += 0.07
        setattr(env, GOAL_ATTR, goal)
    return goal


def _goal_distance(env) -> torch.Tensor:
    return torch.linalg.vector_norm(_goal(env) - _box(env).data.root_pos_w, dim=-1)


def _lifted(env, z_lifted_m: float) -> torch.Tensor:
    """lifted(x_obj): the box is currently at least z_lifted_m above its start."""
    return lift_height(env) > z_lifted_m


def _progress_on(env, key: str, error: torch.Tensor, active: torch.Tensor | None = None) -> torch.Tensor:
    """max(best - error, 0), then best = min(best, error); inactive envs keep no best."""
    best = _progress(env, key, error.shape[1] if error.dim() > 1 else 0)
    active = torch.ones_like(error, dtype=torch.bool) if active is None else active
    previous = torch.where(best.isnan(), error, best)
    reward = torch.where(active, (previous - error).clamp(min=0.0), torch.zeros_like(error))
    best[:] = torch.where(active, torch.minimum(previous, error), best)
    return reward


def dextrah_to_object(env) -> torch.Tensor:
    tips = _robot(env).data.body_pos_w[:, _body_ids(env, FINGERTIP_BODIES)]
    distance = torch.linalg.vector_norm((tips - _box(env).data.root_pos_w.unsqueeze(1)).flatten(1), dim=-1)
    return _progress_on(env, "to_object", distance)


def dextrah_lift(env, z_lifted_m: float = 0.20) -> torch.Tensor:
    lifted = _lifted(env, z_lifted_m)
    error = (z_lifted_m - lift_height(env)).clamp(min=0.0)
    return _progress_on(env, "lift_error", error) * (~lifted).float()


def dextrah_lifted(env, z_lifted_m: float = 0.20) -> torch.Tensor:
    paid = _progress(env, "lifted_paid")  # NaN until paid; cleared at reset
    first = _lifted(env, z_lifted_m) & paid.isnan()
    paid[first] = 1.0
    return first.float()


def dextrah_to_goal(env, z_lifted_m: float = 0.20) -> torch.Tensor:
    return _progress_on(env, "goal", _goal_distance(env), _lifted(env, z_lifted_m))


def dextrah_reached(env, d_success_m: float = 0.03) -> torch.Tensor:
    return (_goal_distance(env) < d_success_m).float()


def dextrah_success(env, d_success_m: float = 0.03, t_success_s: float = 1.0) -> torch.Tensor:
    """Termination: within d_success_m of the goal for t_success_s in a row."""
    streak = _progress(env, "reached_streak")
    streak[streak.isnan()] = 0.0
    streak[:] = torch.where(_goal_distance(env) < d_success_m, streak + 1.0, torch.zeros_like(streak))
    return streak * env.step_dt >= t_success_s - 1e-6


def dextrah_success_bonus(env, term: str = "success") -> torch.Tensor:
    """(T_max - T) in seconds on the success step."""
    left = (env.max_episode_length - env.episode_length_buf).float() * env.step_dt
    return env.termination_manager.get_term(term).float() * left


def goal_in_base(env) -> torch.Tensor:
    """Goal minus box position, in the robot base frame."""
    robot = _robot(env)
    return quat_apply(quat_conjugate(robot.data.root_quat_w), _goal(env) - _box(env).data.root_pos_w)


# -- v2 goal pose: the box as it sat on the table, raised a fixed height --------
#
# Distance to the goal is DexPBT's keypoint distance: the largest gap between a
# box corner and the same corner of the goal pose, so tilting counts as much as
# being off in position.

_CORNER_SIGNS = torch.tensor([[x, y, z] for x in (-1.0, 1.0) for y in (-1.0, 1.0) for z in (-1.0, 1.0)])


def _start_quat(env) -> torch.Tensor:
    quat = getattr(env, "grasp_box_start_quat", None)
    return _box(env).data.root_quat_w if quat is None else quat


def _corners(pos: torch.Tensor, quat: torch.Tensor) -> torch.Tensor:
    local = _CORNER_SIGNS.to(pos.device) * pos.new_tensor(BOX_HALF_EXTENTS_M)
    count = local.shape[0]
    world = quat_apply(quat.unsqueeze(1).expand(-1, count, -1).reshape(-1, 4), local.repeat(pos.shape[0], 1))
    return world.reshape(pos.shape[0], count, 3) + pos.unsqueeze(1)


def goal_corner_error(env) -> torch.Tensor:
    """Largest box-corner distance to the goal pose (m)."""
    box = _box(env)
    current = _corners(box.data.root_pos_w, box.data.root_quat_w)
    target = _corners(_goal(env), _start_quat(env))
    return torch.linalg.vector_norm(current - target, dim=-1).amax(dim=-1)


def _at_goal(env, tolerance_m: float, max_speed_mps: float) -> torch.Tensor:
    speed = torch.linalg.vector_norm(_box(env).data.root_lin_vel_w, dim=-1)
    return (goal_corner_error(env) < tolerance_m) & (speed < max_speed_mps)


def pose_to_goal(env, z_lifted_m: float = 0.05) -> torch.Tensor:
    """to-goal: progress on the corner error, only while lifted."""
    return _progress_on(env, "goal_pose", goal_corner_error(env), _lifted(env, z_lifted_m))


def pose_reached(env, tolerance_m: float = 0.02, max_speed_mps: float = 0.05) -> torch.Tensor:
    return _at_goal(env, tolerance_m, max_speed_mps).float()


def pose_success(env, tolerance_m: float = 0.02, max_speed_mps: float = 0.05, t_success_s: float = 1.0) -> torch.Tensor:
    """Termination: at the goal pose and nearly still for t_success_s in a row."""
    streak = _progress(env, "pose_streak")
    streak[streak.isnan()] = 0.0
    streak[:] = torch.where(_at_goal(env, tolerance_m, max_speed_mps), streak + 1.0, torch.zeros_like(streak))
    return streak * env.step_dt >= t_success_s - 1e-6


def goal_pose_in_box(env) -> torch.Tensor:
    """Goal position minus box position (robot base frame) and goal orientation in the box frame."""
    box = _box(env)
    relative_quat = quat_mul(quat_conjugate(box.data.root_quat_w), _start_quat(env))
    return torch.cat((goal_in_base(env), relative_quat), dim=-1)


# -- v2 "held": a grasp that lasts and carries the box ---------------------------
#
# A flick touches the box for a moment and sends it flying, so a single contact
# step is not enough: thumb + index/middle contact for min_steps in a row. (A
# box-vs-palm speed test did not separate real grasps from flicks: the box
# slides and turns in a real grasp too.) The slow action limits do the rest.


def box_palm_relative_speed(env) -> torch.Tensor:
    """Box speed relative to the palm, as if the box were rigidly attached to the palm at its current spot."""
    robot, box = _robot(env), _box(env)
    palm = _body_ids(env, (PALM_BODY,))[0]
    carried = robot.data.body_lin_vel_w[:, palm] + torch.cross(
        robot.data.body_ang_vel_w[:, palm], box.data.root_pos_w - robot.data.body_pos_w[:, palm], dim=-1
    )
    return torch.linalg.vector_norm(box.data.root_lin_vel_w - carried, dim=-1)


def grasp_held(env, min_steps: int = 3) -> torch.Tensor:
    """Computed once per control step (several terms ask for it)."""
    cache = env.__dict__.setdefault("_grasp_held_cache", {})
    step = int(env.common_step_counter)
    if cache.get("step") != step:
        streak = _progress(env, "contact_streak")
        streak[streak.isnan()] = 0.0
        streak[:] = torch.where(grasp_flag(env), streak + 1.0, torch.zeros_like(streak))
        cache["step"] = step
        cache["value"] = streak >= min_steps
    return cache["value"]


def held_lift(env, z_lifted_m: float = 0.05) -> torch.Tensor:
    return dextrah_lift(env, z_lifted_m) * grasp_held(env).float()


def held_lifted(env, z_lifted_m: float = 0.05) -> torch.Tensor:
    """Once, the first time the box is above z_lifted_m while held."""
    paid = _progress(env, "held_lifted_paid")
    first = _lifted(env, z_lifted_m) & grasp_held(env) & paid.isnan()
    paid[first] = 1.0
    return first.float()


def held_pose_to_goal(env, z_lifted_m: float = 0.05) -> torch.Tensor:
    return _progress_on(env, "held_goal_pose", goal_corner_error(env), _lifted(env, z_lifted_m) & grasp_held(env))


def held_pose_reached(env, tolerance_m: float = 0.02, max_speed_mps: float = 0.05) -> torch.Tensor:
    return (_at_goal(env, tolerance_m, max_speed_mps) & grasp_held(env)).float()


def held_pose_success(env, tolerance_m: float = 0.02, max_speed_mps: float = 0.05,
                      t_success_s: float = 1.0) -> torch.Tensor:
    """Termination: held at the goal pose, nearly still, for t_success_s in a row."""
    streak = _progress(env, "held_pose_streak")
    streak[streak.isnan()] = 0.0
    ok = _at_goal(env, tolerance_m, max_speed_mps) & grasp_held(env)
    streak[:] = torch.where(ok, streak + 1.0, torch.zeros_like(streak))
    return streak * env.step_dt >= t_success_s - 1e-6

