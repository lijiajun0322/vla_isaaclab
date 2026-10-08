"""Observation, reward and termination terms for closed-loop grasp-and-lift RL.

The episode starts at a pregrasp state (see ``vla_isaaclab.rl.pregrasp_table``),
which records the object's start pose on the env as ``grasp_object_start_pos``,
``grasp_object_start_quat`` and ``grasp_object_start_up``. Lift is measured from
that start height. Object size and axes come from ``env.cfg.object_spec``.
"""

from __future__ import annotations

import numpy as np
import torch

from isaaclab.assets import Articulation, RigidObject
from isaaclab.utils.math import quat_apply, quat_conjugate, quat_mul, subtract_frame_transforms

from vla_isaaclab.envs.common import (
    LEFT_HAND_CLOSED_JOINT_POSITIONS,
    LEFT_HAND_JOINT_NAMES,
    GraspObjectSpec,
    object_up_axis,
)


PALM_BODY = "left_hand_palm_link"
FINGERTIP_BODIES = ("left_hand_thumb_2_link", "left_hand_index_1_link", "left_hand_middle_1_link")
# Hand/object contact sensors (one per link; see YCBGraspStateSceneCfg).
THUMB_CONTACT_SENSORS = ("object_contact_thumb_1", "object_contact_thumb_2")
FINGER_CONTACT_SENSORS = (
    "object_contact_index_0",
    "object_contact_index_1",
    "object_contact_middle_0",
    "object_contact_middle_1",
)
HAND_CONTACT_SENSORS = (
    "object_contact_palm",
    "object_contact_thumb_0",
    *THUMB_CONTACT_SENSORS,
    *FINGER_CONTACT_SENSORS,
)
CONTACT_THRESHOLD_N = 0.5
# Hand/table contact sensors, added only where the task uses them (see add_hand_table_contacts).
TABLE_CONTACT_SENSORS = tuple(f"table_contact_{short}" for short in (
    "palm", "thumb_0", "thumb_1", "thumb_2", "index_0", "index_1", "middle_0", "middle_1"))


def _object(env) -> RigidObject:
    return env.scene["object"]


def _robot(env) -> Articulation:
    return env.scene["robot"]


def _spec(env) -> GraspObjectSpec:
    return env.cfg.object_spec


def _body_ids(env, names) -> list[int]:
    cache = env.__dict__.setdefault("_grasp_rl_body_ids", {})
    if names not in cache:
        ids, _ = _robot(env).find_bodies(list(names), preserve_order=True)
        cache[names] = ids
    return cache[names]


def _side(env) -> str:
    """Grasping hand, cfg.grasp_side ("left" by default)."""
    return getattr(env.cfg, "grasp_side", "left")


def _palm_body(env) -> str:
    return f"{_side(env)}_hand_palm_link"


def _fingertip_bodies(env) -> tuple[str, ...]:
    side = _side(env)
    return (f"{side}_hand_thumb_2_link", f"{side}_hand_index_1_link", f"{side}_hand_middle_1_link")


def _fingertip_offsets(env) -> tuple[tuple[float, float, float], ...]:
    """FINGERTIP_OFFSETS; the right hand mirrors the left (palm/link y flipped), so its thumb points along +y."""
    if _side(env) == "right":
        return tuple((x, -y, z) if i == 0 else (x, y, z) for i, (x, y, z) in enumerate(FINGERTIP_OFFSETS))
    return FINGERTIP_OFFSETS


def _sided_sensors(env, sensors) -> tuple[str, ...]:
    """Left-hand sensor names mapped to the right hand's (object_contact_right_*, table_contact_right_*)."""
    if _side(env) != "right":
        return tuple(sensors)
    out = []
    for name in sensors:
        for prefix in ("object_contact_", "table_contact_"):
            if name.startswith(prefix) and not name.startswith(prefix + "right_"):
                name = prefix + "right_" + name[len(prefix):]
                break
        out.append(name)
    return tuple(out)


def _start_pos(env) -> torch.Tensor:
    start = getattr(env, "grasp_object_start_pos", None)
    return _object(env).data.root_pos_w if start is None else start


def contact_forces(env, sensors=HAND_CONTACT_SENSORS) -> torch.Tensor:
    """Hand-link/object contact force magnitudes, (num_envs, len(sensors)).

    Zero on the first step of an episode: until a physics step runs after a
    reset, a contact sensor re-reads PhysX's last report, which is the previous
    episode's final grip. Episodes start from table states without hand contact.
    """
    forces = torch.stack(
        [
            torch.linalg.vector_norm(env.scene.sensors[name].data.force_matrix_w, dim=-1).amax(dim=(1, 2))
            for name in _sided_sensors(env, sensors)
        ],
        dim=-1,
    )
    just_reset = getattr(env, "episode_length_buf", None)
    if just_reset is not None:
        forces = forces * (just_reset > 0).unsqueeze(-1)
    return forces


def grasp_flag(env) -> torch.Tensor:
    """Thumb on the object together with index or middle finger."""
    thumb = contact_forces(env, THUMB_CONTACT_SENSORS).amax(dim=-1) > CONTACT_THRESHOLD_N
    finger = contact_forces(env, FINGER_CONTACT_SENSORS).amax(dim=-1) > CONTACT_THRESHOLD_N
    return thumb & finger


# Object mesh vertices (root frame) for the lowest point: a fixed subsample is
# within a few mm of the true minimum and cheap for thousands of envs.
LOWEST_POINT_SAMPLES = 1024


def object_points_local(env) -> torch.Tensor:
    """Object mesh vertices in its root frame, from the spec's USD, (<= LOWEST_POINT_SAMPLES, 3); cached."""
    points = env.__dict__.get("_object_points_local")
    if points is None:
        from pxr import Usd, UsdGeom

        stage = Usd.Stage.Open(_spec(env).usd_path)
        cache = UsdGeom.XformCache()
        root = stage.GetDefaultPrim()
        to_root = cache.GetLocalToWorldTransform(root).GetInverse()
        parts = []
        for prim in Usd.PrimRange(root):
            if prim.IsA(UsdGeom.Mesh):
                local = torch.from_numpy(np.asarray(UsdGeom.Mesh(prim).GetPointsAttr().Get(), dtype=np.float64))
                matrix = torch.from_numpy(np.asarray(cache.GetLocalToWorldTransform(prim) * to_root, dtype=np.float64))
                parts.append(torch.cat((local, torch.ones(len(local), 1, dtype=torch.float64)), -1) @ matrix)
        points = torch.cat(parts)[:, :3].float()
        if len(points) > LOWEST_POINT_SAMPLES:
            points = points[torch.randperm(len(points), generator=torch.Generator().manual_seed(0))[:LOWEST_POINT_SAMPLES]]
        points = points.to(env.device)
        env.__dict__["_object_points_local"] = points
    return points


def object_lowest_height(env) -> torch.Tensor:
    """Height of the object's lowest point above the table top, (num_envs,)."""
    obj = _object(env)
    points = object_points_local(env)
    count = points.shape[0]
    quat = obj.data.root_quat_w.unsqueeze(1).expand(-1, count, -1).reshape(-1, 4)
    world_z = quat_apply(quat, points.repeat(env.num_envs, 1))[:, 2].reshape(env.num_envs, count)
    return world_z.amin(dim=-1) + obj.data.root_pos_w[:, 2] - _spec(env).support_height_m


def lift_height(env) -> torch.Tensor:
    """Root rise above the start, or, with cfg.lift_from_lowest_point, the lowest point's height above the table
    (a tipped bowl raises its root without leaving the table)."""
    if getattr(env.cfg, "lift_from_lowest_point", False):
        return object_lowest_height(env)
    return _object(env).data.root_pos_w[:, 2] - _start_pos(env)[:, 2]


def object_tilt(env) -> torch.Tensor:
    """Angle (rad) of the object's up axis from its start orientation."""
    start_up = getattr(env, "grasp_object_start_up", None)
    up = object_up_axis(_spec(env), _object(env).data.root_quat_w)
    if start_up is None:
        start_up = torch.zeros_like(up)
        start_up[:, 2] = 1.0
    return torch.acos((up * start_up).sum(-1).clamp(-1.0, 1.0))


def fingertips_in_object(env) -> torch.Tensor:
    """Fingertip link positions in the object frame, (num_envs, 3, 3)."""
    obj = _object(env)
    tips = _robot(env).data.body_pos_w[:, _body_ids(env, _fingertip_bodies(env))]
    count = tips.shape[1]
    quat = quat_conjugate(obj.data.root_quat_w).unsqueeze(1).expand(-1, count, -1).reshape(-1, 4)
    local = quat_apply(quat, (tips - obj.data.root_pos_w.unsqueeze(1)).reshape(-1, 3))
    return local.reshape(env.num_envs, count, 3)


# Far end of each distal link in its own frame (thumb, index, middle; from the
# visual meshes, the links are 52 mm long). The link origins sit at the last joint.
FINGERTIP_OFFSETS = ((0.0, -0.045, 0.0), (0.045, 0.0, 0.0), (0.045, 0.0, 0.0))


def fingertip_points_in_object(env) -> torch.Tensor:
    """Actual fingertip points (not link origins) in the object frame, (num_envs, 3, 3)."""
    robot, obj = _robot(env), _object(env)
    ids = _body_ids(env, _fingertip_bodies(env))
    pos, quat = robot.data.body_pos_w[:, ids], robot.data.body_quat_w[:, ids]
    offsets = pos.new_tensor(_fingertip_offsets(env)).expand_as(pos)
    tips = pos + quat_apply(quat.reshape(-1, 4), offsets.reshape(-1, 3)).reshape(pos.shape)
    count = tips.shape[1]
    inv = quat_conjugate(obj.data.root_quat_w).unsqueeze(1).expand(-1, count, -1).reshape(-1, 4)
    local = quat_apply(inv, (tips - obj.data.root_pos_w.unsqueeze(1)).reshape(-1, 3))
    return local.reshape(env.num_envs, count, 3)


def fingertip_surface_gaps(env) -> torch.Tensor:
    """Fingertip-link distance to the object's collider box (0 inside), (num_envs, 3)."""
    spec = _spec(env)
    tips = fingertips_in_object(env)
    tips = tips - tips.new_tensor(spec.box_center_m)
    outside = (tips.abs() - tips.new_tensor(spec.half_extents_m)).clamp(min=0.0)
    return torch.linalg.vector_norm(outside, dim=-1)


# -- observations -------------------------------------------------------------


def palm_pose_b(env) -> torch.Tensor:
    robot = _robot(env)
    palm = _body_ids(env, (_palm_body(env),))[0]
    pos, quat = subtract_frame_transforms(
        robot.data.root_pos_w, robot.data.root_quat_w, robot.data.body_pos_w[:, palm], robot.data.body_quat_w[:, palm]
    )
    return torch.cat((pos, quat), dim=-1)


def object_pose_in_palm(env) -> torch.Tensor:
    robot = _robot(env)
    obj = _object(env)
    palm = _body_ids(env, (_palm_body(env),))[0]
    pos, quat = subtract_frame_transforms(
        robot.data.body_pos_w[:, palm], robot.data.body_quat_w[:, palm], obj.data.root_pos_w, obj.data.root_quat_w
    )
    return torch.cat((pos, quat), dim=-1)


def object_velocity_b(env) -> torch.Tensor:
    robot = _robot(env)
    obj = _object(env)
    inv = quat_conjugate(robot.data.root_quat_w)
    return torch.cat((quat_apply(inv, obj.data.root_lin_vel_w), quat_apply(inv, obj.data.root_ang_vel_w)), dim=-1)


def fingertips_in_object_obs(env) -> torch.Tensor:
    return fingertips_in_object(env).reshape(env.num_envs, -1)


def lift_obs(env) -> torch.Tensor:
    return torch.stack((lift_height(env), object_tilt(env)), dim=-1)


def hand_contact_force(env, max_force: float = 20.0) -> torch.Tensor:
    return contact_forces(env).clamp(max=max_force) / max_force


def hand_contact_force_clipped(env, max_force: float = 10.0) -> torch.Tensor:
    """Per-link contact force scaled to [0, 1]; the actor sees how hard it presses."""
    return contact_forces(env).clamp(max=max_force) / max_force


def table_contact_force(env, max_force: float = 20.0) -> torch.Tensor:
    """Per-link hand/table contact force scaled to [0, 1] (critic only: a real hand has no such sensor)."""
    return contact_forces(env, TABLE_CONTACT_SENSORS).clamp(max=max_force) / max_force


# -- hand pressing on the table ---------------------------------------------------
#
# With the bowl the lower finger starts a few millimetres above the table and can
# pry the bowl up against it. Light touches are free; pressing costs up to 1 per
# step, and a crushing press ends the episode.


def table_press(env, threshold_n: float = 1.0, max_n: float = 21.0) -> torch.Tensor:
    """Largest hand-link/table force above threshold_n, scaled to [0, 1] at max_n."""
    force = contact_forces(env, TABLE_CONTACT_SENSORS).amax(dim=-1)
    return (force - threshold_n).clamp(min=0.0, max=max_n - threshold_n) / (max_n - threshold_n)


def table_crush(env, max_force_n: float = 30.0) -> torch.Tensor:
    """Termination: a hand link presses on the table harder than max_force_n."""
    return contact_forces(env, TABLE_CONTACT_SENSORS).amax(dim=-1) > max_force_n


# -- progress rewards (DextrAH-G / DexPBT style) -------------------------------
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


def _progress_on(env, key: str, error: torch.Tensor, active: torch.Tensor | None = None) -> torch.Tensor:
    """max(best - error, 0), then best = min(best, error); inactive envs keep no best."""
    best = _progress(env, key, error.shape[1] if error.dim() > 1 else 0)
    active = torch.ones_like(error, dtype=torch.bool) if active is None else active
    previous = torch.where(best.isnan(), error, best)
    reward = torch.where(active, (previous - error).clamp(min=0.0), torch.zeros_like(error))
    best[:] = torch.where(active, torch.minimum(previous, error), best)
    return reward


# -- goal pose: the object as it sat on the table, raised a fixed height ---------
#
# Distance to the goal is DexPBT's keypoint distance: the largest gap between a
# collider-box corner and the same corner of the goal pose, so tilting counts
# as much as being off in position.

GOAL_ATTR = "grasp_goal_pos"
_CORNER_SIGNS = torch.tensor([[x, y, z] for x in (-1.0, 1.0) for y in (-1.0, 1.0) for z in (-1.0, 1.0)])


def sample_lift_goal(env, env_ids: torch.Tensor | None, xy_range_m: float = 0.05,
                     z_range_m: tuple[float, float] = (0.20, 0.30)):
    """Reset event: a goal point above the object's start position (world frame)."""
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
        goal[:, 2] += 0.05
        setattr(env, GOAL_ATTR, goal)
    return goal


def _start_quat(env) -> torch.Tensor:
    quat = getattr(env, "grasp_object_start_quat", None)
    return _object(env).data.root_quat_w if quat is None else quat


def _lifted(env, z_lifted_m: float) -> torch.Tensor:
    """lifted(x_obj): the object is currently at least z_lifted_m above its start."""
    return lift_height(env) > z_lifted_m


def _corners(env, pos: torch.Tensor, quat: torch.Tensor) -> torch.Tensor:
    spec = _spec(env)
    local = _CORNER_SIGNS.to(pos.device) * pos.new_tensor(spec.half_extents_m) + pos.new_tensor(spec.box_center_m)
    count = local.shape[0]
    world = quat_apply(quat.unsqueeze(1).expand(-1, count, -1).reshape(-1, 4), local.repeat(pos.shape[0], 1))
    return world.reshape(pos.shape[0], count, 3) + pos.unsqueeze(1)


def goal_corner_error(env) -> torch.Tensor:
    """Largest collider-box corner distance to the goal pose (m).

    A rim object (spec.rim_radius_m > 0, the bowl) is rotationally symmetric, so
    yaw about its up axis must not count: its error is the collider-center
    offset plus how far tilting moves a rim point (rim radius x tilt angle).
    """
    obj = _object(env)
    spec = _spec(env)
    if spec.rim_radius_m > 0.0:
        center = obj.data.root_pos_w.new_tensor(spec.box_center_m).expand_as(obj.data.root_pos_w)
        current = obj.data.root_pos_w + quat_apply(obj.data.root_quat_w, center)
        target = _goal(env) + quat_apply(_start_quat(env), center)
        return torch.linalg.vector_norm(current - target, dim=-1) + spec.rim_radius_m * object_tilt(env)
    current = _corners(env, obj.data.root_pos_w, obj.data.root_quat_w)
    target = _corners(env, _goal(env), _start_quat(env))
    return torch.linalg.vector_norm(current - target, dim=-1).amax(dim=-1)


def goal_in_base(env) -> torch.Tensor:
    """Goal minus object position, in the robot base frame."""
    robot = _robot(env)
    return quat_apply(quat_conjugate(robot.data.root_quat_w), _goal(env) - _object(env).data.root_pos_w)


def goal_pose_in_object(env) -> torch.Tensor:
    """Goal position minus object position (robot base frame) and goal orientation in the object frame."""
    obj = _object(env)
    relative_quat = quat_mul(quat_conjugate(obj.data.root_quat_w), _start_quat(env))
    return torch.cat((goal_in_base(env), relative_quat), dim=-1)


# -- "held": a grasp that lasts and carries the object ---------------------------
#
# A flick touches the object for a moment and sends it flying, so a single
# contact step is not enough: thumb + index/middle contact for min_steps in a
# row. (An object-vs-palm speed test did not separate real grasps from flicks:
# the object slides and turns in a real grasp too.) The slow action limits do
# the rest.


def object_palm_relative_speed(env) -> torch.Tensor:
    """Object speed relative to the palm, as if rigidly attached to the palm at its current spot."""
    robot, obj = _robot(env), _object(env)
    palm = _body_ids(env, (_palm_body(env),))[0]
    carried = robot.data.body_lin_vel_w[:, palm] + torch.cross(
        robot.data.body_ang_vel_w[:, palm], obj.data.root_pos_w - robot.data.body_pos_w[:, palm], dim=-1
    )
    return torch.linalg.vector_norm(obj.data.root_lin_vel_w - carried, dim=-1)


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


# -- rewards: DextrAH-G (arXiv 2407.02274) progress terms before the lift ----------


def fingertip_rim_distances(env) -> torch.Tensor:
    """Fingertip distance to the object's rim circle (see GraspObjectSpec.rim_radius_m), (num_envs, 3)."""
    spec = _spec(env)
    tips = fingertip_points_in_object(env)
    axial = tips[..., spec.up_axis] * spec.up_sign
    radial = torch.linalg.vector_norm(tips, dim=-1).square() - axial.square()
    radial = radial.clamp(min=0.0).sqrt()
    return torch.hypot(radial - spec.rim_radius_m, axial - spec.rim_height_m)


def dextrah_to_object(env) -> torch.Tensor:
    """Progress on the summed fingertip distance to the object origin, or to its rim for a rim grasp."""
    if _spec(env).rim_radius_m > 0.0:
        distance = torch.linalg.vector_norm(fingertip_rim_distances(env), dim=-1)
    else:
        tips = _robot(env).data.body_pos_w[:, _body_ids(env, _fingertip_bodies(env))]
        distance = torch.linalg.vector_norm((tips - _object(env).data.root_pos_w.unsqueeze(1)).flatten(1), dim=-1)
    return _progress_on(env, "to_object", distance)


def grasp_contact(env) -> torch.Tensor:
    """Dense contact reward: 0.25 thumb on the object, 0.25 index or middle, 0.5 more for both (max 1)."""
    thumb = (contact_forces(env, THUMB_CONTACT_SENSORS).amax(dim=-1) > CONTACT_THRESHOLD_N).float()
    finger = (contact_forces(env, FINGER_CONTACT_SENSORS).amax(dim=-1) > CONTACT_THRESHOLD_N).float()
    return 0.25 * thumb + 0.25 * finger + 0.5 * thumb * finger


def hand_closure(env) -> torch.Tensor:
    """Progress on the summed hand-joint distance to the closed pose, until thumb and a finger touch."""
    cache = env.__dict__.setdefault("_grasp_rl_hand_joints", {})
    if "ids" not in cache:
        # The right hand closes in the mirrored direction (its joint limits are the left's negated).
        side, mirror = _side(env), (-1.0 if _side(env) == "right" else 1.0)
        names = [name.replace("left_", f"{side}_", 1) for name in LEFT_HAND_JOINT_NAMES]
        ids, _ = _robot(env).find_joints(names, preserve_order=True)
        cache["ids"] = ids
        cache["closed"] = torch.tensor(LEFT_HAND_CLOSED_JOINT_POSITIONS, device=env.device) * mirror
    error = (_robot(env).data.joint_pos[:, cache["ids"]] - cache["closed"]).abs().sum(dim=-1)
    return _progress_on(env, "hand_closure", error, active=~grasp_flag(env))


def held_lift(env, z_lifted_m: float = 0.05) -> torch.Tensor:
    """Progress on (z_lifted - lift height) while held, until lifted."""
    lifted = _lifted(env, z_lifted_m)
    error = (z_lifted_m - lift_height(env)).clamp(min=0.0)
    return _progress_on(env, "lift_error", error) * (~lifted).float() * grasp_held(env).float()


# -- lift-and-score: the episode ends when the held object passes 5 cm -----------
#
# The goal pose is the object's start pose raised z_lifted_m. At the moment the
# held object passes that height the episode ends and is scored by how far the
# collider corners are from the goal pose. "success" is the named subset with
# every corner within success_m (about 10 degrees of tilt for the sugar box; for
# the bowl, center offset + rim radius x tilt, about 14 degrees at no offset).

LIFT_CORNER_ERROR_ATTR = "grasp_lift_corner_error"


def held_lift_event(env, z_lifted_m: float) -> tuple[torch.Tensor, torch.Tensor]:
    """(lifted this step while held, corner error); computed once per step."""
    cache = env.__dict__.setdefault("_held_lift_cache", {})
    step = int(env.common_step_counter)
    if cache.get("step") != step:
        error = goal_corner_error(env)
        fired = _lifted(env, z_lifted_m) & grasp_held(env)
        record = env.__dict__.setdefault(LIFT_CORNER_ERROR_ATTR, torch.full((env.num_envs,), float("nan"),
                                                                             device=env.device))
        record[fired] = error[fired]
        cache.update(step=step, value=(fired, error))
    return cache["value"]


def lift_success(env, z_lifted_m: float = 0.05, success_m: float = 0.02) -> torch.Tensor:
    """Termination: held object passed z_lifted_m with every corner within success_m of the goal pose."""
    fired, error = held_lift_event(env, z_lifted_m)
    return fired & (error < success_m)


def lift_off_pose(env, z_lifted_m: float = 0.05, success_m: float = 0.02) -> torch.Tensor:
    """Termination: held object passed z_lifted_m but tilted or shifted beyond success_m."""
    fired, error = held_lift_event(env, z_lifted_m)
    return fired & (error >= success_m)


def lift_pose_score(env, z_lifted_m: float = 0.05, scale_m: float = 0.03) -> torch.Tensor:
    """On the lift step: exp(-corner error / scale_m), 1 for a perfectly upright, unshifted object."""
    fired, error = held_lift_event(env, z_lifted_m)
    return fired.float() * torch.exp(-error / scale_m)
