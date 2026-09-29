"""Analytic pregrasp states for one YCB object: generation checks and table resets for RL.

The scripted policy's pregrasp palm pose is a closed-form function of the
object's planar pose and its calibrated ``GraspObjectSpec``. Instead of simulating the approach, the arm is solved kinematically
onto that pose, the robot and box are written into the simulator, and a short
static hold checks that nothing collides. Accepted states are stored in a table
that RL resets sample from.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import torch

from isaaclab.utils.math import (
    compute_pose_error,
    quat_apply,
    quat_conjugate,
    quat_from_angle_axis,
    quat_mul,
)

from vla_isaaclab.envs.common import (
    LEFT_ARM_JOINT_NAMES,
    LEFT_END_EFFECTOR,
    LEFT_HAND_CLOSED_JOINT_POSITIONS,
    LEFT_HAND_JOINT_NAMES,
    PROJECT_ROOT,
    WAIST_JOINT_NAMES,
    GraspObjectSpec,
    object_up_axis,
)
from vla_isaaclab.envs.common.mdp import POSE_OFFSET_ATTR
from vla_isaaclab.policies.bounded_ik import bounded_dls


# Same IK joints and waist bound as the scripted policy.
ARM_JOINT_NAMES = (WAIST_JOINT_NAMES[0], *LEFT_ARM_JOINT_NAMES)
MAX_WAIST_YAW_DEVIATION_RAD = math.radians(25.0)
FINGERTIP_BODY_NAMES = ("left_hand_thumb_2_link", "left_hand_index_1_link", "left_hand_middle_1_link")
# 3: adds object_name, checked against the env's object_spec on load.
TABLE_FORMAT_VERSION = 3
PREGRASP_TABLE_DIR = PROJECT_ROOT / "outputs/rl"


def pregrasp_table_path(spec: GraspObjectSpec, open_hand: bool = False) -> Path:
    """outputs/rl/<object>/pregrasp_table_{preshape,open}.pt"""
    return PREGRASP_TABLE_DIR / spec.name / f"pregrasp_table_{'open' if open_hand else 'preshape'}.pt"


def hand_preshape_joint_pos(thumb_rotate: float, thumb: float, index: float, middle: float) -> dict[str, float]:
    """Left-hand joint targets by name; each curl follows its closed-pose sign."""
    closed = dict(zip(LEFT_HAND_JOINT_NAMES, LEFT_HAND_CLOSED_JOINT_POSITIONS))
    curl = {"thumb_1": thumb, "thumb_2": thumb, "index_0": index, "index_1": index, "middle_0": middle, "middle_1": middle}
    targets = {"left_hand_thumb_0_joint": math.copysign(thumb_rotate, closed["left_hand_thumb_0_joint"])}
    for short, value in curl.items():
        name = f"left_hand_{short}_joint"
        targets[name] = math.copysign(value, closed[name])
    return targets


def _axis(reference: torch.Tensor, index: int) -> torch.Tensor:
    axis = torch.zeros(reference.shape[0], 3, device=reference.device, dtype=reference.dtype)
    axis[:, index] = 1.0
    return axis


def box_yaw_quat(box_quat: torch.Tensor, nominal_quat: torch.Tensor) -> torch.Tensor:
    """World-Z yaw of the upright object relative to its calibrated default pose."""
    delta = quat_mul(box_quat, quat_conjugate(nominal_quat))
    x_axis = quat_apply(delta, _axis(box_quat, 0))
    yaw = torch.atan2(x_axis[:, 1], x_axis[:, 0])
    return quat_from_angle_axis(yaw, _axis(box_quat, 2))


def turn_point(spec: GraspObjectSpec, box_pos, box_quat, nominal_pos, nominal_quat, env_origins) -> torch.Tensor:
    """Scripted turn point carried rigidly with the object's planar pose (world frame)."""
    yaw_quat = box_yaw_quat(box_quat, nominal_quat)
    # turn_point_world is env-local; nominal_pos and the result are world frame.
    point = env_origins + nominal_pos.new_tensor(spec.turn_point_world)
    pivot = nominal_pos.clone()
    pivot[:, :2] = box_pos[:, :2]
    return pivot + quat_apply(yaw_quat, point - nominal_pos)


def pregrasp_palm_target(spec: GraspObjectSpec, box_pos, box_quat, nominal_quat, robot_root_pos, backoff_m):
    """Replicate SugarBoxPhaseStrategy._compute_grasp, pulled back along the approach.

    Returns world-frame palm position, palm orientation (wxyz) and the unit
    approach direction.
    """
    yaw_quat = box_yaw_quat(box_quat, nominal_quat)
    grasp_quat = quat_mul(yaw_quat, box_quat.new_tensor(spec.grasp_quat_wxyz).expand_as(box_quat))
    approach = quat_apply(grasp_quat, _axis(box_quat, 0))
    offset = quat_apply(yaw_quat, box_pos.new_tensor(spec.grasp_offset_world).expand_as(box_pos))
    # A vertical pinch axis (the bowl's rim pinch) has no horizontal direction to shift along.
    if spec.grasp_away_shift_m != 0.0:
        thickness_axis = quat_apply(box_quat, _axis(box_quat, spec.pinch_axis))
        thickness_axis[:, 2] = 0.0
        thickness_axis = thickness_axis / torch.linalg.vector_norm(thickness_axis, dim=-1, keepdim=True)
        away = box_pos - robot_root_pos
        away_axis = torch.where((thickness_axis * away).sum(-1, keepdim=True) < 0.0, -thickness_axis, thickness_axis)
        offset = offset + spec.grasp_away_shift_m * away_axis
    position = box_pos + offset - backoff_m.unsqueeze(-1) * approach
    return position, grasp_quat, approach


class ArmKinematics:
    """Batched kinematic IK for waist yaw + left arm; nothing is simulated."""

    def __init__(self, env, orientation_weight: float = 0.3, damping: float = 0.03):
        self.env = env
        self.robot = env.scene["robot"]
        self.arm_joint_ids, found = self.robot.find_joints(list(ARM_JOINT_NAMES), preserve_order=True)
        if list(found) != list(ARM_JOINT_NAMES):
            raise RuntimeError(f"Arm joint mismatch: expected {ARM_JOINT_NAMES}, found {found}")
        palm_ids, _ = self.robot.find_bodies([LEFT_END_EFFECTOR], preserve_order=True)
        self.palm_body_id = palm_ids[0]
        self.palm_jacobian_id = self.palm_body_id - 1 if self.robot.is_fixed_base else self.palm_body_id
        self.orientation_weight = orientation_weight
        self.damping = damping
        self.base_joint_pos = self.robot.data.default_joint_pos.clone()
        limits = self.robot.data.soft_joint_pos_limits[:, self.arm_joint_ids]
        self.lower, self.upper = limits[..., 0].clone(), limits[..., 1].clone()
        waist_center = self.base_joint_pos[:, self.arm_joint_ids[0]]
        self.lower[:, 0] = torch.maximum(self.lower[:, 0], waist_center - MAX_WAIST_YAW_DEVIATION_RAD)
        self.upper[:, 0] = torch.minimum(self.upper[:, 0], waist_center + MAX_WAIST_YAW_DEVIATION_RAD)
        self.use_finite_difference = False

    def set_hand(self, targets: dict[str, float]) -> None:
        """Hold the left hand at these joint positions in every written state."""
        ids, names = self.robot.find_joints(list(targets), preserve_order=True)
        self.base_joint_pos[:, ids] = self.base_joint_pos.new_tensor([targets[n] for n in names])

    def full_joint_pos(self, arm_q: torch.Tensor) -> torch.Tensor:
        q = self.base_joint_pos.clone()
        q[:, self.arm_joint_ids] = arm_q
        return q

    def set_arm(self, arm_q: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        q = self.full_joint_pos(arm_q)
        self.robot.write_joint_state_to_sim(q, torch.zeros_like(q))
        # Reading link poses refreshes the articulation kinematics.
        return (
            self.robot.data.body_pos_w[:, self.palm_body_id].clone(),
            self.robot.data.body_quat_w[:, self.palm_body_id].clone(),
        )

    def _analytic_jacobian(self) -> torch.Tensor:
        return self.robot.root_physx_view.get_jacobians()[:, self.palm_jacobian_id, :, self.arm_joint_ids].clone()

    def _finite_difference_jacobian(self, arm_q, pos, quat, eps: float = 1.0e-3) -> torch.Tensor:
        columns = []
        for index in range(arm_q.shape[1]):
            shifted = arm_q.clone()
            shifted[:, index] += eps
            pos_j, quat_j = self.set_arm(shifted)
            dpos, drot = compute_pose_error(pos, quat, pos_j, quat_j, rot_error_type="axis_angle")
            columns.append(torch.cat((dpos, drot), dim=-1) / eps)
        return torch.stack(columns, dim=-1)

    def check_jacobian(self, arm_q: torch.Tensor) -> float:
        """Compare PhysX's Jacobian after a kinematic write with finite differences."""
        pos, quat = self.set_arm(arm_q)
        analytic = self._analytic_jacobian()
        numeric = self._finite_difference_jacobian(arm_q, pos, quat)
        error = torch.linalg.matrix_norm(analytic - numeric) / torch.linalg.matrix_norm(numeric).clamp_min(1e-9)
        relative = error.max().item()
        self.use_finite_difference = relative > 0.05
        return relative

    def solve(self, arm_q, target_pos, target_quat, iterations: int, max_step: float = 0.08):
        arm_q = torch.clamp(arm_q.clone(), self.lower, self.upper)
        weight = self.orientation_weight
        for _ in range(iterations):
            pos, quat = self.set_arm(arm_q)
            jacobian = (
                self._finite_difference_jacobian(arm_q, pos, quat)
                if self.use_finite_difference
                else self._analytic_jacobian()
            )
            pos_err, rot_err = compute_pose_error(pos, quat, target_pos, target_quat, rot_error_type="axis_angle")
            error = torch.cat((pos_err, weight * rot_err), dim=-1)
            jacobian[:, 3:] *= weight
            delta_lower = torch.clamp(self.lower - arm_q, min=-max_step, max=0.0)
            delta_upper = torch.clamp(self.upper - arm_q, min=0.0, max=max_step)
            arm_q = arm_q + bounded_dls(jacobian, error, delta_lower, delta_upper, self.damping)
            arm_q = torch.clamp(arm_q, self.lower, self.upper)
        pos, quat = self.set_arm(arm_q)
        pos_err, rot_err = compute_pose_error(pos, quat, target_pos, target_quat, rot_error_type="axis_angle")
        # Waist yaw only has to stay inside its IK bound (enforced above); RL keeps
        # it fixed, so only the seven arm joints need room to move.
        margin, limiting_joint = torch.minimum(arm_q - self.lower, self.upper - arm_q)[:, 1:].min(dim=-1)
        limiting_joint = limiting_joint + 1
        return arm_q, {
            "ik_position_error_m": torch.linalg.vector_norm(pos_err, dim=-1),
            "ik_rotation_error_rad": torch.linalg.vector_norm(rot_err, dim=-1),
            "joint_limit_margin_rad": margin,
            # Index into ARM_JOINT_NAMES of the joint closest to its limit.
            "limiting_joint": limiting_joint.float(),
        }


OBJECT_CONTACT_SENSOR_PREFIX = "object_contact_"
ARM_CONTACT_SENSOR = "left_arm_contacts"


def hand_box_contact_forces(env) -> torch.Tensor:
    """Per-link hand/object contact force magnitudes, shape (num_envs, num_links)."""
    forces = [
        torch.linalg.vector_norm(sensor.data.force_matrix_w, dim=-1).amax(dim=(1, 2))
        for name, sensor in env.scene.sensors.items()
        if name.startswith(OBJECT_CONTACT_SENSOR_PREFIX)
    ]
    if not forces:
        raise RuntimeError(f"Scene has no '{OBJECT_CONTACT_SENSOR_PREFIX}*' contact sensors")
    return torch.stack(forces, dim=-1)


@dataclass
class SettleResult:
    joint_pos: torch.Tensor
    joint_vel: torch.Tensor
    joint_target: torch.Tensor
    box_state: torch.Tensor
    metrics: dict[str, torch.Tensor]


def settle_check(env, kinematics: ArmKinematics, arm_q, box_pose_w, palm_target_pos, palm_target_quat,
                 control_steps: int) -> SettleResult:
    """Write robot and box, hold the joint targets and measure what moved or touched."""
    robot = env.scene["robot"]
    box = env.scene["object"]
    arm_contacts = env.scene.sensors[ARM_CONTACT_SENSOR]
    q = kinematics.full_joint_pos(arm_q)
    robot.write_joint_state_to_sim(q, torch.zeros_like(q))
    robot.set_joint_position_target(q)
    box.write_root_pose_to_sim(box_pose_w)
    box.write_root_velocity_to_sim(torch.zeros(env.num_envs, 6, device=env.device))
    for sensor in env.scene.sensors.values():
        sensor.reset()

    max_box_force = torch.zeros(env.num_envs, device=env.device)
    max_arm_force = torch.zeros_like(max_box_force)
    for _ in range(control_steps * env.cfg.decimation):
        env.scene.write_data_to_sim()
        env.sim.step(render=False)
        env.scene.update(dt=env.physics_dt)
        max_box_force = torch.maximum(max_box_force, hand_box_contact_forces(env).amax(dim=-1))
        arm_force = torch.linalg.vector_norm(arm_contacts.data.net_forces_w, dim=-1).amax(dim=-1)
        max_arm_force = torch.maximum(max_arm_force, arm_force)

    box_pos, box_quat = box.data.root_pos_w, box.data.root_quat_w
    spec = env.cfg.object_spec
    up_start = object_up_axis(spec, box_pose_w[:, 3:7])
    up_end = object_up_axis(spec, box_quat)
    palm_pos = robot.data.body_pos_w[:, kinematics.palm_body_id]
    palm_quat = robot.data.body_quat_w[:, kinematics.palm_body_id]
    palm_pos_err, palm_rot_err = compute_pose_error(
        palm_pos, palm_quat, palm_target_pos, palm_target_quat, rot_error_type="axis_angle"
    )
    tip_ids, _ = robot.find_bodies(list(FINGERTIP_BODY_NAMES), preserve_order=True)
    tips = robot.data.body_pos_w[:, tip_ids]
    tips_in_box = quat_apply(
        quat_conjugate(box_quat).unsqueeze(1).expand(-1, len(tip_ids), -1).reshape(-1, 4),
        (tips - box_pos.unsqueeze(1)).reshape(-1, 3),
    ).reshape(env.num_envs, len(tip_ids), 3)
    metrics = {
        "box_displacement_m": torch.linalg.vector_norm(box_pos - box_pose_w[:, :3], dim=-1),
        "box_tilt_change_rad": torch.acos((up_start * up_end).sum(-1).clamp(-1.0, 1.0)),
        "box_speed_mps": torch.linalg.vector_norm(box.data.root_lin_vel_w, dim=-1),
        "box_angular_speed_radps": torch.linalg.vector_norm(box.data.root_ang_vel_w, dim=-1),
        "box_contact_force_n": max_box_force,
        # Net force on any left-arm/hand link: box, table or anything else.
        "arm_contact_force_n": max_arm_force,
        "settled_palm_position_error_m": torch.linalg.vector_norm(palm_pos_err, dim=-1),
        "settled_palm_rotation_error_rad": torch.linalg.vector_norm(palm_rot_err, dim=-1),
        # Object-frame fingertip positions, in order thumb, index, middle; used to
        # judge whether the open hand straddles the object.
        "fingertips_in_box_m": tips_in_box.reshape(env.num_envs, -1),
    }
    box_state = box.data.root_state_w.clone()
    box_state[:, :3] -= env.scene.env_origins
    return SettleResult(robot.data.joint_pos.clone(), robot.data.joint_vel.clone(), q, box_state, metrics)


@dataclass
class PregraspTable:
    joint_names: list[str]
    joint_pos: torch.Tensor  # settled state after the static hold
    joint_vel: torch.Tensor
    joint_target: torch.Tensor  # commanded IK solution the hold tracked
    box_state: torch.Tensor  # env-local root state (pos, quat wxyz, lin vel, ang vel)
    box_offset: torch.Tensor  # (x m, y m, yaw rad) relative to the calibrated pose
    backoff_m: torch.Tensor
    object_name: str
    meta: dict

    def __len__(self) -> int:
        return self.joint_pos.shape[0]


def load_pregrasp_table(path: str | Path, device, valid_only: bool = True) -> PregraspTable:
    data = torch.load(Path(path), map_location=device, weights_only=False)
    if data.get("format_version") != TABLE_FORMAT_VERSION:
        raise ValueError(f"Unsupported pregrasp table format: {data.get('format_version')}")
    rows = data["valid"] if valid_only else torch.ones_like(data["valid"])
    if not bool(rows.any()):
        raise ValueError(f"Pregrasp table {path} holds no valid states")
    return PregraspTable(
        joint_names=list(data["joint_names"]),
        joint_pos=data["joint_pos"][rows],
        joint_vel=data["joint_vel"][rows],
        joint_target=data["joint_target"][rows],
        box_state=data["box_state"][rows],
        box_offset=data["box_offset"][rows],
        backoff_m=data["backoff_m"][rows],
        object_name=data["object_name"],
        meta=data["meta"],
    )


def reset_from_pregrasp_table(env, env_ids: torch.Tensor, table_path: str):
    """Reset event: place robot and object at randomly drawn accepted pregrasp states."""
    table: PregraspTable | None = getattr(env, "pregrasp_table", None)
    robot = env.scene["robot"]
    box = env.scene["object"]
    if table is None:
        table = load_pregrasp_table(table_path, env.device)
        if table.joint_names != list(robot.joint_names):
            raise RuntimeError("Pregrasp table joint order does not match the robot articulation")
        if table.object_name != env.cfg.object_spec.name:
            raise RuntimeError(f"Pregrasp table {table_path} is for {table.object_name}, "
                               f"not {env.cfg.object_spec.name}")
        env.pregrasp_table = table
        env.pregrasp_table_index = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)
    rows = torch.randint(len(table), (len(env_ids),), device=env.device)
    # Resume the static hold: settled state, same targets as during the check.
    robot.write_joint_state_to_sim(table.joint_pos[rows], table.joint_vel[rows], env_ids=env_ids)
    robot.set_joint_position_target(table.joint_target[rows], env_ids=env_ids)
    box_state = table.box_state[rows].clone()
    box_state[:, :3] += env.scene.env_origins[env_ids]
    box.write_root_state_to_sim(box_state, env_ids=env_ids)
    env.pregrasp_table_index[env_ids] = rows
    # Start pose for the lift and tilt measurements in the grasp RL terms.
    if getattr(env, "grasp_object_start_pos", None) is None:
        env.grasp_object_start_pos = torch.zeros(env.num_envs, 3, device=env.device)
        env.grasp_object_start_up = torch.zeros(env.num_envs, 3, device=env.device)
        env.grasp_object_start_quat = torch.zeros(env.num_envs, 4, device=env.device)
    env.grasp_object_start_pos[env_ids] = box_state[:, :3]
    env.grasp_object_start_quat[env_ids] = box_state[:, 3:7]
    env.grasp_object_start_up[env_ids] = object_up_axis(env.cfg.object_spec, box_state[:, 3:7])
    offsets = getattr(env, POSE_OFFSET_ATTR, None)
    if offsets is None:
        offsets = torch.zeros(env.num_envs, 3, device=env.device)
        setattr(env, POSE_OFFSET_ATTR, offsets)
    offsets[env_ids] = table.box_offset[rows]
