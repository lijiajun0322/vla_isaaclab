"""Scripted sugar-box strategy plus bounded IK action generation."""

from __future__ import annotations

import math

import torch

from isaaclab.utils.math import compute_pose_error

from vla_isaaclab.envs.common import (
    LEFT_ARM_JOINT_NAMES,
    LEFT_END_EFFECTOR,
    LEFT_HAND_CLOSED_JOINT_POSITIONS,
    LEFT_HAND_JOINT_NAMES,
    LEFT_HAND_OPEN_JOINT_POSITIONS,
    WAIST_JOINT_NAMES,
)
from .bounded_ik import bounded_dls
from .joint_limits import joint_targets_to_normalized, resolved_action_joints
from .ycb_sugar_box_strategy import SugarBoxPhaseStrategy


class _G1Semantics:
    waist_joint_names = WAIST_JOINT_NAMES
    left_arm_joint_names = LEFT_ARM_JOINT_NAMES
    left_hand_joint_names = LEFT_HAND_JOINT_NAMES
    left_end_effector = LEFT_END_EFFECTOR
    left_hand_open_joint_positions = LEFT_HAND_OPEN_JOINT_POSITIONS
    left_hand_closed_joint_positions = LEFT_HAND_CLOSED_JOINT_POSITIONS


class _SugarBoxSemantics:
    metadata = {"sugar_box_dimensions_m": [0.092, 0.045, 0.176]}


class YCBSugarBoxScriptedPolicy:
    """Own task phases, bounded IK, hand targets, and action normalization."""

    def __init__(self, env):
        self.env = env
        self.robot = env.scene["robot"]
        self.strategy = SugarBoxPhaseStrategy(env, _G1Semantics, _SugarBoxSemantics)
        _, self.joint_names, self.action_joint_ids = resolved_action_joints(env)
        self.name_to_action_index = {name: index for index, name in enumerate(self.joint_names)}

        # Keep waist roll/pitch at their default targets.  Only waist yaw joins
        # the left arm in IK so the torso can turn without leaning sideways or
        # pitching forward/backward.
        self.arm_joint_names = [WAIST_JOINT_NAMES[0], *LEFT_ARM_JOINT_NAMES]
        self.locked_waist_joint_names = list(WAIST_JOINT_NAMES[1:])
        self.locked_waist_joint_ids, locked_found = self.robot.find_joints(
            self.locked_waist_joint_names, preserve_order=True
        )
        if list(locked_found) != self.locked_waist_joint_names:
            raise RuntimeError(
                f"Locked-waist joint mismatch: expected {self.locked_waist_joint_names}, "
                f"found {locked_found}"
            )
        self.arm_joint_ids, found = self.robot.find_joints(self.arm_joint_names, preserve_order=True)
        if list(found) != self.arm_joint_names:
            raise RuntimeError(f"Left-arm joint mismatch: expected {self.arm_joint_names}, found {found}")
        self.arm_action_indices = [self.name_to_action_index[name] for name in self.arm_joint_names]
        self.hand_joint_names = list(LEFT_HAND_JOINT_NAMES)
        self.hand_action_indices = [self.name_to_action_index[name] for name in self.hand_joint_names]
        self.open_hand = torch.tensor(LEFT_HAND_OPEN_JOINT_POSITIONS, device=env.device).unsqueeze(0)
        nominal_closed_hand = torch.tensor(
            LEFT_HAND_CLOSED_JOINT_POSITIONS, device=env.device
        ).unsqueeze(0)
        self.grip_closure_scale = 1.10
        self.closed_hand = self.open_hand + self.grip_closure_scale * (
            nominal_closed_hand - self.open_hand
        )

        palm_ids, _ = self.robot.find_bodies([LEFT_END_EFFECTOR], preserve_order=True)
        self.palm_body_id = palm_ids[0]
        self.palm_jacobian_id = self.palm_body_id - 1 if self.robot.is_fixed_base else self.palm_body_id
        self.max_joint_delta = 0.04
        self.max_tracking_error = 0.35
        self.max_waist_yaw_deviation = math.radians(25.0)
        self.ik_gain = 0.15
        self.ik_damping = 0.04
        self.orientation_weight = 0.20
        self.last_joint_targets = self.robot.data.joint_pos[:, self.arm_joint_ids].clone()
        self.last_mode = "pose"
        self.compute_count = 0
        self.debug_samples = []

    @property
    def phase(self):
        return self.strategy.phase

    @property
    def failed(self):
        return self.strategy.failed

    def reset(self):
        self.strategy.reset()
        self.last_joint_targets = self.robot.data.joint_pos[:, self.arm_joint_ids].clone()
        self.compute_count = 0
        self.debug_samples = []

    def compute(self, step: int = 0) -> torch.Tensor:
        target = self.strategy.compute(step)
        palm_pos = self.robot.data.body_pos_w[:, self.palm_body_id]
        palm_quat = self.robot.data.body_quat_w[:, self.palm_body_id]
        jacobian = self.robot.root_physx_view.get_jacobians()[
            :, self.palm_jacobian_id, :, self.arm_joint_ids
        ]
        current_arm = self.robot.data.joint_pos[:, self.arm_joint_ids]
        if target.orientation is None:
            self.last_mode = "position"
            error = target.position - palm_pos
            task_jacobian = jacobian[:, :3].clone()
            task_jacobian[:, :, 0] = 0.0
        else:
            self.last_mode = "pose"
            position_error, rotation_error = compute_pose_error(
                palm_pos,
                palm_quat,
                target.position,
                target.orientation,
                rot_error_type="axis_angle",
            )
            error = torch.cat((position_error, self.orientation_weight * rotation_error), dim=-1)
            task_jacobian = jacobian.clone()
            task_jacobian[:, 3:] *= self.orientation_weight

        limits = self.robot.data.soft_joint_pos_limits[:, self.arm_joint_ids]
        lower, upper = limits[..., 0].clone(), limits[..., 1].clone()
        waist_yaw_center = self.robot.data.default_joint_pos[:, self.arm_joint_ids[0]]
        lower[:, 0] = torch.maximum(
            lower[:, 0], waist_yaw_center - self.max_waist_yaw_deviation
        )
        upper[:, 0] = torch.minimum(
            upper[:, 0], waist_yaw_center + self.max_waist_yaw_deviation
        )
        reference = torch.clamp(
            self.last_joint_targets,
            current_arm - self.max_tracking_error,
            current_arm + self.max_tracking_error,
        )
        reference = torch.clamp(reference, lower, upper)
        delta_lower = torch.maximum(
            lower - reference, torch.full_like(reference, -self.max_joint_delta)
        )
        delta_upper = torch.minimum(
            upper - reference, torch.full_like(reference, self.max_joint_delta)
        )
        delta_lower = torch.maximum(
            delta_lower, current_arm - self.max_tracking_error - reference
        )
        delta_upper = torch.minimum(
            delta_upper, current_arm + self.max_tracking_error - reference
        )
        delta_lower = torch.minimum(delta_lower, torch.zeros_like(delta_lower))
        delta_upper = torch.maximum(delta_upper, torch.zeros_like(delta_upper))
        if target.orientation is None:
            delta_lower[:, 0] = 0.0
            delta_upper[:, 0] = 0.0
        correction = bounded_dls(
            task_jacobian,
            self.ik_gain * error,
            delta_lower,
            delta_upper,
            self.ik_damping,
        )
        raw_targets = reference + correction
        arm_targets = torch.clamp(raw_targets, lower, upper)
        self.last_joint_targets = arm_targets.clone()
        self.compute_count += 1
        if self.compute_count in {1, 2, 5, 10, 25} or self.compute_count % 50 == 0:
            self.debug_samples.append(
                {
                    "step": self.compute_count,
                    "mode": self.last_mode,
                    "ee_position_m": (
                        palm_pos[0] - self.env.scene.env_origins[0]
                    ).detach().cpu().tolist(),
                    "target_position_m": (
                        target.position[0] - self.env.scene.env_origins[0]
                    ).detach().cpu().tolist(),
                    "joint_position_rad": current_arm[0].detach().cpu().tolist(),
                    "raw_joint_delta_rad": (raw_targets[0] - current_arm[0]).detach().cpu().tolist(),
                    "commanded_joint_delta_rad": (
                        arm_targets[0] - current_arm[0]
                    ).detach().cpu().tolist(),
                }
            )

        targets = self.robot.data.default_joint_pos[:, self.action_joint_ids].clone()
        targets[:, self.arm_action_indices] = arm_targets
        fraction = torch.clamp(target.gripper_closed_fraction, 0.0, 1.0)
        hand_targets = self.open_hand + fraction * (self.closed_hand - self.open_hand)
        targets[:, self.hand_action_indices] = hand_targets
        return joint_targets_to_normalized(self.robot, self.action_joint_ids, targets)

    def diagnostics(self) -> dict:
        strategy = self.strategy.diagnostics()
        strategy["control"] = {
            "arm_joint_names": self.arm_joint_names,
            "locked_waist_joint_names": self.locked_waist_joint_names,
            "locked_waist_joint_position_rad": self.robot.data.joint_pos[
                0, self.locked_waist_joint_ids
            ].detach().cpu().tolist(),
            "arm_joint_position_rad": self.robot.data.joint_pos[
                0, self.arm_joint_ids
            ].detach().cpu().tolist(),
            "arm_joint_target_rad": self.last_joint_targets[0].detach().cpu().tolist(),
            "max_joint_delta_rad": self.max_joint_delta,
            "max_waist_yaw_deviation_rad": self.max_waist_yaw_deviation,
            "grip_closure_scale": self.grip_closure_scale,
            "max_tracking_error_rad": self.max_tracking_error,
            "ik_method": "bounded_dls_accumulated_command",
            "ik_mode": self.last_mode,
            "debug_samples": self.debug_samples,
        }
        return strategy
