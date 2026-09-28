"""Deterministic side-approach strategy for the YCB sugar box."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from isaaclab.utils.math import (
    quat_apply,
    quat_conjugate,
    quat_error_magnitude,
    quat_from_angle_axis,
    quat_mul,
    quat_slerp,
)



@dataclass(frozen=True)
class EndEffectorTarget:
    position: torch.Tensor
    orientation: torch.Tensor | None
    gripper_closed_fraction: torch.Tensor


PHASES = (
    "reset",
    "move_to_turn_point",
    "orient_hand",
    "move_pregrasp",
    "approach",
    "close_gripper",
    "lift",
    "move_left",
    "level_box",
    "lower",
    "open_gripper",
    "retreat",
    "done",
    "failed",
)


@dataclass(frozen=True)
class PhaseLimit:
    minimum_steps: int
    maximum_steps: int


LIMITS = {
    # The object is spawned exactly on its support surface.  Start the
    # clearance move immediately so the arm does not settle toward the box
    # before the first commanded motion.
    "reset": PhaseLimit(0, 0),
    "move_to_turn_point": PhaseLimit(40, 300),
    "orient_hand": PhaseLimit(30, 300),
    "move_pregrasp": PhaseLimit(35, 300),
    "approach": PhaseLimit(90, 240),
    "close_gripper": PhaseLimit(120, 120),
    "lift": PhaseLimit(90, 200),
    "move_left": PhaseLimit(60, 150),
    "level_box": PhaseLimit(100, 240),
    "lower": PhaseLimit(120, 300),
    "open_gripper": PhaseLimit(90, 120),
    "retreat": PhaseLimit(80, 180),
    "done": PhaseLimit(10_000_000, 10_000_000),
    "failed": PhaseLimit(10_000_000, 10_000_000),
}

class SugarBoxPhaseStrategy:
    """Generate side-approach EE/gripper targets without implementing control."""

    PREGRASP_CLEARANCE_M = 0.0
    TARGET_WORLD_DELTA = (-0.02, 0.0, 0.0)

    def __init__(self, env, robot_definition, object_spec):
        self.env = env
        self.definition = robot_definition
        # Calibrated approach (turn point, grasp orientation and offset) and axes;
        # see GraspObjectSpec.
        self.spec = object_spec
        self.robot = env.scene["robot"]
        self.sugar_box = env.scene["object"]
        palm_ids, _ = self.robot.find_bodies(
            [robot_definition.left_end_effector], preserve_order=True
        )
        self.palm_body_id = palm_ids[0]
        self.hand_joint_ids, _ = self.robot.find_joints(
            list(robot_definition.left_hand_joint_names), preserve_order=True
        )
        self.sugar_box_height_m = 2.0 * object_spec.half_extents_m[object_spec.up_axis]
        self.reset()

    @property
    def phase(self) -> str:
        return PHASES[self.phase_index]

    def _palm_pose(self) -> tuple[torch.Tensor, torch.Tensor]:
        return (
            self.robot.data.body_pos_w[:, self.palm_body_id],
            self.robot.data.body_quat_w[:, self.palm_body_id],
        )

    def reset(self) -> None:
        palm_pos, palm_quat = self._palm_pose()
        self.phase_index = 0
        self.phase_step = 0
        self.total_step = 0
        self.failed = False
        self.failure_reason = None
        self.failed_target_pos = None
        self.last_position_error = float("inf")
        self.last_orientation_error = float("inf")
        self.phase_history = []
        self.orientation_ready_steps = 0
        self.grasp_ready_steps = 0
        self.measured_lift_rise_m = None
        self.placement_ready_steps = 0
        self.placement_box_pos = None
        self.placement_box_quat = None
        self.phase_start_pos = palm_pos.clone()
        self.phase_start_quat = palm_quat.clone()
        self.target_pos = palm_pos.clone()
        self.target_quat = palm_quat.clone()
        self.gripper_start = 0.0
        self.gripper_target = 0.0
        self.initial_box_pos = self.sugar_box.data.root_pos_w.clone()
        self.pregrasp_pos = None
        self.grasp_pos = None
        self.grasp_offset = None
        self.grasp_quat = None
        self.box_yaw_delta = None
        self.box_yaw_quat = None
        self.travel_quat = None
        self.approach_direction = None
        self.box_top_z = None
        self.grasp_depth_below_box_top_m = None
        self.env.policy_phase = torch.zeros(
            self.env.num_envs, dtype=torch.long, device=self.env.device
        )

    def _measure_box_yaw(self) -> None:
        # World-Z yaw of the upright box relative to its calibrated default pose.
        nominal = self.sugar_box.data.default_root_state[:, 3:7]
        delta = quat_mul(self.sugar_box.data.root_quat_w, quat_conjugate(nominal))
        x_axis = quat_apply(delta, delta.new_tensor([[1.0, 0.0, 0.0]]).repeat(self.env.num_envs, 1))
        self.box_yaw_delta = torch.atan2(x_axis[:, 1], x_axis[:, 0])
        z_axis = delta.new_tensor([[0.0, 0.0, 1.0]]).repeat(self.env.num_envs, 1)
        self.box_yaw_quat = quat_from_angle_axis(self.box_yaw_delta, z_axis)

    def _box_frame_point(self, calibrated_point) -> torch.Tensor:
        # Carry a calibrated env-local point rigidly with the box's planar pose.
        nominal = self.sugar_box.data.default_root_state[:, :3] + self.env.scene.env_origins
        relative = quat_apply(
            self.box_yaw_quat, nominal.new_tensor([calibrated_point]) + self.env.scene.env_origins - nominal
        )
        pivot = nominal.clone()
        pivot[:, :2] = self.sugar_box.data.root_pos_w[:, :2]
        return pivot + relative

    def _compute_grasp(self) -> None:
        # Fitted to the upright, yawed box and actual three-finger pad sweeps.
        # Keep the reachable wrist tilt rather than forcing a level palm.
        self._measure_box_yaw()
        self.grasp_quat = quat_mul(
            self.box_yaw_quat,
            self.sugar_box.data.root_pos_w.new_tensor([self.spec.grasp_quat_wxyz]).repeat(self.env.num_envs, 1),
        )
        self.travel_quat = self.grasp_quat.clone()
        self.approach_direction = quat_apply(
            self.grasp_quat, self.grasp_quat.new_tensor([[1.0, 0.0, 0.0]]).repeat(self.env.num_envs, 1)
        )
        pinch = [0.0, 0.0, 0.0]
        pinch[self.spec.pinch_axis] = 1.0
        thickness_axis = quat_apply(
            self.sugar_box.data.root_quat_w,
            self.grasp_quat.new_tensor([pinch]).repeat(self.env.num_envs, 1),
        )
        thickness_axis[:, 2] = 0.0
        thickness_axis = thickness_axis / torch.linalg.vector_norm(thickness_axis, dim=-1, keepdim=True)
        away = self.sugar_box.data.root_pos_w - self.robot.data.root_pos_w
        away_axis = torch.where(
            (thickness_axis * away).sum(-1, keepdim=True) < 0.0, -thickness_axis, thickness_axis
        )
        self.grasp_offset = quat_apply(
            self.box_yaw_quat,
            self.grasp_quat.new_tensor([self.spec.grasp_offset_world]).repeat(self.env.num_envs, 1),
        ) + self.spec.grasp_away_shift_m * away_axis
        self.grasp_pos = self.sugar_box.data.root_pos_w + self.grasp_offset
        self.pregrasp_pos = self.grasp_pos.clone()
        self.box_top_z = self.sugar_box.data.root_pos_w[:, 2] + 0.5 * self.sugar_box_height_m
        self.grasp_depth_below_box_top_m = None
        self.initial_box_pos = self.sugar_box.data.root_pos_w.clone()

    def _set_phase(self, phase: str, position, orientation, gripper: float) -> None:
        palm_pos, palm_quat = self._palm_pose()
        self.phase_history.append(
            {
                "phase": self.phase,
                "steps": self.phase_step,
                "sugar_box_quat_wxyz": self.sugar_box.data.root_quat_w[0].detach().cpu().tolist(),
                "palm_position_m": (palm_pos[0] - self.env.scene.env_origins[0]).detach().cpu().tolist(),
                "palm_quat_wxyz": palm_quat[0].detach().cpu().tolist(),
                "left_hand_joint_position_rad": self.robot.data.joint_pos[0, self.hand_joint_ids].detach().cpu().tolist(),
                "sugar_box_position_m": (
                    self.sugar_box.data.root_pos_w[0] - self.env.scene.env_origins[0]
                ).detach().cpu().tolist(),
            }
        )
        self.phase_index = PHASES.index(phase)
        self.phase_step = 0
        self.phase_start_pos = palm_pos.clone()
        self.phase_start_quat = palm_quat.clone()
        self.target_pos = position.clone()
        self.target_quat = orientation.clone()
        if phase in ("move_pregrasp", "approach"):
            # Rotation has its own stage. Translation keeps the same command
            # orientation rather than interpolating another rotation near the box.
            self.phase_start_quat = orientation.clone()
        self.gripper_start = self.gripper_target
        self.gripper_target = gripper
        self.env.policy_phase.fill_(self.phase_index)
        print(f"[sugar-box-policy] phase={phase} step={self.total_step}", flush=True)

    def _fail(self, reason: str) -> None:
        if self.failed:
            return
        self.failed = True
        self.failure_reason = reason
        self.failed_target_pos = self.target_pos.clone()
        palm_pos, palm_quat = self._palm_pose()
        self._set_phase("failed", palm_pos, palm_quat, self.gripper_target)
        print(f"[sugar-box-policy] FAILED: {reason}", flush=True)

    def _pose_ready(self) -> bool:
        palm_pos, palm_quat = self._palm_pose()
        pos_error = torch.linalg.vector_norm(palm_pos - self.target_pos, dim=-1)[0]
        rot_error = quat_error_magnitude(palm_quat, self.target_quat)[0]
        self.last_position_error = pos_error.item()
        self.last_orientation_error = rot_error.item()
        orientation_ready = rot_error < math.radians(15.0)
        return bool(pos_error < 0.025 and orientation_ready)

    def _advance_if_ready(self, phase: str, position, orientation, gripper: float) -> None:
        ready = self._pose_ready()
        timed_out = self.phase_step > LIMITS[self.phase].maximum_steps
        if self.phase_step < LIMITS[self.phase].minimum_steps and not timed_out:
            return
        if not (ready or timed_out):
            return
        if timed_out and not ready:
            self._fail(
                f"{self.phase} pose timeout (position={self.last_position_error:.3f} m, "
                f"orientation={math.degrees(self.last_orientation_error):.1f} deg)"
            )
            return
        self._set_phase(phase, position, orientation, gripper)

    def _palm_for_box_pose(self, box_pos, box_quat):
        # Snapshot the physical grasp transform, without attaching the object.
        palm_pos, palm_quat = self._palm_pose()
        rotation = quat_mul(box_quat, quat_conjugate(self.sugar_box.data.root_quat_w))
        return (box_pos + quat_apply(rotation, palm_pos - self.sugar_box.data.root_pos_w),
                quat_mul(rotation, palm_quat))

    def _up_axis_local(self) -> torch.Tensor:
        up = self.target_pos.new_zeros(self.env.num_envs, 3)
        up[:, self.spec.up_axis] = self.spec.up_sign
        return up

    def _box_tilt(self):
        up = self._up_axis_local()
        axis = quat_apply(self.sugar_box.data.root_quat_w, up)
        return torch.acos(axis[:, 2].clamp(-1.0, 1.0))

    def _transition(self) -> None:
        if self.phase in ("done", "failed"):
            return
        if not torch.isfinite(self.sugar_box.data.root_state_w).all():
            self._fail("sugar-box state became non-finite")
            return
        if self.sugar_box.data.root_pos_w[0, 2] < 0.57:
            self._fail("sugar box fell below the tabletop")
            return

        timed_out = self.phase_step > LIMITS[self.phase].maximum_steps
        if self.phase == "reset" and timed_out:
            self._compute_grasp()
            palm_pos, palm_quat = self._palm_pose()
            turn_point = self._box_frame_point(self.spec.turn_point_world)
            self._set_phase("move_to_turn_point", turn_point, palm_quat, 0.0)
        elif self.phase == "move_to_turn_point":
            self._advance_if_ready("orient_hand", self.target_pos, self.grasp_quat, 0.0)
        elif self.phase == "orient_hand":
            self._pose_ready()
            # Require a settled orientation before any forward target is issued.
            ready = (self.last_position_error < 0.025
                     and self.last_orientation_error < math.radians(5.0))
            self.orientation_ready_steps = self.orientation_ready_steps + 1 if ready else 0
            if (self.phase_step >= LIMITS[self.phase].minimum_steps
                    and self.orientation_ready_steps >= 10):
                # Account for initial physical settling before the final approach.
                self.grasp_pos = self.sugar_box.data.root_pos_w + self.grasp_offset
                self.pregrasp_pos = self.grasp_pos.clone()
                self._set_phase("move_pregrasp", self.pregrasp_pos, self.grasp_quat, 0.0)
            elif timed_out:
                self._fail(
                    f"orient_hand pose timeout (position={self.last_position_error:.3f} m, "
                    f"orientation={math.degrees(self.last_orientation_error):.1f} deg)"
                )
        elif self.phase == "move_pregrasp":
            # The fitted open-hand envelope surrounds the box at this target.
            # Require closer alignment before initiating all three fingers.
            self._pose_ready()
            ready = (self.last_position_error < 0.012
                     and self.last_orientation_error < math.radians(5.0))
            self.grasp_ready_steps = self.grasp_ready_steps + 1 if ready else 0
            if (self.phase_step >= LIMITS[self.phase].minimum_steps
                    and self.grasp_ready_steps >= 15):
                palm_pos, palm_quat = self._palm_pose()
                self.grasp_pos = palm_pos.clone()
                self._set_phase("close_gripper", palm_pos, palm_quat, 1.0)
            elif timed_out:
                self._fail(
                    f"move_pregrasp closure gate timeout (position={self.last_position_error:.3f} m, "
                    f"orientation={math.degrees(self.last_orientation_error):.1f} deg)"
                )
        elif self.phase == "approach":
            self._advance_if_ready("close_gripper", self.grasp_pos, self.grasp_quat, 1.0)
        elif self.phase == "close_gripper" and timed_out:
            palm_pos, palm_quat = self._palm_pose()
            lift = palm_pos + palm_pos.new_tensor([[0.0, 0.0, 0.08]])
            self._set_phase("lift", lift, palm_quat, 1.0)
        elif self.phase == "lift":
            ready = self._pose_ready()
            if ready or timed_out:
                rise = (
                    self.sugar_box.data.root_pos_w[0, 2] - self.initial_box_pos[0, 2]
                ).item()
                self.measured_lift_rise_m = rise
                if rise < 0.045:
                    self._fail(f"grasp did not lift the sugar box (rise={rise:.3f} m)")
                elif timed_out and not ready:
                    self._fail("lift pose timeout")
                else:
                    palm_pos, palm_quat = self._palm_pose()
                    self.placement_box_pos = self.initial_box_pos + palm_pos.new_tensor([self.TARGET_WORLD_DELTA])
                    transport_box = self.placement_box_pos.clone()
                    transport_box[:, 2] = self.sugar_box.data.root_pos_w[:, 2]
                    move = palm_pos + transport_box - self.sugar_box.data.root_pos_w
                    self._set_phase("move_left", move, palm_quat, 1.0)
        elif self.phase == "move_left":
            # Rotate the measured box height axis onto world +Z. This levels
            # its bottom with minimum wrist rotation, retaining tabletop yaw.
            box_quat = self.sugar_box.data.root_quat_w
            up = self.target_pos.new_tensor([[0.0, 0.0, 1.0]]).repeat(self.env.num_envs, 1)
            axis = quat_apply(box_quat, self._up_axis_local())
            correction = torch.cat((1.0 + (axis * up).sum(-1, keepdim=True),
                                    torch.linalg.cross(axis, up)), dim=-1)
            correction = correction / torch.linalg.vector_norm(correction, dim=-1, keepdim=True).clamp_min(1e-6)
            self.placement_box_quat = quat_mul(correction, box_quat)
            position, orientation = self._palm_for_box_pose(
                self.sugar_box.data.root_pos_w, self.placement_box_quat)
            self._advance_if_ready("level_box", position, orientation, 1.0)
        elif self.phase == "level_box":
            self._pose_ready()
            if self.phase_step >= LIMITS[self.phase].minimum_steps and self._box_tilt()[0] < math.radians(5):
                position, orientation = self._palm_for_box_pose(self.placement_box_pos, self.placement_box_quat)
                self._set_phase("lower", position, orientation, 1.0)
            elif timed_out:
                self._fail("box leveling timeout")
        elif self.phase == "lower":
            box_pos = self.sugar_box.data.root_pos_w
            error = self.placement_box_pos - box_pos
            # Release near the support pose; a closed hand can sustain contact
            # jitter even after the box reaches the table. Stability is still
            # checked by the Task after release, not used to prevent opening.
            ready = (abs(error[0, 2]) < 0.008 and torch.linalg.vector_norm(error[0, :2]) < 0.015
                     and self._box_tilt()[0] < math.radians(8))
            self.placement_ready_steps = self.placement_ready_steps + 1 if ready else 0
            if self.phase_step >= LIMITS[self.phase].minimum_steps and self.placement_ready_steps >= 5:
                palm_pos, palm_quat = self._palm_pose()
                self._set_phase("open_gripper", palm_pos, palm_quat, 0.0)
            elif timed_out:
                self._fail("box did not settle level at placement pose")
            elif self.phase_step >= LIMITS[self.phase].minimum_steps and self.phase_step % 15 == 0:
                # Small object-based position corrections after the descent.
                self.target_pos += error.clamp(-0.004, 0.004)
        elif self.phase == "open_gripper" and timed_out:
            palm_pos, palm_quat = self._palm_pose()
            # Withdraw along the approach direction before lifting the hand.
            direction = quat_apply(palm_quat, palm_pos.new_tensor([[1.0, 0.0, 0.0]]).repeat(self.env.num_envs, 1))
            direction[:, 2] = 0.0
            direction /= torch.linalg.vector_norm(direction, dim=-1, keepdim=True)
            retreat = palm_pos - 0.06 * direction
            self._set_phase("retreat", retreat, palm_quat, 0.0)
        elif self.phase == "retreat":
            palm_pos, palm_quat = self._palm_pose()
            self._advance_if_ready("done", palm_pos, palm_quat, 0.0)

    def compute(self, step: int) -> EndEffectorTarget:
        del step
        duration = max(LIMITS[self.phase].minimum_steps, 1)
        alpha = min(1.0, (self.phase_step + 1) / duration)
        alpha = alpha * alpha * (3.0 - 2.0 * alpha)
        position = self.phase_start_pos + alpha * (self.target_pos - self.phase_start_pos)
        orientation = torch.stack(
            [
                quat_slerp(
                    self.phase_start_quat[index], self.target_quat[index].clone(), alpha
                )
                for index in range(self.env.num_envs)
            ]
        )
        gripper = self.gripper_start + alpha * (self.gripper_target - self.gripper_start)
        gripper = position.new_full((self.env.num_envs, 1), gripper)
        if self.phase not in ("done", "failed"):
            self.phase_step += 1
            self.total_step += 1
            self._transition()
        return EndEffectorTarget(position, orientation, gripper)

    def diagnostics(self) -> dict:
        origin = self.env.scene.env_origins[0]
        palm_pos, palm_quat = self._palm_pose()
        gripper_axis = quat_apply(
            self.grasp_quat if self.grasp_quat is not None else palm_quat,
            palm_pos.new_tensor([[1.0, 0.0, 0.0]]),
        )
        return {
            "phase": self.phase,
            "failed": self.failed,
            "measured_lift_rise_m": self.measured_lift_rise_m,
            "failure_reason": self.failure_reason,
            "total_steps": self.total_step,
            "sugar_box_position_m": (
                self.sugar_box.data.root_pos_w[0] - origin
            ).detach().cpu().tolist(),
            "left_ee_position_m": (palm_pos[0] - origin).detach().cpu().tolist(),
            "target_left_ee_position_m": (self.target_pos[0] - origin).detach().cpu().tolist(),
            "failed_target_left_ee_position_m": None
            if self.failed_target_pos is None
            else (self.failed_target_pos[0] - origin).detach().cpu().tolist(),
            "pregrasp_left_ee_position_m": None
            if self.pregrasp_pos is None
            else (self.pregrasp_pos[0] - origin).detach().cpu().tolist(),
            "grasp_left_ee_position_m": None
            if self.grasp_pos is None
            else (self.grasp_pos[0] - origin).detach().cpu().tolist(),
            "box_top_height_m": None
            if self.box_top_z is None
            else (self.box_top_z[0] - origin[2]).item(),
            "grasp_depth_below_box_top_m": None
            if self.grasp_depth_below_box_top_m is None
            else self.grasp_depth_below_box_top_m[0].item(),
            "pregrasp_clearance_m": self.PREGRASP_CLEARANCE_M,
            "turn_point_world_m": list(self.spec.turn_point_world),
            "grasp_quat_wxyz": list(self.spec.grasp_quat_wxyz),
            "grasp_offset_world_m": list(self.spec.grasp_offset_world),
            "grasp_away_shift_m": self.spec.grasp_away_shift_m,
            "box_yaw_delta_rad": None if self.box_yaw_delta is None else self.box_yaw_delta[0].item(),
            "applied_grasp_quat_wxyz": None
            if self.grasp_quat is None
            else self.grasp_quat[0].detach().cpu().tolist(),
            "left_hand_joint_names": list(self.definition.left_hand_joint_names),
            "approach_direction_world": None
            if self.approach_direction is None
            else self.approach_direction[0].detach().cpu().tolist(),
            "gripper_forward_axis_world": gripper_axis[0].detach().cpu().tolist(),
            "position_error_m": torch.linalg.vector_norm(
                palm_pos - self.target_pos, dim=-1
            )[0].item(),
            "orientation_error_rad": quat_error_magnitude(
                palm_quat, self.target_quat
            )[0].item(),
            "last_command_position_error_m": self.last_position_error,
            "last_command_orientation_error_rad": self.last_orientation_error,
            "target_displacement_world_m": list(self.TARGET_WORLD_DELTA),
            "robot_right_definition": "base -Y = world +X",
            "phase_history": self.phase_history,
        }
