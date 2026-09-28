"""Closed-loop grasp-and-lift RL for one YCB object, starting from pregrasp states.

Resets draw settled pregrasp states (fingers at the object's safe preshape) from
the table built by ``scripts/rl/build_pregrasp_table.sh --object <name>``. Each
control step the policy moves the palm by a small relative pose (differential
IK on the seven left-arm joints) and the seven left-hand joints by small
deltas, so a zero action holds the current pose. Waist, legs, right arm and
right hand hold the targets written at reset.

Rewards follow DextrAH-G: progress toward the object and in lift height. The
episode ends the moment the held object passes 5 cm and is scored by how
closely it matches its start pose raised 5 cm.
"""

import isaaclab.envs.mdp as base_mdp
from isaaclab.controllers import DifferentialIKControllerCfg
from isaaclab.envs.mdp.actions import DifferentialInverseKinematicsActionCfg, RelativeJointPositionActionCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass

from vla_isaaclab.rl.pregrasp_table import pregrasp_table_path, reset_from_pregrasp_table

from ..common import (
    LEFT_ARM_JOINT_NAMES,
    LEFT_END_EFFECTOR,
    LEFT_HAND_JOINT_NAMES,
    MUSTARD_BOTTLE,
    SUGAR_BOX,
    SUPPORT_HEIGHT,
    GraspObjectSpec,
)
from .env_cfg import YCBGraspStateEnvCfg
from .mdp import grasp_rl, invalid_state, object_fallen


CONTROLLED_JOINTS = SceneEntityCfg("robot", joint_names=[*LEFT_ARM_JOINT_NAMES, *LEFT_HAND_JOINT_NAMES], preserve_order=True)


@configclass
class GraspLiftActionsCfg:
    # One speed limit shared with the scripted place (scripts/rl/play_handoff.py):
    # palm <= 3 cm/s and 0.15 rad/s, fingers <= 0.3 rad/s at 30 Hz with actions
    # clipped to [-1, 1]. Far too slow to flick the object 5 cm up (~1 m/s needed).
    arm = DifferentialInverseKinematicsActionCfg(
        asset_name="robot",
        joint_names=list(LEFT_ARM_JOINT_NAMES),
        body_name=LEFT_END_EFFECTOR,
        controller=DifferentialIKControllerCfg(command_type="pose", use_relative_mode=True, ik_method="dls"),
        scale=(0.001, 0.001, 0.001, 0.005, 0.005, 0.005),
    )
    # Target = current joint position + delta; at most 0.01 rad per step (0.3 rad/s).
    hand = RelativeJointPositionActionCfg(
        asset_name="robot",
        joint_names=list(LEFT_HAND_JOINT_NAMES),
        scale=0.01,
        clip={".*": (-0.01, 0.01)},
    )


@configclass
class GraspLiftPolicyCfg(ObsGroup):
    joint_pos = ObsTerm(func=base_mdp.joint_pos_rel, params={"asset_cfg": CONTROLLED_JOINTS})
    joint_vel = ObsTerm(func=base_mdp.joint_vel_rel, params={"asset_cfg": CONTROLLED_JOINTS})
    palm_pose = ObsTerm(func=grasp_rl.palm_pose_b)
    object_in_palm = ObsTerm(func=grasp_rl.object_pose_in_palm)
    object_velocity = ObsTerm(func=grasp_rl.object_velocity_b)
    fingertips_in_object = ObsTerm(func=grasp_rl.fingertips_in_object_obs)
    lift_and_tilt = ObsTerm(func=grasp_rl.lift_obs)
    hand_contact_force = ObsTerm(func=grasp_rl.hand_contact_force_clipped)
    goal = ObsTerm(func=grasp_rl.goal_pose_in_object)
    last_action = ObsTerm(func=base_mdp.last_action)

    def __post_init__(self):
        self.enable_corruption = False
        self.concatenate_terms = True
        self.history_length = 3
        self.flatten_history_dim = True


@configclass
class GraspLiftCriticCfg(ObsGroup):
    """Current step only, plus unclipped-range contact forces."""

    joint_pos = ObsTerm(func=base_mdp.joint_pos_rel, params={"asset_cfg": CONTROLLED_JOINTS})
    joint_vel = ObsTerm(func=base_mdp.joint_vel_rel, params={"asset_cfg": CONTROLLED_JOINTS})
    palm_pose = ObsTerm(func=grasp_rl.palm_pose_b)
    object_in_palm = ObsTerm(func=grasp_rl.object_pose_in_palm)
    object_velocity = ObsTerm(func=grasp_rl.object_velocity_b)
    fingertips_in_object = ObsTerm(func=grasp_rl.fingertips_in_object_obs)
    lift_and_tilt = ObsTerm(func=grasp_rl.lift_obs)
    hand_contact_force = ObsTerm(func=grasp_rl.hand_contact_force)
    goal = ObsTerm(func=grasp_rl.goal_pose_in_object)
    last_action = ObsTerm(func=base_mdp.last_action)

    def __post_init__(self):
        self.enable_corruption = False
        self.concatenate_terms = True


@configclass
class GraspLiftObservationsCfg:
    policy: GraspLiftPolicyCfg = GraspLiftPolicyCfg()
    critic: GraspLiftCriticCfg = GraspLiftCriticCfg()


@configclass
class GraspLiftEventsCfg:
    reset_scene_to_default = EventTerm(func=base_mdp.reset_scene_to_default, mode="reset")
    # The table path is set from the object_spec.
    reset_to_pregrasp = EventTerm(func=reset_from_pregrasp_table, mode="reset", params={"table_path": ""})
    reset_progress = EventTerm(func=grasp_rl.reset_grasp_progress, mode="reset")
    # After reset_to_pregrasp, which sets the object start position.
    sample_goal = EventTerm(
        func=grasp_rl.sample_lift_goal, mode="reset", params={"xy_range_m": 0.0, "z_range_m": (0.05, 0.05)}
    )


@configclass
class GraspLiftRewardsCfg:
    # The episode ends the moment the held object passes 5 cm; that step is scored
    # 100 * exp(-e / 3 cm), e = largest collider-corner distance to the goal pose
    # (the start pose raised 5 cm). Before that, DextrAH-G's progress terms.
    to_object = RewTerm(func=grasp_rl.dextrah_to_object, weight=5.0)
    lift = RewTerm(func=grasp_rl.held_lift, weight=50.0, params={"z_lifted_m": 0.05})
    lift_pose = RewTerm(func=grasp_rl.lift_pose_score, weight=100.0)


@configclass
class GraspLiftTerminationsCfg:
    time_out = DoneTerm(func=base_mdp.time_out, time_out=True)
    # Held object passed 5 cm: "success" if every corner is within 2 cm of the
    # goal pose, otherwise "lifted_off_pose". Both end the episode.
    success = DoneTerm(func=grasp_rl.lift_success)
    lifted_off_pose = DoneTerm(func=grasp_rl.lift_off_pose)
    # Set from the object_spec's table height.
    fell = DoneTerm(func=object_fallen, params={"support_height": SUPPORT_HEIGHT})
    # Simulator guard only (NaN state), not part of the task design.
    invalid_state = DoneTerm(func=invalid_state)


@configclass
class YCBGraspLiftEnvCfg(YCBGraspStateEnvCfg):
    actions: GraspLiftActionsCfg = GraspLiftActionsCfg()
    observations: GraspLiftObservationsCfg = GraspLiftObservationsCfg()
    events: GraspLiftEventsCfg = GraspLiftEventsCfg()
    rewards: GraspLiftRewardsCfg = GraspLiftRewardsCfg()
    terminations: GraspLiftTerminationsCfg = GraspLiftTerminationsCfg()
    episode_length_s: float = 8.0
    # Policy actions are clamped to [-1, 1] by ClippedRslRlVecEnvWrapper.
    action_clip: float = 1.0
    fast_actions: bool = False
    """5x the palm and finger speed limits and 5 s episodes. From scratch the slow
    limits never lift the object; train fast first, then fine-tune at the slow
    limits (--resume)."""

    def __post_init__(self):
        super().__post_init__()
        if self.fast_actions:
            # Palm <= 15 cm/s and 0.45 rad/s, fingers <= 1.5 rad/s.
            self.actions.arm.scale = (0.005, 0.005, 0.005, 0.015, 0.015, 0.015)
            self.actions.hand.scale = 0.05
            self.actions.hand.clip = {".*": (-0.05, 0.05)}
            self.episode_length_s = 5.0
        self.events.reset_to_pregrasp.params["table_path"] = str(pregrasp_table_path(self.object_spec))
        # Parallel RL: no texture waits, and a tighter env grid.
        self.wait_for_textures = False
        self.rerender_on_reset = False
        self.scene.env_spacing = 2.0


# -- one config per object ------------------------------------------------------


@configclass
class SugarBoxGraspLiftEnvCfg(YCBGraspLiftEnvCfg):
    object_spec: GraspObjectSpec = SUGAR_BOX


@configclass
class MustardBottleGraspLiftEnvCfg(YCBGraspLiftEnvCfg):
    object_spec: GraspObjectSpec = MUSTARD_BOTTLE


@configclass
class SugarBoxGraspLiftFastEnvCfg(SugarBoxGraspLiftEnvCfg):
    fast_actions: bool = True


@configclass
class MustardBottleGraspLiftFastEnvCfg(MustardBottleGraspLiftEnvCfg):
    fast_actions: bool = True
