"""Closed-loop sugar-box grasp-and-lift RL, starting from scripted pregrasp states.

Resets draw settled pregrasp states from the table built by
``scripts/rl/build_pregrasp_table.sh``. Each control step the policy adjusts the
palm pose (relative differential IK on the seven left-arm joints) and sets the
seven left-hand joint targets. Waist, legs, right arm and right hand hold the
targets written at reset.
"""

import isaaclab.envs.mdp as base_mdp
from isaaclab.controllers import DifferentialIKControllerCfg
from isaaclab.envs.mdp.actions import (
    DifferentialInverseKinematicsActionCfg,
    JointPositionActionCfg,
    RelativeJointPositionActionCfg,
)
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass

from vla_isaaclab.rl.pregrasp_table import reset_from_pregrasp_table

from ..common import LEFT_ARM_JOINT_NAMES, LEFT_END_EFFECTOR, LEFT_HAND_JOINT_NAMES, SUPPORT_HEIGHT
from .env_cfg import PROJECT_ROOT, YCBSugarBoxStateEnvCfg
from .mdp import grasp_rl, invalid_state, object_fallen


PREGRASP_TABLE = PROJECT_ROOT / "outputs/rl/pregrasp_table.pt"
PREGRASP_PRESHAPE_TABLE = PROJECT_ROOT / "outputs/rl/pregrasp_table_preshape.pt"
CONTROLLED_JOINTS = SceneEntityCfg("robot", joint_names=[*LEFT_ARM_JOINT_NAMES, *LEFT_HAND_JOINT_NAMES], preserve_order=True)


@configclass
class GraspActionsCfg:
    # Per step: palm translation (m) and axis-angle rotation (rad) in the robot base frame.
    arm = DifferentialInverseKinematicsActionCfg(
        asset_name="robot",
        joint_names=list(LEFT_ARM_JOINT_NAMES),
        body_name=LEFT_END_EFFECTOR,
        controller=DifferentialIKControllerCfg(command_type="pose", use_relative_mode=True, ik_method="dls"),
        scale=(0.01, 0.01, 0.01, 0.03, 0.03, 0.03),
    )
    # Absolute hand targets around the open hand (action 0 = open).
    hand = JointPositionActionCfg(
        asset_name="robot",
        joint_names=list(LEFT_HAND_JOINT_NAMES),
        scale=1.0,
        use_default_offset=True,
    )


@configclass
class GraspPolicyCfg(ObsGroup):
    joint_pos = ObsTerm(func=base_mdp.joint_pos_rel, params={"asset_cfg": CONTROLLED_JOINTS})
    joint_vel = ObsTerm(func=base_mdp.joint_vel_rel, params={"asset_cfg": CONTROLLED_JOINTS})
    palm_pose = ObsTerm(func=grasp_rl.palm_pose_b)
    box_in_palm = ObsTerm(func=grasp_rl.box_pose_in_palm)
    box_velocity = ObsTerm(func=grasp_rl.box_velocity_b)
    fingertips_in_box = ObsTerm(func=grasp_rl.fingertips_in_box_obs)
    lift_and_tilt = ObsTerm(func=grasp_rl.lift_obs)
    hand_contact = ObsTerm(func=grasp_rl.hand_contact_binary)
    last_action = ObsTerm(func=base_mdp.last_action)

    def __post_init__(self):
        self.enable_corruption = False
        self.concatenate_terms = True


@configclass
class GraspCriticCfg(GraspPolicyCfg):
    """Privileged critic: also sees continuous per-link contact forces."""

    hand_contact_force = ObsTerm(func=grasp_rl.hand_contact_force)


@configclass
class GraspObservationsCfg:
    policy: GraspPolicyCfg = GraspPolicyCfg()
    critic: GraspCriticCfg = GraspCriticCfg()


@configclass
class GraspEventsCfg:
    reset_scene_to_default = EventTerm(func=base_mdp.reset_scene_to_default, mode="reset")
    reset_to_pregrasp = EventTerm(
        func=reset_from_pregrasp_table, mode="reset", params={"table_path": str(PREGRASP_TABLE)}
    )


@configclass
class GraspRewardsCfg:
    # r = r_contact + r_lift + r_stable - r_drop - r_tilt, plus a tiny action-rate term.
    fingertip_reach = RewTerm(func=grasp_rl.fingertip_reach, weight=0.5)
    grasp_contact = RewTerm(func=grasp_rl.grasp_contact, weight=1.0)
    lift = RewTerm(func=grasp_rl.lift, weight=5.0)
    stable_hold = RewTerm(func=grasp_rl.stable_hold, weight=2.0)
    hold_success = RewTerm(func=grasp_rl.hold_success, weight=2.0)
    drop = RewTerm(func=base_mdp.is_terminated_term, weight=-20.0, params={"term_keys": "dropped"})
    knocked = RewTerm(
        func=base_mdp.is_terminated_term, weight=-10.0, params={"term_keys": ["tipped", "pushed", "fell"]}
    )
    tilt = RewTerm(func=grasp_rl.tilt_penalty, weight=-1.0)
    action_rate = RewTerm(func=base_mdp.action_rate_l2, weight=-1.0e-3)


@configclass
class GraspTerminationsCfg:
    time_out = DoneTerm(func=base_mdp.time_out, time_out=True)
    dropped = DoneTerm(func=grasp_rl.box_dropped)
    tipped = DoneTerm(func=grasp_rl.box_tipped)
    pushed = DoneTerm(func=grasp_rl.box_pushed)
    fell = DoneTerm(func=object_fallen, params={"support_height": SUPPORT_HEIGHT})
    invalid_state = DoneTerm(func=invalid_state)


@configclass
class YCBSugarBoxGraspRLEnvCfg(YCBSugarBoxStateEnvCfg):
    actions: GraspActionsCfg = GraspActionsCfg()
    observations: GraspObservationsCfg = GraspObservationsCfg()
    events: GraspEventsCfg = GraspEventsCfg()
    rewards: GraspRewardsCfg = GraspRewardsCfg()
    terminations: GraspTerminationsCfg = GraspTerminationsCfg()
    episode_length_s: float = 5.0
    task_instruction: str = "Grasp the YCB 004 sugar box and lift it off the table."

    def __post_init__(self):
        super().__post_init__()
        # Parallel RL: no texture waits, and a tighter env grid.
        self.wait_for_textures = False
        self.rerender_on_reset = False
        self.scene.env_spacing = 2.0


# -- v2 ---------------------------------------------------------------------------
# Rewards and terminations follow DextrAH-G / DexSuite / DexPBT: progress-based
# reach and lift, contact reward, no termination for tipping or pushing. The
# fingers start preshaped (see DEFAULT_HAND_PRESHAPE) and move by relative
# deltas, so a zero action keeps the current hand shape.


@configclass
class GraspV2ActionsCfg:
    # Slow enough that the box cannot be flicked 5 cm up (that needs ~1 m/s):
    # palm <= 15 cm/s and ~0.45 rad/s at 30 Hz with actions clipped to [-1, 1].
    arm = DifferentialInverseKinematicsActionCfg(
        asset_name="robot",
        joint_names=list(LEFT_ARM_JOINT_NAMES),
        body_name=LEFT_END_EFFECTOR,
        controller=DifferentialIKControllerCfg(command_type="pose", use_relative_mode=True, ik_method="dls"),
        scale=(0.005, 0.005, 0.005, 0.015, 0.015, 0.015),
    )
    # Target = current joint position + delta; at most 0.05 rad per step (1.5 rad/s).
    hand = RelativeJointPositionActionCfg(
        asset_name="robot",
        joint_names=list(LEFT_HAND_JOINT_NAMES),
        scale=0.05,
        clip={".*": (-0.05, 0.05)},
    )


@configclass
class GraspV2PolicyCfg(ObsGroup):
    joint_pos = ObsTerm(func=base_mdp.joint_pos_rel, params={"asset_cfg": CONTROLLED_JOINTS})
    joint_vel = ObsTerm(func=base_mdp.joint_vel_rel, params={"asset_cfg": CONTROLLED_JOINTS})
    palm_pose = ObsTerm(func=grasp_rl.palm_pose_b)
    box_in_palm = ObsTerm(func=grasp_rl.box_pose_in_palm)
    box_velocity = ObsTerm(func=grasp_rl.box_velocity_b)
    fingertips_in_box = ObsTerm(func=grasp_rl.fingertips_in_box_obs)
    lift_and_tilt = ObsTerm(func=grasp_rl.lift_obs)
    hand_contact_force = ObsTerm(func=grasp_rl.hand_contact_force_clipped)
    goal = ObsTerm(func=grasp_rl.goal_pose_in_box)
    last_action = ObsTerm(func=base_mdp.last_action)

    def __post_init__(self):
        self.enable_corruption = False
        self.concatenate_terms = True
        self.history_length = 3
        self.flatten_history_dim = True


@configclass
class GraspV2CriticCfg(ObsGroup):
    """Current step only, plus unclipped-range contact forces."""

    joint_pos = ObsTerm(func=base_mdp.joint_pos_rel, params={"asset_cfg": CONTROLLED_JOINTS})
    joint_vel = ObsTerm(func=base_mdp.joint_vel_rel, params={"asset_cfg": CONTROLLED_JOINTS})
    palm_pose = ObsTerm(func=grasp_rl.palm_pose_b)
    box_in_palm = ObsTerm(func=grasp_rl.box_pose_in_palm)
    box_velocity = ObsTerm(func=grasp_rl.box_velocity_b)
    fingertips_in_box = ObsTerm(func=grasp_rl.fingertips_in_box_obs)
    lift_and_tilt = ObsTerm(func=grasp_rl.lift_obs)
    hand_contact_force = ObsTerm(func=grasp_rl.hand_contact_force)
    goal = ObsTerm(func=grasp_rl.goal_pose_in_box)
    last_action = ObsTerm(func=base_mdp.last_action)

    def __post_init__(self):
        self.enable_corruption = False
        self.concatenate_terms = True


@configclass
class GraspV2ObservationsCfg:
    policy: GraspV2PolicyCfg = GraspV2PolicyCfg()
    critic: GraspV2CriticCfg = GraspV2CriticCfg()


@configclass
class GraspV2EventsCfg:
    reset_scene_to_default = EventTerm(func=base_mdp.reset_scene_to_default, mode="reset")
    reset_to_pregrasp = EventTerm(
        func=reset_from_pregrasp_table, mode="reset", params={"table_path": str(PREGRASP_PRESHAPE_TABLE)}
    )
    reset_progress = EventTerm(func=grasp_rl.reset_grasp_progress, mode="reset")
    # After reset_to_pregrasp, which sets the box start position.
    sample_goal = EventTerm(
        func=grasp_rl.sample_lift_goal, mode="reset", params={"xy_range_m": 0.0, "z_range_m": (0.05, 0.05)}
    )


@configclass
class GraspV2RewardsCfg:
    # The episode ends the moment the held box passes 5 cm; that step is scored
    # 100 * exp(-e / 3 cm), e = largest box-corner distance to the goal pose (the
    # start pose raised 5 cm). Before that, DextrAH-G's progress terms.
    to_object = RewTerm(func=grasp_rl.dextrah_to_object, weight=5.0)
    lift = RewTerm(func=grasp_rl.held_lift, weight=50.0, params={"z_lifted_m": 0.05})
    lift_pose = RewTerm(func=grasp_rl.lift_pose_score, weight=100.0)


@configclass
class GraspV2TerminationsCfg:
    time_out = DoneTerm(func=base_mdp.time_out, time_out=True)
    # Held box passed 5 cm: "success" if every corner is within 2 cm of the goal
    # pose (about 10 degrees of tilt), otherwise "lifted_off_pose". Both end the episode.
    success = DoneTerm(func=grasp_rl.lift_success)
    lifted_off_pose = DoneTerm(func=grasp_rl.lift_off_pose)
    fell = DoneTerm(func=object_fallen, params={"support_height": SUPPORT_HEIGHT})
    # Simulator guard only (NaN state), not part of the task design.
    invalid_state = DoneTerm(func=invalid_state)


@configclass
class YCBSugarBoxGraspRLV2EnvCfg(YCBSugarBoxGraspRLEnvCfg):
    actions: GraspV2ActionsCfg = GraspV2ActionsCfg()
    observations: GraspV2ObservationsCfg = GraspV2ObservationsCfg()
    events: GraspV2EventsCfg = GraspV2EventsCfg()
    rewards: GraspV2RewardsCfg = GraspV2RewardsCfg()
    terminations: GraspV2TerminationsCfg = GraspV2TerminationsCfg()
    episode_length_s: float = 5.0
    # Policy actions are clamped to [-1, 1] by ClippedRslRlVecEnvWrapper.
    action_clip: float = 1.0
