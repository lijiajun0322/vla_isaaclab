"""Bimanual bowl pivoting: G1 turns an upside-down bowl upright with both arms and both Dex3 hands.

Every episode starts from one fixed, settled state built by
``scripts/rl/build_bowl_pivot_init.sh``: the bowl rests upside down in front of
the robot and both hands hover about 1 cm from it, palms facing it, fingers
and thumb half curled toward it, out of contact. That state is stored in the pregrasp-table format and loaded by
the grasp-lift task's ``reset_from_pregrasp_table``. No randomization yet.

Each control step the policy moves each palm by a small relative pose
(differential IK on that arm's seven joints) and each hand's seven joints by
small deltas, so a zero action holds the current pose (26-D action). Waist and
legs hold their reset targets. ``VLA-BowlPivot-G1-Bimanual-v0`` runs at 2/3
of the grasp-lift fast speed limits, ``-Fast-v0`` at the full fast limits and
``-HalfSpeed-v0`` at 1/2.

Rewards: progress in the bowl's tilt toward upright (main term, paid on each
new best), a dense upright term, both hands near and on the bowl, a success
bonus and a penalty for running away. Success: upright within
15 deg, resting on the table and still, for 10 steps in a row.
"""

from pathlib import Path

import isaaclab.envs.mdp as base_mdp
from isaaclab.controllers import DifferentialIKControllerCfg
from isaaclab.envs.mdp.actions import DifferentialInverseKinematicsActionCfg, RelativeJointPositionActionCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.sensors import ContactSensorCfg
from isaaclab.utils import configclass

from vla_isaaclab.rl.pregrasp_table import PREGRASP_TABLE_DIR, reset_from_pregrasp_table

from ..common import (
    BOWL,
    LEFT_ARM_JOINT_NAMES,
    LEFT_END_EFFECTOR,
    LEFT_HAND_JOINT_NAMES,
    RIGHT_ARM_JOINT_NAMES,
    RIGHT_END_EFFECTOR,
    RIGHT_HAND_JOINT_NAMES,
    SUPPORT_HEIGHT,
    GraspObjectSpec,
)
from ..ycb_grasp.env_cfg import YCBGraspStateEnvCfg, YCBGraspStateSceneCfg, _hand_object_contacts
from ..ycb_grasp.mdp import grasp_rl
from .mdp import invalid_state, object_fallen, pivot


# The bowl turned 180 deg about its x axis: the foot (the root) on top, the lip
# on the table; 55 mm tall. Centered between the hands, 34 cm in front of the
# robot root (its lip 2 cm from the table edge). With the palms upright and the
# fingers forward the arms cannot reach a lower table: on the grasp-lift bowl's
# 0.68 m table the IK missed the pre-contact palm poses by ~10 cm.
BOWL_UPSIDE_DOWN = BOWL.replace(
    rest_quat_wxyz=(0.0, 1.0, 0.0, 0.0),
    rest_half_height_m=0.055,
    initial_xy=(0.0, -0.30),
    support_height_m=SUPPORT_HEIGHT + 0.15,
    dr_x_range_m=(0.0, 0.0),
    dr_y_range_m=(0.0, 0.0),
    dr_yaw_range_rad=(0.0, 0.0),
)

CONTROLLED_JOINTS = SceneEntityCfg(
    "robot",
    joint_names=[*LEFT_ARM_JOINT_NAMES, *RIGHT_ARM_JOINT_NAMES, *LEFT_HAND_JOINT_NAMES, *RIGHT_HAND_JOINT_NAMES],
    preserve_order=True,
)


def bowl_pivot_init_path(spec: GraspObjectSpec) -> Path:
    """outputs/rl/<object>/bowl_pivot_init.pt"""
    return PREGRASP_TABLE_DIR / spec.name / "bowl_pivot_init.pt"


# Speed limits at 30 Hz with actions clipped to [-1, 1]. Fast (the grasp-lift
# fast limits): palm <= 15 cm/s and 0.45 rad/s, fingers <= 1.5 rad/s; there the
# policy learned to toss the bowl over (airborne and untouched for ~0.3 s in
# every success). The env cfg's speed_fraction scales both (default 2/3: palm
# <= 10 cm/s and 0.3 rad/s, fingers <= 1.0 rad/s). The grasp-lift slow limits,
# 1/5, were too slow to continue from the fast policy.
FAST_ARM_SCALE = (0.005, 0.005, 0.005, 0.015, 0.015, 0.015)
FAST_HAND_STEP_RAD = 0.05


def _arm_action(joint_names, body_name) -> DifferentialInverseKinematicsActionCfg:
    # The scale is set from speed_fraction.
    return DifferentialInverseKinematicsActionCfg(
        asset_name="robot",
        joint_names=list(joint_names),
        body_name=body_name,
        controller=DifferentialIKControllerCfg(command_type="pose", use_relative_mode=True, ik_method="dls"),
        scale=FAST_ARM_SCALE,
    )


def _hand_action(joint_names) -> RelativeJointPositionActionCfg:
    # Target = current joint position + delta; scale and clip are set from speed_fraction.
    return RelativeJointPositionActionCfg(
        asset_name="robot", joint_names=list(joint_names), scale=FAST_HAND_STEP_RAD,
        clip={".*": (-FAST_HAND_STEP_RAD, FAST_HAND_STEP_RAD)},
    )


@configclass
class BowlPivotSceneCfg(YCBGraspStateSceneCfg):
    """The grasp-lift scene (left-hand/bowl sensors included) plus the same sensors on the right."""

    right_arm_contacts = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/right_(shoulder|elbow|wrist|hand)_.*",
        history_length=4,
    )
    object_contact_right_palm = _hand_object_contacts("right_hand_palm_link")
    object_contact_right_thumb_0 = _hand_object_contacts("right_hand_thumb_0_link")
    object_contact_right_thumb_1 = _hand_object_contacts("right_hand_thumb_1_link")
    object_contact_right_thumb_2 = _hand_object_contacts("right_hand_thumb_2_link")
    object_contact_right_index_0 = _hand_object_contacts("right_hand_index_0_link")
    object_contact_right_index_1 = _hand_object_contacts("right_hand_index_1_link")
    object_contact_right_middle_0 = _hand_object_contacts("right_hand_middle_0_link")
    object_contact_right_middle_1 = _hand_object_contacts("right_hand_middle_1_link")


@configclass
class BowlPivotActionsCfg:
    left_arm = _arm_action(LEFT_ARM_JOINT_NAMES, LEFT_END_EFFECTOR)
    right_arm = _arm_action(RIGHT_ARM_JOINT_NAMES, RIGHT_END_EFFECTOR)
    left_hand = _hand_action(LEFT_HAND_JOINT_NAMES)
    right_hand = _hand_action(RIGHT_HAND_JOINT_NAMES)


@configclass
class BowlPivotPolicyCfg(ObsGroup):
    joint_pos = ObsTerm(func=base_mdp.joint_pos_rel, params={"asset_cfg": CONTROLLED_JOINTS})
    joint_vel = ObsTerm(func=base_mdp.joint_vel_rel, params={"asset_cfg": CONTROLLED_JOINTS})
    palm_poses = ObsTerm(func=pivot.palm_poses_b)
    bowl_pose = ObsTerm(func=pivot.bowl_pose_b)
    bowl_up = ObsTerm(func=pivot.bowl_up_b)
    bowl_velocity = ObsTerm(func=grasp_rl.object_velocity_b)
    bowl_in_palms = ObsTerm(func=pivot.bowl_in_palms)
    bowl_displacement = ObsTerm(func=pivot.bowl_displacement)
    hand_contact_force = ObsTerm(func=pivot.hands_contact_force)
    last_action = ObsTerm(func=base_mdp.last_action)

    def __post_init__(self):
        self.enable_corruption = False
        self.concatenate_terms = True
        self.history_length = 3
        self.flatten_history_dim = True


@configclass
class BowlPivotCriticCfg(BowlPivotPolicyCfg):
    """Same terms, current step only."""

    def __post_init__(self):
        self.enable_corruption = False
        self.concatenate_terms = True


@configclass
class BowlPivotObservationsCfg:
    policy: BowlPivotPolicyCfg = BowlPivotPolicyCfg()
    critic: BowlPivotCriticCfg = BowlPivotCriticCfg()


@configclass
class BowlPivotEventsCfg:
    reset_scene_to_default = EventTerm(func=base_mdp.reset_scene_to_default, mode="reset")
    # The path is set from the object_spec; build it with build_bowl_pivot_init.sh.
    reset_to_init = EventTerm(func=reset_from_pregrasp_table, mode="reset", params={"table_path": ""})
    reset_streak = EventTerm(func=pivot.reset_upright_streak, mode="reset")
    # Forget the per-env best tilt of tilt_progress.
    reset_progress = EventTerm(func=grasp_rl.reset_grasp_progress, mode="reset")


@configclass
class BowlPivotRewardsCfg:
    # Isaac Lab scales every weight by the 1/30 s step.
    # tilt_progress (main): each new best tilt pays once; turning the bowl 10 deg
    # further pays 860/30 * 10/180 = 1.6, as much as both hands touching the bowl
    # for 8 s (hands_contact), the full 180 deg about 29.
    tilt_progress = RewTerm(func=pivot.tilt_progress, weight=860.0)
    # Dense hold term: up to 2/30 per step while upright, at most 16 per 8 s episode.
    upright = RewTerm(func=pivot.upright, weight=2.0)
    hands_near = RewTerm(func=pivot.hands_near_bowl, weight=0.2)
    hands_contact = RewTerm(func=pivot.hands_contact, weight=0.2)
    # 2000/30 = 67, more than an episode could still collect after it ends.
    success_bonus = RewTerm(func=base_mdp.is_terminated_term, weight=2000.0, params={"term_keys": "success"})
    failure_penalty = RewTerm(
        func=base_mdp.is_terminated_term, weight=-300.0, params={"term_keys": ["ran_away", "fell", "spinning"]}
    )


@configclass
class BowlPivotTerminationsCfg:
    time_out = DoneTerm(func=base_mdp.time_out, time_out=True)
    success = DoneTerm(func=pivot.pivot_success)
    ran_away = DoneTerm(func=pivot.bowl_ran_away)
    # Set from the object_spec's table height.
    fell = DoneTerm(func=object_fallen, params={"support_height": BOWL_UPSIDE_DOWN.support_height_m})
    spinning = DoneTerm(func=pivot.bowl_spinning)
    # Simulator guard only (NaN state), not part of the task design.
    invalid_state = DoneTerm(func=invalid_state)


@configclass
class BowlPivotEnvCfg(YCBGraspStateEnvCfg):
    object_spec: GraspObjectSpec = BOWL_UPSIDE_DOWN
    scene: BowlPivotSceneCfg = BowlPivotSceneCfg(num_envs=1024, env_spacing=2.0, replicate_physics=True)
    actions: BowlPivotActionsCfg = BowlPivotActionsCfg()
    observations: BowlPivotObservationsCfg = BowlPivotObservationsCfg()
    events: BowlPivotEventsCfg = BowlPivotEventsCfg()
    rewards: BowlPivotRewardsCfg = BowlPivotRewardsCfg()
    terminations: BowlPivotTerminationsCfg = BowlPivotTerminationsCfg()
    episode_length_s: float = 8.0
    # Policy actions are clamped to [-1, 1] by ClippedRslRlVecEnvWrapper.
    action_clip: float = 1.0
    speed_fraction: float = 2.0 / 3.0
    """Palm and finger speed limits as a fraction of the fast limits (FAST_*)."""

    # Pre-contact hand pose, read by build_bowl_pivot_init.py, mirrored for the
    # two hands. Palm +x (fingers) along world +Y (forward), the palm facing the
    # bowl and tilted down toward it by precontact_palm_tilt_deg; index, middle
    # and thumb half curled so that all three point at the bowl.
    precontact_palm_tilt_deg: float = 15.0
    # Palm position relative to the settled bowl root: y in world axes, z above
    # the table top. The lateral distance is not set here: the builder moves the
    # hands in until they would touch and backs off precontact_clearance_m.
    precontact_palm_y_m: float = -0.09
    precontact_palm_z_m: float = 0.06
    precontact_clearance_m: float = 0.01
    precontact_hand_curl: dict[str, float] = {"thumb_rotate": 0.0, "thumb": 0.5, "index": 0.8, "middle": 0.8}

    def __post_init__(self):
        super().__post_init__()
        for arm in (self.actions.left_arm, self.actions.right_arm):
            arm.scale = tuple(v * self.speed_fraction for v in FAST_ARM_SCALE)
        hand_step = FAST_HAND_STEP_RAD * self.speed_fraction
        for hand in (self.actions.left_hand, self.actions.right_hand):
            hand.scale = hand_step
            hand.clip = {".*": (-hand_step, hand_step)}
        self.task_instruction = "Turn the upside-down bowl upright with both hands."
        self.events.reset_to_init.params["table_path"] = str(bowl_pivot_init_path(self.object_spec))
        # Parallel RL: no texture waits.
        self.wait_for_textures = False
        self.rerender_on_reset = False


@configclass
class BowlPivotFastEnvCfg(BowlPivotEnvCfg):
    speed_fraction: float = 1.0


@configclass
class BowlPivotHalfSpeedEnvCfg(BowlPivotEnvCfg):
    speed_fraction: float = 0.5
