"""Bimanual bowl pivoting: G1 turns an upside-down bowl upright with both arms and both Dex3 hands.

Every episode starts from a settled state built by
``scripts/rl/build_bowl_pivot_init.sh``: the bowl rests upside down in front of
the robot and both hands hover about 1 cm from it, palms facing it, fingers
and thumb half curled toward it, out of contact. That state is stored in the pregrasp-table format and loaded by
the grasp-lift task's ``reset_from_pregrasp_table``. With randomize_bowl_init
(default) the bowl starts at a random contact-free offset around that state
(bowl_pivot_init_random.pt, same builder); the hands stay fixed.

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
    enable_g1_self_collisions,
)
from ..ycb_grasp.env_cfg import YCBGraspStateEnvCfg, YCBGraspStateSceneCfg, _hand_object_contacts, add_hand_table_contacts
from ..ycb_grasp.rl_env_cfg import BowlGraspLiftEnvCfg
from ..ycb_grasp.mdp import grasp_rl
from .mdp import invalid_state, lift, object_fallen, pivot


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


def bowl_pivot_init_path(spec: GraspObjectSpec, randomized: bool = False) -> Path:
    """outputs/rl/<object>/bowl_pivot_init.pt, or bowl_pivot_init_random.pt with the bowl start randomized."""
    return PREGRASP_TABLE_DIR / spec.name / f"bowl_pivot_init{'_random' if randomized else ''}.pt"


# Speed limits at 30 Hz with actions clipped to [-1, 1]. Fast (the grasp-lift
# fast limits): palm <= 15 cm/s and 0.45 rad/s, fingers <= 1.5 rad/s; there the
# policy learned to toss the bowl over (airborne and untouched for ~0.3 s in
# every success). The env cfg's speed_fraction scales both (default 2/3: palm
# <= 10 cm/s and 0.3 rad/s, fingers <= 1.0 rad/s). The grasp-lift slow limits,
# 1/5, were too slow to continue from the fast policy.
FAST_ARM_SCALE = (0.005, 0.005, 0.005, 0.015, 0.015, 0.015)
FAST_HAND_STEP_RAD = 0.05


def bowl_flipped_pregrasp_path(spec: GraspObjectSpec, side: str = "left") -> Path:
    """outputs/rl/<object>/bowl_flipped_pregrasp[_right].pt: built by scripts/rl/build_bowl_flipped_pregrasp.sh."""
    return PREGRASP_TABLE_DIR / spec.name / f"bowl_flipped_pregrasp{'' if side == 'left' else '_right'}.pt"


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
    # Bowl on the table: landing impacts.
    object_table_contact = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Object",
        filter_prim_paths_expr=["{ENV_REGEX_NS}/SupportSurface"],
        history_length=4,
    )


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
    reset_gentle = EventTerm(func=pivot.reset_gentle, mode="reset")


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
    # Turn the bowl over on its rim and set it down, not toss it (a toss had the bowl
    # 4-5 cm up, untouched for 5-10 steps, spinning at ~10 rad/s). tilt_progress pays
    # only with the rim on the table; per step, flying untouched costs 120/30 = 4, a
    # bowl/table force above 3 N up to 2, spinning above 4 rad/s 1.5 at 10 rad/s; and a
    # success without flight or a landing above 10 N pays another 1000/30 = 33. Together
    # a toss gives up ~80-110 against a gentle turn, while still beating no turn.
    free_flight = RewTerm(func=pivot.free_flight, weight=-120.0)
    hard_landing = RewTerm(func=pivot.hard_landing, weight=-60.0)
    fast_spin = RewTerm(func=pivot.fast_spin, weight=-30.0)
    gentle_success_bonus = RewTerm(func=pivot.gentle_success, weight=1000.0)


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
    # Bowl start randomization around the fixed state (world x, y), hands fixed.
    # The builder writes candidate offsets and keeps only those where the bowl
    # rests untouched; with ~1 cm hand clearance that truncates x near the hands.
    randomize_bowl_init: bool = True
    # y stops at -1.5 cm: the lip already sits 2 cm from the table's near edge.
    bowl_init_x_range_m: tuple[float, float] = (-0.03, 0.03)
    bowl_init_y_range_m: tuple[float, float] = (-0.015, 0.04)
    precontact_hand_curl: dict[str, float] = {"thumb_rotate": 0.0, "thumb": 0.5, "index": 0.8, "middle": 0.8}

    def __post_init__(self):
        super().__post_init__()
        for arm in (self.actions.left_arm, self.actions.right_arm):
            arm.scale = tuple(v * self.speed_fraction for v in FAST_ARM_SCALE)
        hand_step = FAST_HAND_STEP_RAD * self.speed_fraction
        for hand in (self.actions.left_hand, self.actions.right_hand):
            hand.scale = hand_step
            hand.clip = {".*": (-hand_step, hand_step)}
        # The elbows otherwise pass into the torso.
        enable_g1_self_collisions(self.scene.robot)
        # For the bowl/table contact sensor.
        self.scene.object.spawn.activate_contact_sensors = True
        self.task_instruction = "Turn the upside-down bowl upright with both hands."
        self.events.reset_to_init.params["table_path"] = str(
            bowl_pivot_init_path(self.object_spec, randomized=self.randomize_bowl_init))
        # Parallel RL: no texture waits.
        self.wait_for_textures = False
        self.rerender_on_reset = False


@configclass
class BowlPivotFastEnvCfg(BowlPivotEnvCfg):
    speed_fraction: float = 1.0


@configclass
class BowlPivotHalfSpeedEnvCfg(BowlPivotEnvCfg):
    speed_fraction: float = 0.5


# -- two stages: turn the bowl upright, then lift it 5 cm level ------------------


@configclass
class BowlPivotLiftEventsCfg(BowlPivotEventsCfg):
    reset_lift = EventTerm(func=lift.reset_lift_state, mode="reset")


@configclass
class BowlPivotLiftRewardsCfg(BowlPivotRewardsCfg):
    # Stage 1, about 100 in total: tilt progress (29), the dense upright term until the
    # flip (the policy otherwise turned the bowl and then left it, collecting upright
    # for the rest of the episode), and 67 once when the bowl is upright on the table.
    upright = RewTerm(func=lift.stage1_upright, weight=2.0)
    success_bonus = None
    # Its "success" is the lift here; not tuned for the two-stage task yet.
    gentle_success_bonus = None
    flip_bonus = RewTerm(func=lift.flip_bonus, weight=2000.0)
    # Stage 2 outweighs it. Dense, after the flip, per step: both hands at their support
    # points beside the bowl's foot up to 9/30 = 0.3, both hands on the outside of the
    # bowl 0.3 (0.15 each), each hand on the inner wall -0.5 (30/30 x 0.5): touching
    # inside always costs more than touching outside earns. Held-lift
    # progress over the 5 cm pays 30; on the lift step exp(-error / 3 cm) pays up to 100
    # and a success (error < 2 cm) 200, more than the ~100 of dense terms an episode
    # ending at the lift gives up.
    stage2_support_near = RewTerm(func=lift.stage2_support_near, weight=9.0)
    # Each new closest mean hand distance to the support points pays once: from ~20 cm
    # after the flip down to 0 that is 4500/30 * 0.2 = 30, like tilt_progress.
    stage2_support_progress = RewTerm(func=lift.stage2_support_progress, weight=4500.0)
    stage2_outer_contact = RewTerm(func=lift.stage2_outer_contact, weight=9.0)
    stage2_inner_contact = RewTerm(func=lift.stage2_inner_contact, weight=-30.0)
    # Held off the table: up to 9/30 = 0.3 per step for a level bowl (0 at 90 deg tilt),
    # so the lift keeps the bowl level all the way, not just at the 5 cm mark.
    stage2_level = RewTerm(func=lift.stage2_level, weight=9.0)
    lift_progress = RewTerm(func=lift.held_lift_progress, weight=18000.0)
    lift_pose = RewTerm(func=lift.lift_pose_score, weight=3000.0)
    lift_success_bonus = RewTerm(func=base_mdp.is_terminated_term, weight=6000.0, params={"term_keys": "success"})
    # Stage 3, after a good lift: hold it there for cfg.lift_hold_s. Per step: up to
    # 30/30 = 1.0 at the target pose and still (60 over a 2 s hold); 15/30 = 0.5 for
    # holding it level 3-10 cm up at all (at most ~128 over the rest of an episode, well
    # under the ~260 of a successful hold, so holding without finishing does not pay);
    # up to -0.5 for still rising faster than 2 cm/s. Dropping the bowl after the lift
    # ends the episode with the failure penalty (-10). (The policy first lifted and let
    # go: 61% dropped, hold_still ~0, after 150 iterations at 0.5 alone.)
    hold_still = RewTerm(func=lift.hold_still, weight=30.0)
    held_aloft = RewTerm(func=lift.held_aloft, weight=15.0)
    rising_after_lift = RewTerm(func=lift.rising_after_lift, weight=-15.0)
    failure_penalty = RewTerm(
        func=base_mdp.is_terminated_term, weight=-300.0,
        params={"term_keys": ["ran_away", "fell", "spinning", "dropped"]},
    )


@configclass
class BowlPivotLiftTerminationsCfg(BowlPivotTerminationsCfg):
    # The pivot success no longer ends the episode (it starts stage 2); the success is
    # a good lift held for cfg.lift_hold_s.
    success = DoneTerm(func=lift.lift_success)
    lifted_off_pose = DoneTerm(func=lift.lift_off_pose)
    dropped = DoneTerm(func=lift.bowl_dropped)


@configclass
class BowlPivotLiftEnvCfg(BowlPivotHalfSpeedEnvCfg):
    """Pivot the bowl upright, then lift it 5 cm level (1/2 speed). "Held" is any hand contact."""

    events: BowlPivotLiftEventsCfg = BowlPivotLiftEventsCfg()
    rewards: BowlPivotLiftRewardsCfg = BowlPivotLiftRewardsCfg()
    terminations: BowlPivotLiftTerminationsCfg = BowlPivotLiftTerminationsCfg()
    # The flip takes ~2 s; leave time for the lift.
    episode_length_s: float = 12.0
    lift_hold_s: float = 2.0
    """After a good lift the bowl must stay up, level and held for this long (0: the lift itself is the success)."""
    lift_hold_mode: str = "outer"
    """"outer": a hand on the outside of the bowl; "any": any link of either hand; "grip": thumb, index and middle of one hand."""

    def __post_init__(self):
        super().__post_init__()
        self.task_instruction = "Turn the upside-down bowl upright, then lift it 5 cm keeping it level."


@configclass
class BowlPivotLiftGripEnvCfg(BowlPivotLiftEnvCfg):
    """As BowlPivotLiftEnvCfg, but held means a three-finger grip of one hand, a three-finger
    grip pays per step in stage 2, and a success with the other hand off the bowl pays 100."""

    lift_hold_mode: str = "grip"

    def __post_init__(self):
        super().__post_init__()
        self.rewards.one_hand_grip = RewTerm(func=lift.one_hand_grip, weight=2.0)
        self.rewards.free_hand_bonus = RewTerm(func=lift.free_hand_at_success, weight=3000.0)


# -- lift the turned bowl: the grasp-lift task in the pivot scene ------------------


@configclass
class BowlFlippedLiftEnvCfg(BowlGraspLiftEnvCfg):
    """The bowl grasp-lift task (left arm and hand, same observations and actions, so the
    grasp-lift policy fine-tunes directly) on the pivot table. Resets draw states where the
    pivot policy turned the bowl upright, the right hand moved away and a scripted left
    hand came down from outside the bowl to the pregrasp (build_bowl_flipped_pregrasp.sh).
    Unlike the other pregrasp tables these starts may have the hand touching the bowl
    (the user's choice). The lift is the bowl's lowest point above the table."""

    object_spec: GraspObjectSpec = BOWL_UPSIDE_DOWN
    lift_from_lowest_point: bool = True

    def __post_init__(self):
        super().__post_init__()
        self.events.reset_to_pregrasp.params["table_path"] = str(bowl_flipped_pregrasp_path(self.object_spec))
        enable_g1_self_collisions(self.scene.robot)
        self.task_instruction = "Grasp the upright bowl and lift it 5 cm keeping it level."


@configclass
class BowlFlippedLiftRightEnvCfg(BowlFlippedLiftEnvCfg):
    """BowlFlippedLiftEnvCfg with the right arm and hand: the left hand moved away after the
    flip and the right one came to the mirrored bowl pregrasp (build_bowl_flipped_pregrasp.sh
    --side right). The pivot scene has the right hand's bowl contact sensors."""

    scene: BowlPivotSceneCfg = BowlPivotSceneCfg(num_envs=1024, env_spacing=2.0, replicate_physics=True)
    grasp_side: str = "right"

    def __post_init__(self):
        super().__post_init__()
        joints = SceneEntityCfg("robot", joint_names=[*RIGHT_ARM_JOINT_NAMES, *RIGHT_HAND_JOINT_NAMES], preserve_order=True)
        for group in (self.observations.policy, self.observations.critic):
            group.joint_pos.params["asset_cfg"] = joints
            group.joint_vel.params["asset_cfg"] = joints
        self.actions.arm.joint_names = list(RIGHT_ARM_JOINT_NAMES)
        self.actions.arm.body_name = RIGHT_END_EFFECTOR
        self.actions.hand.joint_names = list(RIGHT_HAND_JOINT_NAMES)
        add_hand_table_contacts(self.scene, side="right")
        # The pivot scene's bowl/table sensor.
        self.scene.object.spawn.activate_contact_sensors = True
        self.events.reset_to_pregrasp.params["table_path"] = str(bowl_flipped_pregrasp_path(self.object_spec, "right"))
        self.task_instruction = "Grasp the upright bowl with the right hand and lift it 5 cm keeping it level."
