"""Manager-based YCB 004 sugar-box pick-and-place environment."""

import math
from pathlib import Path

import isaaclab.envs.mdp as base_mdp
import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.utils import configclass

from ..common import (
    ACTION_JOINT_NAMES,
    EventsCfg,
    JointLimitActionsCfg,
    LEFT_END_EFFECTOR,
    SUPPORT_HEIGHT,
    VLAEnvCfg,
    camera_cfg,
    g1_head_camera_cfg,
    g1_left_wrist_camera_cfg,
    ground_cfg,
    light_cfgs,
    make_g1_cfg,
    table_cfgs,
)
from . import mdp


PROJECT_ROOT = Path(__file__).resolve().parents[4]
SUGAR_BOX_USD = PROJECT_ROOT / "assets/YCB/Axis_Aligned_Physics/004_sugar_box.usd"
CAMERA_EYE = (0.35, 1.90, 1.85)
CAMERA_TARGET = (-0.15, -0.30, 0.72)
SUGAR_BOX_HALF_HEIGHT_M = 0.088
SUGAR_BOX_ORIENTATION_WXYZ = (
    0.6728436464587195,
    0.6728436464587194,
    0.21744292911045365,
    0.21744292911045368,
)
SUGAR_BOX_TABLE_YAW_RAD = 0.6251518035481555
INITIAL_XY = (0.010509244994595768, -0.2904894446620847)
TARGET_DISPLACEMENT_M = -0.02
TARGET_MARKER_HEIGHT = SUPPORT_HEIGHT + 0.002
TARGET_MARKER_ORIENTATION_WXYZ = (
    math.cos(SUGAR_BOX_TABLE_YAW_RAD / 2),
    0.0,
    0.0,
    math.sin(SUGAR_BOX_TABLE_YAW_RAD / 2),
)
# Reset ranges around the calibrated pose.
DR_X_RANGE_M = (-0.02, 0.02)
DR_Y_RANGE_M = (-0.02, 0.02)
DR_YAW_RANGE_RAD = (-math.radians(15.0), math.radians(15.0))
TARGET_POSE = (
    INITIAL_XY[0] + TARGET_DISPLACEMENT_M,
    INITIAL_XY[1],
    SUPPORT_HEIGHT + SUGAR_BOX_HALF_HEIGHT_M,
    *SUGAR_BOX_ORIENTATION_WXYZ,
)


def _sugar_box_cfg() -> RigidObjectCfg:
    if not SUGAR_BOX_USD.is_file():
        raise FileNotFoundError(f"Missing cached YCB asset: {SUGAR_BOX_USD}")
    return RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Object",
        spawn=sim_utils.UsdFileCfg(usd_path=str(SUGAR_BOX_USD)),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(INITIAL_XY[0], INITIAL_XY[1], SUPPORT_HEIGHT + SUGAR_BOX_HALF_HEIGHT_M),
            rot=SUGAR_BOX_ORIENTATION_WXYZ,
        ),
    )


_DOME_LIGHT, _KEY_LIGHT = light_cfgs()
_SURFACE, _LEG_0, _LEG_1, _LEG_2, _LEG_3 = table_cfgs()


@configclass
class YCBSugarBoxSceneCfg(InteractiveSceneCfg):
    # Refresh sensors every physics step so the contact history holds all
    # substeps of one control step.
    lazy_sensor_update: bool = False
    ground = ground_cfg()
    dome_light = _DOME_LIGHT
    key_light = _KEY_LIGHT
    support_surface = _SURFACE
    support_leg_0 = _LEG_0
    support_leg_1 = _LEG_1
    support_leg_2 = _LEG_2
    support_leg_3 = _LEG_3
    robot = make_g1_cfg((0.0, -0.64, 0.80))
    cam_side = camera_cfg(CAMERA_EYE, CAMERA_TARGET)
    cam_left_high = g1_head_camera_cfg()
    cam_left_wrist = g1_left_wrist_camera_cfg()
    object = _sugar_box_cfg()
    # Per-link robot/box contact forces for observations, rewards and checks.
    robot_box_contacts = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*",
        filter_prim_paths_expr=["{ENV_REGEX_NS}/Object"],
        history_length=4,
    )
    target_marker = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/SugarBoxTarget",
        spawn=sim_utils.CuboidCfg(
            size=(0.095, 0.050, 0.003),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.1, 0.8, 0.2), opacity=0.35
            ),
            collision_props=None,
        ),
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=(INITIAL_XY[0] + TARGET_DISPLACEMENT_M, INITIAL_XY[1], TARGET_MARKER_HEIGHT),
            rot=TARGET_MARKER_ORIENTATION_WXYZ,
        ),
    )


@configclass
class CommandsCfg:
    target_pose = mdp.FixedPoseCommandCfg(pose=TARGET_POSE)


@configclass
class SugarBoxPolicyCfg(ObsGroup):
    joint_position = ObsTerm(
        func=base_mdp.joint_pos,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=list(ACTION_JOINT_NAMES))},
    )
    joint_velocity = ObsTerm(
        func=base_mdp.joint_vel,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=list(ACTION_JOINT_NAMES))},
    )
    left_end_effector_pose = ObsTerm(
        func=mdp.left_ee_pose,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot", body_names=[LEFT_END_EFFECTOR], preserve_order=True
            )
        },
    )
    sugar_box_state = ObsTerm(func=mdp.object_state, params={"asset_cfg": SceneEntityCfg("object")})
    target_pose = ObsTerm(func=base_mdp.generated_commands, params={"command_name": "target_pose"})
    last_action = ObsTerm(func=base_mdp.last_action)

    def __post_init__(self):
        self.enable_corruption = False
        self.concatenate_terms = True


@configclass
class ObservationsCfg:
    policy: SugarBoxPolicyCfg = SugarBoxPolicyCfg()


@configclass
class RewardsCfg:
    placement = RewTerm(
        func=mdp.placement_reward,
        weight=2.0,
        params={"palm_body_name": LEFT_END_EFFECTOR, "command_name": "target_pose"},
    )


@configclass
class TerminationsCfg:
    success = DoneTerm(
        func=mdp.task_success,
        params={"palm_body_name": LEFT_END_EFFECTOR, "command_name": "target_pose"},
    )
    object_fallen = DoneTerm(func=mdp.object_fallen, params={"support_height": SUPPORT_HEIGHT})
    invalid_state = DoneTerm(func=mdp.invalid_state)
    time_out = DoneTerm(func=base_mdp.time_out, time_out=True)


@configclass
class YCBSugarBoxEnvCfg(VLAEnvCfg):
    scene: YCBSugarBoxSceneCfg = YCBSugarBoxSceneCfg(
        num_envs=1, env_spacing=3.0, replicate_physics=True
    )
    actions: JointLimitActionsCfg = JointLimitActionsCfg()
    observations: ObservationsCfg = ObservationsCfg()
    commands: CommandsCfg = CommandsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    episode_length_s: float = 60.0
    task_instruction: str = (
        "Grasp the rotated YCB 004 sugar box from the robot-facing side, move it 2 cm "
        "toward robot-left, and place it back on the table."
    )
    camera_eye: tuple[float, float, float] = CAMERA_EYE
    camera_target: tuple[float, float, float] = CAMERA_TARGET


@configclass
class DomainRandCommandsCfg:
    target_pose = mdp.ObjectRelativePoseCommandCfg(
        world_offset=(TARGET_DISPLACEMENT_M, 0.0, 0.0),
        object_nominal_quat=SUGAR_BOX_ORIENTATION_WXYZ,
        marker_prim_name="SugarBoxTarget",
        marker_height=TARGET_MARKER_HEIGHT,
        marker_nominal_quat=TARGET_MARKER_ORIENTATION_WXYZ,
    )


@configclass
class DomainRandEventsCfg(EventsCfg):
    randomize_sugar_box_pose = EventTerm(
        func=mdp.randomize_object_planar_pose,
        mode="reset",
        params={
            "x_range": DR_X_RANGE_M,
            "y_range": DR_Y_RANGE_M,
            "yaw_range": DR_YAW_RANGE_RAD,
            "asset_cfg": SceneEntityCfg("object"),
        },
    )


@configclass
class YCBSugarBoxDREnvCfg(YCBSugarBoxEnvCfg):
    """Sugar-box task with a randomized upright tabletop pose per reset."""

    commands: DomainRandCommandsCfg = DomainRandCommandsCfg()
    events: DomainRandEventsCfg = DomainRandEventsCfg()
    task_instruction: str = (
        "Grasp the YCB 004 sugar box from the robot-facing side, move it 2 cm "
        "toward robot-left, and place it back on the table."
    )


def _hand_box_contacts(link: str) -> ContactSensorCfg:
    # Filtered contact forces only work for one sensor body per environment.
    return ContactSensorCfg(
        prim_path=f"{{ENV_REGEX_NS}}/Robot/{link}",
        filter_prim_paths_expr=["{ENV_REGEX_NS}/Object"],
        history_length=4,
    )


def _gravity_free_g1_cfg():
    # The relative differential-IK action re-targets the measured palm pose every
    # step, so under gravity a PD-held arm creeps down (~9 cm in 5 s at zero
    # action). Real arm controllers compensate gravity; here it is switched off
    # for the robot links only (the box keeps gravity).
    cfg = make_g1_cfg((0.0, -0.64, 0.80))
    cfg.spawn.rigid_props.disable_gravity = True
    return cfg


@configclass
class YCBSugarBoxStateSceneCfg(YCBSugarBoxSceneCfg):
    """Camera-free scene for state-based RL and pregrasp-table generation."""

    robot = _gravity_free_g1_cfg()
    cam_side = None
    cam_left_high = None
    cam_left_wrist = None
    # The marker is moved through USD per environment, which does not scale.
    target_marker = None
    # A filtered sensor spanning every robot link reports nothing; see
    # _hand_box_contacts for the per-link replacement.
    robot_box_contacts = None
    # Unfiltered net contact force (box, table or anything else) on the left arm.
    left_arm_contacts = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/left_(shoulder|elbow|wrist|hand)_.*",
        history_length=4,
    )
    box_contact_palm = _hand_box_contacts("left_hand_palm_link")
    box_contact_thumb_0 = _hand_box_contacts("left_hand_thumb_0_link")
    box_contact_thumb_1 = _hand_box_contacts("left_hand_thumb_1_link")
    box_contact_thumb_2 = _hand_box_contacts("left_hand_thumb_2_link")
    box_contact_index_0 = _hand_box_contacts("left_hand_index_0_link")
    box_contact_index_1 = _hand_box_contacts("left_hand_index_1_link")
    box_contact_middle_0 = _hand_box_contacts("left_hand_middle_0_link")
    box_contact_middle_1 = _hand_box_contacts("left_hand_middle_1_link")


@configclass
class StateCommandsCfg:
    target_pose = mdp.ObjectRelativePoseCommandCfg(
        world_offset=(TARGET_DISPLACEMENT_M, 0.0, 0.0),
        object_nominal_quat=SUGAR_BOX_ORIENTATION_WXYZ,
    )


@configclass
class YCBSugarBoxStateEnvCfg(YCBSugarBoxDREnvCfg):
    """Randomized sugar-box scene without cameras, for many parallel envs."""

    scene: YCBSugarBoxStateSceneCfg = YCBSugarBoxStateSceneCfg(
        num_envs=1024, env_spacing=3.0, replicate_physics=True
    )
    commands: StateCommandsCfg = StateCommandsCfg()
