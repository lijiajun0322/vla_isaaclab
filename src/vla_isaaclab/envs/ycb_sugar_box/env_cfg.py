"""Manager-based YCB 004 sugar-box pick-and-place environment."""

import math

import isaaclab.envs.mdp as base_mdp
import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg
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
    SUGAR_BOX,
    SUPPORT_HEIGHT,
    VLAEnvCfg,
    camera_cfg,
    g1_head_camera_cfg,
    g1_left_wrist_camera_cfg,
    ground_cfg,
    light_cfgs,
    make_g1_cfg,
    object_rigid_cfg,
    table_cfgs,
)
from . import mdp


CAMERA_EYE = (0.35, 1.90, 1.85)
CAMERA_TARGET = (-0.15, -0.30, 0.72)
SUGAR_BOX_TABLE_YAW_RAD = 0.6251518035481555
INITIAL_XY = SUGAR_BOX.initial_xy
TARGET_DISPLACEMENT_M = -0.02
TARGET_MARKER_HEIGHT = SUPPORT_HEIGHT + 0.002
TARGET_MARKER_ORIENTATION_WXYZ = (
    math.cos(SUGAR_BOX_TABLE_YAW_RAD / 2),
    0.0,
    0.0,
    math.sin(SUGAR_BOX_TABLE_YAW_RAD / 2),
)
TARGET_POSE = (
    INITIAL_XY[0] + TARGET_DISPLACEMENT_M,
    INITIAL_XY[1],
    SUPPORT_HEIGHT + SUGAR_BOX.rest_half_height_m,
    *SUGAR_BOX.rest_quat_wxyz,
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
    object = object_rigid_cfg(SUGAR_BOX)
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
        object_nominal_quat=SUGAR_BOX.rest_quat_wxyz,
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
            "x_range": SUGAR_BOX.dr_x_range_m,
            "y_range": SUGAR_BOX.dr_y_range_m,
            "yaw_range": SUGAR_BOX.dr_yaw_range_rad,
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
