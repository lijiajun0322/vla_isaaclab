"""Unitree G1 29-DoF + Dex3 asset configuration and joint semantics."""

from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg


PROJECT_ROOT = Path(__file__).resolve().parents[4]
G1_USD = (
    PROJECT_ROOT
    / "assets/robots/g1-29dof-dex3-base-fix-usd/g1_29dof_with_dex3_base_fix.usd"
)
LEGACY_G1_USD = (
    PROJECT_ROOT / "assets/Isaac/4.5/Isaac/IsaacLab/Robots/Unitree/G1/g1_23dof_legacy.usd"
)

# Simulator names, explicitly ordered to match g1_29body_dex3_43d_v1. Do not
# derive this order from the USD articulation or sort it alphabetically.
LOWER_BODY_JOINT_NAMES = (
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
)
WAIST_JOINT_NAMES = (
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
)
LEFT_ARM_JOINT_NAMES = (
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
)
RIGHT_ARM_JOINT_NAMES = (
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
)
LEFT_HAND_JOINT_NAMES = (
    "left_hand_thumb_0_joint",
    "left_hand_thumb_1_joint",
    "left_hand_thumb_2_joint",
    "left_hand_middle_0_joint",
    "left_hand_middle_1_joint",
    "left_hand_index_0_joint",
    "left_hand_index_1_joint",
)
# The contract intentionally swaps index and middle relative to Unitree's
# right-hand DDS order. Keeping the explicit order here performs that mapping.
RIGHT_HAND_JOINT_NAMES = (
    "right_hand_thumb_0_joint",
    "right_hand_thumb_1_joint",
    "right_hand_thumb_2_joint",
    "right_hand_index_0_joint",
    "right_hand_index_1_joint",
    "right_hand_middle_0_joint",
    "right_hand_middle_1_joint",
)
BODY_JOINT_NAMES = (
    LOWER_BODY_JOINT_NAMES + WAIST_JOINT_NAMES + LEFT_ARM_JOINT_NAMES + RIGHT_ARM_JOINT_NAMES
)
ACTION_JOINT_NAMES = BODY_JOINT_NAMES + LEFT_HAND_JOINT_NAMES + RIGHT_HAND_JOINT_NAMES

CONTRACT_JOINT_NAMES = (
    "kLeftHipPitch", "kLeftHipRoll", "kLeftHipYaw", "kLeftKnee",
    "kLeftAnklePitch", "kLeftAnkleRoll", "kRightHipPitch", "kRightHipRoll",
    "kRightHipYaw", "kRightKnee", "kRightAnklePitch", "kRightAnkleRoll",
    "kWaistYaw", "kWaistRoll", "kWaistPitch", "kLeftShoulderPitch",
    "kLeftShoulderRoll", "kLeftShoulderYaw", "kLeftElbow", "kLeftWristRoll",
    "kLeftWristPitch", "kLeftWristYaw", "kRightShoulderPitch",
    "kRightShoulderRoll", "kRightShoulderYaw", "kRightElbow",
    "kRightWristRoll", "kRightWristPitch", "kRightWristYaw",
    "kLeftHandThumb0", "kLeftHandThumb1", "kLeftHandThumb2",
    "kLeftHandMiddle0", "kLeftHandMiddle1", "kLeftHandIndex0",
    "kLeftHandIndex1", "kRightHandThumb0", "kRightHandThumb1",
    "kRightHandThumb2", "kRightHandIndex0", "kRightHandIndex1",
    "kRightHandMiddle0", "kRightHandMiddle1",
)

if len(ACTION_JOINT_NAMES) != 43 or len(set(ACTION_JOINT_NAMES)) != 43:
    raise RuntimeError("G1 Data Contract mapping must contain 43 unique simulator joints")
if len(CONTRACT_JOINT_NAMES) != 43:
    raise RuntimeError("G1 Data Contract must contain 43 canonical joint names")

LEFT_END_EFFECTOR = "left_hand_palm_link"
RIGHT_END_EFFECTOR = "right_hand_palm_link"
LEFT_HAND_OPEN_JOINT_POSITIONS = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
# The left Dex3 middle/index joints flex toward the palm in the negative
# direction (USD limits [-90, 0] and [-100, 0] degrees).
LEFT_HAND_CLOSED_JOINT_POSITIONS = (0.2, 0.8, 0.8, -1.2, -1.2, -1.2, -1.2)

LEG_HOME_JOINT_POSITIONS = {
    "left_hip_pitch_joint": -0.05,
    "left_knee_joint": 0.2,
    "left_ankle_pitch_joint": -0.15,
    "right_hip_pitch_joint": -0.05,
    "right_knee_joint": 0.2,
    "right_ankle_pitch_joint": -0.15,
}
LEFT_ARM_HOME_JOINT_POSITIONS = {
    "waist_yaw_joint": 0.0,
    "waist_roll_joint": 0.0,
    "waist_pitch_joint": 0.0,
    "left_shoulder_pitch_joint": 0.255715,
    "left_shoulder_roll_joint": 0.431127,
    "left_shoulder_yaw_joint": -0.368285,
    "left_elbow_joint": -0.04442,
    "left_wrist_roll_joint": -0.333947,
    "left_wrist_pitch_joint": 0.0,
    "left_wrist_yaw_joint": 0.0,
}


def _base_g1_cfg() -> ArticulationCfg:
    return ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/Robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path=str(G1_USD),
            activate_contact_sensors=True,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                retain_accelerations=True,
                linear_damping=0.0,
                angular_damping=0.0,
                max_linear_velocity=1000.0,
                max_angular_velocity=1000.0,
                max_depenetration_velocity=1.0,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False,
                solver_position_iteration_count=16,
                solver_velocity_iteration_count=4,
                fix_root_link=True,
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.74),
            joint_pos={name: 0.0 for name in ACTION_JOINT_NAMES},
            joint_vel={".*": 0.0},
        ),
        soft_joint_pos_limit_factor=0.9,
        actuators={
            "legs": ImplicitActuatorCfg(
                joint_names_expr=[".*_hip_.*_joint", ".*_knee_joint"],
                effort_limit=300.0,
                velocity_limit=100.0,
                stiffness=1000.0,
                damping=100.0,
                armature=0.01,
            ),
            "feet": ImplicitActuatorCfg(
                joint_names_expr=[".*_ankle_pitch_joint", ".*_ankle_roll_joint"],
                effort_limit=300.0,
                velocity_limit=100.0,
                stiffness=1000.0,
                damping=100.0,
                armature=0.01,
            ),
            "waist": ImplicitActuatorCfg(
                joint_names_expr=["waist_.*_joint"],
                effort_limit=1000.0,
                velocity_limit=None,
                stiffness=10000.0,
                damping=10000.0,
                armature=None,
            ),
            "arms": ImplicitActuatorCfg(
                joint_names_expr=[".*_shoulder_.*_joint", ".*_elbow_joint", ".*_wrist_.*_joint"],
                effort_limit=300.0,
                velocity_limit=100.0,
                stiffness={
                    ".*_shoulder_.*_joint": 300.0,
                    ".*_elbow_joint": 400.0,
                    ".*_wrist_.*_joint": 400.0,
                },
                damping={
                    ".*_shoulder_.*_joint": 3.0,
                    ".*_elbow_joint": 2.5,
                    ".*_wrist_.*_joint": 2.5,
                },
                armature=None,
            ),
            "hands": ImplicitActuatorCfg(
                joint_names_expr=[".*_hand_(thumb|middle|index)_.*_joint"],
                effort_limit=300.0,
                velocity_limit=100.0,
                stiffness=100.0,
                damping=10.0,
                armature=0.1,
            ),
        },
    )


def make_g1_cfg(
    position: tuple[float, float, float],
    orientation_wxyz: tuple[float, float, float, float] = (0.70710678, 0.0, 0.0, 0.70710678),
) -> ArticulationCfg:
    """Return the fixed-base, full 43-joint G1 used by project environments."""
    if not G1_USD.is_file():
        raise FileNotFoundError(f"Missing cached G1 29-DoF + Dex3 asset: {G1_USD}")
    cfg = _base_g1_cfg()
    cfg.init_state.pos = position
    cfg.init_state.rot = orientation_wxyz
    cfg.init_state.joint_pos.update(LEG_HOME_JOINT_POSITIONS)
    cfg.init_state.joint_pos.update(LEFT_ARM_HOME_JOINT_POSITIONS)
    cfg.init_state.joint_pos.update(dict(zip(LEFT_HAND_JOINT_NAMES, LEFT_HAND_OPEN_JOINT_POSITIONS)))
    return cfg
