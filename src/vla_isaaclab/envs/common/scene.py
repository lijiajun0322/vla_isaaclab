"""Reusable physical scene configuration helpers."""

from __future__ import annotations

import math

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg
from isaaclab.sensors import CameraCfg


SUPPORT_HEIGHT = 0.62
G1_HEAD_CAMERA_POSITION = (0.075, 0.0, 0.025)
G1_HEAD_CAMERA_PITCH_RAD = math.radians(32.0)
G1_HEAD_CAMERA_ORIENTATION_WXYZ = (
    math.cos(G1_HEAD_CAMERA_PITCH_RAD / 2.0),
    0.0,
    math.sin(G1_HEAD_CAMERA_PITCH_RAD / 2.0),
    0.0,
)
G1_LEFT_WRIST_CAMERA_POSITION = (-0.015, -0.040, 0.045)
G1_LEFT_WRIST_CAMERA_PITCH_RAD = math.radians(24.0)
G1_LEFT_WRIST_CAMERA_ORIENTATION_WXYZ = (
    math.cos(G1_LEFT_WRIST_CAMERA_PITCH_RAD / 2.0),
    0.0,
    math.sin(G1_LEFT_WRIST_CAMERA_PITCH_RAD / 2.0),
    0.0,
)


def _static_box(size: tuple[float, float, float]) -> sim_utils.CuboidCfg:
    return sim_utils.CuboidCfg(
        size=size,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True, disable_gravity=True),
        collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
        visual_material=sim_utils.PreviewSurfaceCfg(
            diffuse_color=(0.92, 0.92, 0.92), roughness=0.35, metallic=0.0
        ),
    )


def ground_cfg() -> AssetBaseCfg:
    return AssetBaseCfg(
        prim_path="/World/Ground",
        spawn=sim_utils.GroundPlaneCfg(color=(0.25, 0.25, 0.25)),
        collision_group=-1,
    )


def light_cfgs() -> tuple[AssetBaseCfg, AssetBaseCfg]:
    dome = AssetBaseCfg(
        prim_path="/World/DomeLight",
        spawn=sim_utils.DomeLightCfg(intensity=1100.0, color=(0.85, 0.88, 1.0)),
    )
    key = AssetBaseCfg(
        prim_path="/World/KeyLight",
        spawn=sim_utils.DistantLightCfg(intensity=2200.0, color=(1.0, 0.95, 0.88), angle=0.8),
        init_state=AssetBaseCfg.InitialStateCfg(rot=(0.9239, 0.3827, 0.0, 0.0)),
    )
    return dome, key


def table_cfgs(
    support_height: float = SUPPORT_HEIGHT,
) -> tuple[AssetBaseCfg, AssetBaseCfg, AssetBaseCfg, AssetBaseCfg, AssetBaseCfg]:
    """Table top (at support_height) and four legs."""
    surface = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/SupportSurface",
        spawn=_static_box((1.2, 0.8, 0.05)),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, support_height - 0.025)),
    )
    leg_height = support_height - 0.05
    legs = tuple(
        AssetBaseCfg(
            prim_path=f"{{ENV_REGEX_NS}}/SupportLeg_{index}",
            spawn=_static_box((0.055, 0.055, leg_height)),
            init_state=AssetBaseCfg.InitialStateCfg(pos=position),
        )
        for index, position in enumerate(
            (
                (-0.52, -0.32, leg_height / 2.0),
                (-0.52, 0.32, leg_height / 2.0),
                (0.52, -0.32, leg_height / 2.0),
                (0.52, 0.32, leg_height / 2.0),
            )
        )
    )
    return (surface, *legs)


def _normalize(vector: tuple[float, float, float]) -> tuple[float, float, float]:
    norm = math.sqrt(sum(value * value for value in vector))
    return tuple(value / norm for value in vector)


def _cross(a, b):
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


def _quat_from_matrix(matrix) -> tuple[float, float, float, float]:
    """Convert a 3x3 local-to-world rotation matrix to wxyz."""
    trace = matrix[0][0] + matrix[1][1] + matrix[2][2]
    if trace > 0.0:
        scale = math.sqrt(trace + 1.0) * 2.0
        quat = (0.25 * scale, (matrix[2][1] - matrix[1][2]) / scale,
                (matrix[0][2] - matrix[2][0]) / scale, (matrix[1][0] - matrix[0][1]) / scale)
    else:
        diagonal = [matrix[index][index] for index in range(3)]
        index = diagonal.index(max(diagonal))
        if index == 0:
            scale = math.sqrt(1.0 + matrix[0][0] - matrix[1][1] - matrix[2][2]) * 2.0
            quat = ((matrix[2][1] - matrix[1][2]) / scale, 0.25 * scale,
                    (matrix[0][1] + matrix[1][0]) / scale, (matrix[0][2] + matrix[2][0]) / scale)
        elif index == 1:
            scale = math.sqrt(1.0 + matrix[1][1] - matrix[0][0] - matrix[2][2]) * 2.0
            quat = ((matrix[0][2] - matrix[2][0]) / scale,
                    (matrix[0][1] + matrix[1][0]) / scale, 0.25 * scale,
                    (matrix[1][2] + matrix[2][1]) / scale)
        else:
            scale = math.sqrt(1.0 + matrix[2][2] - matrix[0][0] - matrix[1][1]) * 2.0
            quat = ((matrix[1][0] - matrix[0][1]) / scale,
                    (matrix[0][2] + matrix[2][0]) / scale,
                    (matrix[1][2] + matrix[2][1]) / scale, 0.25 * scale)
    return _normalize_quat(quat)


def _normalize_quat(quat):
    norm = math.sqrt(sum(value * value for value in quat))
    return tuple(value / norm for value in quat)


def _look_at_quat(eye, target) -> tuple[float, float, float, float]:
    """Return the OpenGL camera orientation used by set_world_poses_from_view."""
    backward = _normalize(tuple(eye[index] - target[index] for index in range(3)))
    right = _normalize(_cross((0.0, 0.0, 1.0), backward))
    up = _normalize(_cross(backward, right))
    rotation = tuple(tuple(axis[row] for axis in (right, up, backward)) for row in range(3))
    return _quat_from_matrix(rotation)


def camera_cfg(eye, target) -> CameraCfg:
    return CameraCfg(
        prim_path="{ENV_REGEX_NS}/Camera",
        update_period=1.0 / 30.0,
        width=640,
        height=480,
        data_types=["rgb", "distance_to_image_plane"],
        depth_clipping_behavior="max",
        offset=CameraCfg.OffsetCfg(pos=eye, rot=_look_at_quat(eye, target), convention="opengl"),
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=28.0,
            focus_distance=2.0,
            horizontal_aperture=20.955,
            clipping_range=(0.05, 10.0),
        ),
    )


def robot_rgb_camera_cfg(
    parent_link: str,
    position: tuple[float, float, float],
    orientation_wxyz: tuple[float, float, float, float],
    focal_length: float,
) -> CameraCfg:
    """Create a 30 Hz RGB camera rigidly attached to a robot link.

    The supplied pose uses the parent link frame with the world-style camera
    convention: camera forward is +X and camera up is +Z.
    """
    return CameraCfg(
        prim_path=f"{{ENV_REGEX_NS}}/Robot/{parent_link}/Camera",
        update_period=1.0 / 30.0,
        width=640,
        height=480,
        data_types=["rgb"],
        offset=CameraCfg.OffsetCfg(
            pos=position,
            rot=orientation_wxyz,
            convention="world",
        ),
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=focal_length,
            focus_distance=0.7,
            horizontal_aperture=20.955,
            clipping_range=(0.03, 5.0),
        ),
    )


def g1_head_camera_cfg() -> CameraCfg:
    return robot_rgb_camera_cfg(
        "head_link",
        G1_HEAD_CAMERA_POSITION,
        G1_HEAD_CAMERA_ORIENTATION_WXYZ,
        focal_length=14.0,
    )


def g1_left_wrist_camera_cfg() -> CameraCfg:
    return robot_rgb_camera_cfg(
        "left_hand_palm_link",
        G1_LEFT_WRIST_CAMERA_POSITION,
        G1_LEFT_WRIST_CAMERA_ORIENTATION_WXYZ,
        focal_length=12.0,
    )
