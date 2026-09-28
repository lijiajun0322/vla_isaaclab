"""Per-object parameters for tabletop grasping, shared by every environment that
spawns or grasps the object.

A ``GraspObjectSpec`` holds everything that depends on the object: asset,
collider size, the pose it rests in on the table, reset randomization, and the
calibrated scripted-approach and finger-preshape values. Scene, RL and scripted
code read these fields instead of object-specific constants.
"""

import math
from pathlib import Path

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import quat_apply

from .scene import SUPPORT_HEIGHT


PROJECT_ROOT = Path(__file__).resolve().parents[4]
YCB_USD_DIR = PROJECT_ROOT / "assets/YCB/Axis_Aligned_Physics"


@configclass
class GraspObjectSpec:
    name: str = ""
    """YCB id, also the USD file stem and the output folder name (e.g. ``004_sugar_box``)."""
    usd_path: str = ""
    half_extents_m: tuple[float, float, float] = (0.0, 0.0, 0.0)
    """Collider half extents along the object's root x, y, z axes."""
    up_axis: int = 2
    """Root axis that points up (times up_sign) while the object rests upright on the table."""
    up_sign: float = 1.0
    pinch_axis: int = 0
    """Root axis the hand closes across (the thumb on one side, the fingers on the other)."""

    rest_quat_wxyz: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)
    rest_half_height_m: float = 0.0
    """Root height above the table top at rest."""
    initial_xy: tuple[float, float] = (0.0, 0.0)
    """Env-local x, y of the calibrated default pose."""
    support_height_m: float = SUPPORT_HEIGHT
    """Table top height; raise it for an object whose grasp height is out of the arm's reach."""

    # Reset randomization around the default pose (world x, y and world-Z yaw).
    dr_x_range_m: tuple[float, float] = (0.0, 0.0)
    dr_y_range_m: tuple[float, float] = (0.0, 0.0)
    dr_yaw_range_rad: tuple[float, float] = (0.0, 0.0)

    # Scripted approach, calibrated in world frame for the default pose; a
    # randomized pose carries them rigidly with its planar offset and yaw.
    turn_point_world: tuple[float, float, float] = (0.0, 0.0, 0.0)
    """Env-local palm point where the hand turns to the grasp orientation."""
    grasp_quat_wxyz: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)
    """Palm orientation at the pregrasp (palm +x is the approach direction)."""
    grasp_offset_world: tuple[float, float, float] = (0.0, 0.0, 0.0)
    """Pregrasp palm position minus object position."""
    grasp_away_shift_m: float = 0.0
    """Extra horizontal pregrasp shift along the pinch axis, away from the robot."""

    hand_preshape: dict[str, float] = {}
    """Safe finger curl at the pregrasp (rad magnitudes): thumb_rotate, thumb, index, middle."""


def object_up_axis(spec: GraspObjectSpec, quat: torch.Tensor) -> torch.Tensor:
    """World direction of the object's upright axis for root orientations quat (wxyz)."""
    local = torch.zeros(quat.shape[0], 3, device=quat.device, dtype=quat.dtype)
    local[:, spec.up_axis] = spec.up_sign
    return quat_apply(quat, local)


def object_rigid_cfg(spec: GraspObjectSpec, prim_path: str = "{ENV_REGEX_NS}/Object") -> RigidObjectCfg:
    """The object resting at its calibrated default pose on the table."""
    if not Path(spec.usd_path).is_file():
        raise FileNotFoundError(f"Missing cached YCB asset: {spec.usd_path}")
    return RigidObjectCfg(
        prim_path=prim_path,
        spawn=sim_utils.UsdFileCfg(usd_path=spec.usd_path),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(*spec.initial_xy, spec.support_height_m + spec.rest_half_height_m),
            rot=spec.rest_quat_wxyz,
        ),
    )


SUGAR_BOX = GraspObjectSpec(
    name="004_sugar_box",
    usd_path=str(YCB_USD_DIR / "004_sugar_box.usd"),
    # x width, y height, z thickness.
    half_extents_m=(0.0463, 0.0881, 0.0226),
    up_axis=1,
    pinch_axis=2,
    # Stands on its narrow end, turned toward the robot.
    rest_quat_wxyz=(0.6728436464587195, 0.6728436464587194, 0.21744292911045365, 0.21744292911045368),
    rest_half_height_m=0.088,
    initial_xy=(0.010509244994595768, -0.2904894446620847),
    dr_x_range_m=(-0.02, 0.02),
    dr_y_range_m=(-0.02, 0.02),
    # +-45 deg was unreachable or knocked the box over.
    dr_yaw_range_rad=(-math.radians(15.0), math.radians(15.0)),
    # Offline FK candidate with open-hand clearance and joint-limit margin.
    turn_point_world=(-0.20000000000003018, -0.44859374354722326, 0.8520679715251525),
    grasp_quat_wxyz=(0.838739529885833, -0.0562443579616566, 0.0882924479819968, 0.5343753520080425),
    grasp_offset_world=(-0.10779687797449515, -0.11014619853853247, 0.07505996064897658),
    # Across the wide face, so the two outer fingers clear the box.
    grasp_away_shift_m=0.02,
    # From a 1024-env scan: index/middle at 0.4 rad already push the box.
    hand_preshape={"thumb_rotate": 0.2, "thumb": 0.6, "index": 0.2, "middle": 0.25},
)


# Mesh bounds 96 x 191 x 58 mm; the collider follows the bottle's shape (convex
# decomposition). Unlike the sugar box its cap is on local -y, so it stands on
# local +y: the sugar-box rest pose turned 180 deg about the thickness axis (z),
# which keeps the pinch direction. Measured along the upright axis from the
# root: bottom -95..-60 mm (58 mm thick), body -50..+40 mm (50 mm thick, 96 mm
# wide), shoulder +40..+60 mm, cap above.
MUSTARD_BOTTLE = GraspObjectSpec(
    name="006_mustard_bottle",
    usd_path=str(YCB_USD_DIR / "006_mustard_bottle.usd"),
    half_extents_m=(0.0480, 0.0957, 0.0291),
    up_axis=1,
    up_sign=-1.0,
    pinch_axis=2,
    rest_quat_wxyz=(0.21744292911045365, -0.21744292911045368, 0.6728436464587194, -0.6728436464587195),
    rest_half_height_m=0.0957,
    initial_xy=SUGAR_BOX.initial_xy,
    # 4 cm higher table: the palm stays at the sugar box's (fully reachable)
    # world height while it grips the bottle's body 4 cm lower on the bottle.
    support_height_m=SUPPORT_HEIGHT + 0.04,
    dr_x_range_m=SUGAR_BOX.dr_x_range_m,
    dr_y_range_m=SUGAR_BOX.dr_y_range_m,
    dr_yaw_range_rad=SUGAR_BOX.dr_yaw_range_rad,
    turn_point_world=SUGAR_BOX.turn_point_world,
    grasp_quat_wxyz=SUGAR_BOX.grasp_quat_wxyz,
    # At the sugar box's palm height above the table the thumb sits on the
    # shoulder and the index finger on the cap; 4 cm lower both are on the body.
    grasp_offset_world=(
        SUGAR_BOX.grasp_offset_world[0],
        SUGAR_BOX.grasp_offset_world[1],
        SUGAR_BOX.grasp_offset_world[2] + SUGAR_BOX.rest_half_height_m - 0.0957 - 0.04,
    ),
    grasp_away_shift_m=SUGAR_BOX.grasp_away_shift_m,
    hand_preshape=dict(SUGAR_BOX.hand_preshape),
)
