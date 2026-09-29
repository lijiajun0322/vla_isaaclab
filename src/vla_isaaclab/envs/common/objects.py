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
DINNERWARE_USD_DIR = PROJECT_ROOT / "assets/YCB/dinnerware"


@configclass
class GraspObjectSpec:
    name: str = ""
    """YCB id, also the USD file stem and the output folder name (e.g. ``004_sugar_box``)."""
    usd_path: str = ""
    half_extents_m: tuple[float, float, float] = (0.0, 0.0, 0.0)
    """Collider half extents along the object's root x, y, z axes."""
    box_center_m: tuple[float, float, float] = (0.0, 0.0, 0.0)
    """Center of that collider box in the root frame (the bowl's root is its bottom)."""
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

    rim_radius_m: float = 0.0
    """Nonzero for a rim grasp: RL rewards fingertips for approaching the rim circle
    (this radius about the up axis, at rim_height_m) instead of the root."""
    rim_height_m: float = 0.0


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


# Mesh bounds 161 x 161 x 55 mm, root at the bottom center (the converter moves
# the mesh's lowest point to z = 0), up is local +z. Wall ~4 mm thick, 70-73 mm
# radius up to 47 mm high, then a lip flares out to 81 mm at 48-55 mm.
# Rim pinch on the robot's left of the bowl: fingers point forward along the
# rim outside the bowl, thumb inside it. Found with
#   scripts/rl/probe_rim_pregrasp.sh --frame forward --anchor thumb
#     --rim-point 0.062 0.048 --phi-deg 200 --pitch-deg 35 --roll-deg 12
#     --radial-m -0.06 --object-dy-m 0.03 --support-dz-m 0.06
# (bowl default then 11 cm left of the sugar box). That palm target overlaps
# the bowl, which the hold pushes 4 cm away; the values below are the settled,
# untouched result (hand pose relative to where the bowl came to rest), from
# which closing and raising the palm lifted the bowl 6.8 cm at 9 deg tilt.
# The table is 6 cm higher so the arm reaches the lowered wrist.
BOWL = GraspObjectSpec(
    name="024_bowl",
    usd_path=str(DINNERWARE_USD_DIR / "024_bowl/024_bowl_physics.usd"),
    half_extents_m=(0.0807, 0.0806, 0.0275),
    box_center_m=(0.0, 0.0, 0.0275),
    up_axis=2,
    pinch_axis=2,
    rest_quat_wxyz=(1.0, 0.0, 0.0, 0.0),
    rest_half_height_m=0.0,
    initial_xy=(-0.094827, -0.250232),
    support_height_m=SUPPORT_HEIGHT + 0.06,
    dr_x_range_m=(-0.01, 0.01),
    dr_y_range_m=(-0.01, 0.01),
    # Rotationally symmetric: yaw changes nothing.
    turn_point_world=(-0.081527, -0.446395, 0.903563),
    grasp_quat_wxyz=(0.390657, -0.193817, 0.194189, 0.8787),
    grasp_offset_world=(-0.009365, -0.133352, 0.118871),
    # 0.1 rad short of first contact when closing from the open hand
    # (scripts/rl/preview_preshape.sh, 5 % quantile: thumb 0.62, index 0.32,
    # middle 0.42); the thumb stays a little further open than that.
    hand_preshape={"thumb_rotate": 0.2, "thumb": 0.55, "index": 0.22, "middle": 0.32},
    rim_radius_m=0.076,
    rim_height_m=0.052,
)
