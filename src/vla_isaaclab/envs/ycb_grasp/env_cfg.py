"""Camera-free G1 + one YCB object scene, for state-based RL and pregrasp-table generation.

The object comes from ``object_spec`` (see ``envs/common/objects.py``). Episodes
start from its randomized default pose on the table; the RL task in
``rl_env_cfg.py`` replaces these managers.
"""

from dataclasses import MISSING

import isaaclab.envs.mdp as base_mdp
from isaaclab.assets import RigidObjectCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.utils import configclass

from ..common import (
    EventsCfg,
    GraspObjectSpec,
    JointLimitActionsCfg,
    PreviewObservationsCfg,
    PreviewRewardsCfg,
    SUPPORT_HEIGHT,
    VLAEnvCfg,
    camera_cfg,
    g1_left_wrist_camera_cfg,
    ground_cfg,
    light_cfgs,
    make_g1_cfg,
    object_rigid_cfg,
    table_cfgs,
)
from . import mdp


# Side view for recorded videos, same as the sugar-box VLA scene.
VIDEO_CAMERA_EYE = (0.35, 1.90, 1.85)
VIDEO_CAMERA_TARGET = (-0.15, -0.30, 0.72)
VIDEO_CAMERAS = ("cam_side", "cam_left_wrist")


def add_video_cameras(cfg) -> None:
    """Attach the side and left-wrist cameras (every env gets them; record with one env)."""
    cfg.scene.cam_side = camera_cfg(VIDEO_CAMERA_EYE, VIDEO_CAMERA_TARGET)
    cfg.scene.cam_left_wrist = g1_left_wrist_camera_cfg()


def _hand_object_contacts(link: str) -> ContactSensorCfg:
    # A filtered sensor spanning every robot link reports nothing: filtered
    # contact forces only work for one sensor body per environment.
    return ContactSensorCfg(
        prim_path=f"{{ENV_REGEX_NS}}/Robot/{link}",
        filter_prim_paths_expr=["{ENV_REGEX_NS}/Object"],
        history_length=4,
    )


# Hand links that can press on the table, by the short name used in sensor names.
HAND_LINKS = {
    "palm": "left_hand_palm_link",
    "thumb_0": "left_hand_thumb_0_link",
    "thumb_1": "left_hand_thumb_1_link",
    "thumb_2": "left_hand_thumb_2_link",
    "index_0": "left_hand_index_0_link",
    "index_1": "left_hand_index_1_link",
    "middle_0": "left_hand_middle_0_link",
    "middle_1": "left_hand_middle_1_link",
}


def add_hand_table_contacts(scene, side: str = "left") -> None:
    """One table_contact_<link> sensor per hand link (table_contact_right_<link> for the right hand), filtered to the table top."""
    for short, link in HAND_LINKS.items():
        name = f"table_contact_{short}" if side == "left" else f"table_contact_right_{short}"
        setattr(scene, name, ContactSensorCfg(
            prim_path=f"{{ENV_REGEX_NS}}/Robot/{link.replace('left_', f'{side}_', 1)}",
            filter_prim_paths_expr=["{ENV_REGEX_NS}/SupportSurface"],
            history_length=4,
        ))


def _gravity_free_g1_cfg():
    # The relative differential-IK action re-targets the measured palm pose every
    # step, so under gravity a PD-held arm creeps down (~9 cm in 5 s at zero
    # action). Real arm controllers compensate gravity; here it is switched off
    # for the robot links only (the object keeps gravity).
    cfg = make_g1_cfg((0.0, -0.64, 0.80))
    cfg.spawn.rigid_props.disable_gravity = True
    return cfg


_DOME_LIGHT, _KEY_LIGHT = light_cfgs()
_SURFACE, _LEG_0, _LEG_1, _LEG_2, _LEG_3 = table_cfgs()


@configclass
class YCBGraspStateSceneCfg(InteractiveSceneCfg):
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
    robot = _gravity_free_g1_cfg()
    # Only for videos, see add_video_cameras.
    cam_side = None
    cam_left_wrist = None
    # Set from the env's object_spec.
    object: RigidObjectCfg = MISSING
    # Unfiltered net contact force (object, table or anything else) on the left arm.
    left_arm_contacts = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/left_(shoulder|elbow|wrist|hand)_.*",
        history_length=4,
    )
    object_contact_palm = _hand_object_contacts("left_hand_palm_link")
    object_contact_thumb_0 = _hand_object_contacts("left_hand_thumb_0_link")
    object_contact_thumb_1 = _hand_object_contacts("left_hand_thumb_1_link")
    object_contact_thumb_2 = _hand_object_contacts("left_hand_thumb_2_link")
    object_contact_index_0 = _hand_object_contacts("left_hand_index_0_link")
    object_contact_index_1 = _hand_object_contacts("left_hand_index_1_link")
    object_contact_middle_0 = _hand_object_contacts("left_hand_middle_0_link")
    object_contact_middle_1 = _hand_object_contacts("left_hand_middle_1_link")


@configclass
class StateEventsCfg(EventsCfg):
    # Ranges are set from the object_spec.
    randomize_object_pose = EventTerm(
        func=mdp.randomize_object_planar_pose,
        mode="reset",
        params={"x_range": (0.0, 0.0), "y_range": (0.0, 0.0), "yaw_range": (0.0, 0.0),
                "asset_cfg": SceneEntityCfg("object")},
    )


@configclass
class StateTerminationsCfg:
    time_out = DoneTerm(func=base_mdp.time_out, time_out=True)
    # Set from the object_spec's table height.
    fell = DoneTerm(func=mdp.object_fallen, params={"support_height": SUPPORT_HEIGHT})
    invalid_state = DoneTerm(func=mdp.invalid_state)


@configclass
class YCBGraspStateEnvCfg(VLAEnvCfg):
    """One YCB object at a randomized pose, no cameras, for many parallel envs."""

    object_spec: GraspObjectSpec = MISSING
    scene: YCBGraspStateSceneCfg = YCBGraspStateSceneCfg(num_envs=1024, env_spacing=3.0, replicate_physics=True)
    actions: JointLimitActionsCfg = JointLimitActionsCfg()
    observations: PreviewObservationsCfg = PreviewObservationsCfg()
    events: StateEventsCfg = StateEventsCfg()
    rewards: PreviewRewardsCfg = PreviewRewardsCfg()
    terminations: StateTerminationsCfg = StateTerminationsCfg()
    episode_length_s: float = 60.0

    def __post_init__(self):
        super().__post_init__()
        spec = self.object_spec
        self.scene.object = object_rigid_cfg(spec)
        (self.scene.support_surface, self.scene.support_leg_0, self.scene.support_leg_1,
         self.scene.support_leg_2, self.scene.support_leg_3) = table_cfgs(spec.support_height_m)
        self.terminations.fell.params["support_height"] = spec.support_height_m
        self.task_instruction = f"Grasp the YCB {spec.name} and lift it off the table."
        randomize = getattr(self.events, "randomize_object_pose", None)
        if randomize is not None:
            randomize.params.update(
                x_range=spec.dr_x_range_m, y_range=spec.dr_y_range_m, yaw_range=spec.dr_yaw_range_rad
            )
