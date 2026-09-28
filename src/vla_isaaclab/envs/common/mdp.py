"""Object-agnostic tabletop reset and termination terms."""

import torch

from isaaclab.assets import Articulation, RigidObject
from isaaclab.envs import ManagerBasedEnv, ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import quat_from_angle_axis, quat_mul, sample_uniform


POSE_OFFSET_ATTR = "object_pose_offset"


def randomize_object_planar_pose(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    x_range: tuple[float, float],
    y_range: tuple[float, float],
    yaw_range: tuple[float, float],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("object"),
):
    """Offset the default object pose on the tabletop and yaw it about world +Z.

    Isaac Lab's ``reset_root_state_uniform`` composes yaw in the object's body
    frame, which tips over an object whose up axis is not local +Z.
    """
    obj: RigidObject = env.scene[asset_cfg.name]
    ranges = torch.tensor((x_range, y_range, yaw_range), dtype=torch.float32, device=obj.device)
    samples = sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), 3), device=obj.device)

    default = obj.data.default_root_state[env_ids]
    position = default[:, :3] + env.scene.env_origins[env_ids]
    position[:, :2] += samples[:, :2]
    z_axis = torch.tensor((0.0, 0.0, 1.0), device=obj.device).repeat(len(env_ids), 1)
    orientation = quat_mul(quat_from_angle_axis(samples[:, 2], z_axis), default[:, 3:7])
    obj.write_root_pose_to_sim(torch.cat((position, orientation), dim=-1), env_ids=env_ids)
    obj.write_root_velocity_to_sim(torch.zeros_like(default[:, 7:13]), env_ids=env_ids)

    offsets = getattr(env, POSE_OFFSET_ATTR, None)
    if offsets is None:
        offsets = torch.zeros(env.num_envs, 3, device=obj.device)
        setattr(env, POSE_OFFSET_ATTR, offsets)
    offsets[env_ids] = samples


def object_pose_offset(env: ManagerBasedEnv, env_index: int = 0) -> dict | None:
    """Return the sampled reset offset of one environment, if randomized."""
    offsets = getattr(env, POSE_OFFSET_ATTR, None)
    if offsets is None:
        return None
    x, y, yaw = offsets[env_index].detach().cpu().tolist()
    return {"x_m": x, "y_m": y, "yaw_rad": yaw}


def object_fallen(env: ManagerBasedRLEnv, support_height: float) -> torch.Tensor:
    obj: RigidObject = env.scene["object"]
    return obj.data.root_pos_w[:, 2] < support_height - 0.05


def invalid_state(env: ManagerBasedRLEnv) -> torch.Tensor:
    robot: Articulation = env.scene["robot"]
    obj: RigidObject = env.scene["object"]
    return ~(
        torch.isfinite(robot.data.joint_pos).all(dim=-1)
        & torch.isfinite(obj.data.root_state_w).all(dim=-1)
    )
