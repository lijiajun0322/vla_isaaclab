"""Sugar-box success termination terms."""

import torch

from isaaclab.assets import Articulation, RigidObject
from isaaclab.envs import ManagerBasedRLEnv


def task_metrics(env: ManagerBasedRLEnv, palm_body_name: str, command_name: str) -> dict[str, torch.Tensor]:
    robot: Articulation = env.scene["robot"]
    sugar_box: RigidObject = env.scene["object"]
    palm_ids, _ = robot.find_bodies([palm_body_name], preserve_order=True)
    target = env.command_manager.get_command(command_name)[:, :3] + env.scene.env_origins
    return {
        "xy_error": torch.linalg.vector_norm(sugar_box.data.root_pos_w[:, :2] - target[:, :2], dim=-1),
        "height_error": torch.abs(sugar_box.data.root_pos_w[:, 2] - target[:, 2]),
        "linear_speed": torch.linalg.vector_norm(sugar_box.data.root_lin_vel_w, dim=-1),
        "angular_speed": torch.linalg.vector_norm(sugar_box.data.root_ang_vel_w, dim=-1),
        "hand_distance": torch.linalg.vector_norm(
            robot.data.body_pos_w[:, palm_ids[0]] - sugar_box.data.root_pos_w, dim=-1
        ),
    }


def task_success(
    env: ManagerBasedRLEnv, palm_body_name: str, command_name: str, hold_steps: int = 15
) -> torch.Tensor:
    metrics = task_metrics(env, palm_body_name, command_name)
    instantaneous = (
        (metrics["xy_error"] < 0.015)
        & (metrics["height_error"] < 0.015)
        & (metrics["linear_speed"] < 0.04)
        & (metrics["angular_speed"] < 0.30)
        & (metrics["hand_distance"] > 0.20)
    )
    counter = getattr(env, "task_success_counter", None)
    if counter is None or counter.shape[0] != env.num_envs:
        counter = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
        env.task_success_counter = counter
    counter[:] = torch.where(instantaneous, counter + 1, torch.zeros_like(counter))
    return counter >= hold_steps
