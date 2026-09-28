#!/usr/bin/env python3
"""Sanity-check the grasp-and-lift RL environment with fixed action sequences.

zero:       all-zero actions; the arm and hand should hold the pregrasp and the box stay put.
close:      close the hand toward the scripted closed pose; contact rewards should respond.
close_lift: close, then keep squeezing while raising the palm; lift rewards should respond.

Hand actions are mapped by joint name (the action term follows the USD joint
order). Absolute hand actions get the closed targets; relative (delta) actions
get closing deltas.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import traceback
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task", default="VLA-YCBSugarBox-G1-GraspLift-RL-v1")
parser.add_argument("--modes", nargs="+", default=("zero", "close", "close_lift"),
                    choices=("zero", "close", "close_lift"))
parser.add_argument("--num-envs", type=int, default=64)
parser.add_argument("--steps", type=int, default=150)
parser.add_argument("--robot-gravity", choices=("on", "off", "cfg"), default="cfg",
                    help="Override robot gravity; 'cfg' keeps the environment's setting.")
parser.add_argument("--close-steps", type=int, default=30, help="Steps spent closing before the lift.")
parser.add_argument("--lift-mps", type=float, default=0.06, help="Commanded palm rise speed in 'close_lift'.")
parser.add_argument("--lift-to-m", type=float, default=None, help="Stop raising the palm after this rise.")
parser.add_argument("--close-delta", type=float, default=0.03, help="Relative hand: closing rad per step.")
parser.add_argument("--grip-delta", type=float, default=0.1, help="Relative hand: squeeze rad per step while lifting.")
parser.add_argument("--video-dir", type=Path, default=None, help="Record env 0 of each mode here.")
AppLauncher.add_app_launcher_args(parser)
ARGS = parser.parse_args()
RECORD = ARGS.video_dir is not None
ARGS.enable_cameras = RECORD
kit = ("ycb.python.headless.rendering.kit" if ARGS.headless else "ycb.python.rendering.kit") \
    if RECORD else ("ycb.python.headless.kit" if ARGS.headless else "ycb.python.kit")
ARGS.experience = str(PROJECT_ROOT / "configs" / kit)
ARGS.kit_args = f"--portable-root {PROJECT_ROOT}/outputs/runtime/kit"
APP = AppLauncher(ARGS).app

import gymnasium as gym
import numpy as np
import torch

import vla_isaaclab  # noqa: F401
from isaaclab.envs.mdp.actions import RelativeJointPositionActionCfg
from isaaclab_tasks.utils import parse_env_cfg

from vla_isaaclab.envs.common import (
    LEFT_END_EFFECTOR,
    LEFT_HAND_CLOSED_JOINT_POSITIONS,
    LEFT_HAND_JOINT_NAMES,
    camera_cfg,
    g1_left_wrist_camera_cfg,
)
from vla_isaaclab.envs.ycb_sugar_box.env_cfg import CAMERA_EYE, CAMERA_TARGET
from vla_isaaclab.envs.ycb_sugar_box.mdp import grasp_rl


VIDEO_CAMERAS = ("cam_side", "cam_left_wrist")


class Videos:
    def __init__(self, directory: Path):
        import av

        directory.mkdir(parents=True, exist_ok=True)
        self.streams = {}
        for name in VIDEO_CAMERAS:
            container = av.open(str(directory / f"{name}.mp4"), mode="w")
            stream = container.add_stream("libx264", rate=30)
            stream.width, stream.height, stream.pix_fmt = 640, 480, "yuv420p"
            self.streams[name] = (container, stream)

    def write(self, env):
        import av

        for name, (container, stream) in self.streams.items():
            rgb = env.scene.sensors[name].data.output["rgb"][0, ..., :3].detach().cpu().numpy().astype(np.uint8)
            for packet in stream.encode(av.VideoFrame.from_ndarray(rgb, format="rgb24")):
                container.mux(packet)

    def close(self):
        for container, stream in self.streams.values():
            for packet in stream.encode():
                container.mux(packet)
            container.close()


def hand_actions(env, mode: str, step: int) -> torch.Tensor:
    """Hand action columns in the action term's own joint order."""
    term = env.action_manager.get_term("hand")
    closed = dict(zip(LEFT_HAND_JOINT_NAMES, LEFT_HAND_CLOSED_JOINT_POSITIONS))
    scale = env.cfg.actions.hand.scale
    out = torch.zeros(len(term._joint_names), device=env.device)
    if mode == "zero":
        return out
    relative = isinstance(env.cfg.actions.hand, RelativeJointPositionActionCfg)
    for i, name in enumerate(term._joint_names):
        if relative:
            # The preshape already rotated the thumb base; only curl.
            if name == "left_hand_thumb_0_joint":
                continue
            delta = ARGS.close_delta if step < ARGS.close_steps else ARGS.grip_delta
            out[i] = math.copysign(delta, closed[name]) / scale
        else:
            out[i] = closed[name] * min(1.0, (step + 1) / ARGS.close_steps) / scale
    return out


def run(env, mode: str) -> dict:
    videos = Videos(ARGS.video_dir / mode) if RECORD else None
    env.reset()
    robot = env.scene["robot"]
    box = env.scene["object"]
    palm = robot.find_bodies([LEFT_END_EFFECTOR])[0][0]
    palm_start = robot.data.body_pos_w[:, palm].clone()
    box_start = box.data.root_pos_w.clone()
    arm_dim = env.action_manager.get_term("arm").action_dim
    actions = torch.zeros(env.num_envs, env.action_manager.total_action_dim, device=env.device)
    hand_ids = robot.find_joints(list(LEFT_HAND_JOINT_NAMES), preserve_order=True)[0]
    hand_start = robot.data.joint_pos[:, hand_ids].clone()
    hand_change_max = torch.zeros(env.num_envs, device=env.device)
    dz = ARGS.lift_mps * env.step_dt / env.cfg.actions.arm.scale[2]
    reward_sums = {name: 0.0 for name in env.reward_manager.active_terms}
    done_counts = {name: 0 for name in env.termination_manager.active_terms}
    alive = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)
    peak_lift = torch.zeros(env.num_envs, device=env.device)
    ever_grasped = torch.zeros_like(alive)
    min_corner = torch.full((env.num_envs,), float("inf"), device=env.device)
    high_steps = contact_high = held_high = 0
    rel_speeds = []
    run = torch.zeros(env.num_envs, device=env.device)
    longest = torch.zeros(env.num_envs, device=env.device)
    link_force_max = torch.zeros(env.num_envs, len(grasp_rl.HAND_CONTACT_SENSORS), device=env.device)
    for step in range(ARGS.steps):
        actions[:, arm_dim:] = hand_actions(env, mode, step)
        rising = mode == "close_lift" and step >= ARGS.close_steps
        if rising and ARGS.lift_to_m is not None:
            rising = (robot.data.body_pos_w[:, palm, 2] - palm_start[:, 2]) < ARGS.lift_to_m
        actions[:, 2] = torch.where(torch.as_tensor(rising, device=env.device), dz, 0.0) if torch.is_tensor(rising) \
            else (dz if rising else 0.0)
        env.step(actions)
        if videos is not None and bool(alive[0]):
            videos.write(env)
        hand_change = (robot.data.joint_pos[:, hand_ids] - hand_start).abs().amax(dim=-1)
        hand_change_max = torch.where(alive, torch.maximum(hand_change_max, hand_change), hand_change_max)
        # Only count each env's first episode.
        for i, name in enumerate(env.reward_manager.active_terms):
            reward_sums[name] += (env.reward_manager._step_reward[:, i] * alive).sum().item()
        for name in env.termination_manager.active_terms:
            done_counts[name] += int((env.termination_manager.get_term(name) & alive).sum().item())
        peak_lift = torch.where(alive, torch.maximum(peak_lift, grasp_rl.lift_height(env)), peak_lift)
        high = alive & ~env.termination_manager.dones & (grasp_rl.lift_height(env) > 0.05)
        high_steps += int(high.sum())
        contact_high += int((high & grasp_rl.grasp_flag(env)).sum())
        held_high += int((high & grasp_rl.grasp_held(env)).sum())
        rel_speeds.append(grasp_rl.box_palm_relative_speed(env)[high & grasp_rl.grasp_flag(env)])
        run = torch.where(high & grasp_rl.grasp_flag(env), run + 1, torch.zeros_like(run))
        longest = torch.maximum(longest, run)
        if hasattr(env, grasp_rl.GOAL_ATTR):
            corner = grasp_rl.goal_corner_error(env)
            min_corner = torch.where(alive, torch.minimum(min_corner, corner), min_corner)
        ever_grasped |= alive & grasp_rl.grasp_flag(env)
        link_force_max = torch.where(alive.unsqueeze(-1), torch.maximum(link_force_max, grasp_rl.contact_forces(env)),
                                     link_force_max)
        alive &= ~env.termination_manager.dones
        if step == 0:
            first_step_palm = torch.linalg.vector_norm(robot.data.body_pos_w[:, palm] - palm_start, dim=-1)
    if videos is not None:
        videos.close()
    survivors = alive
    palm_drift = torch.linalg.vector_norm(robot.data.body_pos_w[:, palm] - palm_start, dim=-1)
    box_move = torch.linalg.vector_norm(box.data.root_pos_w - box_start, dim=-1)
    return {
        "mode": mode,
        "envs": env.num_envs,
        "steps": ARGS.steps,
        "episodes_still_running": int(survivors.sum().item()),
        "terminations_first_episode": done_counts,
        "reward_per_env_first_episode": {k: v / env.num_envs for k, v in reward_sums.items()},
        "ever_grasped_fraction": ever_grasped.float().mean().item(),
        # Fraction of envs where each hand link ever touched the box (> threshold).
        "link_touched_fraction": {
            name: (link_force_max[:, i] > grasp_rl.CONTACT_THRESHOLD_N).float().mean().item()
            for i, name in enumerate(grasp_rl.HAND_CONTACT_SENSORS)
        },
        "hand_joint_change_max_rad": {"median": hand_change_max.median().item(), "max": hand_change_max.max().item()},
        "box_above_5cm_steps": high_steps,
        "above_5cm_fraction_contact": contact_high / max(high_steps, 1),
        "above_5cm_fraction_held": held_high / max(high_steps, 1),
            # Per env: longest run of steps with the box above 5 cm and thumb+finger contact.
            "longest_contact_above_5cm_steps": {q: longest.quantile(q).item() for q in (0.1, 0.25, 0.5, 0.75, 0.9)},
            "envs_with_run_ge": {k: (longest >= k).float().mean().item() for k in (3, 9, 15, 30)},
            "above_5cm_contact_rel_speed_mps_quantiles": (lambda v: {q: round(v.quantile(q).item(), 3) for q in (0.25, 0.5, 0.75, 0.9)}
                                                         if v.numel() else None)(torch.cat(rel_speeds)),
        "min_goal_corner_error_mm": {"median": min_corner.median().item() * 1e3,
                                     "p10": min_corner.quantile(0.1).item() * 1e3},
        "peak_lift_mm": {"median": peak_lift.median().item() * 1e3, "max": peak_lift.max().item() * 1e3},
        "palm_move_first_step_mm_max": first_step_palm.max().item() * 1e3,
        "survivor_palm_drift_mm": None if not survivors.any() else {
            "median": palm_drift[survivors].median().item() * 1e3, "max": palm_drift[survivors].max().item() * 1e3},
        "survivor_box_move_mm": None if not survivors.any() else {
            "median": box_move[survivors].median().item() * 1e3, "max": box_move[survivors].max().item() * 1e3},
    }


def main() -> int:
    cfg = parse_env_cfg(ARGS.task, device=ARGS.device, num_envs=ARGS.num_envs)
    if ARGS.robot_gravity != "cfg":
        cfg.scene.robot.spawn.rigid_props.disable_gravity = ARGS.robot_gravity == "off"
    if RECORD:
        cfg.scene.cam_side = camera_cfg(CAMERA_EYE, CAMERA_TARGET)
        cfg.scene.cam_left_wrist = g1_left_wrist_camera_cfg()
    cfg.episode_length_s = max(cfg.episode_length_s, (ARGS.steps + 5) * cfg.decimation * cfg.sim.dt)
    env = gym.make(ARGS.task, cfg=cfg).unwrapped
    try:
        print(f"[check] obs dims: {env.observation_manager.group_obs_dim}, "
              f"action dim: {env.action_manager.total_action_dim}, "
              f"robot gravity disabled: {cfg.scene.robot.spawn.rigid_props.disable_gravity}", flush=True)
        with torch.inference_mode():
            results = [run(env, mode) for mode in ARGS.modes]
        print(json.dumps(results, indent=2), flush=True)
        return 0
    finally:
        env.close()


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception:
        traceback.print_exc()
        raise
    finally:
        APP.close()
