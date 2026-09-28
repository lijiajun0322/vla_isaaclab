#!/usr/bin/env python3
"""RL grasp-and-lift, then a scripted place: move the box 2 cm, lower, release, retreat.

The trained policy runs until the held box passes 5 cm (the RL episode's end).
From then on a script takes over, per env:

  hold     keep the palm at its lift pose for a moment
  move     shift the box by SugarBoxPhaseStrategy.TARGET_WORLD_DELTA in the plane
  lower    bring the box back down to its start height
  release  open the fingers
  retreat  pull the palm back and up

Both parts share one speed limit (--palm-speed, --palm-rot-speed,
--finger-speed): the policy's actions are clipped to it, without retraining.
The palm is driven through the same relative DiffIK action as the policy, as a
pose tracker toward a moving palm target; the fingers keep a light fixed grip
(--grip-rad closing delta per step) until release. Placement is
judged with the original sugar-box task_success thresholds, computed here: this
RL task has no placement success termination, so the report does not claim one.
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
parser.add_argument("--task", default="VLA-YCBGraspLift-SugarBox-G1-v0")
parser.add_argument("--checkpoint", type=Path, required=True)
parser.add_argument("--num-envs", type=int, default=256)
parser.add_argument("--seed", type=int, default=7)
parser.add_argument("--grip-rad", type=float, default=0.005,
                    help="Grip during transport: closing delta per step (rad); force ~ hand stiffness x delta.")
# One speed limit for both the RL and the scripted part; RL actions are clipped to it.
parser.add_argument("--palm-speed", type=float, default=0.03, help="m/s")
parser.add_argument("--palm-rot-speed", type=float, default=0.15, help="rad/s")
parser.add_argument("--finger-speed", type=float, default=0.3, help="rad/s")
parser.add_argument("--video-dir", type=Path, default=None, help="Record env 0 (use --num-envs 1).")
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
from rsl_rl.runners import OnPolicyRunner

import vla_isaaclab  # noqa: F401
import vla_isaaclab.rl.isaaclab_rl_compat  # noqa: F401  Before any isaaclab_rl import.
from isaaclab.utils.math import compute_pose_error, quat_apply, quat_conjugate
from isaaclab_tasks.utils import load_cfg_from_registry, parse_env_cfg

from vla_isaaclab.envs.common import (
    LEFT_HAND_CLOSED_JOINT_POSITIONS,
    LEFT_HAND_JOINT_NAMES,
    LEFT_HAND_OPEN_JOINT_POSITIONS,
)
from vla_isaaclab.envs.ycb_grasp.env_cfg import VIDEO_CAMERAS, add_video_cameras
from vla_isaaclab.envs.ycb_grasp.mdp import grasp_rl
from vla_isaaclab.policies.ycb_sugar_box_strategy import SugarBoxPhaseStrategy
from vla_isaaclab.rl.action_clip import ClippedRslRlVecEnvWrapper


RL, HOLD, MOVE, LOWER, RELEASE, RETREAT, CHECK = range(7)
PHASE_NAMES = ("rl", "hold", "move", "lower", "release", "retreat", "check")
RETREAT_BACK_M, RETREAT_UP_M = 0.08, 0.06
# Seconds per scripted phase at the shared palm/finger speed limits.
MOVE_M = abs(SugarBoxPhaseStrategy.TARGET_WORLD_DELTA[0])
DURATION_S = {
    HOLD: 0.3,
    MOVE: MOVE_M / ARGS.palm_speed + 0.2,
    LOWER: 0.06 / ARGS.palm_speed,
    RELEASE: 1.0 / ARGS.finger_speed,
    RETREAT: math.hypot(RETREAT_BACK_M, RETREAT_UP_M) / ARGS.palm_speed + 0.2,
    CHECK: 1.5,
}


def open_videos():
    import av

    ARGS.video_dir.mkdir(parents=True, exist_ok=True)
    videos = {}
    for name in VIDEO_CAMERAS:
        container = av.open(str(ARGS.video_dir / f"{name}.mp4"), mode="w")
        stream = container.add_stream("libx264", rate=30)
        stream.width, stream.height, stream.pix_fmt = 640, 480, "yuv420p"
        videos[name] = (container, stream)
    return videos


def write_frames(env, videos):
    import av

    for name, (container, stream) in videos.items():
        rgb = env.scene.sensors[name].data.output["rgb"][0, ..., :3].detach().cpu().numpy().astype(np.uint8)
        for packet in stream.encode(av.VideoFrame.from_ndarray(rgb, format="rgb24")):
            container.mux(packet)


def close_videos(videos):
    for container, stream in videos.values():
        for packet in stream.encode():
            container.mux(packet)
        container.close()


def placement_ok(base, target_w, palm_body) -> torch.Tensor:
    """The original sugar-box task_success instantaneous thresholds."""
    box = base.scene["object"]
    robot = base.scene["robot"]
    return (
        (torch.linalg.vector_norm(box.data.root_pos_w[:, :2] - target_w[:, :2], dim=-1) < 0.015)
        & ((box.data.root_pos_w[:, 2] - target_w[:, 2]).abs() < 0.015)
        & (torch.linalg.vector_norm(box.data.root_lin_vel_w, dim=-1) < 0.04)
        & (torch.linalg.vector_norm(box.data.root_ang_vel_w, dim=-1) < 0.30)
        & (torch.linalg.vector_norm(robot.data.body_pos_w[:, palm_body] - box.data.root_pos_w, dim=-1) > 0.20)
    )


def main() -> int:
    env_cfg = parse_env_cfg(ARGS.task, device=ARGS.device, num_envs=ARGS.num_envs)
    env_cfg.seed = ARGS.seed
    # The script continues past the RL episode's lift termination.
    env_cfg.terminations.success = None
    env_cfg.terminations.lifted_off_pose = None
    env_cfg.episode_length_s = 4.0 + sum(DURATION_S.values())
    if RECORD:
        add_video_cameras(env_cfg)
    agent_cfg = load_cfg_from_registry(ARGS.task, "rsl_rl_cfg_entry_point")
    agent_cfg.device = ARGS.device
    env = ClippedRslRlVecEnvWrapper(gym.make(ARGS.task, cfg=env_cfg))
    base = env.unwrapped
    videos = open_videos() if RECORD else None
    try:
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
        runner.load(str(ARGS.checkpoint), load_optimizer=False)
        policy = runner.get_inference_policy(device=base.device)

        n, device, dt = base.num_envs, base.device, base.step_dt
        robot, box = base.scene["robot"], base.scene["object"]
        palm_body = robot.find_bodies([grasp_rl.PALM_BODY])[0][0]
        arm_dim = base.action_manager.get_term("arm").action_dim
        arm_scale = torch.tensor(base.cfg.actions.arm.scale, device=device)
        hand_term = base.action_manager.get_term("hand")
        hand_names = list(hand_term._joint_names)
        hand_ids = robot.find_joints(hand_names, preserve_order=True)[0]
        hand_scale = base.cfg.actions.hand.scale
        closed = dict(zip(LEFT_HAND_JOINT_NAMES, LEFT_HAND_CLOSED_JOINT_POSITIONS))
        opened = torch.tensor([dict(zip(LEFT_HAND_JOINT_NAMES, LEFT_HAND_OPEN_JOINT_POSITIONS))[j] for j in hand_names],
                              device=device)
        squeeze = torch.tensor([0.0 if j == "left_hand_thumb_0_joint" else math.copysign(1.0, closed[j])
                                for j in hand_names], device=device) * ARGS.grip_rad / hand_scale
        steps = {p: round(s / dt) for p, s in DURATION_S.items()}
        # Per-dimension action limit for the shared speeds (1.0 = the trained limit).
        limit = torch.cat((
            (ARGS.palm_speed * dt / arm_scale[:3]), (ARGS.palm_rot_speed * dt / arm_scale[3:]),
            torch.full((len(hand_names),), ARGS.finger_speed * dt / hand_scale, device=device),
        )).clamp(max=1.0)

        phase = torch.full((n,), RL, dtype=torch.long, device=device)
        phase_step = torch.zeros(n, dtype=torch.long, device=device)
        palm_lift_b = torch.zeros(n, 7, device=device)  # palm pose at the lift, base frame
        box_lift_w = torch.zeros(n, 3, device=device)
        start_w = torch.zeros(n, 3, device=device)
        target_w = torch.zeros(n, 3, device=device)
        retreat_b = torch.zeros(n, 3, device=device)
        lifted = torch.zeros(n, dtype=torch.bool, device=device)
        lift_step = torch.full((n,), -1, dtype=torch.long, device=device)
        placed = torch.zeros_like(lifted)
        streak = torch.zeros(n, dtype=torch.long, device=device)
        best_streak = torch.zeros_like(streak)
        fell = torch.zeros_like(lifted)
        final = {}
        delta = torch.tensor(SugarBoxPhaseStrategy.TARGET_WORLD_DELTA, device=device)
        with torch.inference_mode():
            base.reset()
            obs, _ = env.get_observations()
            start_w[:] = base.grasp_object_start_pos
            target_w[:] = start_w + delta
            root_inv = quat_conjugate(robot.data.root_quat_w)
            total = round(env_cfg.episode_length_s / dt) - 2
            for step in range(total):
                actions = policy(obs)
                scripted = phase > RL
                # Palm target: lift pose plus the commanded box displacement (world -> base frame).
                alpha = lambda p: (phase_step.float() / steps[p]).clamp(max=1.0).unsqueeze(-1)
                disp_w = torch.zeros(n, 3, device=device)
                plane = target_w - box_lift_w
                plane[:, 2] = 0.0
                down = torch.zeros(n, 3, device=device)
                down[:, 2] = target_w[:, 2] - box_lift_w[:, 2]
                disp_w = torch.where((phase == MOVE).unsqueeze(-1), alpha(MOVE) * plane, disp_w)
                disp_w = torch.where((phase >= LOWER).unsqueeze(-1), plane + (phase > LOWER).float().unsqueeze(-1) * down
                                     + (phase == LOWER).float().unsqueeze(-1) * alpha(LOWER) * down, disp_w)
                palm_target = palm_lift_b.clone()
                palm_target[:, :3] += quat_apply(root_inv, disp_w)
                palm_target[:, :3] += torch.where((phase == RETREAT).unsqueeze(-1), alpha(RETREAT) * retreat_b,
                                                  (phase == CHECK).float().unsqueeze(-1) * retreat_b)
                palm = grasp_rl.palm_pose_b(base)
                pos_err, rot_err = compute_pose_error(palm[:, :3], palm[:, 3:7], palm_target[:, :3], palm_target[:, 3:7],
                                                      rot_error_type="axis_angle")
                arm_action = (torch.cat((pos_err, rot_err), dim=-1) / arm_scale).clamp(-1.0, 1.0)
                # Fingers: light grip until release, then open toward the open pose.
                joint = robot.data.joint_pos[:, hand_ids]
                open_action = (opened - joint) / hand_scale
                hand_action = torch.where((phase >= RELEASE).unsqueeze(-1), open_action, squeeze.expand(n, -1))
                actions[scripted, :arm_dim] = arm_action[scripted]
                actions[scripted, arm_dim:] = hand_action[scripted]
                actions = torch.maximum(torch.minimum(actions, limit), -limit)
                obs, _, dones, _ = env.step(actions)
                if videos is not None:
                    write_frames(base, videos)
                fell |= dones.bool() & ~placed

                # Phase transitions.
                phase_step += scripted.long()
                fired, _ = grasp_rl.held_lift_event(base, 0.05)
                new = (phase == RL) & fired
                lifted |= new
                lift_step[new] = step
                palm_lift_b[new] = grasp_rl.palm_pose_b(base)[new]
                box_lift_w[new] = box.data.root_pos_w[new]
                # Retreat opposite the palm's approach axis, then up (base frame).
                back = quat_apply(palm_lift_b[:, 3:7], torch.tensor([[1.0, 0.0, 0.0]], device=device).expand(n, -1))
                back[:, 2] = 0.0
                back = -back / torch.linalg.vector_norm(back, dim=-1, keepdim=True).clamp_min(1e-6)
                retreat_b[new] = (RETREAT_BACK_M * back + torch.tensor([0.0, 0.0, RETREAT_UP_M], device=device))[new]
                phase[new] = HOLD
                phase_step[new] = 0
                for p in (HOLD, MOVE, LOWER, RELEASE, RETREAT):
                    advance = (phase == p) & (phase_step >= steps[p])
                    phase[advance] = p + 1
                    phase_step[advance] = 0
                checking = phase == CHECK
                streak = torch.where(checking & placement_ok(base, target_w, palm_body), streak + 1,
                                     torch.zeros_like(streak))
                placed |= streak >= 15
                best_streak = torch.maximum(best_streak, streak)
                if bool(((phase == CHECK) & (phase_step >= steps[CHECK]) | fell).all()):
                    break

            box_pos = box.data.root_pos_w
            xy_error = torch.linalg.vector_norm(box_pos[:, :2] - target_w[:, :2], dim=-1)
            height_error = (box_pos[:, 2] - target_w[:, 2]).abs()
            tilt = torch.rad2deg(grasp_rl.object_tilt(base))
            done = phase == CHECK
            q = lambda x: None if not bool(done.any()) else {k: round(x[done].quantile(k).item(), 4) for k in (0.1, 0.5, 0.9)}
            from vla_isaaclab.envs.common.mdp import POSE_OFFSET_ATTR

            offset = getattr(base, POSE_OFFSET_ATTR)[0]
            final = {
                "episodes": n,
                "seed": ARGS.seed,
                # Env 0's box pose relative to the calibrated one (from the pregrasp table row).
                "env0_box_offset": {"x_mm": offset[0].item() * 1e3, "y_mm": offset[1].item() * 1e3,
                                    "yaw_deg": math.degrees(offset[2].item())},
                "rl_lifted_fraction": lifted.float().mean().item(),
                "reached_check_phase": int(done.sum()),
                "fell_or_ended_early": int(fell.sum()),
                # Original sugar-box task_success thresholds held 15 steps, computed in this script.
                "placed_fraction": placed.float().mean().item(),
                "final_xy_error_m": q(xy_error),
                "final_height_error_m": q(height_error),
                "final_tilt_deg": q(tilt),
                "env0_final_checks": {
                    "box_speed_mps": torch.linalg.vector_norm(box.data.root_lin_vel_w[0]).item(),
                    "box_angular_speed_radps": torch.linalg.vector_norm(box.data.root_ang_vel_w[0]).item(),
                    "hand_distance_m": torch.linalg.vector_norm(
                        robot.data.body_pos_w[0, palm_body] - box.data.root_pos_w[0]).item(),
                    "longest_ok_streak_steps": int(best_streak[0]),
                },
                "env0": {"phase": PHASE_NAMES[int(phase[0])], "placed": bool(placed[0]),
                         "xy_error_mm": xy_error[0].item() * 1e3, "height_error_mm": height_error[0].item() * 1e3,
                         "tilt_deg": tilt[0].item()},
                "grip_rad_per_step": ARGS.grip_rad,
                "speed_limits": {"palm_mps": ARGS.palm_speed, "palm_radps": ARGS.palm_rot_speed,
                                 "finger_radps": ARGS.finger_speed},
                "rl_lift_step_median": None if not bool(lifted.any()) else lift_step[lifted].float().median().item(),
                "videos": None if videos is None else [str(ARGS.video_dir / f"{c}.mp4") for c in VIDEO_CAMERAS],
            }
        print(json.dumps(final, indent=2), flush=True)
        out = ARGS.checkpoint.with_name(f"{ARGS.checkpoint.stem}_handoff_seed{ARGS.seed}_n{ARGS.num_envs}_eval.json")
        out.write_text(json.dumps(final, indent=2) + "\n")
        return 0
    finally:
        if videos is not None:
            close_videos(videos)
        env.close()


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception:
        traceback.print_exc()
        raise
    finally:
        APP.close()
