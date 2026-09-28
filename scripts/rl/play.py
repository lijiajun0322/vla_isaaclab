#!/usr/bin/env python3
"""Evaluate a trained sugar-box grasp-and-lift policy, optionally with videos.

Every env runs exactly one episode from a random pregrasp-table state. With a
"success" termination (v1 task) that is the success signal; otherwise success
means the box was held at least --success-height-m above its start, grasped
(thumb plus index/middle contact) and tilted less than --success-tilt-deg, for
--success-hold-steps consecutive steps, without the drop termination firing.
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
parser.add_argument("--checkpoint", type=Path, required=True)
parser.add_argument("--num-envs", type=int, default=256)
parser.add_argument("--seed", type=int, default=7)
parser.add_argument("--success-height-m", type=float, default=0.06)
parser.add_argument("--success-tilt-deg", type=float, default=20.0)
parser.add_argument("--success-hold-steps", type=int, default=15)
parser.add_argument("--video-dir", type=Path, default=None,
                    help="Record env 0 from the side and left-wrist cameras into this directory.")
AppLauncher.add_app_launcher_args(parser)
ARGS = parser.parse_args()
# AppLauncher consumes enable_cameras, so keep our own flag.
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
from vla_isaaclab.rl.action_clip import ClippedRslRlVecEnvWrapper
from isaaclab_tasks.utils import load_cfg_from_registry, parse_env_cfg

from vla_isaaclab.envs.common import camera_cfg, g1_left_wrist_camera_cfg
from vla_isaaclab.envs.ycb_sugar_box.env_cfg import CAMERA_EYE, CAMERA_TARGET
from vla_isaaclab.envs.ycb_sugar_box.mdp import grasp_rl


VIDEO_CAMERAS = ("cam_side", "cam_left_wrist")


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


def main() -> int:
    env_cfg = parse_env_cfg(ARGS.task, device=ARGS.device, num_envs=ARGS.num_envs)
    env_cfg.seed = ARGS.seed
    if RECORD:
        env_cfg.scene.cam_side = camera_cfg(CAMERA_EYE, CAMERA_TARGET)
        env_cfg.scene.cam_left_wrist = g1_left_wrist_camera_cfg()
    agent_cfg = load_cfg_from_registry(ARGS.task, "rsl_rl_cfg_entry_point")
    agent_cfg.device = ARGS.device
    env = ClippedRslRlVecEnvWrapper(gym.make(ARGS.task, cfg=env_cfg))
    base = env.unwrapped
    videos = open_videos() if RECORD else None
    try:
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
        runner.load(str(ARGS.checkpoint), load_optimizer=False)
        policy = runner.get_inference_policy(device=base.device)

        n = base.num_envs
        alive = torch.ones(n, dtype=torch.bool, device=base.device)
        success = torch.zeros_like(alive)
        streak = torch.zeros(n, dtype=torch.long, device=base.device)
        grasped = torch.zeros_like(alive)
        peak_lift = torch.zeros(n, device=base.device)
        outcome = {name: torch.zeros_like(alive) for name in base.termination_manager.active_terms}
        max_tilt = math.radians(ARGS.success_tilt_deg)
        high_steps = contact_high = held_high = 0
        rel_speeds = []
        run = torch.zeros(base.num_envs, device=base.device)
        longest = torch.zeros(base.num_envs, device=base.device)
        with torch.inference_mode():
            base.reset()
            obs, _ = env.get_observations()
            for _ in range(base.max_episode_length):
                obs, _, dones, _ = env.step(policy(obs))
                if videos is not None and bool(alive[0]):
                    write_frames(base, videos)
                done = dones.bool()
                for name in outcome:
                    outcome[name] |= alive & done & base.termination_manager.get_term(name)
                # Terminal-step state is already reset; score only continuing steps.
                live = alive & ~done
                lift = grasp_rl.lift_height(base)
                holding = grasp_rl.grasp_flag(base) & (lift > ARGS.success_height_m) & (grasp_rl.box_tilt(base) < max_tilt)
                streak = torch.where(live & holding, streak + 1, torch.zeros_like(streak))
                success |= live & (streak >= ARGS.success_hold_steps)
                grasped |= live & grasp_rl.grasp_flag(base)
                high = live & (lift > 0.05)
                high_steps += int(high.sum())
                contact_high += int((high & grasp_rl.grasp_flag(base)).sum())
                held_high += int((high & grasp_rl.grasp_held(base)).sum())
                rel_speeds.append(grasp_rl.box_palm_relative_speed(base)[high & grasp_rl.grasp_flag(base)])
                run = torch.where(high & grasp_rl.grasp_flag(base), run + 1, torch.zeros_like(run))
                longest = torch.maximum(longest, run)
                peak_lift = torch.where(live, torch.maximum(peak_lift, lift), peak_lift)
                alive &= ~done
                if not bool(alive.any()):
                    break
        if "success" in outcome:
            # The environment's named success termination is the only success signal.
            success = outcome["success"].clone()
        elif "dropped" in outcome:
            success &= ~outcome["dropped"]
        report = {
            "checkpoint": str(ARGS.checkpoint),
            "episodes": n,
            "success_rate": success.float().mean().item(),
            "grasp_rate": grasped.float().mean().item(),
            "peak_lift_mm": {"median": peak_lift.median().item() * 1e3,
                             "p90": peak_lift.quantile(0.9).item() * 1e3},
            # Among steps with the box more than 5 cm up: raw thumb+finger contact vs held.
            "box_above_5cm_steps": high_steps,
            "above_5cm_fraction_contact": contact_high / max(high_steps, 1),
            "above_5cm_fraction_held": held_high / max(high_steps, 1),
            # Per env: longest run of steps with the box above 5 cm and thumb+finger contact.
            "longest_contact_above_5cm_steps": {q: longest.quantile(q).item() for q in (0.1, 0.25, 0.5, 0.75, 0.9)},
            "envs_with_run_ge": {k: (longest >= k).float().mean().item() for k in (3, 9, 15, 30)},
            "above_5cm_contact_rel_speed_mps_quantiles": (lambda v: {q: round(v.quantile(q).item(), 3) for q in (0.25, 0.5, 0.75, 0.9)}
                                                         if v.numel() else None)(torch.cat(rel_speeds)),
            "termination_rate": {name: flag.float().mean().item() for name, flag in outcome.items()},
            "success_source": "success termination" if "success" in outcome else "play.py hold rule",
            "success_definition": {"height_m": ARGS.success_height_m, "tilt_deg": ARGS.success_tilt_deg,
                                   "hold_steps": ARGS.success_hold_steps},
            "env0_success": bool(success[0].item()),
            "videos": None if videos is None else [str(ARGS.video_dir / f"{c}.mp4") for c in VIDEO_CAMERAS],
        }
        print(json.dumps(report, indent=2), flush=True)
        out = ARGS.checkpoint.with_name(ARGS.checkpoint.stem + "_eval.json")
        out.write_text(json.dumps(report, indent=2) + "\n")
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
