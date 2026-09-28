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
parser.add_argument("--hold-after-lift-s", type=float, default=None,
                    help="Do not end the episode at the lift; afterwards hold the palm at its lift pose (scripted "
                         "arm) for this long and report whether the box stays held.")
parser.add_argument("--hold-hand", choices=("policy", "squeeze"), default="policy",
                    help="During --hold-after-lift-s: hand from the policy, or a scripted full-rate squeeze.")
parser.add_argument("--squeeze-fraction", type=float, default=1.0,
                    help="Scripted squeeze strength as a fraction of the hand action limit (1.0 = 0.05 rad/step).")
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
from isaaclab.utils.math import compute_pose_error
import numpy as np
import torch
from rsl_rl.runners import OnPolicyRunner

import vla_isaaclab  # noqa: F401
import vla_isaaclab.rl.isaaclab_rl_compat  # noqa: F401  Before any isaaclab_rl import.
from vla_isaaclab.rl.action_clip import ClippedRslRlVecEnvWrapper
from isaaclab_tasks.utils import load_cfg_from_registry, parse_env_cfg

from vla_isaaclab.envs.common import (
    LEFT_HAND_CLOSED_JOINT_POSITIONS,
    LEFT_HAND_JOINT_NAMES,
    camera_cfg,
    g1_left_wrist_camera_cfg,
)
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


def hold_after_lift(env, base, policy, videos) -> int:
    """Run the policy to the held 5 cm lift, then hold the palm still and watch the box."""
    n, device = base.num_envs, base.device
    arm_dim = base.action_manager.get_term("arm").action_dim
    hold_steps = round(ARGS.hold_after_lift_s / base.step_dt)
    # Scripted squeeze: every curl joint toward its closed pose at the action limit (thumb base left alone).
    closed = dict(zip(LEFT_HAND_JOINT_NAMES, LEFT_HAND_CLOSED_JOINT_POSITIONS))
    squeeze = torch.tensor([0.0 if name == "left_hand_thumb_0_joint" else math.copysign(1.0, closed[name])
                            for name in base.action_manager.get_term("hand")._joint_names], device=device)
    alive = torch.ones(n, dtype=torch.bool, device=device)
    lift_step = torch.full((n,), -1, dtype=torch.long, device=device)
    lift_error = torch.full((n,), float("nan"), device=device)
    lift_height0 = torch.zeros(n, device=device)
    box_in_palm0 = torch.zeros(n, 3, device=device)
    palm_z0 = torch.zeros(n, device=device)
    palm_hold = torch.zeros(n, 7, device=device)  # palm pose (base frame) recorded at the lift
    arm_scale = torch.tensor(base.cfg.actions.arm.scale, device=device)
    max_palm_drop = torch.zeros(n, device=device)
    held_count = torch.zeros(n, device=device)
    min_height = torch.full((n,), float("inf"), device=device)
    max_error = torch.zeros(n, device=device)
    max_slip = torch.zeros(n, device=device)
    done_hold = torch.zeros(n, dtype=torch.bool, device=device)
    with torch.inference_mode():
        base.reset()
        obs, _ = env.get_observations()
        for step in range(base.max_episode_length):
            actions = policy(obs)
            holding = lift_step >= 0
            # Relative DiffIK with a zero action lets the loaded palm sag (its target follows
            # the measured pose), so pull the palm back to the pose recorded at the lift.
            palm = grasp_rl.palm_pose_b(base)
            pos_err, rot_err = compute_pose_error(palm[:, :3], palm[:, 3:7], palm_hold[:, :3], palm_hold[:, 3:7],
                                                  rot_error_type="axis_angle")
            hold_action = (torch.cat((pos_err, rot_err), dim=-1) / arm_scale).clamp(-1.0, 1.0)
            actions[holding, :arm_dim] = hold_action[holding]
            if ARGS.hold_hand == "squeeze":
                actions[holding, arm_dim:] = squeeze * ARGS.squeeze_fraction
            obs, _, dones, _ = env.step(actions)
            if videos is not None and bool(alive[0]) and not bool(done_hold[0]):
                write_frames(base, videos)
            done = dones.bool()
            live = alive & ~done
            fired, error = grasp_rl._held_lift_event(base, 0.05)
            new = live & (lift_step < 0) & fired
            lift_step[new] = step
            lift_error[new] = error[new]
            lift_height0[new] = grasp_rl.lift_height(base)[new]
            box_in_palm0[new] = grasp_rl.box_pose_in_palm(base)[new, :3]
            palm_z = grasp_rl.palm_pose_b(base)[:, 2]
            palm_z0[new] = palm_z[new]
            palm_hold[new] = grasp_rl.palm_pose_b(base)[new]
            watching = live & (lift_step >= 0) & ~new & ~done_hold
            held_count += (watching & grasp_rl.grasp_held(base)).float()
            min_height = torch.where(watching, torch.minimum(min_height, grasp_rl.lift_height(base)), min_height)
            max_error = torch.where(watching, torch.maximum(max_error, error), max_error)
            slip = torch.linalg.vector_norm(grasp_rl.box_pose_in_palm(base)[:, :3] - box_in_palm0, dim=-1)
            max_slip = torch.where(watching, torch.maximum(max_slip, slip), max_slip)
            max_palm_drop = torch.where(watching, torch.maximum(max_palm_drop, palm_z0 - palm_z), max_palm_drop)
            done_hold |= (lift_step >= 0) & (step - lift_step >= hold_steps)
            # Box fell off the table or episode ended before the hold finished.
            alive &= ~done
            if bool((~alive | done_hold).all()):
                break
    lifted = lift_step >= 0
    watched = lifted & done_hold
    dropped = watched & (min_height < 0.02)
    q = lambda x: None if not bool(watched.any()) else {k: round(x[watched].quantile(k).item(), 4) for k in (0.1, 0.5, 0.9)}
    report = {
        "checkpoint": str(ARGS.checkpoint),
        "episodes": n,
        "hold_after_lift_s": ARGS.hold_after_lift_s,
        "hold_hand": ARGS.hold_hand,
        "squeeze_fraction": ARGS.squeeze_fraction if ARGS.hold_hand == "squeeze" else None,
        "lifted_fraction": lifted.float().mean().item(),
        "lift_step_median": None if not bool(lifted.any()) else lift_step[lifted].float().median().item(),
        "lift_corner_error_mm": None if not bool(lifted.any()) else
            {k: round(lift_error[lifted].quantile(k).item() * 1e3, 1) for k in (0.1, 0.5, 0.9)},
        "watched_full_hold": int(watched.sum()),
        "lost_before_hold_ended": int((lifted & ~done_hold).sum()),
        # Over the hold: box back below 2 cm counts as dropped.
        "dropped_fraction": None if not bool(watched.any()) else dropped[watched].float().mean().item(),
        "held_fraction_of_hold_steps": None if not bool(watched.any()) else
            (held_count[watched] / hold_steps).mean().item(),
        "min_lift_height_m": q(min_height),
        "max_corner_error_m": q(max_error),
        "max_box_slip_in_palm_m": q(max_slip),
        "max_palm_drop_m": q(max_palm_drop),
        "env0": None if not bool(lifted[0]) else {
            "lift_step": int(lift_step[0]), "lift_corner_error_mm": lift_error[0].item() * 1e3,
            "min_lift_height_mm": min_height[0].item() * 1e3, "max_corner_error_mm": max_error[0].item() * 1e3,
            "max_slip_mm": max_slip[0].item() * 1e3},
        "videos": None if videos is None else [str(ARGS.video_dir / f"{c}.mp4") for c in VIDEO_CAMERAS],
    }
    print(json.dumps(report, indent=2), flush=True)
    out = ARGS.checkpoint.with_name(f"{ARGS.checkpoint.stem}_hold_{ARGS.hold_hand}"
                                  + (f"{ARGS.squeeze_fraction:g}" if ARGS.hold_hand == "squeeze" else "") + "_eval.json")
    out.write_text(json.dumps(report, indent=2) + "\n")
    return 0


def main() -> int:
    env_cfg = parse_env_cfg(ARGS.task, device=ARGS.device, num_envs=ARGS.num_envs)
    env_cfg.seed = ARGS.seed
    if RECORD:
        env_cfg.scene.cam_side = camera_cfg(CAMERA_EYE, CAMERA_TARGET)
        env_cfg.scene.cam_left_wrist = g1_left_wrist_camera_cfg()
    if ARGS.hold_after_lift_s is not None:
        env_cfg.terminations.success = None
        env_cfg.terminations.lifted_off_pose = None
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
        lift_error = torch.full((n,), float("nan"), device=base.device)
        rel_speeds = []
        run = torch.zeros(base.num_envs, device=base.device)
        longest = torch.zeros(base.num_envs, device=base.device)
        if ARGS.hold_after_lift_s is not None:
            return hold_after_lift(env, base, policy, videos)
        with torch.inference_mode():
            base.reset()
            obs, _ = env.get_observations()
            for _ in range(base.max_episode_length):
                obs, _, dones, _ = env.step(policy(obs))
                if videos is not None and bool(alive[0]):
                    write_frames(base, videos)
                done = dones.bool()
                if "lifted_off_pose" in outcome:
                    lifted_now = alive & done & (base.termination_manager.get_term("success")
                                                 | base.termination_manager.get_term("lifted_off_pose"))
                    lift_error[lifted_now] = getattr(base, grasp_rl.LIFT_CORNER_ERROR_ATTR)[lifted_now]
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
            # Episodes that ended by lifting the held box past 5 cm: corner error to the goal pose.
            "lifted_episodes": int((~lift_error.isnan()).sum()),
            "lift_corner_error_mm": None if bool(lift_error.isnan().all()) else {
                q: torch.nanquantile(lift_error, q).item() * 1e3 for q in (0.1, 0.5, 0.9)},
            "lift_score_mean": None if bool(lift_error.isnan().all()) else
                torch.exp(-lift_error[~lift_error.isnan()] / 0.03).mean().item() * 100,
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
