#!/usr/bin/env python3
"""Run the scripted sugar-box policy up to its pregrasp pose and record a video."""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from isaaclab.app import AppLauncher


# The scripted policy leaves "move_pregrasp" for "close_gripper" once the palm
# has settled on the pregrasp pose; stop before any finger closes.
STOP_PHASE = "close_gripper"
# A pregrasp only counts while the box is still standing where it was.
MAX_BOX_TILT_DEG = 10.0


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", default="VLA-YCBSugarBox-G1-JointPos-DR-v0", help="Registered sugar-box Gym ID.")
    parser.add_argument("--seed", type=int, default=42, help="Reset seed (selects the randomized box pose).")
    parser.add_argument("--max-steps", type=int, default=600, help="Give up if the pregrasp is not reached.")
    parser.add_argument("--camera", default="cam_side", help="Scene camera for the video, e.g. cam_left_wrist.")
    parser.add_argument(
        "--video", type=Path, default=PROJECT_ROOT / "outputs/videos/pregrasp/sugarbox_pregrasp.mp4"
    )
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    args.enable_cameras = True
    return args


ARGS = parse_args()
ARGS.experience = str(
    PROJECT_ROOT
    / "configs"
    / ("ycb.python.headless.rendering.kit" if ARGS.headless else "ycb.python.rendering.kit")
)
ARGS.kit_args = f"--portable-root {PROJECT_ROOT}/outputs/runtime/kit"
APP = AppLauncher(ARGS).app

import av
import gymnasium as gym
import numpy as np
import torch

import vla_isaaclab  # noqa: F401  Register environments.
from isaaclab.utils.math import quat_error_magnitude
from isaaclab_tasks.utils import parse_env_cfg

from vla_isaaclab.policies import YCBSugarBoxScriptedPolicy


def main() -> int:
    cfg = parse_env_cfg(ARGS.task, device=ARGS.device, num_envs=1)
    cfg.episode_length_s = (ARGS.max_steps + 1) * cfg.decimation * cfg.sim.dt
    env = gym.make(ARGS.task, cfg=cfg).unwrapped
    video = ARGS.video.resolve()
    video.parent.mkdir(parents=True, exist_ok=True)
    container = av.open(str(video), mode="w")
    stream = container.add_stream("libx264", rate=30)
    stream.width = 640
    stream.height = 480
    stream.pix_fmt = "yuv420p"
    try:
        env.reset(seed=ARGS.seed)
        if ARGS.camera not in env.scene.sensors:
            raise ValueError(f"Unknown camera {ARGS.camera!r}. Available: {sorted(env.scene.sensors)}")
        camera = env.scene.sensors[ARGS.camera]
        policy = YCBSugarBoxScriptedPolicy(env)
        strategy = policy.strategy
        origin = env.scene.env_origins[0]
        box_start = (env.scene["object"].data.root_pos_w[0] - origin).tolist()
        steps = 0
        terminations = {}
        box_tilt_deg = 0.0
        with torch.inference_mode():
            while APP.is_running() and steps < ARGS.max_steps:
                action = policy.compute(steps)
                _, _, terminated, timed_out, _ = env.step(action)
                rgb = camera.data.output["rgb"][0, ..., :3].detach().cpu().numpy()
                frame = av.VideoFrame.from_ndarray(rgb.astype(np.uint8), format="rgb24")
                for packet in stream.encode(frame):
                    container.mux(packet)
                steps += 1
                box_tilt_deg = float(np.degrees(strategy._box_tilt()[0].item()))
                if bool(terminated[0].item()) or bool(timed_out[0].item()):
                    # The environment auto-resets on termination; keep what ended it.
                    terminations = {
                        name: bool(env.termination_manager.get_term(name)[0].item())
                        for name in env.termination_manager.active_terms
                    }
                    break
                if policy.phase == STOP_PHASE or policy.failed:
                    break

            palm_pos, palm_quat = strategy._palm_pose()
            reached = (
                policy.phase == STOP_PHASE and not policy.failed and not terminations
                and box_tilt_deg < MAX_BOX_TILT_DEG
            )
            report = {
                "task": ARGS.task,
                "seed": ARGS.seed,
                "reached_pregrasp": reached,
                "steps": steps,
                "phase": policy.phase,
                "failure_reason": strategy.failure_reason,
                "terminations": terminations,
                "sugar_box_tilt_deg": box_tilt_deg,
                "pregrasp_target_m": None
                if strategy.pregrasp_pos is None
                else (strategy.pregrasp_pos[0] - origin).tolist(),
                "palm_position_m": (palm_pos[0] - origin).tolist(),
                "palm_position_error_m": None
                if strategy.pregrasp_pos is None
                else torch.linalg.vector_norm(palm_pos[0] - strategy.pregrasp_pos[0]).item(),
                "palm_orientation_error_deg": None
                if strategy.grasp_quat is None
                else float(np.degrees(quat_error_magnitude(palm_quat, strategy.grasp_quat)[0].item())),
                "sugar_box_start_m": box_start,
                "sugar_box_now_m": (env.scene["object"].data.root_pos_w[0] - origin).tolist(),
                "phase_history": [{"phase": h["phase"], "steps": h["steps"]} for h in strategy.phase_history],
                "camera": ARGS.camera,
                "video": str(video),
            }
        output_dir = PROJECT_ROOT / "outputs/environments" / ARGS.task
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "pregrasp.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report, indent=2), flush=True)
        return 0 if reached else 2
    finally:
        for packet in stream.encode():
            container.mux(packet)
        container.close()
        env.close()


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception:
        traceback.print_exc()
        raise
    finally:
        APP.close()
