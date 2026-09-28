#!/usr/bin/env python3
"""Preview a finger preshape from open-hand pregrasp-table states, with videos.

Runs the grasp-lift task with two changes: resets from the object's open-hand
table (``build_pregrasp_table.sh --open-hand``) and absolute hand targets
(action 0 = open hand). The arm holds its pregrasp pose. The thumb rotates to face the fingers, then
the thumb, index and middle each close slowly until they touch the box and
back off a little. Env 0 is recorded from the side and left-wrist cameras;
the printed summary covers all envs.
"""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task", default="VLA-YCBGraspLift-SugarBox-G1-v0")
parser.add_argument("--num-envs", type=int, default=16)
parser.add_argument("--seed", type=int, default=7)
parser.add_argument("--close-rate", type=float, default=0.02, help="Finger closing speed, rad per step.")
parser.add_argument("--back-off", type=float, default=0.12, help="Joint back-off after contact, rad.")
parser.add_argument("--video-dir", type=Path, default=PROJECT_ROOT / "outputs/rl/videos/preshape_preview")
parser.add_argument("--no-video", action="store_true", help="Skip cameras (for many-env statistics).")
parser.add_argument("--fixed", type=float, nargs=3, default=None, metavar=("THUMB", "INDEX", "MIDDLE"),
                    help="Instead of closing until contact, ramp to this fixed curl (rad, magnitudes).")
AppLauncher.add_app_launcher_args(parser)
ARGS = parser.parse_args()
ARGS.enable_cameras = not ARGS.no_video
if ARGS.no_video:
    kit = "ycb.python.headless.kit" if ARGS.headless else "ycb.python.kit"
else:
    kit = "ycb.python.headless.rendering.kit" if ARGS.headless else "ycb.python.rendering.kit"
ARGS.experience = str(PROJECT_ROOT / "configs" / kit)
ARGS.kit_args = f"--portable-root {PROJECT_ROOT}/outputs/runtime/kit"
APP = AppLauncher(ARGS).app

import gymnasium as gym
import numpy as np
import torch

import vla_isaaclab  # noqa: F401
from isaaclab.envs.mdp.actions import JointPositionActionCfg
from isaaclab_tasks.utils import parse_env_cfg

from vla_isaaclab.envs.common import (
    LEFT_HAND_CLOSED_JOINT_POSITIONS,
    LEFT_HAND_JOINT_NAMES,
)
from vla_isaaclab.envs.ycb_grasp.env_cfg import VIDEO_CAMERAS, add_video_cameras
from vla_isaaclab.envs.ycb_grasp.mdp import grasp_rl
from vla_isaaclab.rl.pregrasp_table import pregrasp_table_path


CLOSED = dict(zip(LEFT_HAND_JOINT_NAMES, LEFT_HAND_CLOSED_JOINT_POSITIONS))
# Each finger closes as one unit and stops when any of its links touches the box.
FINGERS = {
    "thumb": (("left_hand_thumb_1_joint", "left_hand_thumb_2_joint"), ("object_contact_thumb_1", "object_contact_thumb_2")),
    "index": (("left_hand_index_0_joint", "left_hand_index_1_joint"), ("object_contact_index_0", "object_contact_index_1")),
    "middle": (("left_hand_middle_0_joint", "left_hand_middle_1_joint"), ("object_contact_middle_0", "object_contact_middle_1")),
}
HOLD_STEPS, ROTATE_STEPS, CLOSE_STEPS, SETTLE_STEPS = 20, 20, 130, 40


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


def tip_gaps_mm(env) -> torch.Tensor:
    """Fingertip-link origin distance to the object's collider box, (num_envs, 3) in mm."""
    return grasp_rl.fingertip_surface_gaps(env) * 1e3


def main() -> int:
    cfg = parse_env_cfg(ARGS.task, device=ARGS.device, num_envs=ARGS.num_envs)
    cfg.seed = ARGS.seed
    cfg.events.reset_to_pregrasp.params["table_path"] = str(pregrasp_table_path(cfg.object_spec, open_hand=True))
    cfg.actions.hand = JointPositionActionCfg(
        asset_name="robot", joint_names=list(LEFT_HAND_JOINT_NAMES), scale=1.0, use_default_offset=True
    )
    if not ARGS.no_video:
        add_video_cameras(cfg)
    total = HOLD_STEPS + ROTATE_STEPS + CLOSE_STEPS + SETTLE_STEPS
    cfg.episode_length_s = max(cfg.episode_length_s, (total + 5) * cfg.decimation * cfg.sim.dt)
    env = gym.make(ARGS.task, cfg=cfg).unwrapped
    videos = None if ARGS.no_video else open_videos()
    try:
        with torch.inference_mode():
            env.reset()
            hand = env.action_manager.get_term("hand")
            arm_dim = env.action_manager.get_term("arm").action_dim
            # Hand action order follows the USD joint order; map everything by name.
            names = list(hand._joint_names)
            col = {name: arm_dim + i for i, name in enumerate(names)}
            n, device = env.num_envs, env.device
            actions = torch.zeros(n, env.action_manager.total_action_dim, device=device)
            thumb0 = col["left_hand_thumb_0_joint"]
            closing = {f: torch.ones(n, dtype=torch.bool, device=device) for f in FINGERS}
            touched = {f: torch.zeros(n, dtype=torch.bool, device=device) for f in FINGERS}
            alive = torch.ones(n, dtype=torch.bool, device=device)
            start_gap = tip_gaps_mm(env)
            box = env.scene["object"]
            box_start = box.data.root_pos_w.clone()
            tilt_start = grasp_rl.object_tilt(env).clone()
            # Probe: curl (rad magnitude) at first touch; NaN if the finger reached its limit untouched.
            hit_curl = {f: torch.full((n,), float("nan"), device=device) for f in FINGERS}
            ever_contact = {f: torch.zeros(n, dtype=torch.bool, device=device) for f in FINGERS}
            box_move_max = torch.zeros(n, device=device)

            for step in range(total):
                phase_step = step - HOLD_STEPS
                if 0 <= phase_step < ROTATE_STEPS:
                    actions[:, thumb0] = CLOSED["left_hand_thumb_0_joint"] * (phase_step + 1) / ROTATE_STEPS
                elif ARGS.fixed is not None and step < HOLD_STEPS + ROTATE_STEPS + CLOSE_STEPS:
                    ramp = min(1.0, (step - HOLD_STEPS - ROTATE_STEPS + 1) / (CLOSE_STEPS / 2))
                    for curl, (joints, _) in zip(ARGS.fixed, FINGERS.values()):
                        for joint in joints:
                            sign = 1.0 if CLOSED[joint] > 0 else -1.0
                            actions[:, col[joint]] = sign * curl * max(ramp, 0.0)
                elif HOLD_STEPS + ROTATE_STEPS <= step < HOLD_STEPS + ROTATE_STEPS + CLOSE_STEPS:
                    forces = dict(zip(grasp_rl.HAND_CONTACT_SENSORS, grasp_rl.contact_forces(env).unbind(-1)))
                    for finger, (joints, sensors) in FINGERS.items():
                        hit = torch.stack([forces[s] > grasp_rl.CONTACT_THRESHOLD_N for s in sensors]).any(0)
                        backing = closing[finger] & hit
                        for joint in joints:
                            c, target = col[joint], CLOSED[joint]
                            sign = 1.0 if target > 0 else -1.0
                            moved = actions[:, c] + sign * ARGS.close_rate
                            moved = torch.minimum(moved, torch.full_like(moved, target)) if sign > 0 else \
                                torch.maximum(moved, torch.full_like(moved, target))
                            actions[:, c] = torch.where(closing[finger], moved, actions[:, c])
                            actions[:, c] = torch.where(backing, actions[:, c] - sign * ARGS.back_off, actions[:, c])
                        touched[finger] |= backing
                        hit_curl[finger] = torch.where(backing, actions[:, col[joints[0]]].abs() + ARGS.back_off,
                                                       hit_curl[finger])
                        closing[finger] &= ~hit
                env.step(actions)
                if videos is not None and bool(alive[0]):
                    write_frames(env, videos)
                forces = dict(zip(grasp_rl.HAND_CONTACT_SENSORS, grasp_rl.contact_forces(env).unbind(-1)))
                for finger, (_, sensors) in FINGERS.items():
                    ever_contact[finger] |= alive & torch.stack([forces[s] > grasp_rl.CONTACT_THRESHOLD_N
                                                                 for s in sensors]).any(0)
                moved = torch.linalg.vector_norm(box.data.root_pos_w - box_start, dim=-1)
                box_move_max = torch.where(alive, torch.maximum(box_move_max, moved), box_move_max)
                alive &= ~env.termination_manager.dones

            gap = tip_gaps_mm(env)
            joint_ids = [env.scene["robot"].joint_names.index(j) for j in names]
            quant = lambda x: {q: round(torch.nanquantile(x, q).item(), 3) for q in (0.05, 0.1, 0.25, 0.5)}
            report = {
                "envs": n,
                "mode": "probe" if ARGS.fixed is None else f"fixed {ARGS.fixed}",
                "probe_hit_curl_rad_quantiles": None if ARGS.fixed is not None else {
                    f: {**quant(c), "no_touch_fraction": c.isnan().float().mean().item()} for f, c in hit_curl.items()},
                "ever_contact_fraction": {f: c.float().mean().item() for f, c in ever_contact.items()},
                "box_move_mm": {"median": box_move_max.median().item() * 1e3,
                                "p95": box_move_max.quantile(0.95).item() * 1e3,
                                "max": box_move_max.max().item() * 1e3},
                "box_tilt_change_deg_max": torch.rad2deg((grasp_rl.object_tilt(env) - tilt_start).abs()).max().item(),
                "still_running": int(alive.sum().item()),
                "finger_touched_then_backed_off": {f: t.float().mean().item() for f, t in touched.items()},
                "tip_gap_mm_before": dict(zip(("thumb", "index", "middle"), start_gap.median(0).values.tolist())),
                "tip_gap_mm_after": dict(zip(("thumb", "index", "middle"), gap.median(0).values.tolist())),
                "object_contact_after_n": dict(zip(grasp_rl.HAND_CONTACT_SENSORS,
                                                grasp_rl.contact_forces(env)[0].tolist())),
                "env0_hand_joints_rad": dict(zip(names, env.scene["robot"].data.joint_pos[0, joint_ids].tolist())),
                "videos": None if videos is None else [str(ARGS.video_dir / f"{c}.mp4") for c in VIDEO_CAMERAS],
            }
            print(json.dumps(report, indent=2), flush=True)
            ARGS.video_dir.mkdir(parents=True, exist_ok=True)
            (ARGS.video_dir / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
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
