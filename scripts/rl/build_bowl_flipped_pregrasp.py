#!/usr/bin/env python3
"""Build start states for lifting the turned bowl (VLA-BowlFlippedLift-G1-v0).

In many parallel pivot envs the pivot policy turns the bowl upright. The moment
it is upright and still, each env switches to a scripted handoff: the right
palm moves up and away, and the left palm goes fast to a waypoint outside and
above the grasp-lift bowl pregrasp (turning the palm first at a point further
out and up, away from the torso), then down a straight diagonal from outside
onto the pregrasp, with the fingers at the bowl preshape. After a short hold
the state is kept if the bowl is upright, on the table and nearly still. The
hand may touch the bowl (the user's choice; other pregrasp tables exclude
contact). States are saved in the pregrasp-table format.
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
parser.add_argument("--pivot-checkpoint", type=Path, required=True)
parser.add_argument("--side", choices=("left", "right"), default="left",
                    help="Hand that goes to the pregrasp (the right one uses the left calibration mirrored); the other moves away.")
parser.add_argument("--num-envs", type=int, default=1024)
parser.add_argument("--target-rows", type=int, default=2000)
parser.add_argument("--max-rounds", type=int, default=6)
parser.add_argument("--outward-m", type=float, default=0.06, help="Waypoint offset from the pregrasp, away from the bowl axis.")
parser.add_argument("--above-m", type=float, default=0.06, help="Waypoint offset above the pregrasp.")
parser.add_argument("--free-out-m", type=float, default=0.08,
                    help="Free-space point, away from the body's midline from the waypoint, where the palm turns first.")
parser.add_argument("--free-up-m", type=float, default=0.10, help="Free-space point height above the waypoint.")
parser.add_argument("--transit-speedup", type=float, default=3.0, help="Fast move to the waypoint at this x the env speed limit.")
parser.add_argument("--descent-mps", type=float, default=0.045, help="Speed along the diagonal onto the pregrasp.")
parser.add_argument("--hold-steps", type=int, default=10)
parser.add_argument("--max-tilt-deg", type=float, default=15.0)
parser.add_argument("--max-lowest-m", type=float, default=0.01)
parser.add_argument("--max-speed-mps", type=float, default=0.05)
parser.add_argument("--max-angular-speed-radps", type=float, default=0.5)
parser.add_argument("--output", type=Path, default=None, help="Default: outputs/rl/024_bowl/bowl_flipped_pregrasp.pt")
AppLauncher.add_app_launcher_args(parser)
ARGS = parser.parse_args()
ARGS.enable_cameras = False
ARGS.experience = str(PROJECT_ROOT / "configs" / ("ycb.python.headless.kit" if ARGS.headless else "ycb.python.kit"))
ARGS.kit_args = f"--portable-root {PROJECT_ROOT}/outputs/runtime/kit"
APP = AppLauncher(ARGS).app

import gymnasium as gym
import torch

import vla_isaaclab  # noqa: F401  Register environments.
import vla_isaaclab.rl.isaaclab_rl_compat  # noqa: F401  Before any isaaclab_rl import.
from isaaclab.utils.math import compute_pose_error, matrix_from_quat, quat_from_matrix, subtract_frame_transforms
from isaaclab_tasks.utils import load_cfg_from_registry, parse_env_cfg
from rsl_rl.runners import OnPolicyRunner

from vla_isaaclab.envs.bowl_pivot import BOWL_PIVOT_HALF_SPEED_ENV_ID
from vla_isaaclab.envs.bowl_pivot.env_cfg import bowl_flipped_pregrasp_path
from vla_isaaclab.envs.bowl_pivot.mdp import pivot
from vla_isaaclab.envs.common import BOWL
from vla_isaaclab.rl.action_clip import ClippedRslRlVecEnvWrapper
from vla_isaaclab.rl.pregrasp_table import TABLE_FORMAT_VERSION, hand_preshape_joint_pos

FLIP, PREP, TRANSIT, DESCENT, HOLD, IDLE = range(6)
PREP_ERRORS = []  # (position, rotation) error of the palm when PREP timed out
FLIP_TIMEOUT_S, PREP_TIMEOUT_S, TRANSIT_TIMEOUT_S, DESCENT_TIMEOUT_S = 8.0, 4.0, 4.0, 5.0


def main() -> int:
    cfg = parse_env_cfg(BOWL_PIVOT_HALF_SPEED_ENV_ID, device=ARGS.device, num_envs=ARGS.num_envs)
    cfg.terminations.success = None  # the flip hands over instead of ending the episode
    cfg.rewards.success_bonus = None
    cfg.rewards.gentle_success_bonus = None
    cfg.episode_length_s = 30.0
    env = ClippedRslRlVecEnvWrapper(gym.make(BOWL_PIVOT_HALF_SPEED_ENV_ID, cfg=cfg))
    env.action_clip = ARGS.transit_speedup  # the policy's actions are clipped to [-1, 1] below
    base = env.unwrapped
    try:
        runner = OnPolicyRunner(env, load_cfg_from_registry(BOWL_PIVOT_HALF_SPEED_ENV_ID, "rsl_rl_cfg_entry_point").to_dict(),
                                log_dir=None, device=base.device)
        runner.load(str(ARGS.pivot_checkpoint), load_optimizer=False)
        policy = runner.get_inference_policy(device=base.device)
        robot, bowl = base.scene["robot"], base.scene["object"]
        n, dev, dt = base.num_envs, base.device, base.step_dt
        am = base.action_manager
        arm_scale = torch.tensor(cfg.actions.left_arm.scale, device=dev)
        hand_step = cfg.actions.left_hand.scale
        palm = robot.find_bodies(["left_hand_palm_link", "right_hand_palm_link"], preserve_order=True)[0]
        # G: the grasping side, O: the other; action layout left_arm | right_arm | left_hand | right_hand.
        side = ARGS.side
        other = "right" if side == "left" else "left"
        G, O = (0, 1) if side == "left" else (1, 0)
        arm_slice = {0: slice(0, 6), 1: slice(6, 12)}
        hand_slice = {0: slice(12, 19), 1: slice(19, 26)}
        lh_names = am.get_term(f"{side}_hand")._joint_names
        lh_ids = robot.find_joints(lh_names, preserve_order=True)[0]
        rh_ids = robot.find_joints(am.get_term(f"{other}_hand")._joint_names, preserve_order=True)[0]
        preshape = hand_preshape_joint_pos(**BOWL.hand_preshape, side=side)
        lh_target = torch.tensor([[preshape.get(name, 0.0) for name in lh_names]], device=dev)
        grasp_quat_w = torch.tensor([BOWL.grasp_quat_wxyz], device=dev).expand(n, -1)
        grasp_offset = torch.tensor([BOWL.grasp_offset_world], device=dev)
        if side == "right":
            # Mirror the left calibration through the robot's sagittal plane (world x = 0). The right
            # palm frame is the left one mirrored with its y axis flipped (thumb on +y instead of -y):
            # R_right = M R_left diag(1, -1, 1), M = diag(-1, 1, 1).
            grasp_offset = grasp_offset * grasp_offset.new_tensor([-1.0, 1.0, 1.0])
            mirror = torch.diag(torch.tensor([-1.0, 1.0, 1.0], device=dev))
            flip_y = torch.diag(torch.tensor([1.0, -1.0, 1.0], device=dev))
            grasp_quat_w = quat_from_matrix(mirror @ matrix_from_quat(grasp_quat_w[:1]) @ flip_y).expand(n, -1)
        retreat = torch.tensor([[0.10 if other == "right" else -0.10, -0.06, 0.10]], device=dev)

        def to_base(pos_w, quat_w):
            return subtract_frame_transforms(robot.data.root_pos_w, robot.data.root_quat_w, pos_w, quat_w)

        def palm_base(i):
            return to_base(robot.data.body_pos_w[:, palm[i]], robot.data.body_quat_w[:, palm[i]])

        def arm_cmd(i, pos_b, quat_b, limit):
            pos, quat = palm_base(i)
            dp, dr = compute_pose_error(pos, quat, pos_b, quat_b, rot_error_type="axis_angle")
            cmd = (torch.cat((dp, dr), -1) / arm_scale).clamp(-limit, limit)
            return cmd, torch.linalg.vector_norm(dp, dim=-1), torch.linalg.vector_norm(dr, dim=-1)

        rows = {k: [] for k in ("joint_pos", "joint_vel", "joint_target", "box_state")}
        counts = {"flipped": 0, "flip_timeout": 0, "prep_timeout": 0, "transit_timeout": 0, "descent_timeout": 0, "rejected_after_hold": 0,
                  "terminated": 0, "kept": 0, "kept_with_contact": 0}
        kept = 0
        for round_index in range(ARGS.max_rounds):
            with torch.inference_mode():  # per-env buffers created in inference mode are reset in place
                base.reset()
                obs, _ = env.get_observations()
            phase = torch.full((n,), FLIP, dtype=torch.long, device=dev)
            t_phase = torch.zeros(n, device=dev)
            t = torch.zeros(n, device=dev)
            g_pos = torch.zeros(n, 3, device=dev); g_quat = torch.zeros(n, 4, device=dev)
            w_pos = torch.zeros(n, 3, device=dev)
            f_pos = torch.zeros(n, 3, device=dev)
            r_pos = torch.zeros(n, 3, device=dev); r_quat = torch.zeros(n, 4, device=dev)
            hold_contact = torch.zeros(n, device=dev)
            with torch.inference_mode():
                for _ in range(round(30.0 / dt)):
                    act = policy(obs).clamp(-1.0, 1.0)
                    scripted = phase != FLIP
                    if bool(scripted.any()):
                        s_act = torch.zeros_like(act)
                        # Grasping palm: fast to the waypoint, then a setpoint sliding down the diagonal.
                        frac = ((t_phase * dt * ARGS.descent_mps) / torch.linalg.vector_norm(g_pos - w_pos, dim=-1).clamp_min(1e-6)).clamp(max=1.0)
                        setpoint = torch.where((phase == TRANSIT).unsqueeze(-1), w_pos, w_pos + frac.unsqueeze(-1) * (g_pos - w_pos))
                        setpoint = torch.where((phase == PREP).unsqueeze(-1), f_pos, setpoint)
                        fast = (phase == PREP) | (phase == TRANSIT)
                        limit = torch.where(fast, ARGS.transit_speedup, 1.0).unsqueeze(-1)
                        cmd, _, _ = arm_cmd(G, setpoint, g_quat, ARGS.transit_speedup)
                        s_act[:, arm_slice[G]] = torch.maximum(torch.minimum(cmd, limit), -limit)
                        cmd_r, _, _ = arm_cmd(O, r_pos, r_quat, ARGS.transit_speedup)
                        s_act[:, arm_slice[O]] = cmd_r
                        s_act[:, hand_slice[G]] = ((lh_target - robot.data.joint_pos[:, lh_ids]) / hand_step).clamp(-1, 1)
                        s_act[:, hand_slice[O]] = ((0.0 - robot.data.joint_pos[:, rh_ids]) / hand_step).clamp(-1, 1)
                        s_act[phase == HOLD, arm_slice[G]] = 0.0
                        s_act[phase == HOLD, hand_slice[G]] = 0.0
                        s_act[phase == IDLE] = 0.0
                        act = torch.where(scripted.unsqueeze(-1), s_act, act)
                    obs, _, dones, _ = env.step(act)
                    t += 1; t_phase += 1
                    done = dones.bool()
                    # An env that ends restarts from a fresh episode (counted, then flips again).
                    counts["terminated"] += int((done & (phase != IDLE)).sum())
                    phase[done] = FLIP; t_phase[done] = 0; t[done] = 0

                    # FLIP -> TRANSIT: the bowl is upright and still.
                    flipped = (phase == FLIP) & pivot.pivot_success(base) & ~done
                    if bool(flipped.any()):
                        counts["flipped"] += int(flipped.sum())
                        g_w = bowl.data.root_pos_w + grasp_offset
                        out = grasp_offset[:, :2] / torch.linalg.vector_norm(grasp_offset[:, :2], dim=-1, keepdim=True)
                        w_w = g_w.clone(); w_w[:, :2] += ARGS.outward_m * out; w_w[:, 2] += ARGS.above_m
                        gp, gq = to_base(g_w, grasp_quat_w)
                        wp, _ = to_base(w_w, grasp_quat_w)
                        # Turn the palm first out here, away from the torso: DiffIK from the post-flip
                        # arm pose otherwise stalls ~1 rad short, with the wrist against the torso.
                        f_w = w_w.clone(); f_w[:, 0] += (ARGS.free_out_m if side == "right" else -ARGS.free_out_m)
                        f_w[:, 2] += ARGS.free_up_m
                        fp, _ = to_base(f_w, grasp_quat_w)
                        rp_w = robot.data.body_pos_w[:, palm[O]] + retreat
                        rp, rq = to_base(rp_w, robot.data.body_quat_w[:, palm[O]])
                        for dst, src in ((g_pos, gp), (g_quat, gq), (w_pos, wp), (f_pos, fp), (r_pos, rp), (r_quat, rq)):
                            dst[flipped] = src[flipped]
                        phase[flipped] = PREP; t_phase[flipped] = 0
                    counts["flip_timeout"] += int(((phase == FLIP) & (t > FLIP_TIMEOUT_S / dt)).sum())
                    phase[(phase == FLIP) & (t > FLIP_TIMEOUT_S / dt)] = IDLE

                    # PREP -> TRANSIT once the palm is turned at the free-space point.
                    _, dp_f, dr_f = arm_cmd(G, f_pos, g_quat, 1.0)
                    turned = (phase == PREP) & (dp_f < 0.02) & (dr_f < 0.1)
                    phase[turned] = TRANSIT; t_phase[turned] = 0
                    late = (phase == PREP) & (t_phase > PREP_TIMEOUT_S / dt)
                    counts["prep_timeout"] += int(late.sum()); phase[late] = IDLE
                    if bool(late.any()):
                        PREP_ERRORS.extend(zip(dp_f[late].tolist(), dr_f[late].tolist()))

                    # TRANSIT -> DESCENT at the waypoint.
                    _, dp_w, dr_w = arm_cmd(G, w_pos, g_quat, 1.0)
                    at_w = (phase == TRANSIT) & (dp_w < 0.01) & (dr_w < 0.1)
                    phase[at_w] = DESCENT; t_phase[at_w] = 0
                    late = (phase == TRANSIT) & (t_phase > TRANSIT_TIMEOUT_S / dt)
                    counts["transit_timeout"] += int(late.sum()); phase[late] = IDLE

                    # DESCENT -> HOLD at the pregrasp with the fingers shaped.
                    _, dp_g, dr_g = arm_cmd(G, g_pos, g_quat, 1.0)
                    fingers = (lh_target - robot.data.joint_pos[:, lh_ids]).abs().amax(dim=-1)
                    at_g = (phase == DESCENT) & (dp_g < 0.005) & (dr_g < 0.05) & (fingers < 0.03)
                    phase[at_g] = HOLD; t_phase[at_g] = 0; hold_contact[at_g] = 0.0
                    late = (phase == DESCENT) & (t_phase > DESCENT_TIMEOUT_S / dt)
                    counts["descent_timeout"] += int(late.sum()); phase[late] = IDLE

                    # HOLD: keep the state if the bowl is upright, on the table and nearly still.
                    holding = phase == HOLD
                    touch = (pivot.hand_contact_forces(base, side).amax(-1) > pivot.CONTACT_THRESHOLD_N)
                    hold_contact = torch.where(holding, torch.maximum(hold_contact, touch.float()), hold_contact)
                    ready = holding & (t_phase >= ARGS.hold_steps)
                    if bool(ready.any()):
                        ok = (ready & (pivot.bowl_tilt(base) < math.radians(ARGS.max_tilt_deg))
                              & (pivot.bowl_lowest_height(base) < ARGS.max_lowest_m)
                              & (torch.linalg.vector_norm(bowl.data.root_lin_vel_w, dim=-1) < ARGS.max_speed_mps)
                              & (torch.linalg.vector_norm(bowl.data.root_ang_vel_w, dim=-1) < ARGS.max_angular_speed_radps))
                        counts["rejected_after_hold"] += int((ready & ~ok).sum())
                        if bool(ok.any()):
                            state = bowl.data.root_state_w.clone(); state[:, :3] -= base.scene.env_origins
                            for key, value in (("joint_pos", robot.data.joint_pos), ("joint_vel", robot.data.joint_vel),
                                               ("joint_target", robot.data.joint_pos_target), ("box_state", state)):
                                rows[key].append(value[ok].clone().cpu())
                            counts["kept"] += int(ok.sum()); counts["kept_with_contact"] += int((ok & (hold_contact > 0)).sum())
                            kept += int(ok.sum())
                        phase[ready] = IDLE
                    if bool((phase == IDLE).all()):
                        break
            print(f"[flipped-pregrasp] round {round_index}: kept {kept} so far, counts {counts}", flush=True)
            if PREP_ERRORS:
                e = torch.tensor(PREP_ERRORS)
                print(f"[flipped-pregrasp] PREP timeouts: position error median {e[:, 0].median():.3f} m, "
                      f"rotation error median {e[:, 1].median():.2f} rad", flush=True)
            if kept >= ARGS.target_rows:
                break

        output = ARGS.output or bowl_flipped_pregrasp_path(cfg.object_spec, ARGS.side)
        output.parent.mkdir(parents=True, exist_ok=True)
        table = {k: torch.cat(v)[:ARGS.target_rows] for k, v in rows.items() if v}
        count = len(table["joint_pos"]) if table else 0
        summary = {"object": cfg.object_spec.name, "side": ARGS.side, "rows": count, "pivot_checkpoint": str(ARGS.pivot_checkpoint),
                   "counts": counts, "hand_contact_allowed": True, "args": {k: str(v) for k, v in vars(ARGS).items()}}
        print("[flipped-pregrasp] " + json.dumps(summary), flush=True)
        output.with_suffix(".json").write_text(json.dumps(summary, indent=2))
        if count == 0:
            print("[flipped-pregrasp] no state kept, nothing saved", flush=True)
            return 1
        torch.save({
            "format_version": TABLE_FORMAT_VERSION,
            "object_name": cfg.object_spec.name,
            "joint_names": list(robot.joint_names),
            **table,
            "box_offset": torch.zeros(count, 3),
            "backoff_m": torch.zeros(count),
            "valid": torch.ones(count, dtype=torch.bool),
            "meta": summary,
        }, output)
        print(f"[flipped-pregrasp] saved {output}", flush=True)
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
