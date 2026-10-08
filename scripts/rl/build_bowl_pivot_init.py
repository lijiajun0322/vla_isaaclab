#!/usr/bin/env python3
"""Build the fixed start state of the bimanual bowl-pivoting task.

The upside-down bowl first settles alone on the table. Both hands are set to the
env cfg's pre-contact shape (palm tilt, height, finger curl), and the arms are
solved kinematically onto palm poses beside the bowl at a range of lateral
distances, one distance per parallel env. Each candidate is held still for a
moment; it passes if the IK is exact, the bowl did not move and no hand or arm
link touched anything. The hands go to the closest passing distance plus the
cfg's clearance. That state is saved in the pregrasp-table format (one row), so
the task resets with the grasp-lift ``reset_from_pregrasp_table``.

A second stage then keeps those hands and moves the bowl to random offsets in
the cfg's bowl_init_{x,y}_range_m, holds each one still, and saves the offsets
where the bowl rests untouched as bowl_pivot_init_random.pt (--random-rows).
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
parser.add_argument("--bowl-settle-s", type=float, default=2.0, help="Let the bowl come to rest alone first.")
parser.add_argument("--hold-steps", type=int, default=30, help="Control steps of the static hold check.")
parser.add_argument("--ik-iterations", type=int, default=200)
parser.add_argument("--lateral-range-m", type=float, nargs=2, default=(0.09, 0.22),
                    help="Palm distance from the bowl axis (world x) swept, one candidate per env.")
parser.add_argument("--lateral-step-m", type=float, default=0.0025)
# Acceptance thresholds.
parser.add_argument("--max-ik-position-m", type=float, default=0.005)
parser.add_argument("--max-ik-rotation-deg", type=float, default=5.0)
parser.add_argument("--min-joint-margin-rad", type=float, default=0.05)
parser.add_argument("--max-bowl-displacement-m", type=float, default=0.002)
parser.add_argument("--max-bowl-tilt-deg", type=float, default=1.0)
parser.add_argument("--max-contact-force-n", type=float, default=1.0,
                    help="Any hand link on the bowl, or any arm/hand link on anything.")
parser.add_argument("--max-settled-palm-m", type=float, default=0.01)
parser.add_argument("--output", type=Path, default=None, help="Default: outputs/rl/024_bowl/bowl_pivot_init.pt")
parser.add_argument("--random-rows", type=int, default=2000,
                    help="Accepted randomized bowl starts to collect (0: skip the randomized table).")
parser.add_argument("--max-random-batches", type=int, default=100)
parser.add_argument("--seed", type=int, default=0)
AppLauncher.add_app_launcher_args(parser)
ARGS = parser.parse_args()
ARGS.enable_cameras = False
ARGS.experience = str(PROJECT_ROOT / "configs" / ("ycb.python.headless.kit" if ARGS.headless else "ycb.python.kit"))
ARGS.kit_args = f"--portable-root {PROJECT_ROOT}/outputs/runtime/kit"
APP = AppLauncher(ARGS).app

import torch

import vla_isaaclab  # noqa: F401  Register environments.
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.utils.math import compute_pose_error, quat_from_angle_axis, quat_mul

from vla_isaaclab.envs.bowl_pivot.env_cfg import BowlPivotEnvCfg, bowl_pivot_init_path
from vla_isaaclab.envs.common import (
    LEFT_ARM_JOINT_NAMES,
    LEFT_END_EFFECTOR,
    RIGHT_ARM_JOINT_NAMES,
    RIGHT_END_EFFECTOR,
    object_up_axis,
)
from vla_isaaclab.rl.pregrasp_table import (
    TABLE_FORMAT_VERSION,
    ArmKinematics,
    hand_box_contact_forces,
    hand_preshape_joint_pos,
)


ARM_CONTACT_SENSORS = ("left_arm_contacts", "right_arm_contacts")
# Palm +x (fingers) along world +Y, palm +z up: the left palm (-y) faces world
# +x and the right palm (+y) faces world -x, i.e. both face the bowl between them.
UPRIGHT_PALM_QUAT = (0.70710678, 0.0, 0.0, 0.70710678)
# Which side of the bowl each hand is on (world x).
SIDE_SIGN = {"left": -1.0, "right": 1.0}


def step_physics(env, control_steps: int, on_step=None):
    for _ in range(control_steps * env.cfg.decimation):
        env.scene.write_data_to_sim()
        env.sim.step(render=False)
        env.scene.update(dt=env.physics_dt)
        if on_step is not None:
            on_step()


def palm_quat(cfg, side: str, count: int, device) -> torch.Tensor:
    """Upright palm turned about world +Y so that its face points down toward the bowl by the tilt."""
    base = torch.tensor([UPRIGHT_PALM_QUAT], device=device).repeat(count, 1)
    y_axis = torch.tensor([[0.0, 1.0, 0.0]], device=device).repeat(count, 1)
    angle = torch.full((count,), -SIDE_SIGN[side] * math.radians(cfg.precontact_palm_tilt_deg), device=device)
    return quat_mul(quat_from_angle_axis(angle, y_axis), base)


def random_table(env, cfg, q_row, bowl_state_local, output: Path):
    """Fixed hands, bowl at random offsets; keep the offsets where it rests untouched."""
    robot, bowl = env.scene["robot"], env.scene["object"]
    count, device = env.num_envs, env.device
    q = q_row.to(device).expand(count, -1).clone()
    ranges = torch.tensor((cfg.bowl_init_x_range_m, cfg.bowl_init_y_range_m), device=device)
    torch.manual_seed(ARGS.seed)
    kept = {"joint_pos": [], "joint_vel": [], "box_state": [], "box_offset": []}
    tried = accepted = 0
    for batch in range(ARGS.max_random_batches):
        offset = torch.zeros(count, 3, device=device)
        offset[:, :2] = ranges[:, 0] + torch.rand(count, 2, device=device) * (ranges[:, 1] - ranges[:, 0])
        start = bowl_state_local[:, :7].to(device).expand(count, -1).clone()
        start[:, :3] += env.scene.env_origins + offset
        robot.write_joint_state_to_sim(q, torch.zeros_like(q))
        robot.set_joint_position_target(q)
        bowl.write_root_pose_to_sim(start)
        bowl.write_root_velocity_to_sim(torch.zeros(count, 6, device=device))
        for sensor in env.scene.sensors.values():
            sensor.reset()
        peak = {"hand": torch.zeros(count, device=device), "arm": torch.zeros(count, device=device)}

        def track():
            peak["hand"] = torch.maximum(peak["hand"], hand_box_contact_forces(env).amax(dim=-1))
            for name in ARM_CONTACT_SENSORS:
                peak["arm"] = torch.maximum(peak["arm"], torch.linalg.vector_norm(
                    env.scene.sensors[name].data.net_forces_w, dim=-1).amax(dim=-1))

        step_physics(env, ARGS.hold_steps, track)
        up_start = object_up_axis(cfg.object_spec, start[:, 3:7])
        up_end = object_up_axis(cfg.object_spec, bowl.data.root_quat_w)
        ok = ((torch.linalg.vector_norm(bowl.data.root_pos_w - start[:, :3], dim=-1) <= ARGS.max_bowl_displacement_m)
              & (torch.acos((up_start * up_end).sum(-1).clamp(-1.0, 1.0)) <= math.radians(ARGS.max_bowl_tilt_deg))
              & (peak["hand"] <= ARGS.max_contact_force_n) & (peak["arm"] <= ARGS.max_contact_force_n))
        state = bowl.data.root_state_w.clone()
        state[:, :3] -= env.scene.env_origins
        for key, value in (("joint_pos", robot.data.joint_pos), ("joint_vel", robot.data.joint_vel),
                           ("box_state", state), ("box_offset", offset)):
            kept[key].append(value[ok].clone().cpu())
        tried += count
        accepted += int(ok.sum())
        if accepted >= ARGS.random_rows:
            break
    table = {key: torch.cat(values)[:ARGS.random_rows] for key, values in kept.items()}
    n = len(table["joint_pos"])
    off = table["box_offset"]
    summary = {
        "object": cfg.object_spec.name, "rows": n, "tried": tried, "acceptance_rate": accepted / max(tried, 1),
        "x_range_m": list(cfg.bowl_init_x_range_m), "y_range_m": list(cfg.bowl_init_y_range_m),
        "accepted_x_m": [round(off[:, 0].min().item(), 4), round(off[:, 0].max().item(), 4)] if n else None,
        "accepted_y_m": [round(off[:, 1].min().item(), 4), round(off[:, 1].max().item(), 4)] if n else None,
    }
    print("[bowl-pivot-init] randomized: " + json.dumps(summary), flush=True)
    output.with_suffix(".json").write_text(json.dumps(summary, indent=2))
    if n == 0:
        print(f"[bowl-pivot-init] no randomized start passed, nothing saved to {output}", flush=True)
        return
    torch.save({
        "format_version": TABLE_FORMAT_VERSION,
        "object_name": cfg.object_spec.name,
        "joint_names": list(robot.joint_names),
        **table,
        "joint_target": q_row.cpu().expand(n, -1).clone(),
        "backoff_m": torch.zeros(n),
        "valid": torch.ones(n, dtype=torch.bool),
        "meta": summary,
    }, output)
    print(f"[bowl-pivot-init] saved {output}", flush=True)


def main() -> int:
    lo, hi = ARGS.lateral_range_m
    count = int(round((hi - lo) / ARGS.lateral_step_m)) + 1
    cfg = BowlPivotEnvCfg()
    cfg.scene.num_envs = count
    cfg.sim.device = ARGS.device
    # The state this script builds is what that event would load.
    cfg.events.reset_to_init = None
    env = ManagerBasedRLEnv(cfg)
    try:
        env.reset()
        spec = cfg.object_spec
        robot, bowl = env.scene["robot"], env.scene["object"]
        lateral = torch.linspace(lo, hi, count, device=env.device)
        arms = {
            "left": ArmKinematics(env, joint_names=LEFT_ARM_JOINT_NAMES, end_effector=LEFT_END_EFFECTOR),
            "right": ArmKinematics(env, joint_names=RIGHT_ARM_JOINT_NAMES, end_effector=RIGHT_END_EFFECTOR),
        }
        for side, arm in arms.items():
            home = arm.base_joint_pos[:, arm.arm_joint_ids]
            error = arm.check_jacobian(torch.clamp(home + 0.2, arm.lower, arm.upper))
            print(f"[bowl-pivot-init] {side} Jacobian vs finite difference: {error:.3g} -> "
                  f"{'finite differences' if arm.use_finite_difference else 'PhysX Jacobian'}", flush=True)

        # 1. Hands at the pre-contact shape, arms at home; the bowl settles alone.
        home_q = arms["left"].base_joint_pos.clone()
        for side in arms:
            shape = hand_preshape_joint_pos(**cfg.precontact_hand_curl, side=side)
            ids, names = robot.find_joints(list(shape), preserve_order=True)
            home_q[:, ids] = home_q.new_tensor([shape[name] for name in names])
        robot.write_joint_state_to_sim(home_q, torch.zeros_like(home_q))
        robot.set_joint_position_target(home_q)
        spawn = bowl.data.default_root_state[:, :7].clone()
        spawn[:, :3] += env.scene.env_origins
        bowl.write_root_pose_to_sim(spawn)
        bowl.write_root_velocity_to_sim(torch.zeros(count, 6, device=env.device))
        step_physics(env, round(ARGS.bowl_settle_s / env.step_dt))
        bowl_pose = bowl.data.root_state_w[:, :7].clone()
        print(f"[bowl-pivot-init] bowl settled: moved "
              f"{torch.linalg.vector_norm(bowl_pose[:, :3] - spawn[:, :3], dim=-1).max().item() * 1e3:.1f} mm, "
              f"speed {torch.linalg.vector_norm(bowl.data.root_lin_vel_w, dim=-1).max().item():.4f} m/s", flush=True)

        # 2. Both palms onto the pre-contact poses (the arms are separate chains).
        targets, ik = {}, {}
        q = home_q
        for side, arm in arms.items():
            arm.base_joint_pos = q.clone()
            pos = bowl_pose[:, :3].clone()
            pos[:, 0] += SIDE_SIGN[side] * lateral
            pos[:, 1] += cfg.precontact_palm_y_m
            pos[:, 2] = spec.support_height_m + cfg.precontact_palm_z_m
            quat = palm_quat(cfg, side, count, env.device)
            arm_q, ik[side] = arm.solve(arm.base_joint_pos[:, arm.arm_joint_ids], pos, quat,
                                        iterations=ARGS.ik_iterations)
            targets[side] = (pos, quat)
            q = arm.full_joint_pos(arm_q)

        # 3. Static hold with the bowl in place.
        robot.write_joint_state_to_sim(q, torch.zeros_like(q))
        robot.set_joint_position_target(q)
        bowl.write_root_pose_to_sim(bowl_pose)
        bowl.write_root_velocity_to_sim(torch.zeros(count, 6, device=env.device))
        for sensor in env.scene.sensors.values():
            sensor.reset()
        peak = {"hand_bowl_force_n": torch.zeros(count, device=env.device),
                "arm_any_force_n": torch.zeros(count, device=env.device)}

        def track():
            peak["hand_bowl_force_n"] = torch.maximum(peak["hand_bowl_force_n"],
                                                      hand_box_contact_forces(env).amax(dim=-1))
            for name in ARM_CONTACT_SENSORS:
                force = torch.linalg.vector_norm(env.scene.sensors[name].data.net_forces_w, dim=-1).amax(dim=-1)
                peak["arm_any_force_n"] = torch.maximum(peak["arm_any_force_n"], force)

        step_physics(env, ARGS.hold_steps, track)

        up_start = object_up_axis(spec, bowl_pose[:, 3:7])
        up_end = object_up_axis(spec, bowl.data.root_quat_w)
        metrics = {
            "bowl_displacement_m": torch.linalg.vector_norm(bowl.data.root_pos_w - bowl_pose[:, :3], dim=-1),
            "bowl_tilt_change_rad": torch.acos((up_start * up_end).sum(-1).clamp(-1.0, 1.0)),
            **peak,
        }
        for side, arm in arms.items():
            metrics[f"{side}_ik_position_error_m"] = ik[side]["ik_position_error_m"]
            metrics[f"{side}_ik_rotation_error_rad"] = ik[side]["ik_rotation_error_rad"]
            metrics[f"{side}_joint_limit_margin_rad"] = ik[side]["joint_limit_margin_rad"]
            metrics[f"{side}_limiting_joint"] = ik[side]["limiting_joint"]
            pos_err, _ = compute_pose_error(robot.data.body_pos_w[:, arm.palm_body_id],
                                            robot.data.body_quat_w[:, arm.palm_body_id],
                                            *targets[side], rot_error_type="axis_angle")
            metrics[f"{side}_settled_palm_error_m"] = torch.linalg.vector_norm(pos_err, dim=-1)
        failures = {
            "bowl_moved": metrics["bowl_displacement_m"] > ARGS.max_bowl_displacement_m,
            "bowl_tilted": metrics["bowl_tilt_change_rad"] > math.radians(ARGS.max_bowl_tilt_deg),
            "hand_touches_bowl": metrics["hand_bowl_force_n"] > ARGS.max_contact_force_n,
            "arm_contact": metrics["arm_any_force_n"] > ARGS.max_contact_force_n,
        }
        for side in arms:
            failures[f"{side}_ik_position"] = metrics[f"{side}_ik_position_error_m"] > ARGS.max_ik_position_m
            failures[f"{side}_ik_rotation"] = (metrics[f"{side}_ik_rotation_error_rad"]
                                               > math.radians(ARGS.max_ik_rotation_deg))
            failures[f"{side}_joint_limit"] = metrics[f"{side}_joint_limit_margin_rad"] < ARGS.min_joint_margin_rad
            failures[f"{side}_palm_sag"] = metrics[f"{side}_settled_palm_error_m"] > ARGS.max_settled_palm_m
        passed = ~torch.stack(list(failures.values()), dim=-1).any(dim=-1)
        for i in range(count):
            reasons = [name for name, failed in failures.items() if failed[i]]
            print(f"[bowl-pivot-init] lateral {lateral[i].item() * 100:5.2f} cm: "
                  f"{'pass' if passed[i] else 'fail ' + ','.join(reasons)}", flush=True)

        # 4. Closest touch-free distance plus the clearance (that candidate must pass too).
        chosen = None
        if bool(passed.any()):
            closest = lateral[passed].min().item()
            wanted = (lateral >= closest + cfg.precontact_clearance_m - 1e-6) & passed
            if bool(wanted.any()):
                chosen = int(torch.nonzero(wanted)[0].item())
        output = ARGS.output or bowl_pivot_init_path(spec)
        output.parent.mkdir(parents=True, exist_ok=True)
        index = 0 if chosen is None else chosen
        bowl_state = bowl.data.root_state_w[index:index + 1].clone()
        bowl_state[:, :3] -= env.scene.env_origins[index:index + 1]
        summary = {
            "object": spec.name,
            "valid": chosen is not None,
            "passing_lateral_m": [round(v, 4) for v in lateral[passed].tolist()],
            "chosen_lateral_m": None if chosen is None else lateral[chosen].item(),
            "clearance_m": cfg.precontact_clearance_m,
            "palm_y_m": cfg.precontact_palm_y_m,
            "palm_z_m": cfg.precontact_palm_z_m,
            "palm_tilt_deg": cfg.precontact_palm_tilt_deg,
            "hand_curl": dict(cfg.precontact_hand_curl),
            "bowl_state_env_local": bowl_state[0].tolist(),
            "metrics": {k: v[index].item() for k, v in metrics.items()},
        }
        print(json.dumps(summary, indent=2), flush=True)
        output.with_suffix(".json").write_text(json.dumps(summary, indent=2))
        if chosen is None:
            print(f"[bowl-pivot-init] REJECTED: no passing distance with {cfg.precontact_clearance_m} m clearance, "
                  f"nothing saved to {output}", flush=True)
            return 1
        rows = slice(chosen, chosen + 1)
        torch.save({
            "format_version": TABLE_FORMAT_VERSION,
            "object_name": spec.name,
            "joint_names": list(robot.joint_names),
            "joint_pos": robot.data.joint_pos[rows].clone().cpu(),
            "joint_vel": robot.data.joint_vel[rows].clone().cpu(),
            "joint_target": q[rows].cpu(),
            "box_state": bowl_state.cpu(),
            "box_offset": torch.zeros(1, 3),
            "backoff_m": torch.zeros(1),
            "valid": torch.ones(1, dtype=torch.bool),
            "meta": summary,
        }, output)
        print(f"[bowl-pivot-init] saved {output}", flush=True)
        if ARGS.random_rows > 0:
            random_table(env, cfg, q[rows], bowl_state, output.with_name(bowl_pivot_init_path(spec, True).name))
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
