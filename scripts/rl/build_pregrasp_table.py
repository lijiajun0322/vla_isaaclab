#!/usr/bin/env python3
"""Build a table of collision-free sugar-box pregrasp states for RL resets.

For each sampled box pose the scripted pregrasp palm pose is computed in closed
form, the arm is solved kinematically onto it (turn point -> orient -> pregrasp,
like the scripted approach), and a short static hold with the box in place
checks for contact, box motion and table contact. A failing pose is retried
with the palm pulled back along the approach direction.
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
import time
import traceback
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from isaaclab.app import AppLauncher


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-envs", type=int, default=1024, help="Candidates evaluated in parallel per batch.")
    parser.add_argument("--target-valid", type=int, default=10000, help="Stop once this many states pass.")
    parser.add_argument("--max-batches", type=int, default=40)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--x-range-m", type=float, nargs=2, default=(-0.02, 0.02))
    parser.add_argument("--y-range-m", type=float, nargs=2, default=(-0.02, 0.02))
    parser.add_argument("--yaw-range-deg", type=float, nargs=2, default=(-15.0, 15.0))
    parser.add_argument("--backoffs-m", type=float, nargs="+", default=(0.0, 0.005, 0.01, 0.015),
                        help="Palm pull-back distances tried in order until a state passes.")
    parser.add_argument("--palm-noise-m", type=float, default=0.0, help="Uniform per-axis palm target noise.")
    parser.add_argument("--palm-noise-deg", type=float, default=0.0, help="Uniform per-axis palm rotation noise.")
    parser.add_argument("--settle-steps", type=int, default=20, help="Control steps of the static hold.")
    parser.add_argument("--box-settle-s", type=float, default=2.0,
                        help="Let the box come to rest alone first; the calibrated pose rocks ~3 deg.")
    # Acceptance thresholds.
    parser.add_argument("--max-ik-position-m", type=float, default=0.003)
    parser.add_argument("--max-ik-rotation-deg", type=float, default=3.0)
    parser.add_argument("--min-joint-margin-rad", type=float, default=0.02)
    parser.add_argument("--max-box-displacement-m", type=float, default=0.002)
    parser.add_argument("--max-box-tilt-deg", type=float, default=1.0)
    parser.add_argument("--max-box-speed-mps", type=float, default=0.003)
    parser.add_argument("--max-box-angular-speed-radps", type=float, default=0.05,
                        help="A box still rocking after it settled keeps moving after reset.")
    parser.add_argument("--max-box-force-n", type=float, default=1.0)
    parser.add_argument("--max-arm-force-n", type=float, default=1.0,
                        help="Net contact force on any left-arm/hand link (box, table, ...).")
    parser.add_argument("--max-settled-palm-m", type=float, default=0.01)
    parser.add_argument("--hand-preshape", type=float, nargs=4, default=None,
                        metavar=("THUMB_ROTATE", "THUMB", "INDEX", "MIDDLE"),
                        help="Finger preshape curl (rad magnitudes); default is DEFAULT_HAND_PRESHAPE.")
    parser.add_argument("--open-hand", action="store_true", help="Keep the hand fully open (the original table).")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "outputs/rl/pregrasp_table_preshape.pt")
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    args.enable_cameras = False
    return args


ARGS = parse_args()
ARGS.experience = str(PROJECT_ROOT / "configs" / ("ycb.python.headless.kit" if ARGS.headless else "ycb.python.kit"))
ARGS.kit_args = f"--portable-root {PROJECT_ROOT}/outputs/runtime/kit"
APP = AppLauncher(ARGS).app

import torch

import vla_isaaclab  # noqa: F401  Register environments.
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.utils.math import quat_from_angle_axis, quat_mul, sample_uniform

from vla_isaaclab.envs.ycb_sugar_box.env_cfg import YCBSugarBoxStateEnvCfg
from vla_isaaclab.rl.pregrasp_table import (
    ARM_JOINT_NAMES,
    DEFAULT_HAND_PRESHAPE,
    FINGERTIP_BODY_NAMES,
    TABLE_FORMAT_VERSION,
    ArmKinematics,
    hand_preshape_joint_pos,
    load_pregrasp_table,
    pregrasp_palm_target,
    reset_from_pregrasp_table,
    settle_check,
    turn_point,
)


def sample_box_poses(env, generator_seed: int):
    box = env.scene["object"]
    ranges = torch.tensor(
        (ARGS.x_range_m, ARGS.y_range_m, [math.radians(v) for v in ARGS.yaw_range_deg]),
        dtype=torch.float32, device=env.device,
    )
    torch.manual_seed(generator_seed)
    offsets = sample_uniform(ranges[:, 0], ranges[:, 1], (env.num_envs, 3), device=env.device)
    default = box.data.default_root_state
    position = default[:, :3] + env.scene.env_origins
    position[:, :2] += offsets[:, :2]
    z_axis = torch.zeros(env.num_envs, 3, device=env.device)
    z_axis[:, 2] = 1.0
    orientation = quat_mul(quat_from_angle_axis(offsets[:, 2], z_axis), default[:, 3:7])
    return offsets, torch.cat((position, orientation), dim=-1)


def settle_box_alone(env, kinematics, home_arm, box_pose):
    """Rest the box on the table with the arm at home; return the settled pose."""
    robot = env.scene["robot"]
    box = env.scene["object"]
    q = kinematics.full_joint_pos(home_arm)
    robot.write_joint_state_to_sim(q, torch.zeros_like(q))
    robot.set_joint_position_target(q)
    box.write_root_pose_to_sim(box_pose)
    box.write_root_velocity_to_sim(torch.zeros(env.num_envs, 6, device=env.device))
    for _ in range(round(ARGS.box_settle_s / env.physics_dt)):
        env.scene.write_data_to_sim()
        env.sim.step(render=False)
        env.scene.update(dt=env.physics_dt)
    return box.data.root_state_w[:, :7].clone(), torch.linalg.vector_norm(box.data.root_lin_vel_w, dim=-1)


def noisy_target(pos, quat):
    if ARGS.palm_noise_m > 0.0:
        pos = pos + (torch.rand_like(pos) * 2.0 - 1.0) * ARGS.palm_noise_m
    if ARGS.palm_noise_deg > 0.0:
        rotvec = (torch.rand_like(pos) * 2.0 - 1.0) * math.radians(ARGS.palm_noise_deg)
        angle = torch.linalg.vector_norm(rotvec, dim=-1)
        axis = rotvec / angle.clamp_min(1e-9).unsqueeze(-1)
        quat = quat_mul(quat_from_angle_axis(angle, axis), quat)
    return pos, quat


def failure_reasons(ik: dict, settle: dict) -> dict[str, torch.Tensor]:
    return {
        "ik_position": ik["ik_position_error_m"] > ARGS.max_ik_position_m,
        "ik_rotation": ik["ik_rotation_error_rad"] > math.radians(ARGS.max_ik_rotation_deg),
        "joint_limit": ik["joint_limit_margin_rad"] < ARGS.min_joint_margin_rad,
        "box_moved": settle["box_displacement_m"] > ARGS.max_box_displacement_m,
        "box_tilted": settle["box_tilt_change_rad"] > math.radians(ARGS.max_box_tilt_deg),
        "box_moving": (settle["box_speed_mps"] > ARGS.max_box_speed_mps)
        | (settle["box_angular_speed_radps"] > ARGS.max_box_angular_speed_radps),
        "box_contact": settle["box_contact_force_n"] > ARGS.max_box_force_n,
        "arm_contact": settle["arm_contact_force_n"] > ARGS.max_arm_force_n,
        "palm_sag": settle["settled_palm_position_error_m"] > ARGS.max_settled_palm_m,
    }


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "-C", str(PROJECT_ROOT), "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def save_plot(offsets, valid, backoff, path: Path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    offsets = offsets.cpu().numpy()
    valid = valid.cpu().numpy()
    backoff = backoff.cpu().numpy()
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, (i, j, xlabel, ylabel, scale) in zip(
        axes, ((0, 1, "box x offset (mm)", "box y offset (mm)", (1e3, 1e3)),
               (1, 2, "box y offset (mm)", "box yaw offset (deg)", (1e3, 180.0 / math.pi)))
    ):
        x = offsets[:, i] * scale[0]
        y = offsets[:, j] * scale[1]
        ax.scatter(x[~valid], y[~valid], s=4, c="#c0392b", label="rejected")
        points = ax.scatter(x[valid], y[valid], s=4, c=backoff[valid] * 1e3, cmap="viridis", label="accepted")
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.legend(loc="upper right", markerscale=3)
    fig.colorbar(points, ax=axes, label="palm pull-back (mm)")
    fig.suptitle(f"Pregrasp candidates: {int(valid.sum())} accepted / {len(valid)}")
    fig.savefig(path, dpi=120)
    plt.close(fig)


def check_against_strategy(env, nominal_pos, nominal_quat) -> dict:
    """The closed-form targets must equal the scripted strategy's on the same box poses."""
    from vla_isaaclab.policies.ycb_sugar_box import _G1Semantics, _SugarBoxSemantics
    from vla_isaaclab.policies.ycb_sugar_box_strategy import SugarBoxPhaseStrategy

    box = env.scene["object"]
    _, box_pose = sample_box_poses(env, 12345)
    box.write_root_pose_to_sim(box_pose)
    strategy = SugarBoxPhaseStrategy(env, _G1Semantics, _SugarBoxSemantics)
    strategy._compute_grasp()
    zero = torch.zeros(env.num_envs, device=env.device)
    pos, quat, _ = pregrasp_palm_target(
        box_pose[:, :3], box_pose[:, 3:7], nominal_quat, env.scene["robot"].data.root_pos_w, zero
    )
    turn = turn_point(box_pose[:, :3], box_pose[:, 3:7], nominal_pos, nominal_quat, env.scene.env_origins)
    return {
        "pregrasp_position_max_diff_m": torch.linalg.vector_norm(pos - strategy.grasp_pos, dim=-1).max().item(),
        "pregrasp_quat_max_diff": (quat - strategy.grasp_quat).abs().max().item(),
        "turn_point_max_diff_m": torch.linalg.vector_norm(
            turn - strategy._box_frame_point(strategy.TURN_POINT_WORLD), dim=-1
        ).max().item(),
    }


def verify_table(env, kinematics, path: Path) -> dict:
    """Reset every env from the saved table and hold still: nothing should move."""
    table = load_pregrasp_table(path, env.device)
    env.pregrasp_table = None
    reset_from_pregrasp_table(env, None, str(path))
    rows = env.pregrasp_table_index
    robot = env.scene["robot"]
    box = env.scene["object"]
    box_start = box.data.root_pos_w.clone()
    for _ in range(ARGS.settle_steps * env.cfg.decimation):
        env.scene.write_data_to_sim()
        env.sim.step(render=False)
        env.scene.update(dt=env.physics_dt)
    moved = torch.linalg.vector_norm(box.data.root_pos_w - box_start, dim=-1)
    joint_error = (robot.data.joint_pos[:, kinematics.arm_joint_ids]
                   - table.joint_pos[rows][:, kinematics.arm_joint_ids]).abs().amax(dim=-1)
    hand_ids = kinematics.hand_joint_ids
    hand_error = (robot.data.joint_pos[:, hand_ids] - table.joint_target[rows][:, hand_ids]).abs().amax(dim=-1)
    return {
        "states_checked": env.num_envs,
        "box_displacement_max_m": moved.max().item(),
        "box_displacement_p99_m": moved.quantile(0.99).item(),
        "states_over_displacement_threshold": int((moved > ARGS.max_box_displacement_m).sum().item()),
        "arm_joint_drift_max_rad": joint_error.max().item(),
        "hand_joint_error_to_preshape_max_rad": hand_error.max().item(),
    }


def main() -> int:
    cfg = YCBSugarBoxStateEnvCfg()
    cfg.scene.num_envs = ARGS.num_envs
    cfg.sim.device = ARGS.device
    cfg.seed = ARGS.seed
    env = ManagerBasedRLEnv(cfg)
    try:
        env.reset(seed=ARGS.seed)
        robot = env.scene["robot"]
        box = env.scene["object"]
        kinematics = ArmKinematics(env)
        if ARGS.open_hand:
            hand_preshape = {}
        else:
            curls = dict(zip(DEFAULT_HAND_PRESHAPE, ARGS.hand_preshape)) if ARGS.hand_preshape else DEFAULT_HAND_PRESHAPE
            hand_preshape = hand_preshape_joint_pos(**curls)
            kinematics.set_hand(hand_preshape)
        kinematics.hand_joint_ids = robot.find_joints([n for n in robot.joint_names if n.startswith("left_hand_")])[0]
        print(f"[pregrasp-table] hand preshape: {hand_preshape or 'open'}", flush=True)
        print(f"[pregrasp-table] left-hand bodies: {[n for n in robot.body_names if n.startswith('left_hand')]}")
        missing = [name for name in FINGERTIP_BODY_NAMES if name not in robot.body_names]
        if missing:
            raise RuntimeError(f"Fingertip bodies not found: {missing}")

        home_arm = kinematics.base_joint_pos[:, kinematics.arm_joint_ids].clone()
        probe = torch.clamp(home_arm + 0.2, kinematics.lower, kinematics.upper)
        jacobian_error = kinematics.check_jacobian(probe)
        print(f"[pregrasp-table] PhysX Jacobian vs finite difference after kinematic write: "
              f"relative error {jacobian_error:.3g} -> "
              f"{'finite differences' if kinematics.use_finite_difference else 'PhysX Jacobian'}", flush=True)
        _, home_quat = kinematics.set_arm(home_arm)

        nominal_pos = box.data.default_root_state[:, :3] + env.scene.env_origins
        nominal_quat = box.data.default_root_state[:, 3:7]
        strategy_check = check_against_strategy(env, nominal_pos, nominal_quat)
        print(f"[pregrasp-table] closed form vs scripted strategy: {strategy_check}", flush=True)
        if strategy_check["pregrasp_position_max_diff_m"] > 1e-4 or strategy_check["turn_point_max_diff_m"] > 1e-4:
            raise RuntimeError("Closed-form pregrasp target disagrees with SugarBoxPhaseStrategy")
        attempt_failures = [dict() for _ in ARGS.backoffs_m]
        attempt_passed = [0 for _ in ARGS.backoffs_m]
        attempt_tried = [0 for _ in ARGS.backoffs_m]
        backoffs = [float(b) for b in ARGS.backoffs_m]
        columns = {name: [] for name in ("joint_pos", "joint_vel", "joint_target", "box_state", "box_offset", "backoff_m",
                                          "palm_target", "valid", "first_failure")}
        metric_columns: dict[str, list] = {}
        reason_names = None
        started = time.time()
        valid_total = 0
        batch_times = []
        for batch in range(ARGS.max_batches):
            batch_start = time.time()
            offsets, spawn_pose = sample_box_poses(env, ARGS.seed * 100003 + batch)
            box_pose, rest_speed = settle_box_alone(env, kinematics, home_arm, spawn_pose)
            print(f"[pregrasp-table] box settled alone: moved "
                  f"{torch.linalg.vector_norm(box_pose[:, :3] - spawn_pose[:, :3], dim=-1).median().item() * 1e3:.1f} mm "
                  f"(median), rest speed max {rest_speed.max().item():.4f} m/s", flush=True)
            box_pos, box_quat = box_pose[:, :3], box_pose[:, 3:7]
            # Scripted order: go to the turn point keeping the home orientation,
            # rotate there, then move to the pregrasp.
            turn = turn_point(box_pos, box_quat, nominal_pos, nominal_quat, env.scene.env_origins)
            zero = torch.zeros(env.num_envs, device=env.device)
            _, grasp_quat, _ = pregrasp_palm_target(box_pos, box_quat, nominal_quat, robot.data.root_pos_w, zero)
            arm_q, _ = kinematics.solve(home_arm, turn, home_quat, iterations=40)
            arm_q, _ = kinematics.solve(arm_q, turn, grasp_quat, iterations=40)

            done = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
            record = None
            for attempt, backoff in enumerate(backoffs):
                target_pos, target_quat, _ = pregrasp_palm_target(
                    box_pos, box_quat, nominal_quat, robot.data.root_pos_w, torch.full_like(zero, backoff)
                )
                target_pos, target_quat = noisy_target(target_pos, target_quat)
                arm_q, ik = kinematics.solve(arm_q, target_pos, target_quat, iterations=80 if attempt == 0 else 40)
                result = settle_check(env, kinematics, arm_q, box_pose, target_pos, target_quat, ARGS.settle_steps)
                reasons = failure_reasons(ik, result.metrics)
                reason_names = list(reasons)
                failed = torch.stack(list(reasons.values()), dim=-1)
                passed = ~failed.any(dim=-1)
                first_failure = torch.where(passed, -1, failed.float().argmax(dim=-1))
                tried = ~done
                attempt_tried[attempt] += int(tried.sum().item())
                attempt_passed[attempt] += int((tried & passed).sum().item())
                for index, name in enumerate(reasons):
                    count = int((tried & failed[:, index]).sum().item())
                    attempt_failures[attempt][name] = attempt_failures[attempt].get(name, 0) + count
                # Keep the first passing attempt; for never-passing poses keep the last.
                take = (~done) & (passed | (attempt == len(backoffs) - 1))
                current = {
                    "joint_pos": result.joint_pos, "joint_vel": result.joint_vel,
                    "joint_target": result.joint_target,
                    "box_state": result.box_state, "box_offset": offsets,
                    "backoff_m": torch.full_like(zero, backoff),
                    "palm_target": torch.cat((target_pos - env.scene.env_origins, target_quat), dim=-1),
                    "valid": passed, "first_failure": first_failure,
                    **{f"metric/{k}": v for k, v in {**ik, **result.metrics}.items()},
                }
                if record is None:
                    record = {k: v.clone() for k, v in current.items()}
                else:
                    for k, v in current.items():
                        record[k][take] = v[take]
                done |= take
                if bool(done.all()):
                    break

            for key, value in record.items():
                if key.startswith("metric/"):
                    metric_columns.setdefault(key[len("metric/"):], []).append(value.cpu())
                else:
                    columns[key].append(value.cpu())
            batch_valid = int(record["valid"].sum().item())
            valid_total += batch_valid
            batch_times.append(time.time() - batch_start)
            print(f"[pregrasp-table] batch {batch}: {batch_valid}/{env.num_envs} accepted, "
                  f"total {valid_total}, {batch_times[-1]:.1f} s", flush=True)
            if valid_total >= ARGS.target_valid:
                break

        table = {key: torch.cat(values) for key, values in columns.items()}
        metrics = {key: torch.cat(values) for key, values in metric_columns.items()}
        valid = table["valid"]
        failure_counts = {
            name: int((table["first_failure"] == index).sum().item()) for index, name in enumerate(reason_names)
        }
        accepted_backoff = {
            f"{b * 1e3:.1f}mm": int((valid & torch.isclose(table["backoff_m"], torch.tensor(b))).sum().item())
            for b in backoffs
        }

        def stats(name, scale=1.0):
            values = metrics[name][valid] * scale
            if values.numel() == 0:
                return None
            return {"median": values.median().item(), "p99": values.quantile(0.99).item(), "max": values.max().item()}

        tips = metrics["fingertips_in_box_m"][valid].reshape(-1, len(FINGERTIP_BODY_NAMES), 3)
        summary = {
            "output": str(ARGS.output),
            "git_commit": git_commit(),
            "candidates": len(valid),
            "accepted": int(valid.sum().item()),
            "acceptance_rate": valid.float().mean().item(),
            "accepted_by_backoff": accepted_backoff,
            "first_failure_counts": failure_counts,
            # Every failed criterion is counted, so one candidate can appear under several.
            "per_backoff_attempts": {
                f"{b * 1e3:.1f}mm": {"tried": attempt_tried[i], "passed": attempt_passed[i],
                                      "failed_criteria": attempt_failures[i]}
                for i, b in enumerate(backoffs)
            },
            "strategy_check": strategy_check,
            "hand_preshape_rad": hand_preshape,
            "limiting_joint_counts": {
                "accepted": {ARM_JOINT_NAMES[int(i)]: int(c) for i, c in zip(
                    *metrics["limiting_joint"][valid].unique(return_counts=True))},
                "rejected": {ARM_JOINT_NAMES[int(i)]: int(c) for i, c in zip(
                    *metrics["limiting_joint"][~valid].unique(return_counts=True))},
            },
            "timing_s": {
                "total": time.time() - started,
                "per_batch_mean": sum(batch_times) / len(batch_times),
                "num_envs": ARGS.num_envs,
            },
            "jacobian": {"relative_error": jacobian_error, "finite_difference": kinematics.use_finite_difference},
            "accepted_stats": {
                "ik_position_error_mm": stats("ik_position_error_m", 1e3),
                "ik_rotation_error_deg": stats("ik_rotation_error_rad", 180.0 / math.pi),
                "joint_limit_margin_rad": stats("joint_limit_margin_rad"),
                "box_displacement_mm": stats("box_displacement_m", 1e3),
                "box_contact_force_n": stats("box_contact_force_n"),
                "arm_contact_force_n": stats("arm_contact_force_n"),
                "settled_palm_position_error_mm": stats("settled_palm_position_error_m", 1e3),
            },
            # Box frame: x width (92 mm), y height, z thickness (45 mm).
            "accepted_fingertips_in_box_mm_median": {
                name: (tips[:, i].median(dim=0).values * 1e3).tolist() for i, name in enumerate(FINGERTIP_BODY_NAMES)
            } if tips.numel() else None,
            "args": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(ARGS).items()
                     if k not in ("experience", "kit_args")},
        }
        ARGS.output.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "format_version": TABLE_FORMAT_VERSION,
                "joint_names": list(robot.joint_names),
                "arm_joint_names": list(ARM_JOINT_NAMES),
                "metric_names": list(metrics),
                "failure_names": reason_names,
                **table,
                "metrics": metrics,
                "meta": summary,
            },
            ARGS.output,
        )
        if summary["accepted"]:
            summary["replay_check"] = verify_table(env, kinematics, ARGS.output)
        save_plot(table["box_offset"], valid, table["backoff_m"], ARGS.output.with_suffix(".png"))
        ARGS.output.with_suffix(".json").write_text(json.dumps(summary, indent=2) + "\n")
        print(json.dumps(summary, indent=2), flush=True)
        return 0 if summary["accepted"] else 2
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
