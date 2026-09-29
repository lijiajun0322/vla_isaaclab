#!/usr/bin/env python3
"""Try rim-pinch pregrasp poses for a bowl-like object (spec.rim_radius_m > 0).

Each env gets one candidate: the rim point at azimuth PHI around the object's up
axis, with the index / middle fingertips (their mean, at the preshape; or with
--anchor mid the midpoint between them and the thumb tip) placed RADIAL / DZ
away from that rim point. A fingertip is the far end of its 52 mm distal link.
The fingers point inward toward the center, tilted down by PITCH; the thumb
side is up at pitch 0 and inward at pitch 90 (fingers down the outer wall). The arm is solved kinematically onto that palm pose and held still,
like the pregrasp table's check. Then, if --grasp-test: close each finger until
it touches and squeeze a little more, and raise the palm (joint targets only,
the arm is never written) to see whether the pinch carries the object.

Prints one row per candidate plus the spec values (grasp_quat_wxyz,
grasp_offset_world) for the best one. --image-dir saves a side and a close-up
camera view of each env at pregrasp, closed and lifted; --snapshot-dir instead
saves link meshes and poses for scripts/rl/draw_probe_snapshots.py, which draws
them without the RTX renderer.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import sys
import traceback
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task", default="VLA-YCBGraspLift-Bowl-G1-v0")
parser.add_argument("--phi-deg", type=float, nargs="+", default=(180.0, 210.0, 240.0, 270.0))
parser.add_argument("--pitch-deg", type=float, nargs="+", default=(0.0, 25.0, 50.0))
parser.add_argument("--roll-deg", type=float, nargs="+", default=(0.0,),
                    help="Rotation about the finger direction (0 = thumb straight up).")
parser.add_argument("--radial-m", type=float, nargs="+", default=(0.0,),
                    help="Pinch midpoint minus rim point along the outward radial.")
parser.add_argument("--dz-m", type=float, nargs="+", default=(0.0,))
parser.add_argument("--rim-point", type=float, nargs=2, default=(0.078, 0.050), metavar=("R", "Z"),
                    help="Pinch target in the object frame: radius and height (m) of the lip edge.")
parser.add_argument("--anchor", choices=("fingers", "mid", "thumb"), default="fingers",
                    help="Hand point placed at the rim point: fingertips, the open pinch's midpoint, or the thumb tip.")
parser.add_argument("--frame", choices=("radial", "forward"), default="radial",
                    help="radial: fingers point inward (tilted down by pitch). forward: fingers point along the "
                         "rim tangent toward world +y (tilted down by pitch), thumb toward the center.")
parser.add_argument("--object-shift-m", type=float, nargs=2, default=(0.0, 0.0), metavar=("DX", "DY"),
                    help="Move the object's default position on the table (m, world x / y).")
parser.add_argument("--object-dx-m", type=float, nargs="+", default=(0.0,),
                    help="Per-candidate object offset from its default position (world x); no reset randomization.")
parser.add_argument("--object-dy-m", type=float, nargs="+", default=(0.0,))
parser.add_argument("--support-dz-m", type=float, default=0.0, help="Raise the table by this much (m).")
parser.add_argument("--preshape", type=float, nargs=4, default=None,
                    metavar=("THUMB_ROTATE", "THUMB", "INDEX", "MIDDLE"))
parser.add_argument("--settle-steps", type=int, default=20)
parser.add_argument("--grasp-test", action="store_true")
parser.add_argument("--close-rate", type=float, default=0.02, help="Finger curl rad per control step.")
parser.add_argument("--squeeze", type=float, default=0.25, help="Extra curl after first contact (rad).")
parser.add_argument("--close-steps", type=int, default=80)
parser.add_argument("--lift-m", type=float, default=0.06)
parser.add_argument("--lift-steps", type=int, default=90)
parser.add_argument("--image-dir", type=Path, default=None)
parser.add_argument("--snapshot-dir", type=Path, default=None)
AppLauncher.add_app_launcher_args(parser)
ARGS = parser.parse_args()
RECORD = ARGS.image_dir is not None
ARGS.enable_cameras = RECORD
kit = ("ycb.python.headless.rendering.kit" if ARGS.headless else "ycb.python.rendering.kit") \
    if RECORD else ("ycb.python.headless.kit" if ARGS.headless else "ycb.python.kit")
ARGS.experience = str(PROJECT_ROOT / "configs" / kit)
ARGS.kit_args = f"--portable-root {PROJECT_ROOT}/outputs/runtime/kit"
APP = AppLauncher(ARGS).app

import torch

import vla_isaaclab  # noqa: F401
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.utils.math import compute_pose_error, quat_apply, quat_conjugate, quat_from_matrix, quat_mul
from isaaclab_tasks.utils import parse_env_cfg

from vla_isaaclab.envs.common import LEFT_HAND_CLOSED_JOINT_POSITIONS, LEFT_HAND_JOINT_NAMES, camera_cfg
from vla_isaaclab.envs.ycb_grasp.env_cfg import YCBGraspStateEnvCfg, add_video_cameras
from vla_isaaclab.policies.bounded_ik import bounded_dls
from vla_isaaclab.rl.pregrasp_table import (
    FINGERTIP_BODY_NAMES,
    ArmKinematics,
    hand_box_contact_forces,
    hand_preshape_joint_pos,
    settle_check,
)


CLOSED = dict(zip(LEFT_HAND_JOINT_NAMES, LEFT_HAND_CLOSED_JOINT_POSITIONS))
FINGERS = {
    "thumb": (("left_hand_thumb_1_joint", "left_hand_thumb_2_joint"), ("object_contact_thumb_1", "object_contact_thumb_2")),
    "index": (("left_hand_index_0_joint", "left_hand_index_1_joint"), ("object_contact_index_0", "object_contact_index_1")),
    "middle": (("left_hand_middle_0_joint", "left_hand_middle_1_joint"), ("object_contact_middle_0", "object_contact_middle_1")),
}


def frame(first: torch.Tensor, second: torch.Tensor) -> torch.Tensor:
    """Right-handed orthonormal columns: first, second's part orthogonal to it, their cross."""
    e1 = first / torch.linalg.vector_norm(first, dim=-1, keepdim=True)
    e2 = second - (second * e1).sum(-1, keepdim=True) * e1
    e2 = e2 / torch.linalg.vector_norm(e2, dim=-1, keepdim=True)
    return torch.stack((e1, e2, torch.cross(e1, e2, dim=-1)), dim=-1)


def step_physics(env, steps: int):
    for _ in range(steps * env.cfg.decimation):
        env.scene.write_data_to_sim()
        env.sim.step(render=False)
        env.scene.update(dt=env.physics_dt)


def save_images(env, tag: str):
    if not RECORD:
        return
    from PIL import Image

    for _ in range(4):
        env.sim.render()
    ARGS.image_dir.mkdir(parents=True, exist_ok=True)
    for name in ("cam_side", "cam_close"):
        sensor = env.scene.sensors[name]
        sensor.update(dt=0.0, force_recompute=True)
        rgb = sensor.data.output["rgb"][..., :3].cpu().numpy()
        for i in range(env.num_envs):
            Image.fromarray(rgb[i]).save(ARGS.image_dir / f"env{i:02d}_{tag}_{name}.png")


SNAPSHOT_BODIES = ("left_elbow", "left_wrist", "left_hand")
# Far end of each distal link in its own frame (from the visual meshes), thumb, index, middle.
TIP_OFFSETS = ((0.0, -0.045, 0.0), (0.045, 0.0, 0.0), (0.045, 0.0, 0.0))


def fingertips_w(robot, tip_ids) -> torch.Tensor:
    """World fingertip points, (num_envs, 3, 3) in order thumb, index, middle."""
    pos, quat = robot.data.body_pos_w[:, tip_ids], robot.data.body_quat_w[:, tip_ids]
    offsets = pos.new_tensor(TIP_OFFSETS).expand_as(pos)
    return pos + quat_apply(quat.reshape(-1, 4), offsets.reshape(-1, 3)).reshape(pos.shape)


def link_meshes(env) -> dict:
    """Visual mesh vertices and triangles of the left forearm and hand links, in each link's frame."""
    import numpy as np
    from pxr import Usd, UsdGeom

    stage = env.sim.stage
    root = stage.GetPrimAtPath("/World/envs/env_0/Robot")
    names = [n for n in env.scene["robot"].body_names if n.startswith(SNAPSHOT_BODIES)]
    cache = UsdGeom.XformCache()
    # The link's visuals/<link name> and collisions/<link name> children reuse its name: keep the outermost.
    links = {}
    for prim in Usd.PrimRange(root, Usd.TraverseInstanceProxies()):
        if prim.GetName() in names:
            links.setdefault(prim.GetName(), prim)
    meshes = {}
    for name, link in links.items():
        to_link = cache.GetLocalToWorldTransform(link).GetInverse()
        verts, faces, offset = [], [], 0
        for prim in Usd.PrimRange(link, Usd.TraverseInstanceProxies()):
            if not prim.IsA(UsdGeom.Mesh) or "collision" in str(prim.GetPath()).lower():
                continue
            mesh = UsdGeom.Mesh(prim)
            points, counts, indices = mesh.GetPointsAttr().Get(), mesh.GetFaceVertexCountsAttr().Get(), \
                mesh.GetFaceVertexIndicesAttr().Get()
            if not points:
                continue
            matrix = cache.GetLocalToWorldTransform(prim) * to_link
            verts.append(np.array([matrix.Transform(pt) for pt in points], dtype=np.float32))
            start = 0
            for count in counts:
                for k in range(1, count - 1):  # fan-triangulate
                    faces.append((indices[start] + offset, indices[start + k] + offset, indices[start + k + 1] + offset))
                start += count
            offset += len(points)
        if verts:
            meshes[name] = (np.concatenate(verts), np.array(faces, dtype=np.int64))
    return meshes


def save_snapshot(env, tag: str):
    if ARGS.snapshot_dir is None:
        return
    robot, obj = env.scene["robot"], env.scene["object"]
    origins = env.scene.env_origins.unsqueeze(1)
    data = {
        "body_names": robot.body_names,
        "body_pos": (robot.data.body_pos_w - origins).cpu(),
        "body_quat": robot.data.body_quat_w.cpu(),
        "object_pos": (obj.data.root_pos_w - env.scene.env_origins).cpu(),
        "object_quat": obj.data.root_quat_w.cpu(),
    }
    ARGS.snapshot_dir.mkdir(parents=True, exist_ok=True)
    torch.save(data, ARGS.snapshot_dir / f"{tag}.pt")


def main() -> int:
    spec = parse_env_cfg(ARGS.task, device=ARGS.device, num_envs=1).object_spec
    if spec.rim_radius_m <= 0.0:
        raise ValueError(f"{spec.name} has no rim (rim_radius_m)")
    spec = spec.replace(
        support_height_m=spec.support_height_m + ARGS.support_dz_m,
        initial_xy=(spec.initial_xy[0] + ARGS.object_shift_m[0], spec.initial_xy[1] + ARGS.object_shift_m[1]),
    )
    candidates = list(itertools.product(ARGS.phi_deg, ARGS.pitch_deg, ARGS.roll_deg, ARGS.radial_m, ARGS.dz_m,
                                        ARGS.object_dx_m, ARGS.object_dy_m))
    cfg = YCBGraspStateEnvCfg(object_spec=spec)
    cfg.scene.num_envs = len(candidates)
    cfg.sim.device = ARGS.device
    if RECORD:
        add_video_cameras(cfg)
        center = (*spec.initial_xy, spec.support_height_m + ARGS.rim_point[1])
        close = camera_cfg((center[0] - 0.40, center[1] - 0.05, center[2] + 0.12), center)
        close.prim_path = "{ENV_REGEX_NS}/CloseCamera"
        cfg.scene.cam_close = close
    env = ManagerBasedRLEnv(cfg)
    try:
        env.reset()
        robot, obj = env.scene["robot"], env.scene["object"]
        n, device = env.num_envs, env.device
        kinematics = ArmKinematics(env)
        curls = dict(zip(spec.hand_preshape, ARGS.preshape)) if ARGS.preshape else spec.hand_preshape
        kinematics.set_hand(hand_preshape_joint_pos(**curls))
        home_arm = kinematics.base_joint_pos[:, kinematics.arm_joint_ids].clone()

        # Hand geometry at the preshape, in the palm frame.
        palm_pos, palm_quat = kinematics.set_arm(home_arm)
        tip_ids, _ = robot.find_bodies(list(FINGERTIP_BODY_NAMES), preserve_order=True)
        tips = fingertips_w(robot, tip_ids) - palm_pos.unsqueeze(1)
        inv = quat_conjugate(palm_quat)
        tips_p = torch.stack([quat_apply(inv, tips[:, i]) for i in range(3)], dim=1)[0]
        thumb_p, fingers_p = tips_p[0], tips_p[1:].mean(0)
        pinch_mid_p = 0.5 * (thumb_p + fingers_p)
        anchor_p = {"mid": pinch_mid_p, "fingers": fingers_p, "thumb": thumb_p}[ARGS.anchor]
        print(f"[probe] palm-frame tips (mm): thumb {(thumb_p * 1e3).tolist()}, "
              f"index {(tips_p[1] * 1e3).tolist()}, middle {(tips_p[2] * 1e3).tolist()}", flush=True)
        palm_axes = frame(pinch_mid_p, thumb_p - fingers_p)  # columns in palm frame

        # Object at its default pose plus the candidate's offset (replaces the reset randomization), at rest alone.
        c = torch.tensor(candidates, device=device, dtype=torch.float32)
        start = obj.data.default_root_state.clone()
        start[:, :3] += env.scene.env_origins
        start[:, :2] += c[:, 5:7]
        obj.write_root_state_to_sim(start)
        step_physics(env, 60)
        obj_pos, obj_quat = obj.data.root_pos_w.clone(), obj.data.root_quat_w.clone()

        phi, pitch, roll = torch.deg2rad(c[:, 0]), torch.deg2rad(c[:, 1]), torch.deg2rad(c[:, 2])
        outward = torch.stack((torch.cos(phi), torch.sin(phi), torch.zeros_like(phi)), dim=-1)
        up = torch.zeros_like(outward)
        up[:, 2] = 1.0
        rim, rim_z = ARGS.rim_point
        pinch_w = obj_pos + rim * outward + rim_z * up + c[:, 3:4] * outward + c[:, 4:5] * up
        # Fingers point inward and down by pitch. The thumb side turns with them: up at
        # pitch 0, inward at pitch 90 (fingers down the outer wall, thumb inside the
        # rim); roll turns it about the finger direction.
        cos_p, sin_p = torch.cos(pitch).unsqueeze(-1), torch.sin(pitch).unsqueeze(-1)
        finger_dir = -cos_p * outward - sin_p * up
        thumb_side = cos_p * up - sin_p * outward
        if ARGS.frame == "forward":
            tangent = torch.cross(up, outward, dim=-1)
            tangent = torch.where(tangent[:, 1:2] < 0.0, -tangent, tangent)
            finger_dir = cos_p * tangent - sin_p * up
            thumb_side = -outward
        side = torch.cross(finger_dir, thumb_side, dim=-1)
        side = side / torch.linalg.vector_norm(side, dim=-1, keepdim=True)
        thumb_dir = torch.cos(roll).unsqueeze(-1) * thumb_side + torch.sin(roll).unsqueeze(-1) * side
        world_axes = frame(finger_dir, thumb_dir)
        rotation = world_axes @ palm_axes.T.unsqueeze(0)
        target_quat = quat_from_matrix(rotation)
        target_pos = pinch_w - quat_apply(target_quat, anchor_p.expand(n, 3))

        # Scripted order: above-and-outside turn point at home orientation, rotate, then pregrasp.
        turn = target_pos - 0.08 * finger_dir + 0.06 * up
        _, home_quat = kinematics.set_arm(home_arm)
        arm_q, _ = kinematics.solve(home_arm, turn, home_quat, iterations=40)
        arm_q, _ = kinematics.solve(arm_q, turn, target_quat, iterations=40)
        arm_q, ik = kinematics.solve(arm_q, target_pos, target_quat, iterations=120)
        pose = torch.cat((obj_pos, obj_quat), dim=-1)
        # Which links touch the object right after the kinematic write (one control step).
        sensor_names = [s for s in env.scene.sensors if s.startswith("object_contact_")]
        q0 = kinematics.full_joint_pos(arm_q)
        robot.write_joint_state_to_sim(q0, torch.zeros_like(q0))
        robot.set_joint_position_target(q0)
        obj.write_root_pose_to_sim(pose)
        obj.write_root_velocity_to_sim(torch.zeros(n, 6, device=device))
        for sensor in env.scene.sensors.values():
            sensor.reset()
        step_physics(env, 1)
        first_contacts = hand_box_contact_forces(env)
        table_contact = torch.linalg.vector_norm(env.scene.sensors["left_arm_contacts"].data.net_forces_w, dim=-1)
        arm_body_names = env.scene.sensors["left_arm_contacts"].body_names
        result = settle_check(env, kinematics, arm_q, pose, target_pos, target_quat, ARGS.settle_steps)
        m = result.metrics
        # Palm pose the arm actually reached (IK may fall short of the target): usable as the spec's pregrasp.
        reached_pos = robot.data.body_pos_w[:, kinematics.palm_body_id].clone()
        reached_quat = robot.data.body_quat_w[:, kinematics.palm_body_id].clone()
        settled_obj_pos = obj.data.root_pos_w.clone()
        settled_obj_quat = obj.data.root_quat_w.clone()
        settled_contact = hand_box_contact_forces(env).amax(dim=-1)
        settled_speed = torch.linalg.vector_norm(obj.data.root_lin_vel_w, dim=-1)
        settled_up = quat_apply(settled_obj_quat, up)
        settled_tilt = torch.rad2deg(torch.acos(settled_up[:, 2].clamp(-1.0, 1.0)))
        rel = fingertips_w(robot, tip_ids) - obj.data.root_pos_w.unsqueeze(1)
        tips_obj = quat_apply(quat_conjugate(obj.data.root_quat_w).repeat_interleave(3, 0), rel.reshape(-1, 3)).reshape(n, 3, 3)
        tip_r = torch.linalg.vector_norm(tips_obj[..., :2], dim=-1)
        save_images(env, "0_pregrasp")
        save_snapshot(env, "0_pregrasp")
        if ARGS.snapshot_dir is not None:
            torch.save({"meshes": link_meshes(env), "support_height_m": spec.support_height_m,
                        "object_usd": spec.usd_path}, ARGS.snapshot_dir / "geometry.pt")

        rows = []
        for i, cand in enumerate(candidates):
            rows.append({
                "env": i, "phi": cand[0], "pitch": cand[1], "roll": cand[2], "radial_mm": cand[3] * 1e3,
                "dz_mm": cand[4] * 1e3,
                "object_dxy_mm": [round(cand[5] * 1e3), round(cand[6] * 1e3)],
                # Spec values for this candidate (palm target relative to the object at rest).
                "grasp_quat_wxyz": [round(v, 6) for v in target_quat[i].tolist()],
                "grasp_offset_world": [round(v, 6) for v in (target_pos[i] - obj_pos[i]).tolist()],
                "turn_point_world": [round(v, 6) for v in (turn[i] - env.scene.env_origins[i]).tolist()],
                "object_xy": [round(v, 6) for v in (obj_pos[i] - env.scene.env_origins[i])[:2].tolist()],
                "ik_mm": round(ik["ik_position_error_m"][i].item() * 1e3, 1),
                "ik_deg": round(math.degrees(ik["ik_rotation_error_rad"][i].item()), 1),
                "margin_rad": round(ik["joint_limit_margin_rad"][i].item(), 3),
                "obj_contact_n": round(m["box_contact_force_n"][i].item(), 2),
                "arm_contact_n": round(m["arm_contact_force_n"][i].item(), 2),
                "obj_moved_mm": round(m["box_displacement_m"][i].item() * 1e3, 1),
                "palm_sag_mm": round(m["settled_palm_position_error_m"][i].item() * 1e3, 1),
                # After the hold: the object may have been pushed off the hand. If it is at rest,
                # upright and untouched, this relative pose is a collision-free pregrasp.
                "settled": {
                    "contact_n": round(settled_contact[i].item(), 2),
                    "speed_mps": round(settled_speed[i].item(), 4),
                    "tilt_deg": round(settled_tilt[i].item(), 2),
                    "grasp_quat_wxyz": [round(v, 6) for v in reached_quat[i].tolist()],
                    "grasp_offset_world": [round(v, 6) for v in (reached_pos[i] - settled_obj_pos[i]).tolist()],
                    "object_xy": [round(v, 6) for v in (settled_obj_pos[i] - env.scene.env_origins[i])[:2].tolist()],
                },
                "first_step_object_contact_n": {nm[len("object_contact_"):]: round(f, 1) for nm, f in
                                                zip(sensor_names, first_contacts[i].tolist()) if f > 0.5},
                "first_step_arm_contact_n": {nm: round(f, 1) for nm, f in
                                             zip(arm_body_names, table_contact[i].tolist()) if f > 0.5},
                # Fingertips in the object frame: (radius, height) mm for thumb, index, middle.
                "tips_r_z_mm": [[round(tip_r[i, k].item() * 1e3), round(tips_obj[i, k, 2].item() * 1e3)]
                                for k in range(3)],
            })

        if ARGS.grasp_test:
            hand_ids = {name: robot.joint_names.index(name) for name in LEFT_HAND_JOINT_NAMES}
            targets = result.joint_target.clone()
            curl = {f: targets[:, hand_ids[j[0]]].abs().clone() for f, (j, _) in FINGERS.items()}
            stop = {f: torch.full((n,), float("inf"), device=device) for f in FINGERS}
            for _ in range(ARGS.close_steps):
                forces = dict(zip(sensor_names, hand_box_contact_forces(env).unbind(-1)))
                for finger, (joints, sensors) in FINGERS.items():
                    hit = torch.stack([forces[s] > 0.5 for s in sensors]).any(0)
                    stop[finger] = torch.where(hit & stop[finger].isinf(), curl[finger] + ARGS.squeeze, stop[finger])
                    limit = torch.minimum(stop[finger], torch.full_like(stop[finger], abs(CLOSED[joints[0]]) * 1.2))
                    curl[finger] = torch.minimum(curl[finger] + ARGS.close_rate, limit)
                    for joint in joints:
                        targets[:, hand_ids[joint]] = math.copysign(1.0, CLOSED[joint]) * curl[finger]
                robot.set_joint_position_target(targets)
                step_physics(env, 1)
            closed_contacts = hand_box_contact_forces(env)
            save_images(env, "1_closed")
            save_snapshot(env, "1_closed")

            # Raise the palm by joint targets from the current Jacobian; the arm is never written.
            ids = kinematics.arm_joint_ids
            start_pos = robot.data.body_pos_w[:, kinematics.palm_body_id].clone()
            start_quat = robot.data.body_quat_w[:, kinematics.palm_body_id].clone()
            lift_start = obj.data.root_pos_w[:, 2].clone()
            for step in range(ARGS.lift_steps):
                goal = start_pos.clone()
                goal[:, 2] += ARGS.lift_m * min(1.0, (step + 1) / (0.8 * ARGS.lift_steps))
                pos = robot.data.body_pos_w[:, kinematics.palm_body_id]
                quat = robot.data.body_quat_w[:, kinematics.palm_body_id]
                dpos, drot = compute_pose_error(pos, quat, goal, start_quat, rot_error_type="axis_angle")
                jac = kinematics._analytic_jacobian()
                q = targets[:, ids]
                dq = bounded_dls(jac, torch.cat((dpos, drot), -1).clamp(-0.01, 0.01),
                                 (kinematics.lower - q).clamp(max=0.0), (kinematics.upper - q).clamp(min=0.0), 0.05)
                dq[:, 0] = 0.0  # waist stays
                targets[:, ids] = q + dq
                robot.set_joint_position_target(targets)
                step_physics(env, 1)
            lifted = obj.data.root_pos_w[:, 2] - lift_start
            up_now = quat_apply(obj.data.root_quat_w, up)
            tilt = torch.rad2deg(torch.acos(up_now[:, 2].clamp(-1.0, 1.0)))
            final = hand_box_contact_forces(env)
            save_images(env, "2_lifted")
            save_snapshot(env, "2_lifted")
            names = [s[len("object_contact_"):] for s in sensor_names]
            for i, row in enumerate(rows):
                row["closed_touching"] = [nm for nm, f in zip(names, closed_contacts[i].tolist()) if f > 0.5]
                row["lift_mm"] = round(lifted[i].item() * 1e3, 1)
                row["tilt_deg"] = round(tilt[i].item(), 1)
                row["lifted_touching"] = [nm for nm, f in zip(names, final[i].tolist()) if f > 0.5]

        for row in rows:
            print(json.dumps(row), flush=True)
        if ARGS.snapshot_dir is not None:
            (ARGS.snapshot_dir / "rows.json").write_text(json.dumps(rows, indent=1) + "\n")
        ok = [r for r in rows if r["ik_mm"] < 3 and r["ik_deg"] < 3 and r["margin_rad"] > 0.02
              and r["obj_contact_n"] < 1 and r["arm_contact_n"] < 1]
        if not ok and ARGS.grasp_test:
            # Nothing clean: report the best lift anyway (its reached pose is still a valid start).
            ok = [r for r in rows if r["lift_mm"] > 0]
        if ARGS.grasp_test:
            ok.sort(key=lambda r: (-r["lift_mm"], r["tilt_deg"]))
        if ok:
            i = ok[0]["env"]
            offset = (target_pos[i] - obj_pos[i]).tolist()
            print("[probe] best:", json.dumps({
                "env": i, "grasp_quat_wxyz": target_quat[i].tolist(), "grasp_offset_world": offset,
                "reached_grasp_quat_wxyz": reached_quat[i].tolist(),
                "reached_grasp_offset_world": (reached_pos[i] - settled_obj_pos[i]).tolist(),
                "turn_point_world": (turn[i] - env.scene.env_origins[i]).tolist(),
                "object_xy": (obj_pos[i] - env.scene.env_origins[i])[:2].tolist(),
            }), flush=True)
        else:
            print("[probe] no candidate passed the IK / collision checks", flush=True)
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
