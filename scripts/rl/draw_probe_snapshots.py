#!/usr/bin/env python3
"""Draw probe_rim_pregrasp.py --snapshot-dir output without Isaac Sim's renderer.

One PNG per env: columns are the stages (pregrasp, closed, lifted), rows two
views of the rim point: along the rim tangent (thumb over the rim, fingers under
the lip) and from outside and above. Meshes are the USD link visuals and the
object's visual mesh, merged on a coarse grid so matplotlib stays fast.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


# probe_rim_pregrasp.py stages, then scripts/rl/play.py --snapshot-dir stages.
STAGES = ("0_pregrasp", "1_closed", "2_lifted", "1_mid", "2_late")
COLORS = {"thumb": (0.85, 0.25, 0.2), "index": (0.2, 0.65, 0.3), "middle": (0.2, 0.4, 0.85)}
HAND_COLOR, ARM_COLOR, OBJECT_COLOR = (0.95, 0.65, 0.3), (0.6, 0.6, 0.6), (0.8, 0.85, 0.9)


def simplify(vertices: np.ndarray, faces: np.ndarray, cell: float) -> tuple[np.ndarray, np.ndarray]:
    """Merge vertices on a grid of this cell size and drop collapsed triangles."""
    keys = np.round(vertices / cell).astype(np.int64)
    _, first, inverse = np.unique(keys, axis=0, return_index=True, return_inverse=True)
    faces = inverse.reshape(-1)[faces]
    keep = (faces[:, 0] != faces[:, 1]) & (faces[:, 1] != faces[:, 2]) & (faces[:, 0] != faces[:, 2])
    return vertices[first], np.unique(np.sort(faces[keep], axis=1), axis=0)


def load_obj(path: Path) -> tuple[np.ndarray, np.ndarray]:
    vertices, faces = [], []
    for line in path.read_text().splitlines():
        if line.startswith("v "):
            vertices.append([float(x) for x in line.split()[1:4]])
        elif line.startswith("f "):
            idx = [int(tok.split("/")[0]) - 1 for tok in line.split()[1:]]
            faces.extend((idx[0], idx[k], idx[k + 1]) for k in range(1, len(idx) - 1))
    return np.array(vertices), np.array(faces)


def quat_matrix(q) -> np.ndarray:
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


def link_color(name: str):
    for finger, color in COLORS.items():
        if finger in name:
            return color
    return HAND_COLOR if "hand" in name else ARM_COLOR


def shaded(triangles: np.ndarray, color, light=np.array([0.3, -0.4, 0.85])) -> np.ndarray:
    normal = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    normal /= np.linalg.norm(normal, axis=1, keepdims=True) + 1e-12
    brightness = 0.45 + 0.55 * np.abs(normal @ (light / np.linalg.norm(light)))
    return np.clip(np.outer(brightness, color), 0, 1)


def draw(ax, parts, center, elev, azim, half=0.11, support=None):
    for triangles, color, alpha in parts:
        dist = np.linalg.norm(triangles.mean(1) - center, axis=1)
        triangles = triangles[dist < 2.2 * half]
        if len(triangles):
            ax.add_collection3d(Poly3DCollection(triangles, facecolors=shaded(triangles, color), alpha=alpha,
                                                 linewidths=0))
    if support is not None:
        x = np.array([center[0] - half, center[0] + half])
        y = np.array([center[1] - half, center[1] + half])
        xx, yy = np.meshgrid(x, y)
        ax.plot_surface(xx, yy, np.full_like(xx, support), color=(0.55, 0.4, 0.3), alpha=0.35)
    ax.set_xlim(center[0] - half, center[0] + half)
    ax.set_ylim(center[1] - half, center[1] + half)
    ax.set_zlim(center[2] - half, center[2] + half)
    ax.set_box_aspect((1, 1, 1))
    ax.view_init(elev=elev, azim=azim)
    ax.set_proj_type("ortho")
    ax.set_axis_off()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("snapshot_dir", type=Path)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--cell-mm", type=float, default=2.5)
    parser.add_argument("--rim-z-m", type=float, default=0.050)
    args = parser.parse_args()
    out_dir = args.out_dir or args.snapshot_dir / "images"
    out_dir.mkdir(parents=True, exist_ok=True)

    geometry = torch.load(args.snapshot_dir / "geometry.pt", weights_only=False)
    cell = args.cell_mm * 1e-3
    meshes = {name: simplify(v.astype(np.float64), f, cell) for name, (v, f) in geometry["meshes"].items()}
    obj_path = Path(geometry["object_usd"]).with_name("visual.obj")
    obj_mesh = simplify(*load_obj(obj_path), cell)
    rows = json.loads((args.snapshot_dir / "rows.json").read_text())
    stages = [s for s in STAGES if (args.snapshot_dir / f"{s}.pt").is_file()]
    snaps = {s: torch.load(args.snapshot_dir / f"{s}.pt", weights_only=False) for s in stages}

    for row in rows:
        i = row["env"]
        fig = plt.figure(figsize=(4.2 * len(stages), 8.4))
        for col, stage in enumerate(stages):
            snap = snaps[stage]
            parts = []
            for name, (v, f) in meshes.items():
                b = snap["body_names"].index(name)
                world = v @ quat_matrix(snap["body_quat"][i, b].numpy()).T + snap["body_pos"][i, b].numpy()
                parts.append((world[f], link_color(name), 1.0))
            obj_pos = snap["object_pos"][i].numpy()
            obj_world = obj_mesh[0] @ quat_matrix(snap["object_quat"][i].numpy()).T + obj_pos
            parts.append((obj_world[obj_mesh[1]], OBJECT_COLOR, 0.55))
            # Views follow the start pose so every stage is framed the same.
            start = snaps[stages[0]]["object_pos"][i].numpy()
            phi = math.radians(row["phi"])
            center = start + np.array([0.078 * math.cos(phi), 0.078 * math.sin(phi), args.rim_z_m])
            # Matplotlib's azim is the camera's direction from the center, in degrees.
            tangent_azim = row["phi"] + 90.0
            for r, (elev, azim) in enumerate(((0.0, tangent_azim), (35.0, row["phi"] + 35.0))):
                ax = fig.add_subplot(2, len(stages), r * len(stages) + col + 1, projection="3d")
                draw(ax, parts, center, elev, azim, support=geometry["support_height_m"])
                if r == 0:
                    ax.set_title(stage[2:], fontsize=11)
        if "title" in row:
            title = row["title"]
        else:
            title = None
        extra = ""
        if title is None and "lift_mm" in row:
            extra = (f"\nclosed touching: {', '.join(row['closed_touching']) or '-'}   lift {row['lift_mm']} mm, "
                     f"tilt {row['tilt_deg']} deg, still touching: {', '.join(row['lifted_touching']) or '-'}")
        fig.suptitle(title if title is not None else
            f"env {i}: phi {row['phi']:.0f} deg, pitch {row['pitch']:.0f} deg, roll {row['roll']:.0f} deg, "
            f"radial {row['radial_mm']:.0f} mm, dz {row['dz_mm']:.0f} mm\n"
            f"IK {row['ik_mm']} mm / {row['ik_deg']} deg, object contact {row['obj_contact_n']} N, "
            f"arm contact {row['arm_contact_n']} N" + extra,
            fontsize=10,
        )
        fig.text(0.01, 0.01, "top row: along the rim tangent; bottom: outside, above.  "
                 "thumb red, index green, middle blue, bowl translucent, table brown", fontsize=8)
        fig.tight_layout(rect=(0, 0.02, 1, 0.93))
        path = out_dir / f"env{i:02d}.png"
        fig.savefig(path, dpi=90)
        plt.close(fig)
        print(path, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
