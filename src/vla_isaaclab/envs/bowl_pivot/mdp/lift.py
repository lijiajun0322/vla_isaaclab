"""Second stage of the pivot-then-lift task: lift the turned bowl 5 cm, level.

Stage 1 is the pivot task. The moment the bowl is upright on the table
(``upright_on_table``: within 15 deg, foot within 2 cm of the table top, no
stillness required, so the hands are still close) the env is latched
"flipped" and the bowl's root position recorded; stage 2 then scores a
held lift from there, as the grasp-lift task does: progress in lift height
while held, and at the step the held bowl's lowest point passes ``z_lifted_m``
above the table the episode ends, scored by how far it is from the flip pose raised ``z_lifted_m``
(collider-center offset + rim radius x tilt from upright).

"Held" depends on ``env.cfg.lift_hold_mode``: "any" (any link of either hand
on the bowl) or "grip" (thumb, index and middle of one hand on the bowl), for
``min_steps`` control steps in a row, so a flick does not count.
"""

from __future__ import annotations

import math

import torch

from isaaclab.utils.math import quat_apply, quat_conjugate

from vla_isaaclab.envs.ycb_grasp.mdp import grasp_rl

from . import pivot


_STATE_ATTR = "bowl_lift_state"
# Hand link short names (pivot.HAND_LINK_SHORT_NAMES order) that make a three-finger grip.
GRIP_GROUPS = (("thumb_1", "thumb_2"), ("index_0", "index_1"), ("middle_0", "middle_1"))


def _state(env) -> dict[str, torch.Tensor]:
    state = env.__dict__.get(_STATE_ATTR)
    if state is None:
        n, device = env.num_envs, env.device
        state = {
            "flipped": torch.zeros(n, dtype=torch.bool, device=device),
            "flip_pos": torch.zeros(n, 3, device=device),
            "held_streak": torch.zeros(n, device=device),
            "lifted": torch.zeros(n, dtype=torch.bool, device=device),
            "hold_steps": torch.zeros(n, device=device),
        }
        env.__dict__[_STATE_ATTR] = state
    return state


def reset_lift_state(env, env_ids: torch.Tensor | None):
    """Reset event: back to stage 1."""
    ids = slice(None) if env_ids is None else env_ids
    state = _state(env)
    state["flipped"][ids] = False
    state["held_streak"][ids] = 0.0
    state["lifted"][ids] = False
    state["hold_steps"][ids] = 0.0


def _cached(env, key: str, compute):
    cache = env.__dict__.setdefault("_bowl_lift_cache", {})
    step = int(env.common_step_counter)
    if cache.get((key, "step")) != step:
        cache[(key, "step")] = step
        cache[key] = compute()
    return cache[key]


def upright_on_table(env, max_tilt_deg: float = 15.0, max_height_m: float = 0.02) -> torch.Tensor:
    """Bowl upright within max_tilt_deg with its foot within max_height_m of the table top (moving or not)."""
    foot = pivot._bowl(env).data.root_pos_w[:, 2] - env.cfg.object_spec.support_height_m
    return (pivot.bowl_tilt(env) < math.radians(max_tilt_deg)) & (foot < max_height_m)


def stage(env) -> tuple[torch.Tensor, torch.Tensor]:
    """(flipped, flipped this step); the switch is upright_on_table."""

    def compute():
        state = _state(env)
        newly = upright_on_table(env) & ~state["flipped"]
        state["flip_pos"][newly] = pivot._bowl(env).data.root_pos_w[newly]
        state["flipped"] |= newly
        return state["flipped"].clone(), newly

    return _cached(env, "stage", compute)


def grips(env) -> torch.Tensor:
    """Three-finger grip per hand (thumb, index and middle each touching), (num_envs, 2) for left, right."""

    def compute():
        per_hand = []
        for side in pivot.SIDES:
            forces = dict(zip(pivot.HAND_LINK_SHORT_NAMES, pivot.hand_contact_forces(env, side).unbind(-1)))
            touching = [torch.stack([forces[link] for link in group], -1).amax(-1) > pivot.CONTACT_THRESHOLD_N
                        for group in GRIP_GROUPS]
            per_hand.append(touching[0] & touching[1] & touching[2])
        return torch.stack(per_hand, dim=-1)

    return _cached(env, "grips", compute)


# -- where a hand touches the bowl: inside (the cavity) or outside ------------------
#
# Each touching hand link is placed in the bowl frame (root = foot center, +z = opening):
# the distal links by their fingertip point, the others by their origin. Within
# INNER_RADIUS_M of the axis, between the floor and the rim, it is on the inner wall.
INNER_RADIUS_M = 0.068   # inner wall ~66-69 mm (outer 70-73 mm, wall ~4 mm)
INNER_Z_M = (0.008, 0.056)
_DISTAL = {"thumb_2": (0.0, -0.045, 0.0), "index_1": (0.045, 0.0, 0.0), "middle_1": (0.045, 0.0, 0.0)}


def _hand_points_in_bowl(env, side: str) -> torch.Tensor:
    """Hand link points (pivot.HAND_LINK_SHORT_NAMES order) in the bowl frame, (num_envs, 8, 3)."""
    robot, bowl = pivot._robot(env), pivot._bowl(env)
    ids = pivot._body_ids(env, tuple(f"{side}_hand_{short}_link" for short in pivot.HAND_LINK_SHORT_NAMES))
    pos, quat = robot.data.body_pos_w[:, ids], robot.data.body_quat_w[:, ids]
    mirror = -1.0 if side == "right" else 1.0  # the right hand's links are the left's with y flipped
    offsets = pos.new_tensor([[x, mirror * y, z] for x, y, z in
                              (_DISTAL.get(short, (0.0, 0.0, 0.0)) for short in pivot.HAND_LINK_SHORT_NAMES)])
    count = pos.shape[1]
    points = pos + quat_apply(quat.reshape(-1, 4), offsets.expand(env.num_envs, -1, -1).reshape(-1, 3)).reshape(pos.shape)
    inv = quat_conjugate(bowl.data.root_quat_w).unsqueeze(1).expand(-1, count, -1).reshape(-1, 4)
    return quat_apply(inv, (points - bowl.data.root_pos_w.unsqueeze(1)).reshape(-1, 3)).reshape(env.num_envs, count, 3)


def contact_sides(env) -> tuple[torch.Tensor, torch.Tensor]:
    """(touching the inner wall, touching the outside) per hand, each (num_envs, 2) for left, right; once per step."""

    def compute():
        inner, outer = [], []
        for side in pivot.SIDES:
            touching = pivot.hand_contact_forces(env, side) > pivot.CONTACT_THRESHOLD_N
            local = _hand_points_in_bowl(env, side)
            radial = torch.linalg.vector_norm(local[..., :2], dim=-1)
            inside = (radial < INNER_RADIUS_M) & (local[..., 2] > INNER_Z_M[0]) & (local[..., 2] < INNER_Z_M[1])
            inner.append((touching & inside).any(dim=-1))
            outer.append((touching & ~inside).any(dim=-1))
        return torch.stack(inner, dim=-1), torch.stack(outer, dim=-1)

    return _cached(env, "contact_sides", compute)


# -- where to hold it: on the outside of the wall, low, on each hand's side ----------
SUPPORT_OUT_M = 0.09   # from the bowl axis: outer wall ~70-73 mm plus the fingers
SUPPORT_UP_M = 0.015   # above the foot along the bowl axis


def support_points(env) -> torch.Tensor:
    """World support points beside the bowl for the left and right hand, (num_envs, 2, 3).

    From the bowl axis SUPPORT_UP_M above the foot, SUPPORT_OUT_M out along world -x
    (left hand) and +x (right hand), taken perpendicular to the bowl axis.
    """
    bowl = pivot._bowl(env)
    up = pivot.bowl_up(env)
    base = bowl.data.root_pos_w + SUPPORT_UP_M * up
    points = []
    for sign in (-1.0, 1.0):
        lateral = torch.zeros_like(up); lateral[:, 0] = sign
        lateral = lateral - (lateral * up).sum(-1, keepdim=True) * up
        lateral = lateral / torch.linalg.vector_norm(lateral, dim=-1, keepdim=True).clamp_min(1e-6)
        points.append(base + SUPPORT_OUT_M * lateral)
    return torch.stack(points, dim=1)


def hand_centers(env) -> torch.Tensor:
    """Mean of the palm origin and the three fingertip points per hand (world), (num_envs, 2, 3)."""
    robot = pivot._robot(env)
    centers = []
    for side in pivot.SIDES:
        local = _hand_points_in_bowl(env, side)[:, [0, 3, 5, 7]].mean(dim=1)  # palm, thumb/index/middle tips
        bowl = pivot._bowl(env)
        centers.append(bowl.data.root_pos_w + quat_apply(bowl.data.root_quat_w, local))
    return torch.stack(centers, dim=1)


def held(env, min_steps: int = 3) -> torch.Tensor:
    """The bowl held for min_steps steps in a row (cfg.lift_hold_mode): "any" hand contact, a three-finger
    "grip", or "outer": a hand on the outside of the bowl (a finger hooked inside does not count)."""

    def compute():
        if env.cfg.lift_hold_mode == "grip":
            now = grips(env).any(dim=-1)
        elif env.cfg.lift_hold_mode == "outer":
            now = contact_sides(env)[1].any(dim=-1)
        else:
            now = pivot.hand_touches_bowl(env, "left") | pivot.hand_touches_bowl(env, "right")
        streak = _state(env)["held_streak"]
        streak[:] = torch.where(now, streak + 1.0, torch.zeros_like(streak))
        return streak >= min_steps

    return _cached(env, "held", compute)


def lift_height(env) -> torch.Tensor:
    """Height of the bowl's lowest point above the table (0 before the flip).

    Not the root (the foot center): a bowl tipped onto its side raises its foot
    ~8 cm without leaving the table, which counted as a 5 cm lift.
    """
    flipped, _ = stage(env)
    height = pivot.bowl_lowest_height(env)
    return torch.where(flipped, height, torch.zeros_like(height))


def lift_error(env, z_lifted_m: float = 0.05) -> torch.Tensor:
    """Distance to the flip pose raised z_lifted_m, upright: collider-center offset + rim radius x tilt."""
    spec = env.cfg.object_spec
    target = _state(env)["flip_pos"] + pivot._bowl(env).data.root_pos_w.new_tensor(spec.box_center_m)
    target[:, 2] += z_lifted_m
    offset = torch.linalg.vector_norm(pivot.bowl_center(env) - target, dim=-1)
    return offset + spec.rim_radius_m * pivot.bowl_tilt(env)


def lift_event(env, z_lifted_m: float = 0.05) -> tuple[torch.Tensor, torch.Tensor]:
    """(held bowl passed z_lifted_m above the flip pose this step, lift error); once per step."""

    def compute():
        flipped, _ = stage(env)
        fired = flipped & held(env) & (lift_height(env) > z_lifted_m)
        return fired, lift_error(env, z_lifted_m)

    return _cached(env, "lift_event", compute)


# -- hold: after a good lift the bowl must stay up, level and still -------------
#
# The first good lift (held, lowest point past z_lifted_m, error < success_m) no longer
# ends the episode: it latches "lifted". From then a step counts toward the hold while
# the bowl is held, its lowest point above hold_min_m and its lift error below
# hold_error_m; cfg.lift_hold_s of such steps in a row is the success. Falling back
# below drop_m ends the episode as "dropped".


def hold_phase(env, z_lifted_m: float = 0.05, success_m: float = 0.02, hold_min_m: float = 0.04,
               hold_error_m: float = 0.03, drop_m: float = 0.01) -> dict[str, torch.Tensor]:
    """first_lift (this step), lifted_now (latched this step, good), in_hold, hold_steps, dropped; once per step."""

    def compute():
        state = _state(env)
        fired, error = lift_event(env, z_lifted_m)
        first_lift = fired & ~state["lifted"]
        lifted_now = first_lift & (error < success_m)
        state["lifted"] |= lifted_now
        height = lift_height(env)
        in_hold = state["lifted"] & held(env) & (height > hold_min_m) & (error < hold_error_m)
        state["hold_steps"][:] = torch.where(in_hold, state["hold_steps"] + 1.0, torch.zeros_like(state["hold_steps"]))
        dropped = state["lifted"] & ~lifted_now & (height < drop_m)
        return {"first_lift": first_lift, "lifted_now": lifted_now, "in_hold": in_hold,
                "hold_steps": state["hold_steps"].clone(), "dropped": dropped, "error": error}

    return _cached(env, "hold_phase", compute)


def hold_still(env, speed_scale_mps: float = 0.05, spin_scale_radps: float = 0.5, error_scale_m: float = 0.02) -> torch.Tensor:
    """While holding: closeness to the target pose times stillness, 1 for a bowl at the target and not moving."""
    phase = hold_phase(env)
    bowl = pivot._bowl(env)
    still = torch.exp(-torch.linalg.vector_norm(bowl.data.root_lin_vel_w, dim=-1) / speed_scale_mps) * torch.exp(
        -torch.linalg.vector_norm(bowl.data.root_ang_vel_w, dim=-1) / spin_scale_radps)
    return phase["in_hold"].float() * torch.exp(-phase["error"] / error_scale_m) * still


def held_aloft(env, min_height_m: float = 0.03, max_height_m: float = 0.10) -> torch.Tensor:
    """After a good lift: held with its lowest point in [min, max] above the table, times levelness (1 - tilt / 90 deg)."""
    phase = hold_phase(env)
    height = lift_height(env)
    up = _state(env)["lifted"] & held(env) & (height > min_height_m) & (height < max_height_m)
    level = (1.0 - pivot.bowl_tilt(env) / (math.pi / 2.0)).clamp(min=0.0)
    return up.float() * level * (~phase["dropped"]).float()


def rising_after_lift(env, free_mps: float = 0.02, scale_mps: float = 0.10) -> torch.Tensor:
    """After a good lift: the bowl's upward speed above free_mps, scaled to [0, 1] (it should stop, not keep rising)."""
    vz = pivot._bowl(env).data.root_lin_vel_w[:, 2]
    return _state(env)["lifted"].float() * ((vz - free_mps) / scale_mps).clamp(0.0, 1.0)


# -- rewards ------------------------------------------------------------------


def flip_bonus(env) -> torch.Tensor:
    """1 on the step stage 1 completes (the old pivot success)."""
    return stage(env)[1].float()


def stage1_upright(env) -> torch.Tensor:
    """pivot.upright before the flip only: once turned, staying upright alone earns nothing."""
    return pivot.upright(env) * (~stage(env)[0]).float()


def stage2_hands_near(env) -> torch.Tensor:
    """After the flip: pivot.hands_near_bowl (1 when both hands reach the bowl)."""
    return pivot.hands_near_bowl(env) * stage(env)[0].float()


def stage2_contact(env) -> torch.Tensor:
    """After the flip: 0.5 per hand with any link on the bowl, to bring the hands back to it."""
    return pivot.hands_contact(env) * stage(env)[0].float()


def stage2_support_near(env, scale_m: float = 0.10) -> torch.Tensor:
    """After the flip: mean over hands of 1 - tanh(distance from hand center to its support point / scale_m)."""
    distance = torch.linalg.vector_norm(hand_centers(env) - support_points(env), dim=-1)
    return (1.0 - torch.tanh(distance / scale_m)).mean(dim=-1) * stage(env)[0].float()


def stage2_support_progress(env) -> torch.Tensor:
    """After the flip: progress in the hands' mean distance to their support points (pays on each new best, m)."""
    flipped, _ = stage(env)
    distance = torch.linalg.vector_norm(hand_centers(env) - support_points(env), dim=-1).mean(dim=-1)
    return grasp_rl._progress_on(env, "bowl_support_distance", distance, active=flipped)


def stage2_outer_contact(env) -> torch.Tensor:
    """After the flip: 0.5 per hand touching the outside of the bowl."""
    return 0.5 * contact_sides(env)[1].float().sum(dim=-1) * stage(env)[0].float()


def stage2_inner_contact(env) -> torch.Tensor:
    """After the flip: 0.5 per hand touching the inner wall (penalized)."""
    return 0.5 * contact_sides(env)[0].float().sum(dim=-1) * stage(env)[0].float()


def stage2_level(env, min_height_m: float = 0.005) -> torch.Tensor:
    """After the flip, while held and off the table: 1 - tilt / 90 deg (1 level, 0 on its side or worse).

    Paid every step of the lift, not only at the 5 cm mark: before it the policy
    lifted the bowl gripped on its side.
    """
    flipped, _ = stage(env)
    airborne = flipped & held(env) & (lift_height(env) > min_height_m)
    level = (1.0 - pivot.bowl_tilt(env) / (math.pi / 2.0)).clamp(min=0.0)
    return level * airborne.float()


def held_lift_progress(env, z_lifted_m: float = 0.05) -> torch.Tensor:
    """Progress on (z_lifted - lift height) while flipped and held, until lifted (grasp-lift's held_lift)."""
    flipped, _ = stage(env)
    height = lift_height(env)
    active = flipped & held(env) & (height <= z_lifted_m) & ~_state(env)["lifted"]
    error = (z_lifted_m - height).clamp(min=0.0)
    return grasp_rl._progress_on(env, "bowl_lift_error", error, active=active)


def lift_pose_score(env, z_lifted_m: float = 0.05, scale_m: float = 0.03) -> torch.Tensor:
    """On the first lift step: exp(-lift error / scale_m), 1 for a level, unshifted bowl."""
    phase = hold_phase(env, z_lifted_m)
    return phase["first_lift"].float() * torch.exp(-phase["error"] / scale_m)


def one_hand_grip(env) -> torch.Tensor:
    """Grip variant, per step in stage 2: 1 while a hand holds the bowl with thumb, index and middle."""
    flipped, _ = stage(env)
    return (flipped & grips(env).any(dim=-1)).float()


def free_hand_at_success(env, z_lifted_m: float = 0.05, success_m: float = 0.02) -> torch.Tensor:
    """Grip variant, on success: 1 if one hand grips with three fingers and the other does not touch the bowl."""
    grip = grips(env)
    touch = torch.stack([pivot.hand_touches_bowl(env, side) for side in pivot.SIDES], dim=-1)
    one_handed = (grip[:, 0] & ~touch[:, 1]) | (grip[:, 1] & ~touch[:, 0])
    return (lift_success(env, z_lifted_m, success_m) & one_handed).float()


# -- terminations -------------------------------------------------------------


def lift_success(env, z_lifted_m: float = 0.05, success_m: float = 0.02) -> torch.Tensor:
    """Termination: after a good lift, the bowl held up, level and still near the target for cfg.lift_hold_s."""
    hold_steps = round(getattr(env.cfg, "lift_hold_s", 0.0) / env.step_dt)
    phase = hold_phase(env, z_lifted_m, success_m)
    if hold_steps <= 0:
        return phase["lifted_now"]
    return phase["hold_steps"] >= hold_steps


def lift_off_pose(env, z_lifted_m: float = 0.05, success_m: float = 0.02) -> torch.Tensor:
    """Termination: the first time the held bowl passes z_lifted_m it is tilted or shifted beyond success_m."""
    phase = hold_phase(env, z_lifted_m, success_m)
    return phase["first_lift"] & (phase["error"] >= success_m)


def bowl_dropped(env) -> torch.Tensor:
    """Termination: after a good lift the bowl fell back onto the table."""
    return hold_phase(env)["dropped"]


__all__ = [
    "bowl_dropped",
    "flip_bonus",
    "hold_phase",
    "hold_still",
    "held_aloft",
    "rising_after_lift",
    "free_hand_at_success",
    "held_lift_progress",
    "lift_off_pose",
    "lift_pose_score",
    "lift_success",
    "one_hand_grip",
    "reset_lift_state",
    "stage",
    "stage1_upright",
    "stage2_contact",
    "stage2_hands_near",
    "stage2_inner_contact",
    "stage2_outer_contact",
    "stage2_support_near",
    "stage2_support_progress",
    "support_points",
    "stage2_level",
    "upright_on_table",
]
