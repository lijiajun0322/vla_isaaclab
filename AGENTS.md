# AGENTS.md

Read this file before editing or running the simulator.

This file can fall behind the project. If any rule here seems not to fit the
current task, tell the user which rule and why, and ask before working around
it; do not silently ignore it or silently comply.

## Environment

- Activate a developer-owned Conda environment first, or set `VLA_ISAACLAB_ENV`.
- Shared Isaac Lab: `${ISAACLAB_ROOT:-/media/data-ssd/software/IsaacLab-v2.0.2}`.
- Required Isaac Lab tag: `v2.0.2`; Isaac Sim: 4.5.0.0.
- Do not clone, checkout, pull, or modify the shared Isaac Lab dependency.
- Do not use sudo, alter drivers/system CUDA, or touch another user's files.

Before simulator work:

```bash
conda activate <developer-env>
source scripts/activate.sh
check_install_environment
```

## Safety

- Preserve unrelated work and generated outputs.
- Do not reset, clean, stash, or commit unless explicitly asked.
- Never teleport, parent, kinematically move, or invisibly attach an object.
  This guards grasps: holding, lifting and transport must come from simulated
  contact and friction. Reset events may still write robot and object state
  (randomized poses, `reset_from_pregrasp_table`), as long as the object starts
  at rest on its support and out of contact with the hand.
- Never claim task success unless its named success termination fires.
- Never save or label a failed manipulation as a successful demonstration.
- If the same task reproduction fails three times, stop and request human inspection.
- Report problems to the user as soon as they come up (a crash, an approach that
  does not work, a result that contradicts the plan): say what happened, what
  you tried and the options, then wait. Do not keep debugging or switching
  approaches on your own for long stretches.

## Architecture

Complete environments are registered Gym IDs. There is no independent
World/Object/Task/Expert/Controller registry and no custom joint ActionTerm.

```text
Gym ID -> EnvCfg -> Scene + Isaac Lab Managers -> normalized action -> robot
                     ↑
               optional scripted policy
```

- `envs/common/`: reusable G1 and physical scene/config helpers, plus
  `objects.py`: one `GraspObjectSpec` per graspable object (see below).
- `envs/<task>/env_cfg.py`: concrete robot, object, camera, action and managers.
- `envs/<task>/mdp/`: command, observation, reward, event and termination terms.
- `policies/`: optional scripted strategy and action generation.
- `recording/`: HDF5 staging and contract-aligned LeRobot v3/v2.1 export; v3 is default.
- `envs/ycb_grasp/`: object-generic grasp-and-lift RL task, one Gym ID per
  object (`VLA-YCBGraspLift-<Object>-G1-v0`), plus a `-Fast-v0` variant with
  5x speed limits for training from scratch before the slow fine-tune.
- `rl/`: RL support such as the per-object pregrasp state table and its reset event.

Use Isaac Lab `JointPositionToLimitsActionCfg` for the 43-D normalized action.
Its mapping is `-1=soft lower limit`, `0=midpoint`, `+1=soft upper limit`.
Policies that calculate physical joint targets must invert that exact mapping.
The action term follows the USD's internal joint order because Isaac Lab v2.0.2
does not expose `preserve_order` on this action config. Dataset recording must
explicitly remap state and processed targets into `g1_29body_dex3_43d_v1` order.
The only retained custom control algorithm is bounded DLS IK, used by the scripted
sugar-box policy and reused for the kinematic pregrasp solve in `rl/`.

Exception: RL training environments (separate Gym IDs) may use a smaller action
space built only from Isaac Lab's own action terms, e.g.
`DifferentialInverseKinematicsActionCfg` for the palm plus
`JointPositionActionCfg` for the hand. They still must not add a custom joint
ActionTerm, and the 43-D environments and data contract stay unchanged. Data
recorded from RL policies for VLA training uses the 43-D absolute joint targets.

Task-specific configs stay in their concrete environment. Do not recreate
top-level `robots/`, `objects/`, `worlds/`, `sensors/`, `controllers/`, or
`experts/` component layers.

Exception: per-object parameters used by more than one environment (asset,
collider size and axes, rest pose, reset randomization, calibrated scripted
approach, finger preshape) live in one `GraspObjectSpec` in
`envs/common/objects.py`. Environments, scripted policies and RL code read the
spec instead of redefining those values. Task-level values (goal offsets,
success thresholds, phase timing) stay in the concrete environment or policy.

Camera placement is owned by the concrete scene EnvCfg. Shared intrinsics and
modalities are owned by `envs/common/scene.py`.

## Reference behavior

- Preview IDs: YCB, dinnerware, microwave; 240 control steps at 30 Hz.
- Sugar-box ID: `VLA-YCBSugarBox-G1-JointPos-v0`.
- Sugar-box table contains only `004_sugar_box`.
- Validated success occurs at step 961 using physical contact and three fingers.
- Preserve the calibrated phase thresholds, pose, grasp, and success criteria
  unless the user explicitly requests behavioral changes.
- The sugar box's cooked convex-hull collider does not rest at the calibrated
  upright pose: written there, it rocks for ~1 s and settles ~3° tilted and
  ~5 mm away. Pregrasp-table states store the settled box; do not swap the
  collider without re-validating the scripted grasp.
- A filtered `ContactSensorCfg` only reports for one sensor body per env. For
  per-link object contact, use one sensor per link (see
  `YCBGraspStateSceneCfg`); the DR scene's `robot_box_contacts` spans every
  robot link and reports zero.
- RL pregrasp table: `./scripts/rl/build_pregrasp_table.sh --headless
  [--task VLA-YCBGraspLift-<Object>-G1-v0]` writes
  `outputs/rl/<object>/pregrasp_table_preshape.{pt,json,png}` (~1 min for 10k
  states at 1024 envs) with the fingers at the spec's `hand_preshape`;
  `--open-hand` writes `pregrasp_table_open.*` instead. A table records its
  object and refuses to load into another object's env.

Useful commands:

```bash
./scripts/run_env.sh --headless --physics-only --list-tasks
./scripts/run_env.sh --headless --physics-only \
  --task VLA-ScenePreview-YCB-G1-v0 --steps 240
./scripts/run_env.sh --headless --physics-only \
  --task VLA-YCBSugarBox-G1-JointPos-v0 --steps 1200
```
