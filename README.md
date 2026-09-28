# vla_isaaclab

`vla_isaaclab` is a robotics simulation framework for developing, running,
validating, and recording manager-based environments. It uses Isaac Sim as the
simulation runtime and Isaac Lab as the environment framework.

## Environment model

Isaac Lab is a shared, read-only external dependency on the lab server:

```text
shared
└── /media/data-ssd/software/IsaacLab-v2.0.2

per developer
├── own Conda environment
├── own vla_isaaclab checkout
└── own outputs and Kit caches
```

All developers use the pinned Isaac Lab `v2.0.2` checkout. They do not clone or
modify Isaac Lab inside this repository. Each developer installs this project
in their own Conda environment and keeps an independent checkout, generated
outputs, and local caches.

| Component | Version |
| --- | --- |
| Python | 3.10 |
| Isaac Sim | 4.5.0.0 |
| Isaac Lab | v2.0.2 |
| PyTorch | 2.5.1+cu121 |
| LeRobot dataset | v3 (default), v2.1 (optional) |

## Installation

Prerequisites are Linux, an NVIDIA GPU/driver compatible with Isaac Sim 4.5,
Git, Conda, and read access to the shared Isaac Lab checkout.

Choose your own Conda environment name:

```bash
git clone <this-repository-url> vla_isaaclab
cd vla_isaaclab

conda env create --name <your-env> --file environment.yml
conda activate <your-env>

export ISAACLAB_ROOT=/media/data-ssd/software/IsaacLab-v2.0.2
git -C "$ISAACLAB_ROOT" describe --tags --exact-match

python -m pip install -r requirements.txt
python -m pip install --no-deps lerobot==0.4.3
```

The Git command must print `v2.0.2`. Link the shared packages into this Conda
environment without writing to the shared checkout:

```bash
python - <<'PY'
import os
import site
from pathlib import Path

root = Path(os.environ["ISAACLAB_ROOT"]).resolve()
packages = ("isaaclab", "isaaclab_assets", "isaaclab_mimic", "isaaclab_rl", "isaaclab_tasks")
paths = [root / "source" / package for package in packages]
missing = [path for path in paths if not (path / path.name).is_dir()]
if missing:
    raise SystemExit(f"Missing shared Isaac Lab packages: {missing}")

link = Path(site.getsitepackages()[0]) / "isaaclab_shared.pth"
link.write_text("".join(f"{path}\n" for path in paths), encoding="utf-8")
print(f"Wrote {link}")
PY

python -m pip install -e .
source scripts/activate.sh
check_install_environment
```

There is intentionally no bootstrap script. `activate.sh` never guesses a
developer environment: activate it first, or explicitly set
`VLA_ISAACLAB_ENV=<your-env>`.

## Daily use

```bash
conda activate <your-env>
source scripts/activate.sh
check_install_environment
./scripts/run_env.sh --headless --physics-only --list-tasks
```

Wrappers source `activate.sh` but preserve the already selected non-`base`
environment. `ISAACLAB_ROOT` defaults to the shared server path and remains
overridable.

## Architecture

```text
vla_isaaclab/
├── assets/                          local robot, YCB, dinnerware, microwave assets
├── configs/                         Isaac Sim Kit configurations
├── scripts/
│   ├── activate.sh                 environment verification and local caches
│   ├── run_env.py/.sh              generic Gym environment runner
│   ├── record_lerobot.sh            LeRobot v3/v2.1 data generation
│   ├── replay_lerobot.py/.sh        normalized-action replay
│   └── prepare_*.py/.sh             asset preparation utilities
├── src/vla_isaaclab/
│   ├── envs/
│   │   ├── common/
│   │   │   ├── base.py             common simulation timing/material settings
│   │   │   ├── g1.py               reusable G1 config and joint semantics
│   │   │   ├── managers.py         built-in action, observations, reset events
│   │   │   ├── mdp.py              object-agnostic reset/termination terms
│   │   │   ├── objects.py          GraspObjectSpec per graspable object
│   │   │   └── scene.py            shared table/light/camera helpers
│   │   ├── scene_preview/
│   │   │   ├── __init__.py         three Gym registrations
│   │   │   └── env_cfg.py          YCB, dinnerware, microwave complete scenes
│   │   ├── ycb_grasp/
│   │   │   ├── __init__.py         one grasp-lift RL Gym ID per object
│   │   │   ├── env_cfg.py          camera-free G1 + one object scene
│   │   │   ├── rl_env_cfg.py       grasp-lift actions/observations/rewards
│   │   │   ├── agents/             RSL-RL PPO configs
│   │   │   └── mdp/grasp_rl.py     grasp-lift observation/reward/termination terms
│   │   └── ycb_sugar_box/
│   │       ├── __init__.py         sugar-box Gym registration
│   │       ├── env_cfg.py          G1 + one sugar box + all manager configs
│   │       └── mdp/                 command, observation, reward, termination terms
│   ├── policies/
│   │   ├── standing.py             normalized default-pose policy
│   │   ├── ycb_sugar_box*.py       scripted phases and action generation
│   │   ├── bounded_ik.py           bounded DLS IK helper
│   │   └── joint_limits.py         inverse of the built-in action mapping
│   ├── rl/                          pregrasp state table, RSL-RL helpers
│   └── recording/                   HDF5 staging and LeRobot v3/v2.1 pipeline
├── tests/                           simulator-free unit tests
└── outputs/                         local caches/videos/datasets; ignored by Git
```

Reusable G1, table, lighting, camera, action, observation, and reset
configuration lives under `envs/common`. Each concrete environment owns its
scene assets and task-specific configuration.

Each simulation setup is registered as a complete Gym environment:

```text
Gym environment ID
└── concrete EnvCfg
    ├── InteractiveSceneCfg     robot, scene assets, camera
    ├── ActionManager          normalized 43-D joint action
    ├── CommandManager         task goal, when applicable
    ├── ObservationManager     policy/task observations
    ├── EventManager           reset and joint targets
    ├── RewardManager          task rewards
    ├── TerminationManager     success/failure/time-out
    └── RecorderManager        HDF5 state/action/camera recording
```

The environment interface is a normalized 43-D joint action. Isaac Lab's
`JointPositionToLimitsActionCfg` maps it to G1 joint-position targets.

## Registration and extension

Importing `vla_isaaclab` registers these IDs with Gymnasium:

- `VLA-ScenePreview-YCB-G1-v0`
- `VLA-ScenePreview-Dinnerware-G1-v0`
- `VLA-ScenePreview-Microwave-G1-v0`
- `VLA-YCBSugarBox-G1-JointPos-v0`
- `VLA-YCBSugarBox-G1-JointPos-DR-v0` (randomized box pose)
- `VLA-YCBGraspLift-<Object>-G1-v0` for `SugarBox` and `MustardBottle` (RL
  grasp-and-lift; smaller action space)
- `VLA-YCBGraspLift-<Object>-G1-Fast-v0`: the same task at 5x the palm and
  finger speed limits with 5 s episodes, for training from scratch

To add an object to the grasp-lift RL task: add a `GraspObjectSpec` in
`envs/common/objects.py`, env config classes (normal and `fast_actions`) and a
PPO runner config class at the end of `envs/ycb_grasp/rl_env_cfg.py` and
`agents/rsl_rl_ppo_cfg.py`, and an entry in `OBJECTS` in
`envs/ycb_grasp/__init__.py`; then build its pregrasp table with
`./scripts/rl/build_pregrasp_table.sh --headless --task <ID>`. An object whose
grasp height is out of the arm's reach can get a higher table
(`support_height_m`).

Grasp-lift training and evaluation:

```bash
# From scratch: fast limits first, then fine-tune at the normal (slow) limits.
./scripts/rl/train.sh --headless --task VLA-YCBGraspLift-<Object>-G1-Fast-v0 \
  --num-envs 1024 --max-iterations 200 --run-name fast200
./scripts/rl/train.sh --headless --task VLA-YCBGraspLift-<Object>-G1-v0 \
  --num-envs 1024 --max-iterations 200 --resume <fast run>/model_199.pt
# One episode per env, mean actions; --stochastic samples with the exploration noise.
./scripts/rl/play.sh --headless --task VLA-YCBGraspLift-<Object>-G1-v0 \
  --checkpoint <run>/model_<N>.pt --num-envs 256
```

`train.sh` prints a short summary per iteration: how episodes end (success,
lifted but off pose, object fell, timed out; weighted by episode duration) and
the time left. `--verbose-log` also prints RSL-RL's full block. Runs go to
`outputs/rl/runs/grasp_lift_<object>/`, and `play.sh` writes
`<checkpoint>_eval.json` next to the checkpoint.

To add a task, create a complete EnvCfg under `src/vla_isaaclab/envs/`, keep its
MDP terms beside it, and register the EnvCfg in that environment package's
`__init__.py`. Add a scripted policy only when deterministic demonstrations are
needed. Use Isaac Lab manager terms and action configurations for environment
behavior and interfaces.

Camera placement belongs to the concrete scene's EnvCfg because framing depends
on the object layout. Shared camera intrinsics/modalities belong in the common
camera helper. Thus, change `CAMERA_EYE`/`CAMERA_TARGET` in the concrete env to
reframe one task, and change `camera_cfg()` to alter camera hardware across all
tasks.

## Scene preview examples

Use one command and choose one of three task IDs:

```bash
./scripts/run_env.sh --headless \
  --task <TASK_ID> --steps 240 \
  --preview-video outputs/previews/<NAME>.mp4
```

| Preview | `TASK_ID` | `NAME` |
| --- | --- | --- |
| YCB objects | `VLA-ScenePreview-YCB-G1-v0` | `ycb` |
| Bowl and plate | `VLA-ScenePreview-Dinnerware-G1-v0` | `dinnerware` |
| Microwave | `VLA-ScenePreview-Microwave-G1-v0` | `microwave` |

## Reference task: YCB sugar box

The table contains only `004_sugar_box`. Relevant code is:

- complete environment: `src/vla_isaaclab/envs/ycb_sugar_box/env_cfg.py`
- MDP terms: `src/vla_isaaclab/envs/ycb_sugar_box/mdp/`
- G1 configuration: `src/vla_isaaclab/envs/common/g1.py`
- built-in action configuration: `src/vla_isaaclab/envs/common/managers.py`
- scripted phases: `src/vla_isaaclab/policies/ycb_sugar_box_strategy.py`
- IK and action generation: `src/vla_isaaclab/policies/ycb_sugar_box.py`
- bounded IK helper: `src/vla_isaaclab/policies/bounded_ik.py`

The scripted policy and environment interact through the standard action
interface:

```text
registered EnvCfg + manager state
        ↓
scripted policy (task phases + bounded IK + three-finger targets)
        ↓
physical joint targets
        ↓  q -> 2 * (q - lower) / (upper - lower) - 1
normalized 43-D action
        ↓
Isaac Lab JointPositionToLimitsActionCfg
        ↓
G1 joint-position targets
```

The box starts upright at XY `(0.010509, -0.290489)` m with world-Z yaw
`35.81856°`. The three-finger side grasp lifts it 8 cm, moves it 2 cm toward
robot-left (world `-X`), levels and releases it, then withdraws the hand.

```bash
./scripts/run_env.sh --headless \
  --task VLA-YCBSugarBox-G1-JointPos-v0 \
  --steps 1200 \
  --preview-video outputs/videos/sugar_box/attempt.mp4
```

The validated reference fires the named `success` termination at step 961.
Success requires <=15 mm XY/height error, low object speed, palm separation
over 20 cm, and a 15-step hold. A failed run must never be labeled or saved as
a successful demonstration.

## Recording

Generate a contract-aligned LeRobot v3 dataset directly; no user-run conversion step is needed:

```bash
./scripts/record_lerobot.sh --dataset-name sugar_box_demo
```

LeRobot v3 is the default. Select the v2.1 layout explicitly when required:

```bash
./scripts/record_lerobot.sh --dataset-name sugar_box_demo_v21 \
  --lerobot-version 2.1
```

The dataset contains 43-D measured joint state and processed absolute targets,
normalized simulator action, environment state, source timestamps, phase,
reward/done/success, seed, and RGB. The writer retains an HDF5 source recording,
preserves unsuccessful episodes with `success: false`, and atomically converts
to `outputs/lerobot/<dataset-name>/`.

By default, only successful episodes are exported for imitation training. To
materialize failed episodes for source/contract review, add
`--include-failed-episodes`; the resulting `collection.json` marks that export
as training-ineligible. The retained HDF5 source always keeps the true outcome.

Validate the result (the flag records the currently approved right-wrist-camera
exception):

```bash
python scripts/inspect_lerobot.py outputs/lerobot/sugar_box_demo \
  --allow-missing-right-wrist
```

Replay normalized actions with:

```bash
./scripts/replay_lerobot.sh outputs/lerobot/sugar_box_demo --episode 0 --headless
```

`outputs/` is ignored by Git. Publish validated datasets to a dataset registry
or object store rather than committing them.
