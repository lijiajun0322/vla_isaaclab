#!/usr/bin/env python3
"""Train an RSL-RL PPO policy on a registered VLA Isaac Lab RL environment."""

from __future__ import annotations

import argparse
import sys
import traceback
from datetime import datetime
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task", default="VLA-YCBSugarBox-G1-GraspLift-RL-v0")
parser.add_argument("--num-envs", type=int, default=2048)
parser.add_argument("--max-iterations", type=int, default=None)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--run-name", default="")
parser.add_argument("--resume", type=Path, default=None, help="Checkpoint (.pt) to continue from.")
parser.add_argument("--log-root", type=Path, default=PROJECT_ROOT / "outputs/rl/runs")
AppLauncher.add_app_launcher_args(parser)
ARGS = parser.parse_args()
ARGS.enable_cameras = False
ARGS.experience = str(PROJECT_ROOT / "configs" / ("ycb.python.headless.kit" if ARGS.headless else "ycb.python.kit"))
ARGS.kit_args = f"--portable-root {PROJECT_ROOT}/outputs/runtime/kit"
APP = AppLauncher(ARGS).app

import gymnasium as gym
import torch
from rsl_rl.runners import OnPolicyRunner

import vla_isaaclab  # noqa: F401  Register environments.
import vla_isaaclab.rl.isaaclab_rl_compat  # noqa: F401  Before any isaaclab_rl import.
from vla_isaaclab.rl.action_clip import ClippedRslRlVecEnvWrapper
from isaaclab.utils.io import dump_pickle, dump_yaml
from isaaclab_tasks.utils import load_cfg_from_registry, parse_env_cfg

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


def main() -> int:
    env_cfg = parse_env_cfg(ARGS.task, device=ARGS.device, num_envs=ARGS.num_envs)
    agent_cfg = load_cfg_from_registry(ARGS.task, "rsl_rl_cfg_entry_point")
    agent_cfg.seed = ARGS.seed
    agent_cfg.device = ARGS.device
    if ARGS.max_iterations is not None:
        agent_cfg.max_iterations = ARGS.max_iterations
    env_cfg.seed = agent_cfg.seed

    run = datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + (f"_{ARGS.run_name}" if ARGS.run_name else "")
    log_dir = ARGS.log_root / agent_cfg.experiment_name / run
    print(f"[train] logging to {log_dir}", flush=True)

    env = ClippedRslRlVecEnvWrapper(gym.make(ARGS.task, cfg=env_cfg))
    try:
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=str(log_dir), device=agent_cfg.device)
        runner.add_git_repo_to_log(__file__)
        if ARGS.resume is not None:
            print(f"[train] resuming from {ARGS.resume}", flush=True)
            runner.load(str(ARGS.resume))
        dump_yaml(str(log_dir / "params/env.yaml"), env_cfg)
        dump_yaml(str(log_dir / "params/agent.yaml"), agent_cfg)
        dump_pickle(str(log_dir / "params/env.pkl"), env_cfg)
        dump_pickle(str(log_dir / "params/agent.pkl"), agent_cfg)
        runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)
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
