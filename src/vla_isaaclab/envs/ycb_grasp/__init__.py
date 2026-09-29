"""Registered YCB grasp-and-lift RL environments, one Gym ID per object.

Adding an object: a ``GraspObjectSpec`` in ``envs/common/objects.py``, env
config classes (normal and ``fast_actions``) and a runner config class, and one
``OBJECTS`` entry. ``VLA-YCBGraspLift-<Object>-G1-Fast-v0`` is the same task
at 5x the speed limits, for training from scratch before the slow fine-tune.
"""

import gymnasium as gym


# Gym name -> (env cfg class, PPO runner cfg class).
OBJECTS = {
    "SugarBox": ("SugarBoxGraspLiftEnvCfg", "SugarBoxGraspLiftPPORunnerCfg"),
    "MustardBottle": ("MustardBottleGraspLiftEnvCfg", "MustardBottleGraspLiftPPORunnerCfg"),
    "Bowl": ("BowlGraspLiftEnvCfg", "BowlGraspLiftPPORunnerCfg"),
}


def grasp_lift_env_id(name: str, fast: bool = False) -> str:
    return f"VLA-YCBGraspLift-{name}-G1-{'Fast-' if fast else ''}v0"


for _name, (_env_cfg, _runner_cfg) in OBJECTS.items():
    for _fast in (False, True):
        if grasp_lift_env_id(_name, _fast) not in gym.registry:
            gym.register(
                id=grasp_lift_env_id(_name, _fast),
                entry_point="isaaclab.envs:ManagerBasedRLEnv",
                kwargs={
                    "env_cfg_entry_point": f"{__name__}.rl_env_cfg:{_env_cfg.replace('EnvCfg', 'FastEnvCfg') if _fast else _env_cfg}",
                    "rsl_rl_cfg_entry_point": f"{__name__}.agents.rsl_rl_ppo_cfg:{_runner_cfg}",
                },
                disable_env_checker=True,
            )


__all__ = ["OBJECTS", "grasp_lift_env_id"]
