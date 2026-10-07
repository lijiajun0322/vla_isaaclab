"""Registered bimanual bowl-pivoting RL environment (G1, both arms, both Dex3 hands).

Build the fixed start state once with ``scripts/rl/build_bowl_pivot_init.sh --headless``.
"""

import gymnasium as gym


BOWL_PIVOT_ENV_ID = "VLA-BowlPivot-G1-Bimanual-v0"
# Same task at 1.5x the speed limits (the grasp-lift fast limits; the v2 run, which learned to toss the bowl).
BOWL_PIVOT_FAST_ENV_ID = "VLA-BowlPivot-G1-Bimanual-Fast-v0"
# Half the fast speed limits.
BOWL_PIVOT_HALF_SPEED_ENV_ID = "VLA-BowlPivot-G1-Bimanual-HalfSpeed-v0"

for _env_id, _env_cfg in ((BOWL_PIVOT_ENV_ID, "BowlPivotEnvCfg"), (BOWL_PIVOT_FAST_ENV_ID, "BowlPivotFastEnvCfg"),
                          (BOWL_PIVOT_HALF_SPEED_ENV_ID, "BowlPivotHalfSpeedEnvCfg")):
    if _env_id not in gym.registry:
        gym.register(
            id=_env_id,
            entry_point="isaaclab.envs:ManagerBasedRLEnv",
            kwargs={
                "env_cfg_entry_point": f"{__name__}.env_cfg:{_env_cfg}",
                "rsl_rl_cfg_entry_point": f"{__name__}.agents.rsl_rl_ppo_cfg:BowlPivotPPORunnerCfg",
            },
            disable_env_checker=True,
        )


__all__ = ["BOWL_PIVOT_ENV_ID", "BOWL_PIVOT_FAST_ENV_ID", "BOWL_PIVOT_HALF_SPEED_ENV_ID"]
