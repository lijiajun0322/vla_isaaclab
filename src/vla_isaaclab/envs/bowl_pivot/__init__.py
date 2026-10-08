"""Registered bimanual bowl-pivoting RL environment (G1, both arms, both Dex3 hands).

Build the fixed start state once with ``scripts/rl/build_bowl_pivot_init.sh --headless``.
"""

import gymnasium as gym


BOWL_PIVOT_ENV_ID = "VLA-BowlPivot-G1-Bimanual-v0"
# Same task at 1.5x the speed limits (the grasp-lift fast limits; the v2 run, which learned to toss the bowl).
BOWL_PIVOT_FAST_ENV_ID = "VLA-BowlPivot-G1-Bimanual-Fast-v0"
# Half the fast speed limits.
BOWL_PIVOT_HALF_SPEED_ENV_ID = "VLA-BowlPivot-G1-Bimanual-HalfSpeed-v0"

# Two stages at 1/2 speed: turn the bowl upright, then lift it 5 cm level. The Grip
# variant asks for a three-finger grip of one hand and the other hand off the bowl.
BOWL_PIVOT_LIFT_ENV_ID = "VLA-BowlPivotLift-G1-Bimanual-HalfSpeed-v0"
BOWL_PIVOT_LIFT_GRIP_ENV_ID = "VLA-BowlPivotLift-Grip-G1-Bimanual-HalfSpeed-v0"

for _env_id, _env_cfg in ((BOWL_PIVOT_ENV_ID, "BowlPivotEnvCfg"), (BOWL_PIVOT_FAST_ENV_ID, "BowlPivotFastEnvCfg"),
                          (BOWL_PIVOT_HALF_SPEED_ENV_ID, "BowlPivotHalfSpeedEnvCfg"),
                          (BOWL_PIVOT_LIFT_ENV_ID, "BowlPivotLiftEnvCfg"),
                          (BOWL_PIVOT_LIFT_GRIP_ENV_ID, "BowlPivotLiftGripEnvCfg")):
    if _env_id not in gym.registry:
        gym.register(
            id=_env_id,
            entry_point="isaaclab.envs:ManagerBasedRLEnv",
            kwargs={
                "env_cfg_entry_point": f"{__name__}.env_cfg:{_env_cfg}",
                "rsl_rl_cfg_entry_point": f"{__name__}.agents.rsl_rl_ppo_cfg:"
                                          + ("BowlPivotLiftPPORunnerCfg" if "Lift" in _env_id else "BowlPivotPPORunnerCfg"),
            },
            disable_env_checker=True,
        )


# The bowl grasp-lift task in the pivot scene, from states the pivot policy turned upright.
BOWL_FLIPPED_LIFT_ENV_ID = "VLA-BowlFlippedLift-G1-v0"
if BOWL_FLIPPED_LIFT_ENV_ID not in gym.registry:
    gym.register(
        id=BOWL_FLIPPED_LIFT_ENV_ID,
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        kwargs={
            "env_cfg_entry_point": f"{__name__}.env_cfg:BowlFlippedLiftEnvCfg",
            "rsl_rl_cfg_entry_point": f"{__name__}.agents.rsl_rl_ppo_cfg:BowlFlippedLiftPPORunnerCfg",
        },
        disable_env_checker=True,
    )


# The same with the right arm and hand.
BOWL_FLIPPED_LIFT_RIGHT_ENV_ID = "VLA-BowlFlippedLift-Right-G1-v0"
if BOWL_FLIPPED_LIFT_RIGHT_ENV_ID not in gym.registry:
    gym.register(
        id=BOWL_FLIPPED_LIFT_RIGHT_ENV_ID,
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        kwargs={
            "env_cfg_entry_point": f"{__name__}.env_cfg:BowlFlippedLiftRightEnvCfg",
            "rsl_rl_cfg_entry_point": f"{__name__}.agents.rsl_rl_ppo_cfg:BowlFlippedLiftRightPPORunnerCfg",
        },
        disable_env_checker=True,
    )


__all__ = [
    "BOWL_FLIPPED_LIFT_ENV_ID",
    "BOWL_FLIPPED_LIFT_RIGHT_ENV_ID",
    "BOWL_PIVOT_ENV_ID",
    "BOWL_PIVOT_FAST_ENV_ID",
    "BOWL_PIVOT_HALF_SPEED_ENV_ID",
    "BOWL_PIVOT_LIFT_ENV_ID",
    "BOWL_PIVOT_LIFT_GRIP_ENV_ID",
]
