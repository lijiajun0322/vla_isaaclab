"""Registered YCB sugar-box manipulation environments."""

import gymnasium as gym


ENV_ID = "VLA-YCBSugarBox-G1-JointPos-v0"
DR_ENV_ID = "VLA-YCBSugarBox-G1-JointPos-DR-v0"
# Closed-loop grasp-and-lift RL from table pregrasp states (not the 43-D contract).
GRASP_RL_ENV_ID = "VLA-YCBSugarBox-G1-GraspLift-RL-v0"
# v2 design: preshaped fingers, relative hand deltas, progress-based rewards.
GRASP_RL_V2_ENV_ID = "VLA-YCBSugarBox-G1-GraspLift-RL-v1"

for env_id, cfg_name in ((ENV_ID, "YCBSugarBoxEnvCfg"), (DR_ENV_ID, "YCBSugarBoxDREnvCfg")):
    if env_id not in gym.registry:
        gym.register(
            id=env_id,
            entry_point="isaaclab.envs:ManagerBasedRLEnv",
            kwargs={"env_cfg_entry_point": f"{__name__}.env_cfg:{cfg_name}"},
            disable_env_checker=True,
        )

if GRASP_RL_ENV_ID not in gym.registry:
    gym.register(
        id=GRASP_RL_ENV_ID,
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        kwargs={
            "env_cfg_entry_point": f"{__name__}.rl_env_cfg:YCBSugarBoxGraspRLEnvCfg",
            "rsl_rl_cfg_entry_point": f"{__name__}.agents.rsl_rl_ppo_cfg:SugarBoxGraspPPORunnerCfg",
        },
        disable_env_checker=True,
    )

if GRASP_RL_V2_ENV_ID not in gym.registry:
    gym.register(
        id=GRASP_RL_V2_ENV_ID,
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        kwargs={
            "env_cfg_entry_point": f"{__name__}.rl_env_cfg:YCBSugarBoxGraspRLV2EnvCfg",
            "rsl_rl_cfg_entry_point": f"{__name__}.agents.rsl_rl_ppo_cfg:SugarBoxGraspV2PPORunnerCfg",
        },
        disable_env_checker=True,
    )


__all__ = ["DR_ENV_ID", "ENV_ID", "GRASP_RL_ENV_ID", "GRASP_RL_V2_ENV_ID"]
