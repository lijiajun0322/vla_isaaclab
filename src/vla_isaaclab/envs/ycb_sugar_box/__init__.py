"""Registered YCB sugar-box manipulation environments."""

import gymnasium as gym


ENV_ID = "VLA-YCBSugarBox-G1-JointPos-v0"
DR_ENV_ID = "VLA-YCBSugarBox-G1-JointPos-DR-v0"

for env_id, cfg_name in ((ENV_ID, "YCBSugarBoxEnvCfg"), (DR_ENV_ID, "YCBSugarBoxDREnvCfg")):
    if env_id not in gym.registry:
        gym.register(
            id=env_id,
            entry_point="isaaclab.envs:ManagerBasedRLEnv",
            kwargs={"env_cfg_entry_point": f"{__name__}.env_cfg:{cfg_name}"},
            disable_env_checker=True,
        )


__all__ = ["DR_ENV_ID", "ENV_ID"]
