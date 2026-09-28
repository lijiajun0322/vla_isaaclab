"""Clip policy actions before they reach the environment.

Isaac Lab v2.0.2's RSL-RL wrapper has no action clipping, and the DiffIK action
term's ``clip`` resolves joint names against its 6-D pose action, so it cannot
bound the palm command. Environments that set ``action_clip`` on their cfg get
every action clamped to [-action_clip, action_clip] here instead. PPO still
learns from the unclipped samples.
"""

from __future__ import annotations

import torch

import vla_isaaclab.rl.isaaclab_rl_compat  # noqa: F401  Before any isaaclab_rl import.
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper


class ClippedRslRlVecEnvWrapper(RslRlVecEnvWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.action_clip = getattr(self.unwrapped.cfg, "action_clip", None)

    def step(self, actions: torch.Tensor):
        if self.action_clip is not None:
            actions = actions.clamp(-self.action_clip, self.action_clip)
        return super().step(actions)
