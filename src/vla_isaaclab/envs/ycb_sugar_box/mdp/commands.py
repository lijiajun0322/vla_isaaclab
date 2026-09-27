"""Deterministic goal command used by the sugar-box task."""

from collections.abc import Sequence
from dataclasses import MISSING

import torch

from isaaclab.managers import CommandTerm, CommandTermCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import quat_conjugate, quat_mul


class FixedPoseCommand(CommandTerm):
    """Expose one fixed environment-local pose through CommandManager."""

    cfg: "FixedPoseCommandCfg"

    def __init__(self, cfg: "FixedPoseCommandCfg", env):
        super().__init__(cfg, env)
        self._command = torch.tensor(cfg.pose, dtype=torch.float32, device=self.device).repeat(self.num_envs, 1)

    @property
    def command(self) -> torch.Tensor:
        return self._command

    def _update_metrics(self):
        pass

    def _resample_command(self, env_ids: Sequence[int]):
        self._command[env_ids] = torch.tensor(self.cfg.pose, dtype=torch.float32, device=self.device)

    def _update_command(self):
        pass


@configclass
class FixedPoseCommandCfg(CommandTermCfg):
    class_type: type[CommandTerm] = FixedPoseCommand
    pose: tuple[float, float, float, float, float, float, float] = MISSING
    resampling_time_range: tuple[float, float] = (1.0e9, 1.0e9)
    debug_vis: bool = False


class ObjectRelativePoseCommand(CommandTerm):
    """Goal pose = object pose at reset + a fixed world-frame displacement.

    Reset events run before commands resample, so a randomized reset pose is
    already written when the goal is captured. An optional visual-only marker
    prim (no physics) is moved to the goal on the tabletop.
    """

    cfg: "ObjectRelativePoseCommandCfg"

    def __init__(self, cfg: "ObjectRelativePoseCommandCfg", env):
        super().__init__(cfg, env)
        self.object = env.scene[cfg.asset_name]
        self._command = torch.zeros(self.num_envs, 7, device=self.device)
        self._command[:, 3] = 1.0
        self._offset = torch.tensor(cfg.world_offset, dtype=torch.float32, device=self.device)

    @property
    def command(self) -> torch.Tensor:
        return self._command

    def _update_metrics(self):
        pass

    def _resample_command(self, env_ids: Sequence[int]):
        position = self.object.data.root_pos_w[env_ids] - self._env.scene.env_origins[env_ids]
        orientation = self.object.data.root_quat_w[env_ids]
        self._command[env_ids, :3] = position + self._offset
        self._command[env_ids, 3:] = orientation
        if self.cfg.marker_prim_name is not None:
            self._move_markers(env_ids, orientation)

    def _move_markers(self, env_ids: Sequence[int], orientation: torch.Tensor):
        import omni.usd
        from pxr import Gf

        # The marker keeps its authored orientation relative to the object.
        nominal = torch.tensor(self.cfg.object_nominal_quat, device=self.device).expand_as(orientation)
        marker = torch.tensor(self.cfg.marker_nominal_quat, device=self.device).expand_as(orientation)
        marker_quat = quat_mul(quat_mul(orientation, quat_conjugate(nominal)), marker)
        stage = omni.usd.get_context().get_stage()
        env_ids = torch.arange(self.num_envs, device=self.device)[env_ids].tolist()
        for row, env_id in enumerate(env_ids):
            prim = stage.GetPrimAtPath(f"{self._env.scene.env_prim_paths[env_id]}/{self.cfg.marker_prim_name}")
            x, y = self._command[env_id, :2].tolist()
            translate = prim.GetAttribute("xformOp:translate")
            orient = prim.GetAttribute("xformOp:orient")
            translate.Set(type(translate.Get())(x, y, self.cfg.marker_height))
            orient.Set(type(orient.Get())(*marker_quat[row].tolist()))

    def _update_command(self):
        pass


@configclass
class ObjectRelativePoseCommandCfg(CommandTermCfg):
    class_type: type[CommandTerm] = ObjectRelativePoseCommand
    asset_name: str = "object"
    world_offset: tuple[float, float, float] = MISSING
    object_nominal_quat: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)
    marker_prim_name: str | None = None
    marker_height: float = 0.0
    marker_nominal_quat: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)
    resampling_time_range: tuple[float, float] = (1.0e9, 1.0e9)
    debug_vis: bool = False
