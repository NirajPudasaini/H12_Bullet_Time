from __future__ import annotations

import math
import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

from .capacitive_sensor import CapacitiveSensor

if TYPE_CHECKING:
    from .cone_sensor_cfg import ConeSensorCfg


class ConeSensor(CapacitiveSensor):
    """Proximity sensor with a conical receptive field.

    Inherits all frame-tracking from CapacitiveSensor (FieldSensor) but restricts
    detection to targets within a cone defined by each sensor's orientation and
    the configured half-angle.
    """

    cfg: ConeSensorCfg

    def _initialize_impl(self):
        super()._initialize_impl()
        if len(self.cfg.relative_sensor_quat) == 1 and self._num_sensors > 1:
            self._relative_sensor_quat = torch.tensor(
                self.cfg.relative_sensor_quat * self._num_sensors, device=self.device
            ).view(self._num_sensors, 4)
        else:
            self._relative_sensor_quat = torch.tensor(self.cfg.relative_sensor_quat, device=self.device)
        self._cone_cos_threshold = math.cos(self.cfg.cone_half_angle_deg * math.pi / 180.0)

    def _update_buffers_impl(self, env_ids: Sequence[int]):
        saved_last = self._last_dist_est_normalized.clone()
        super()._update_buffers_impl(env_ids)

        z_axis = torch.tensor([0.0, 0.0, 1.0], device=self.device)
        sensor_forward = self._quat_rotate_vec(self._relative_sensor_quat, z_axis)  # (S, 3)

        target_pos = self._data.target_pos_sensor  # (N, S, M, 3)
        target_dir = target_pos / (torch.linalg.norm(target_pos, dim=-1, keepdim=True) + 1e-8)
        cos_angle = (target_dir * sensor_forward.view(1, self._num_sensors, 1, 3)).sum(dim=-1)  # (N, S, M)
        within_cone = cos_angle >= self._cone_cos_threshold

        self._data.dist_est[:] = torch.where(
            within_cone, self._data.dist_est, torch.full_like(self._data.dist_est, self.cfg.max_range)
        )
        self._data.dist_est_normalized[:] = torch.where(
            within_cone, self._data.dist_est_normalized, torch.ones_like(self._data.dist_est_normalized)
        )
        self._data.binary_detection[:] = self._data.binary_detection * within_cone.float()
        self._data.dist_est_change_normalized[:] = self._data.dist_est_normalized - saved_last
        self._last_dist_est_normalized[:] = self._data.dist_est_normalized

    def _quat_rotate_vec(self, quat: torch.Tensor, vec: torch.Tensor) -> torch.Tensor:
        w = quat[..., 0:1]
        xyz = quat[..., 1:4]
        if vec.dim() == 1:
            vec = vec.expand(quat.shape[:-1] + (3,))
        t = 2.0 * torch.linalg.cross(xyz, vec)
        return vec + w * t + torch.linalg.cross(xyz, t)
