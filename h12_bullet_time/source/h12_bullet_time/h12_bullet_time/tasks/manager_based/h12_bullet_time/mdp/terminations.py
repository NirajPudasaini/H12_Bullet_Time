# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def is_fallen(env: ManagerBasedRLEnv, threshold: float) -> torch.Tensor:
    """Check if robot has fallen (COM too low)."""
    robot: Articulation = env.scene["robot"]
    com_z = robot.data.root_pos_w[:, 2]
    return (com_z < threshold).float()


def robot_out_of_bounds(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    bounds: tuple[float, float, float, float],
) -> torch.Tensor:
    """Check if robot is out of bounds."""
    robot: Articulation = env.scene[asset_cfg.name]
    pos = robot.data.root_pos_w
    
    x_min, x_max, y_min, y_max = bounds
    
    # Check if outside bounds
    out_of_bounds = (
        (pos[:, 0] < x_min) | (pos[:, 0] > x_max) |
        (pos[:, 1] < y_min) | (pos[:, 1] > y_max)
    ).float()
    
    return out_of_bounds
