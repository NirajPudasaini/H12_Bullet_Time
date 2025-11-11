
from __future__ import annotations

import sys
import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def base_height_below_threshold(
    env: ManagerBasedRLEnv, threshold: float, asset_cfg: SceneEntityCfg
) -> torch.Tensor:
    # extract robot asset
    asset: Articulation = env.scene[asset_cfg.name]
    # get base height (z-position of root body)
    base_height = asset.data.body_pos_w[:, 0, 2]
    # compute termination condition
    is_terminated = base_height < threshold

    return is_terminated


def projectile_hit(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    projectile_names: list | None = None,
    threshold: float = 0.1,
) -> torch.Tensor:

    # Get robot and projectiles
    robot: Articulation = env.scene[asset_cfg.name]
    
    # Get all body positions of the robot: shape (num_envs, num_bodies, 3)
    robot_body_positions = robot.data.body_pos_w  # shape: (num_envs, num_bodies, 3)
    
    # Find projectiles
    scene_names = list(env.scene.keys())
    candidates = [] if projectile_names is None else list(projectile_names)
    if projectile_names is None:
        for n in scene_names:
            if "projectile" in n.lower() or "obstacle" in n.lower():
                candidates.append(n)

    if len(candidates) == 0:
        return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

    hit = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    
    # Check distance from projectile to each robot body
    for name in candidates:

        obj = env.scene[name]
        # Get projectile position
        try:
            proj_pos = obj.data.root_pos_w  # shape: (num_envs, 3)
        except AttributeError:
            proj_pos = obj.data.body_pos_w[:, 0, :]  # shape: (num_envs, 3)
        
        # Compute distance from projectile to each robot body
        # proj_pos: (num_envs, 3) -> (num_envs, 1, 3)
        # robot_body_positions: (num_envs, num_bodies, 3)
        # distance: (num_envs, num_bodies)
        distances = torch.norm(
            robot_body_positions - proj_pos.unsqueeze(1),
            dim=-1
        )
        
        # Find minimum distance to any body for each environment
        min_dist_per_env = distances.min(dim=1)[0]  # shape: (num_envs,)
        
        # Update hit mask
        hit = hit | (min_dist_per_env < float(threshold))
            
    return hit