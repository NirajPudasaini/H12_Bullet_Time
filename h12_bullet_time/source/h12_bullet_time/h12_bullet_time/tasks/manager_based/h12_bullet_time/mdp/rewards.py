"""Minimal reward functions for Phase 1 (standing).

These functions provide simple scalar rewards so the environment can start and
the curriculum logic can operate. Replace with more sophisticated terms later.
"""
from __future__ import annotations

import torch

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg

from isaaclab.envs import ManagerBasedRLEnv


def alive_bonus(env: ManagerBasedRLEnv) -> torch.Tensor:

    # Return constant reward per environment (batch)
    return torch.ones(env.num_envs, dtype=torch.float32, device=env.device)


def base_velocity_reward(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    scale: float = 10.0,
) -> torch.Tensor:

    asset: Articulation = env.scene[asset_cfg.name]

    lin_vel = asset.data.root_lin_vel_w[:, :2]  # shape: (num_envs, 2)
    vel_norm2 = torch.sum(lin_vel ** 2, dim=1)
    reward = torch.exp(-float(scale) * vel_norm2)

    return reward
 
def projectile_hit_penalty(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    projectile_name: str = "Projectile",
    penalty: float = -10.0,
    threshold: float = 0.5,
) -> torch.Tensor:

    # Get robot
    robot: Articulation = env.scene[asset_cfg.name]
    robot_body_positions = robot.data.body_pos_w  # shape: (num_envs, num_bodies, 3)
    
    # Get projectile
    try:
        projectile = env.scene[projectile_name]
        proj_pos = projectile.data.root_pos_w  # shape: (num_envs, 3)
    except (KeyError, AttributeError):
        # Projectile not found, no penalty
        return torch.zeros(env.num_envs, dtype=torch.float32, device=env.device)
    
    # Compute distance from projectile to each robot body
    # proj_pos: (num_envs, 3) -> (num_envs, 1, 3)
    # robot_body_positions: (num_envs, num_bodies, 3)
    distances = torch.norm(
        robot_body_positions - proj_pos.unsqueeze(1),
        dim=-1
    )
    
    # Find minimum distance to any body for each environment
    min_dist_per_env = distances.min(dim=1)[0]  # shape: (num_envs,)
    
    # Apply penalty if within threshold
    reward = torch.zeros(env.num_envs, dtype=torch.float32, device=env.device)
    hit = min_dist_per_env < float(threshold)
    reward[hit] = float(penalty)
    
    return reward