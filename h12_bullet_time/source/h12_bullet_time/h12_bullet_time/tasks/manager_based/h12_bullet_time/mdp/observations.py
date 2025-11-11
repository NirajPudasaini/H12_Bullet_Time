"""Observation functions - import from Isaac Lab instead of using stubs."""
from __future__ import annotations

import torch
from typing import TYPE_CHECKING

# Import the real observation functions from Isaac Lab
from isaaclab.envs.mdp import (
    base_ang_vel,
    joint_pos_rel,
    joint_vel_rel,
    projected_gravity,
    last_action,
)

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.managers import SceneEntityCfg

__all__ = [
    "base_ang_vel",
    "joint_pos_rel",
    "joint_vel_rel",
    "projected_gravity",
    "last_action",
    "projectile_position_relative",
    "projectile_velocity",
    "projectile_distance",
]


def projectile_position_relative(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = None,
    projectile_name: str = "Projectile",
) -> torch.Tensor:
    """Get projectile position relative to robot base.
    
    Returns:
        Tensor of shape (num_envs, 3) with projectile position relative to robot center.
    """
    robot = env.scene["robot"]
    robot_pos = robot.data.root_pos_w  # (num_envs, 3)
    
    try:
        projectile = env.scene[projectile_name]
        proj_pos = projectile.data.root_pos_w  # (num_envs, 3)
    except (KeyError, AttributeError):
        # If projectile not found, return zeros
        return torch.zeros((env.num_envs, 3), device=env.device, dtype=torch.float32)
    
    # Relative position: projectile - robot
    rel_pos = proj_pos - robot_pos
    return rel_pos


def projectile_velocity(
    env: ManagerBasedRLEnv,
    projectile_name: str = "Projectile",
) -> torch.Tensor:
    """Get projectile velocity.
    
    Returns:
        Tensor of shape (num_envs, 3) with projectile linear velocity.
    """
    try:
        projectile = env.scene[projectile_name]
        proj_vel = projectile.data.root_lin_vel_w  # (num_envs, 3)
    except (KeyError, AttributeError):
        # If projectile not found, return zeros
        return torch.zeros((env.num_envs, 3), device=env.device, dtype=torch.float32)
    
    return proj_vel


def projectile_distance(
    env: ManagerBasedRLEnv,
    projectile_name: str = "Projectile",
) -> torch.Tensor:
    """Get distance from robot to projectile.
    
    Returns:
        Tensor of shape (num_envs, 1) with distance to projectile.
    """
    robot = env.scene["robot"]
    robot_pos = robot.data.root_pos_w  # (num_envs, 3)
    
    try:
        projectile = env.scene[projectile_name]
        proj_pos = projectile.data.root_pos_w  # (num_envs, 3)
    except (KeyError, AttributeError):
        # If projectile not found, return large distance
        return torch.ones((env.num_envs, 1), device=env.device, dtype=torch.float32) * 100.0
    
    distance = torch.norm(proj_pos - robot_pos, dim=-1, keepdim=True)
    return distance
