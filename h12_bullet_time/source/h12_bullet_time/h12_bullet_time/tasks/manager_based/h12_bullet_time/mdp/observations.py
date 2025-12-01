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
    "projectile_distance_obs",
]


def projectile_position_relative(
    env: ManagerBasedRLEnv,
    projectile_name: str = "Projectile",
    robot_link_name: str = "head",
) -> torch.Tensor:

    robot = env.scene["robot"]

    # Determine robot link/world position to use
    robot_pos = None
    try:
        body_names = list(robot.body_names)
    except Exception:
        body_names = []

    if robot_link_name is not None and robot_link_name in body_names:
        idx = body_names.index(robot_link_name)
        # body_pos_w shape: (num_envs, num_bodies, 3)
        robot_pos = robot.data.body_pos_w[:, idx, :]
    else:
        # Fallback to root_pos_w (num_envs, 3)
        robot_pos = robot.data.root_pos_w

    try:
        projectile = env.scene[projectile_name]
        proj_pos = projectile.data.root_pos_w  # (num_envs, 3)
    except (KeyError, AttributeError):
        # If projectile not found, return zeros
        return torch.zeros((env.num_envs, 3), device=env.device, dtype=torch.float32)

    # Relative position: projectile - robot_link
    return proj_pos - robot_pos


def projectile_velocity(
    env: ManagerBasedRLEnv,
    projectile_name: str = "Projectile",
) -> torch.Tensor:
    """Get projectile velocity in world frame.

    Returns a (num_envs, 3) tensor. If the projectile is not present returns zeros.
    """
    try:
        projectile = env.scene[projectile_name]
        return projectile.data.root_lin_vel_w  # (num_envs, 3)
    except (KeyError, AttributeError):
        # If projectile not found, return zeros
        return torch.zeros((env.num_envs, 3), device=env.device, dtype=torch.float32)


def projectile_distance_obs(
    env: ManagerBasedRLEnv,
    projectile_name: str = "Projectile",
    robot_link_name: str = "head",
) -> torch.Tensor:

    robot = env.scene["robot"]

    # Choose robot link position
    try:
        body_names = list(robot.body_names)
    except Exception:
        body_names = []

    if robot_link_name is not None and robot_link_name in body_names:
        idx = body_names.index(robot_link_name)
        robot_pos = robot.data.body_pos_w[:, idx, :]
    else:
        robot_pos = robot.data.root_pos_w

    try:
        projectile = env.scene[projectile_name]
        proj_pos = projectile.data.root_pos_w  # (num_envs, 3)
    except (KeyError, AttributeError):
        # If projectile not found, return zeros (no threat)
        return torch.zeros((env.num_envs, 1), device=env.device, dtype=torch.float32)

    distance = torch.norm(proj_pos - robot_pos, dim=-1, keepdim=True)
    return distance
