# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import wrap_to_pi

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


##
# Stage 1: Standing and Balance Rewards
##


def is_alive(env: ManagerBasedRLEnv, threshold: float) -> torch.Tensor:
    """Reward for being alive (maintaining minimum COM height)."""
    # Get robot COM height
    asset: Articulation = env.scene["robot"]
    # Assuming COM height is tracked; if not, we use root position as proxy
    com_z = asset.data.root_pos_w[:, 2]
    # Reward if above threshold
    return (com_z > threshold).float()


def is_fallen(env: ManagerBasedRLEnv, threshold: float) -> torch.Tensor:
    """Penalize falling - returns 1.0 when fallen, 0.0 when standing."""
    asset: Articulation = env.scene["robot"]
    com_z = asset.data.root_pos_w[:, 2]
    # Return 1.0 when COM is below threshold (fallen)
    return (com_z < threshold).float()


def upright_torso(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Reward for maintaining upright torso orientation."""
    asset: Articulation = env.scene["robot"]
    # Get quaternion orientation
    quat = asset.data.root_quat_w  # shape: (num_envs, 4)
    
    # Extract roll and pitch from quaternion
    # For small angles: roll ≈ 2 * atan2(qx, qw), pitch ≈ 2 * atan2(qy, qw)
    qw, qx, qy, qz = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
    
    roll = torch.atan2(2 * (qw * qx + qy * qz), 1 - 2 * (qx**2 + qy**2))
    pitch = torch.asin(2 * (qw * qy - qz * qx))
    
    # Reward is based on how upright (small roll and pitch)
    angle_penalty = torch.abs(roll) + torch.abs(pitch)
    return torch.exp(-3.0 * angle_penalty)  # Gaussian-like decay


def joint_vel_penalty(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize excessive joint velocities for smooth motion."""
    asset: Articulation = env.scene[asset_cfg.name]
    # Sum of absolute joint velocities
    joint_vel = torch.abs(asset.data.joint_vel)
    return torch.sum(joint_vel, dim=1)


##
# Stage 2: Height Control and Agility Rewards
##


def height_control(env: ManagerBasedRLEnv, target_height: float) -> torch.Tensor:
    """Reward for controlling COM height to target."""
    asset: Articulation = env.scene["robot"]
    com_z = asset.data.root_pos_w[:, 2]
    
    # Reward based on proximity to target height
    height_error = torch.abs(com_z - target_height)
    return torch.exp(-2.0 * height_error)  # Gaussian-like reward


def lateral_stability(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Reward for lateral stability - maintaining balanced weight distribution."""
    asset: Articulation = env.scene["robot"]
    
    # Get quaternion
    quat = asset.data.root_quat_w
    qw, qx, qy, qz = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
    
    # Focus on roll (lateral tilt)
    roll = torch.atan2(2 * (qw * qx + qy * qz), 1 - 2 * (qx**2 + qy**2))
    
    # Reward small roll angles
    return torch.exp(-5.0 * torch.abs(roll))


##
# Stage 3: Obstacle Dodging Rewards
##


def dodge_projectile(
    env: ManagerBasedRLEnv,
    projectile_cfg: SceneEntityCfg,
    robot_cfg: SceneEntityCfg,
    safe_distance: float,
) -> torch.Tensor:
    """Reward for maintaining safe distance from projectile."""
    projectile: RigidObject = env.scene[projectile_cfg.name]
    robot: Articulation = env.scene[robot_cfg.name]
    
    # Calculate distance between robot COM and projectile
    robot_pos = robot.data.root_pos_w
    projectile_pos = projectile.data.body_pos_w
    
    distance = torch.linalg.norm(robot_pos - projectile_pos, dim=1)
    
    # Reward based on distance from projectile (up to safe_distance)
    reward = torch.clamp(distance / safe_distance, 0.0, 1.0)
    return reward


def projectile_collision_penalty(
    env: ManagerBasedRLEnv,
    projectile_cfg: SceneEntityCfg,
    robot_cfg: SceneEntityCfg,
    collision_distance: float,
) -> torch.Tensor:
    """Penalty for collision with projectile."""
    projectile: RigidObject = env.scene[projectile_cfg.name]
    robot: Articulation = env.scene[robot_cfg.name]
    
    robot_pos = robot.data.root_pos_w
    projectile_pos = projectile.data.body_pos_w
    
    distance = torch.linalg.norm(robot_pos - projectile_pos, dim=1)
    
    # Return 1.0 if collision (distance < threshold), 0.0 otherwise
    return (distance < collision_distance).float()


def movement_reward(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Reward for movement/displacement to encourage active dodging."""
    asset: Articulation = env.scene[asset_cfg.name]
    
    # Get linear velocity magnitude (in xy plane, horizontal movement)
    lin_vel = asset.data.root_lin_vel_w
    horizontal_vel = torch.linalg.norm(lin_vel[:, :2], dim=1)
    
    # Reward based on horizontal velocity (encourage movement)
    return horizontal_vel


def com_height(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Get COM height - used for monitoring."""
    asset: Articulation = env.scene["robot"]
    return asset.data.root_pos_w[:, 2]
