# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


##
# Base State Observations
##


def base_lin_vel(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Linear velocity of the robot base."""
    robot: Articulation = env.scene["robot"]
    return robot.data.root_lin_vel_w


def base_ang_vel(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Angular velocity of the robot base."""
    robot: Articulation = env.scene["robot"]
    return robot.data.root_ang_vel_w


def base_euler_angles(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Euler angles (roll, pitch, yaw) of the robot base."""
    robot: Articulation = env.scene["robot"]
    quat = robot.data.root_quat_w  # shape: (num_envs, 4) - [w, x, y, z]
    
    qw, qx, qy, qz = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
    
    # Roll (rotation around x-axis)
    roll = torch.atan2(2 * (qw * qx + qy * qz), 1 - 2 * (qx**2 + qy**2))
    
    # Pitch (rotation around y-axis)
    pitch = torch.asin(2 * (qw * qy - qz * qx))
    
    # Yaw (rotation around z-axis)
    yaw = torch.atan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy**2 + qz**2))
    
    return torch.stack([roll, pitch, yaw], dim=1)


##
# Joint State Observations
##


def joint_pos_rel(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Relative joint positions (current - default)."""
    asset: Articulation = env.scene[asset_cfg.name]
    # Get joint positions
    joint_pos = asset.data.joint_pos
    # Get default joint positions
    default_pos = asset.data.default_joint_pos
    # Return relative positions
    return joint_pos - default_pos


def joint_vel_rel(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Relative joint velocities."""
    asset: Articulation = env.scene[asset_cfg.name]
    return asset.data.joint_vel


##
# Center of Mass Observations
##


def com_height(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Height of the robot's center of mass."""
    robot: Articulation = env.scene["robot"]
    return robot.data.root_pos_w[:, 2:3]


def com_lin_vel(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Linear velocity of the robot's center of mass."""
    robot: Articulation = env.scene["robot"]
    return robot.data.root_lin_vel_w


##
# Projectile Relative Observations
##


def projectile_relative_pos(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Relative position of projectile w.r.t. robot base."""
    try:
        projectile: RigidObject = env.scene["projectile"]
        robot: Articulation = env.scene["robot"]
        
        robot_pos = robot.data.root_pos_w
        projectile_pos = projectile.data.body_pos_w
        
        # Return relative position
        relative_pos = projectile_pos - robot_pos
        return relative_pos
    except Exception:
        # If projectile not in scene, return zeros
        robot: Articulation = env.scene["robot"]
        return torch.zeros((robot.data.root_pos_w.shape[0], 3), device=robot.device)


def projectile_relative_vel(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Relative velocity of projectile w.r.t. robot base."""
    try:
        projectile: RigidObject = env.scene["projectile"]
        robot: Articulation = env.scene["robot"]
        
        robot_vel = robot.data.root_lin_vel_w
        projectile_vel = projectile.data.body_lin_vel_w
        
        # Return relative velocity
        relative_vel = projectile_vel - robot_vel
        return relative_vel
    except Exception:
        # If projectile not in scene, return zeros
        robot: Articulation = env.scene["robot"]
        return torch.zeros((robot.data.root_lin_vel_w.shape[0], 3), device=robot.device)


def projectile_distance(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Distance from robot to projectile."""
    try:
        projectile: RigidObject = env.scene["projectile"]
        robot: Articulation = env.scene["robot"]
        
        robot_pos = robot.data.root_pos_w
        projectile_pos = projectile.data.body_pos_w
        
        # Calculate distance
        distance = torch.linalg.norm(robot_pos - projectile_pos, dim=1, keepdim=True)
        return distance
    except Exception:
        # If projectile not in scene, return large distance
        robot: Articulation = env.scene["robot"]
        return torch.ones((robot.data.root_pos_w.shape[0], 1), device=robot.device) * 1000.0
