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


def reset_robot_to_standing(env: ManagerBasedRLEnv) -> None:
    """Reset robot to standing position."""
    robot: Articulation = env.scene["robot"]
    
    # Define standing joint positions
    standing_joint_pos = {
        # Left leg
        "left_hip_yaw_joint": 0.0,
        "left_hip_roll_joint": 0.0,
        "left_hip_pitch_joint": -0.16,
        "left_knee_joint": 0.36,
        "left_ankle_pitch_joint": -0.15,
        "left_ankle_roll_joint": 0.0,
        # Right leg
        "right_hip_yaw_joint": 0.0,
        "right_hip_roll_joint": 0.0,
        "right_hip_pitch_joint": -0.16,
        "right_knee_joint": 0.36,
        "right_ankle_pitch_joint": -0.15,
        "right_ankle_roll_joint": 0.0,

        #torso
        "torso_joint": 0.0,

        # Left arm
        "left_shoulder_pitch_joint": 0.4,
        "left_shoulder_roll_joint": 0.0,
        "left_elbow_joint": 0.3,
        # Right arm
        "right_shoulder_pitch_joint": 0.4,
        "right_shoulder_roll_joint": 0.0,
        "right_elbow_joint": 0.3,
    }
    
    # Reset joint positions
    for joint_name, joint_pos in standing_joint_pos.items():
        # Find joint index
        joint_ids = robot.find_joints(joint_name)
        if len(joint_ids) > 0:
            robot.data.joint_pos[:, joint_ids[0]] = joint_pos
    
    # Reset joint velocities
    robot.data.joint_vel.zero_()
    
    # Reset root state: position at (0, 0, 1.05), no velocity
    robot.data.root_pos_w[:, 0] = 0.0
    robot.data.root_pos_w[:, 1] = 0.0
    robot.data.root_pos_w[:, 2] = 1.05
    robot.data.root_lin_vel_w.zero_()
    robot.data.root_ang_vel_w.zero_()
    
    # Reset quaternion to identity (no rotation)
    robot.data.root_quat_w[:, 0] = 1.0  # w
    robot.data.root_quat_w[:, 1] = 0.0  # x
    robot.data.root_quat_w[:, 2] = 0.0  # y
    robot.data.root_quat_w[:, 3] = 0.0  # z


def spawn_projectile_randomly(
    env: ManagerBasedRLEnv,
    projectile_cfg: SceneEntityCfg,
    robot_cfg: SceneEntityCfg,
    env_ids: torch.Tensor | None = None,
) -> None:
    """Spawn projectile with random position and velocity targeting the robot."""
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)
    
    projectile: RigidObject = env.scene[projectile_cfg.name]
    robot: Articulation = env.scene[robot_cfg.name]
    
    # Random spawn positions around the robot
    num_resets = len(env_ids)
    
    # Spawn from a distance away (3-5 meters horizontally)
    distance = torch.rand(num_resets, device=env.device) * 2.0 + 3.0  # 3-5 meters
    angle = torch.rand(num_resets, device=env.device) * 2 * 3.14159  # 0-2π
    
    spawn_x = robot.data.root_pos_w[env_ids, 0] + distance * torch.cos(angle)
    spawn_y = robot.data.root_pos_w[env_ids, 1] + distance * torch.sin(angle)
    spawn_z = robot.data.root_pos_w[env_ids, 2] + torch.rand(num_resets, device=env.device) * 0.5  # 0-0.5m above robot
    
    # Random velocities targeting the robot's upper body
    target_height_offset = 0.5  # Target upper body
    target_x = robot.data.root_pos_w[env_ids, 0]
    target_y = robot.data.root_pos_w[env_ids, 1]
    target_z = robot.data.root_pos_w[env_ids, 2] + target_height_offset
    
    # Direction from spawn to target
    direction = torch.stack([
        target_x - spawn_x,
        target_y - spawn_y,
        target_z - spawn_z
    ], dim=1)
    
    # Normalize direction
    direction_norm = torch.linalg.norm(direction, dim=1, keepdim=True)
    direction = direction / (direction_norm + 1e-6)
    
    # Random speed (5-15 m/s)
    speed = torch.rand(num_resets, device=env.device) * 10.0 + 5.0
    
    velocity = direction * speed.unsqueeze(1)
    
    # Update projectile state
    projectile.data.body_pos_w[env_ids] = torch.stack([spawn_x, spawn_y, spawn_z], dim=1)
    projectile.data.body_lin_vel_w[env_ids] = velocity


def reset_projectile_on_ground_collision(
    env: ManagerBasedRLEnv,
    projectile_cfg: SceneEntityCfg,
    robot_cfg: SceneEntityCfg,
    threshold: float = 0.1,
) -> None:
    """Reset projectile when it hits the ground or goes out of bounds."""
    projectile: RigidObject = env.scene[projectile_cfg.name]
    robot: Articulation = env.scene[robot_cfg.name]
    
    # Check if projectile is below ground
    ground_collision = projectile.data.body_pos_w[:, 2] < threshold
    
    # Check if projectile is too far from robot
    robot_pos = robot.data.root_pos_w
    projectile_pos = projectile.data.body_pos_w
    distance = torch.linalg.norm(robot_pos - projectile_pos, dim=1)
    out_of_bounds = distance > 20.0  # 20 meters max distance
    
    # Spawn new projectiles for reset environments
    env_ids = torch.where(ground_collision | out_of_bounds)[0]
    
    if len(env_ids) > 0:
        spawn_projectile_randomly(env, projectile_cfg, robot_cfg, env_ids)
