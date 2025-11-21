from __future__ import annotations

import math
import sys
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

from isaaclab.managers import SceneEntityCfg


def launch_projectile(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("Projectile"),
) -> None:

    # Internal configuration
    spawn_distance = 3.0     # distance in front of robot (meters)
    spawn_height = 1.0       # height at robot eye level
    throw_speed = 2.0        # m/s velocity magnitude toward robot

    # Get projectile and robot from scene
    proj = env.scene[asset_cfg.name]
    robot = env.scene["robot"]
    
    # Get robot base positions and orientation
    robot_base_pos = robot.data.root_pos_w  # shape: (num_envs, 3)
    robot_quat = robot.data.root_quat_w  # shape: (num_envs, 4) - [w, x, y, z]
    device = robot_base_pos.device
    
    # Get positions for the envs being reset
    center = robot_base_pos[env_ids]  # shape (n, 3) - robot center
    quat = robot_quat[env_ids]  # shape (n, 4) - robot orientation
    n = env_ids.numel()
    
    # Extract robot forward direction from quaternion
    # Forward in robot frame is +X, which maps to the robot's heading direction
    # Using quaternion rotation: forward_world = quat * [1, 0, 0] * quat_conj
    w, x, y, z = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
    
    # Rotate +X forward direction through quaternion
    forward_x = 1.0 - 2.0 * (y**2 + z**2)
    forward_y = 2.0 * (x*y - w*z)
    forward_z = 2.0 * (x*z + w*y)
    forward = torch.stack([forward_x, forward_y, forward_z], dim=-1)  # shape (n, 3)
    
    # Spawn position: distance in front of robot, at spawn_height
    spawn_pos = center + forward * spawn_distance
    spawn_pos[:, 2] = center[:, 2] + spawn_height
    
    # Create identity quaternions (no rotation) [w, x, y, z]
    quats = torch.zeros((n, 4), device=device, dtype=torch.float32)
    quats[:, 0] = 1.0
    
    # Velocity: normalize direction from spawn to center, multiply by speed
    direction = center - spawn_pos
    direction_norm = torch.norm(direction, dim=-1, keepdim=True).clamp(min=1e-6)
    direction_normalized = direction / direction_norm
    lin_vel = direction_normalized * throw_speed
    
    # Zero angular velocity
    ang_vel = torch.zeros((n, 3), device=device, dtype=torch.float32)
    
    # # Use RigidObject's separate write methods (similar to multi_asset.py demo)
    # print(f"[PROJ] Writing state for {n} projectiles: pos shape {spawn_pos.shape}, vel shape {lin_vel.shape}", file=sys.stderr, flush=True)
    # print(f"[PROJ] Sample spawn pos: {spawn_pos[0].cpu()}, vel: {lin_vel[0].cpu()}", file=sys.stderr, flush=True)
    
    # Build pose tensor: [pos (3), quat (4)] per env
    pose = torch.cat([spawn_pos, quats], dim=-1)  # shape (n, 7)
    # Build velocity tensor: [lin_vel (3), ang_vel (3)] per env
    velocity = torch.cat([lin_vel, ang_vel], dim=-1)  # shape (n, 6)
    
    # Write pose and velocity using the proper RigidObject API
    proj.write_root_pose_to_sim(pose, env_ids)
    proj.write_root_velocity_to_sim(velocity, env_ids)
    
    # print(f"[PROJ] ✓ Launched {n} projectiles successfully", file=sys.stderr, flush=True)
