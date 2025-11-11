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
    spawn_height = 2.0       # height above robot base
    area_size = 5.0          # 5x5 meter area (tight spawn to guarantee hit)
    throw_speed = 2.0        # m/s velocity magnitude toward center

    # Get projectile and robot from scene
    proj = env.scene[asset_cfg.name]
    robot = env.scene["robot"]
    
    # Get robot base positions (center point for spawning)
    robot_base_pos = robot.data.root_pos_w  # shape: (num_envs, 3)
    device = robot_base_pos.device
    
    # Get the center positions for the envs being reset
    center = robot_base_pos[env_ids]  # shape (n, 3)
    n = env_ids.numel()
    
    # Random spawn positions within 2x2 meter area around center
    spawn_x = center[:, 0] - area_size / 2.0 + torch.rand(n, device=device) * area_size
    spawn_y = center[:, 1] - area_size / 2.0 + torch.rand(n, device=device) * area_size
    spawn_z = center[:, 2] + spawn_height 
    
    spawn_pos = torch.stack([spawn_x, spawn_y, spawn_z], dim=-1)
    
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
