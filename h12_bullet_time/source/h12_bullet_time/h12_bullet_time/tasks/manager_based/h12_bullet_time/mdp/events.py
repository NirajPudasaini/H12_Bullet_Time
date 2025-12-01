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
    asset_cfg: SceneEntityCfg | None = None
) -> None:
    """Event handler to spawn and launch spherical projectiles toward the robot.

    This is intended to be called at episode reset (EventTerm with mode="reset").
    The function spawns projectiles 3 meters above the robot at random azimuth
    angles and 5m horizontal distance, then gives them velocity to fall toward the robot.
    
    Args:
        env: The ManagerBasedRLEnv environment instance.
        env_ids: Indices of environments to reset projectiles for.
        asset_cfg: Optional scene entity config for projectile (defaults to "Projectile").
    """
    # Internal configuration
    projectile_name = "Projectile"
    spawn_distance_xy = 5.0  # horizontal distance from robot
    spawn_height = 3.0       # height above robot base
    min_speed = 4.0
    max_speed = 8.0

    # Try to get projectile and robot from scene
    try:
        proj = env.scene[projectile_name]
        robot = env.scene["robot"]
        robot_base_pos = robot.data.body_pos_w[:, 0, :]  # Body 0 is the base
    except (KeyError, AttributeError, IndexError):
        # Assets not available, exit gracefully
        return

    device = robot_base_pos.device
    
    # Convert env_ids to long tensor
    env_ids_long = env_ids.long() if isinstance(env_ids, torch.Tensor) else torch.tensor(env_ids, device=device, dtype=torch.long)
    n = env_ids_long.numel()
    if n == 0:
        return
    
    # Get robot base positions for the envs being reset
    base_pos = robot_base_pos[env_ids_long]  # shape (n, 3)
    
    # Random azimuth angles (around the robot)
    az = torch.rand(n, device=device) * 2 * math.pi
    
    # Spawn position: offset in XY by spawn_distance_xy, add spawn_height to Z
    dx_spawn = torch.cos(az) * spawn_distance_xy
    dy_spawn = torch.sin(az) * spawn_distance_xy
    
    spawn_pos = base_pos.clone()
    spawn_pos[:, 0] += dx_spawn
    spawn_pos[:, 1] += dy_spawn
    spawn_pos[:, 2] += spawn_height
    
    # Create identity quaternions (no rotation) [w, x, y, z]
    quats = torch.zeros((n, 4), device=device, dtype=torch.float32)
    quats[:, 0] = 1.0
    
    # Velocity: toward robot + downward
    to_base_xy = base_pos[:, :2] - spawn_pos[:, :2]
    to_base_z = torch.full_like(to_base_xy[:, :1], -spawn_height)
    
    to_base = torch.cat([to_base_xy, to_base_z], dim=-1)
    to_base_dist = torch.norm(to_base, dim=-1, keepdim=True).clamp(min=1e-6)
    direction_to_base = to_base / to_base_dist
    
    # Random speed for each projectile
    speeds = (min_speed + (max_speed - min_speed) * torch.rand(n, device=device)).unsqueeze(-1)
    vel = direction_to_base * speeds
    
    # Zero angular velocity
    ang_vel = torch.zeros((n, 3), device=device, dtype=torch.float32)
    
    # Update FULL batch buffers (not just reset indices!)
    # This is critical: write_data_to_sim() needs the full batch
    full_pos = proj.data.body_pos_w[:, 0, :].clone()
    full_quat = proj.data.body_quat_w[:, 0, :].clone()
    full_lin_vel = proj.data.body_lin_vel_w[:, 0, :].clone()
    full_ang_vel = proj.data.body_ang_vel_w[:, 0, :].clone()
    
    # Update only reset environments
    full_pos[env_ids_long] = spawn_pos
    full_quat[env_ids_long] = quats
    full_lin_vel[env_ids_long] = vel
    full_ang_vel[env_ids_long] = ang_vel
    
    # Write full batch back
    proj.data.body_pos_w[:, 0, :] = full_pos
    proj.data.body_quat_w[:, 0, :] = full_quat
    proj.data.body_lin_vel_w[:, 0, :] = full_lin_vel
    proj.data.body_ang_vel_w[:, 0, :] = full_ang_vel
    
    # Critically important: write to physics engine
    proj.write_data_to_sim()
    
    # Also update default state to ensure it persists through resets
    try:
        proj.data.default_root_state[env_ids_long, 0:3] = spawn_pos
        proj.data.default_root_state[env_ids_long, 3:7] = quats
        proj.data.default_root_state[env_ids_long, 7:10] = vel
        proj.data.default_root_state[env_ids_long, 10:13] = ang_vel
    except Exception:
        pass
    
    print(f"[PROJ] Spawn {n} projectiles at {spawn_pos[0].cpu()}", file=sys.stderr)
    sys.stderr.flush()
