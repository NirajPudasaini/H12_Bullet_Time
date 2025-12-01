"""Observation functions - import from Isaac Lab instead of using stubs."""
from __future__ import annotations

import torch
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg

# Import the real observation functions from Isaac Lab
from isaaclab.envs.mdp import (
    base_ang_vel,
    joint_pos_rel,
    joint_vel_rel,
    projected_gravity,
    last_action,
)

__all__ = [
    "base_ang_vel",
    "joint_pos_rel",
    "joint_vel_rel",
    "projected_gravity",
    "last_action",
    "projectile_position_relative",
    "projectile_velocity",
    "projectile_distance_obs",
    "tof_distances_obs",
]


def projectile_position_relative(env: ManagerBasedRLEnv, projectile_name: str = "Projectile") -> torch.Tensor:
    """Projectile position relative to base frame.
    
    Args:
        env: Environment instance
        projectile_name: Name of the projectile entity in the scene
        
    Returns:
        Position of projectile relative to robot base (num_envs, 3)
    """
    projectile = env.scene[projectile_name]
    base = env.scene["robot"]
    
    # Get projectile position in world frame
    projectile_pos_world = projectile.data.root_pos_w  # (num_envs, 3)
    base_pos_world = base.data.root_pos_w  # (num_envs, 3)
    
    # Get relative position
    pos_rel = projectile_pos_world - base_pos_world
    
    return pos_rel


def projectile_velocity(env: ManagerBasedRLEnv, projectile_name: str = "Projectile") -> torch.Tensor:
    """Projectile velocity in world frame.
    
    Args:
        env: Environment instance
        projectile_name: Name of the projectile entity in the scene
        
    Returns:
        Velocity of projectile (num_envs, 3)
    """
    projectile = env.scene[projectile_name]
    return projectile.data.root_lin_vel_w  # (num_envs, 3)


def projectile_distance_obs(env: ManagerBasedRLEnv, projectile_name: str = "Projectile") -> torch.Tensor:
    """Distance from base to projectile.
    
    Args:
        env: Environment instance
        projectile_name: Name of the projectile entity in the scene
        
    Returns:
        Distance to projectile (num_envs, 1)
    """
    projectile = env.scene[projectile_name]
    base = env.scene["robot"]
    
    projectile_pos = projectile.data.root_pos_w
    base_pos = base.data.root_pos_w
    
    distance = torch.norm(projectile_pos - base_pos, dim=1, keepdim=True)
    
    return distance


def tof_distances_obs(
    env: ManagerBasedRLEnv,
    max_range: float = 4.0,
    handle_nan: str = "replace_with_max",
) -> torch.Tensor:

    # Check if environment has sensors
    if not hasattr(env.scene, "sensors") or len(env.scene.sensors) == 0:
        # No sensors in scene, return empty tensor
        num_envs = env.num_envs
        return torch.zeros((num_envs, 0), dtype=torch.float32, device=env.device)
    
    # Collect distances from all sensors
    all_distances = []
    
    for sensor in env.scene.sensors:
        # Try to get TOF distance data
        if hasattr(sensor, "data") and hasattr(sensor.data, "distances"):
            distances = sensor.data.distances  # Shape: (num_envs, num_frames, num_targets)
            
            # Flatten each sensor's measurements
            distances_flat = distances.reshape(distances.shape[0], -1)  # (num_envs, flattened_dims)
            all_distances.append(distances_flat)
    
    if not all_distances:
        # No valid sensor data found, return empty tensor
        num_envs = env.num_envs
        return torch.zeros((num_envs, 0), dtype=torch.float32, device=env.device)
    
    # Concatenate all sensor readings
    tof_readings = torch.cat(all_distances, dim=1)  # (num_envs, total_dims)
    
    # Handle NaN values
    if handle_nan == "replace_with_max":
        tof_readings = torch.where(torch.isnan(tof_readings), torch.tensor(max_range, device=env.device), tof_readings)
    elif handle_nan == "zero":
        tof_readings = torch.where(torch.isnan(tof_readings), torch.tensor(0.0, device=env.device), tof_readings)
    # else: keep as-is
    
    # Normalize by max_range
    tof_normalized = tof_readings / max_range
    
    return tof_normalized
