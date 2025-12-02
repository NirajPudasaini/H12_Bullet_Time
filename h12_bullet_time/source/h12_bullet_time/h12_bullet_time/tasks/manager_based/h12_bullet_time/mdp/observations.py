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
    """TOF sensor distance readings aggregated across all sensors.
    
    Args:
        env: Environment instance
        max_range: Maximum range of TOF sensors (used for normalization)
        handle_nan: How to handle NaN values
        
    Returns:
        Flattened TOF sensor distances (num_envs, total_num_measurements)
        Normalized by max_range so values are in [0, 1]
    """
    from h12_bullet_time.sensors.tof_sensor import TofSensor
    
    num_envs = env.num_envs
    all_sensor_data = []
    
    # Get sensors from env.scene._sensors dict (IsaacLab's official sensor registry)
    if hasattr(env.scene, '_sensors') and isinstance(env.scene._sensors, dict):
        for sensor_name, sensor_obj in env.scene._sensors.items():
            # Check if this is a TofSensor
            if isinstance(sensor_obj, TofSensor):
                sensor_data = sensor_obj.data
                
                # Get distance measurements
                if hasattr(sensor_data, "tof_distances"):
                    distances = sensor_data.tof_distances
                    
                    # Flatten everything and reshape to (num_envs, features_per_env)
                    all_flat = distances.reshape(-1)
                    total_per_env = all_flat.numel() // num_envs
                    
                    # Reshape to (num_envs, features_per_env)
                    flattened = all_flat.reshape(num_envs, total_per_env)
                    all_sensor_data.append(flattened)
    
    # If no valid sensors found, return empty observation
    if not all_sensor_data:
        return torch.zeros((num_envs, 0), dtype=torch.float32, device=env.device)
    
    # Concatenate all sensor readings
    tof_readings = torch.cat(all_sensor_data, dim=1)
    
    # Handle NaN values
    if handle_nan == "replace_with_max":
        tof_readings = torch.nan_to_num(tof_readings, nan=max_range)
    elif handle_nan == "zero":
        tof_readings = torch.nan_to_num(tof_readings, nan=0.0)
    elif handle_nan == "mean":
        # Replace NaN with mean of valid values per environment
        valid_mask = ~torch.isnan(tof_readings)
        for env_idx in range(num_envs):
            valid = tof_readings[env_idx, valid_mask[env_idx]]
            if valid.numel() > 0:
                mean_val = valid.mean()
            else:
                mean_val = max_range
            tof_readings[env_idx, ~valid_mask[env_idx]] = mean_val
    
    # Normalize to [0, 1]
    tof_normalized = torch.clamp(tof_readings / max_range, min=0.0, max=1.0)
    
    return tof_normalized
