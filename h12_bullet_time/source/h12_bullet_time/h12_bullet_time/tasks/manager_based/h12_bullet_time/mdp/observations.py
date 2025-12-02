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
    
    Accesses sensor.data property which triggers automatic updates via SensorBase._update_outdated_buffers().
    This is the standard IsaacLab pattern for lazy-evaluated sensor data.
    
    Args:
        env: Environment instance
        max_range: Maximum range of TOF sensors (used for normalization)
        handle_nan: How to handle NaN values:
            - "replace_with_max": Replace NaN with max_range
            - "zero": Replace NaN with 0
            - "keep": Keep NaN values as-is
        
    Returns:
        Flattened TOF sensor distances (num_envs, total_num_measurements)
        Normalized by max_range so values are in [0, 1]
    """
    # Check if environment has sensors
    if not hasattr(env.scene, "sensors") or len(env.scene.sensors) == 0:
        # No sensors in scene, return empty tensor
        num_envs = env.num_envs
        return torch.zeros((num_envs, 0), dtype=torch.float32, device=env.device)
    
    # Collect distances from all sensors
    all_distances = []
    
    # env.scene.sensors is a list of sensor names (strings)
    for sensor_name in env.scene.sensors:
        # Get the actual sensor object from the scene
        try:
            sensor = env.scene[sensor_name]
            sensor_data = sensor.data
            
            # Try tof_distances first (preferred, includes FOV-based culling)
            if hasattr(sensor_data, "tof_distances"):
                distances = sensor_data.tof_distances  # Shape: (num_envs, num_sensors, num_targets)
            # Fallback to raw_target_distances
            elif hasattr(sensor_data, "raw_target_distances"):
                distances = sensor_data.raw_target_distances
            # Final fallback to distances attribute
            elif hasattr(sensor_data, "distances"):
                distances = sensor_data.distances
            else:
                # Skip this sensor if it has no distance data
                continue
            
            # Flatten each sensor's measurements: (num_envs, num_sensors, num_targets) -> (num_envs, flattened)
            distances_flat = distances.reshape(distances.shape[0], -1)  # (num_envs, flattened_dims)
            all_distances.append(distances_flat)
        except Exception:
            # Skip sensors that fail to access data
            continue
    
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
