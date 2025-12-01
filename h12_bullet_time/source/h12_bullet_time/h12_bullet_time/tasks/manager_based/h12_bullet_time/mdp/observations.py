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

__all__ = [
    "base_ang_vel",
    "joint_pos_rel",
    "joint_vel_rel",
    "projected_gravity",
    "last_action",
    "tof_distances_obs",
    "projectile_position_relative",
    "projectile_velocity",
    "projectile_distance_obs",
]


def tof_distances_obs(
    env: ManagerBasedRLEnv,
    max_range: float = 4.0,
    handle_nan: str = "replace_with_max",
) -> torch.Tensor:
    """Extract TOF distance readings from all sensors in the scene.
    
    Args:
        env: The RL environment.
        max_range: Maximum detection range for normalization. Default: 4.0 meters.
        handle_nan: How to handle NaN values (sensor missed target):
            - "replace_with_max": Replace NaN with max_range
            - "zero": Replace NaN with 0
            - "keep": Keep NaN (will cause issues in training)
    
    Returns:
        Flattened tensor of TOF distances from all sensors. Shape: (num_envs, num_sensors_total)
        Values are normalized by max_range.
    """
    # Collect all TOF sensor data from the scene
    tof_distances_list = []
    
    # Iterate through all entities in the scene
    for sensor_name, sensor in env.scene.sensors.items():
        if hasattr(sensor, 'data') and hasattr(sensor.data, 'tof_distances'):
            # Get the TOF distances: shape (num_envs, num_sensors, num_targets)
            tof_data = sensor.data.tof_distances  # (N, S, M)
            
            # Flatten sensor dimensions: (N, S*M)
            N = tof_data.shape[0]
            tof_data_flat = tof_data.reshape(N, -1)
            
            # Handle NaN values
            if handle_nan == "replace_with_max":
                tof_data_flat = torch.where(
                    torch.isnan(tof_data_flat),
                    torch.full_like(tof_data_flat, max_range),
                    tof_data_flat
                )
            elif handle_nan == "zero":
                tof_data_flat = torch.where(
                    torch.isnan(tof_data_flat),
                    torch.zeros_like(tof_data_flat),
                    tof_data_flat
                )
            # else: keep NaN (risky)
            
            # Normalize by max_range
            tof_data_normalized = tof_data_flat / max_range
            
            tof_distances_list.append(tof_data_normalized)
    
    # Concatenate all sensor data
    if tof_distances_list:
        tof_obs = torch.cat(tof_distances_list, dim=1)
    else:
        # No TOF sensors found - return empty tensor
        tof_obs = torch.zeros(env.num_envs, 0, device=env.device)
    
    return tof_obs


def projectile_position_relative(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Projectile position relative to robot base.
    
    Args:
        env: The RL environment.
    
    Returns:
        Relative position vector (x, y, z). Shape: (num_envs, 3)
    """
    robot = env.scene["robot"]
    projectile = env.scene["Projectile"]
    
    # Get positions
    robot_pos = robot.data.root_pos_w  # (num_envs, 3)
    projectile_pos = projectile.data.root_pos_w  # (num_envs, 3)
    
    # Relative position: projectile w.r.t. robot
    projectile_pos_rel = projectile_pos - robot_pos
    
    return projectile_pos_rel


def projectile_velocity(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Projectile velocity in world frame.
    
    Args:
        env: The RL environment.
    
    Returns:
        Projectile linear velocity. Shape: (num_envs, 3)
    """
    projectile = env.scene["Projectile"]
    return projectile.data.root_lin_vel_w  # (num_envs, 3)


def projectile_distance_obs(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Distance between robot and projectile.
    
    Args:
        env: The RL environment.
    
    Returns:
        Euclidean distance (scalar per env). Shape: (num_envs, 1)
    """
    robot = env.scene["robot"]
    projectile = env.scene["Projectile"]
    
    # Get positions
    robot_pos = robot.data.root_pos_w  # (num_envs, 3)
    projectile_pos = projectile.data.root_pos_w  # (num_envs, 3)
    
    # Distance
    distance = torch.linalg.norm(projectile_pos - robot_pos, dim=1, keepdim=True)
    
    return distance

