"""Observation functions - import from Isaac Lab instead of using stubs."""
from __future__ import annotations

import torch
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg
from h12_bullet_time.sensors.capacitive_sensor import CapacitiveSensor
from h12_bullet_time.sensors.tof_sensor import TofSensor

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
    "distances_obs",
    "distance_change_obs",
    "min_distances_obs",
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

def distances_obs(env: ManagerBasedRLEnv) -> torch.Tensor:

    num_envs = env.num_envs
    all_sensor_data = []

    # Get sensors from env.scene._sensors dict (IsaacLab's official sensor registry)
    if hasattr(env.scene, '_sensors') and isinstance(env.scene._sensors, dict):
        for sensor_name, sensor_obj in env.scene._sensors.items():
            # Check if this is a TofSensor
            if isinstance(sensor_obj, CapacitiveSensor) or isinstance(sensor_obj, TofSensor):
                sensor_data = sensor_obj.data
                
                # Get distance measurements
                if hasattr(sensor_data, "dist_est_normalized"):
                    distances = sensor_data.dist_est_normalized
                    
                    # Flatten everything and reshape to (num_envs, features_per_env)
                    all_flat = distances.reshape(-1)
                    total_per_env = all_flat.numel() // num_envs
                    
                    # Reshape to (num_envs, features_per_env)
                    flattened = all_flat.reshape(num_envs, total_per_env)
                    all_sensor_data.append(flattened)

    if not all_sensor_data:
        return torch.zeros((num_envs, 0), dtype=torch.float32, device=env.device)
    
    distances_readings = torch.cat(all_sensor_data, dim=1)
    
    return distances_readings

def distance_change_obs(env: ManagerBasedRLEnv) -> torch.Tensor:
    
    num_envs = env.num_envs
    all_sensor_data = []

    # Get sensors from env.scene._sensors dict (IsaacLab's official sensor registry)
    if hasattr(env.scene, '_sensors') and isinstance(env.scene._sensors, dict):
        for sensor_name, sensor_obj in env.scene._sensors.items():
            # Check if this is a TofSensor
            if isinstance(sensor_obj, CapacitiveSensor) or isinstance(sensor_obj, TofSensor):
                sensor_data = sensor_obj.data
                
                # Get distance measurements
                if hasattr(sensor_data, "dist_est_change_normalized"):
                    dist_diffs = sensor_data.dist_est_change_normalized
                    
                    # Flatten everything and reshape to (num_envs, features_per_env)
                    all_flat = dist_diffs.reshape(-1)
                    total_per_env = all_flat.numel() // num_envs
                    
                    # Reshape to (num_envs, features_per_env)
                    flattened = all_flat.reshape(num_envs, total_per_env)
                    all_sensor_data.append(flattened)

    if not all_sensor_data:
        return torch.zeros((num_envs, 0), dtype=torch.float32, device=env.device)
    
    dist_diffs_readings = torch.cat(all_sensor_data, dim=1)
    
    return dist_diffs_readings

def min_distances_obs(env: ManagerBasedRLEnv) -> torch.Tensor:

    num_envs = env.num_envs
    all_sensor_data = []

    # Get sensors from env.scene._sensors dict (IsaacLab's official sensor registry)
    if hasattr(env.scene, '_sensors') and isinstance(env.scene._sensors, dict):
        for sensor_name, sensor_obj in env.scene._sensors.items():
            # Check if this is a TofSensor
            if isinstance(sensor_obj, CapacitiveSensor) or isinstance(sensor_obj, TofSensor):
                sensor_data = sensor_obj.data
                
                # Get distance measurements
                if hasattr(sensor_data, "dist_est_normalized"):
                    distances = sensor_data.dist_est_normalized

                    if isinstance(sensor_obj, TofSensor):
                        # Take min across pixel dimension (dim=3) to get closest detection per sensor-target
                        # Shape: (N, S, M, P) -> (N, S, M)
                        # .min() returns (values, indices) tuple, so extract .values
                        distances = distances.min(dim=3).values
                    
                    # Flatten everything and reshape to (num_envs, features_per_env)
                    all_flat = distances.reshape(-1)
                    total_per_env = all_flat.numel() // num_envs
                    
                    # Reshape to (num_envs, features_per_env)
                    flattened = all_flat.reshape(num_envs, total_per_env)
                    all_sensor_data.append(flattened)

    if not all_sensor_data:
        return torch.zeros((num_envs, 0), dtype=torch.float32, device=env.device)
    
    distances_readings = torch.cat(all_sensor_data, dim=1)
    
    return distances_readings

