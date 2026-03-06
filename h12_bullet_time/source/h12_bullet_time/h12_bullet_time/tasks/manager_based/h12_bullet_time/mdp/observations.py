"""Observation functions for H12 Bullet Time environment."""
from __future__ import annotations

import torch
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg
from h12_bullet_time.sensors.capacitive_sensor import CapacitiveSensor
from h12_bullet_time.sensors.tof_sensor import TofSensor
from h12_bullet_time.sensors.binary_sensor import BinarySensor

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
    "true_pos_obs",
]


def projectile_position_relative(env: ManagerBasedRLEnv, projectile_name: str = "Projectile") -> torch.Tensor:
    projectile = env.scene[projectile_name]
    base = env.scene["robot"]
    return projectile.data.root_pos_w - base.data.root_pos_w


def projectile_velocity(env: ManagerBasedRLEnv, projectile_name: str = "Projectile") -> torch.Tensor:
    return env.scene[projectile_name].data.root_lin_vel_w


def projectile_distance_obs(env: ManagerBasedRLEnv, projectile_name: str = "Projectile") -> torch.Tensor:
    projectile = env.scene[projectile_name]
    base = env.scene["robot"]
    return torch.norm(projectile.data.root_pos_w - base.data.root_pos_w, dim=1, keepdim=True)


def distances_obs(env: ManagerBasedRLEnv) -> torch.Tensor:
    num_envs = env.num_envs
    all_sensor_data = []

    if hasattr(env.scene, '_sensors') and isinstance(env.scene._sensors, dict):
        for sensor_name, sensor_obj in env.scene._sensors.items():
            if isinstance(sensor_obj, (CapacitiveSensor, TofSensor)):
                sensor_data = sensor_obj.data
                if hasattr(sensor_data, "dist_est_normalized"):
                    distances = sensor_data.dist_est_normalized
                    all_flat = distances.reshape(-1)
                    total_per_env = all_flat.numel() // num_envs
                    all_sensor_data.append(all_flat.reshape(num_envs, total_per_env))
            elif isinstance(sensor_obj, BinarySensor):
                detection = sensor_obj.data.binary_detection
                all_flat = detection.reshape(-1)
                total_per_env = all_flat.numel() // num_envs
                all_sensor_data.append(all_flat.reshape(num_envs, total_per_env))

    if not all_sensor_data:
        return torch.zeros((num_envs, 0), dtype=torch.float32, device=env.device)
    return torch.cat(all_sensor_data, dim=1)


def distance_change_obs(env: ManagerBasedRLEnv) -> torch.Tensor:
    num_envs = env.num_envs
    all_sensor_data = []

    if hasattr(env.scene, '_sensors') and isinstance(env.scene._sensors, dict):
        for sensor_name, sensor_obj in env.scene._sensors.items():
            if isinstance(sensor_obj, (CapacitiveSensor, TofSensor)):
                sensor_data = sensor_obj.data
                if hasattr(sensor_data, "dist_est_change_normalized"):
                    dist_diffs = sensor_data.dist_est_change_normalized
                    all_flat = dist_diffs.reshape(-1)
                    total_per_env = all_flat.numel() // num_envs
                    all_sensor_data.append(all_flat.reshape(num_envs, total_per_env))
            elif isinstance(sensor_obj, BinarySensor):
                change = sensor_obj.data.binary_detection_change
                all_flat = change.reshape(-1)
                total_per_env = all_flat.numel() // num_envs
                all_sensor_data.append(all_flat.reshape(num_envs, total_per_env))

    if not all_sensor_data:
        return torch.zeros((num_envs, 0), dtype=torch.float32, device=env.device)
    return torch.cat(all_sensor_data, dim=1)


def min_distances_obs(env: ManagerBasedRLEnv) -> torch.Tensor:
    num_envs = env.num_envs
    all_sensor_data = []

    if hasattr(env.scene, '_sensors') and isinstance(env.scene._sensors, dict):
        for sensor_name, sensor_obj in env.scene._sensors.items():
            if isinstance(sensor_obj, (CapacitiveSensor, TofSensor)):
                sensor_data = sensor_obj.data
                if hasattr(sensor_data, "dist_est_normalized"):
                    distances = sensor_data.dist_est_normalized
                    if isinstance(sensor_obj, TofSensor):
                        distances = distances.min(dim=3).values
                    all_flat = distances.reshape(-1)
                    total_per_env = all_flat.numel() // num_envs
                    all_sensor_data.append(all_flat.reshape(num_envs, total_per_env))
            elif isinstance(sensor_obj, BinarySensor):
                detection = sensor_obj.data.binary_detection
                all_flat = detection.reshape(-1)
                total_per_env = all_flat.numel() // num_envs
                all_sensor_data.append(all_flat.reshape(num_envs, total_per_env))

    if not all_sensor_data:
        return torch.zeros((num_envs, 0), dtype=torch.float32, device=env.device)
    return torch.cat(all_sensor_data, dim=1)


def true_pos_obs(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Returns XYZ relative position of projectile from robot torso, gated by binary sensor detection.

    If at least one binary sensor detects the target (within max_range), returns the
    relative position vector. Otherwise returns zeros.
    """
    num_envs = env.num_envs
    any_detected = torch.zeros(num_envs, dtype=torch.bool, device=env.device)

    if hasattr(env.scene, '_sensors') and isinstance(env.scene._sensors, dict):
        for sensor_name, sensor_obj in env.scene._sensors.items():
            if isinstance(sensor_obj, BinarySensor):
                detection = sensor_obj.data.binary_detection  # (N, S, M)
                any_detected |= (detection > 0.5).any(dim=-1).any(dim=-1)  # (N,)

    rel_pos = env.scene["Projectile"].data.root_pos_w - env.scene["robot"].data.root_pos_w  # (N, 3)
    return rel_pos * any_detected.unsqueeze(-1).float()
