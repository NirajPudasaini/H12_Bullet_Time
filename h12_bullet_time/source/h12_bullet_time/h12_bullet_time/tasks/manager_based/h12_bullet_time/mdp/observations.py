"""Observation functions for H12 Bullet Time environment."""
from __future__ import annotations

import torch
from isaaclab.envs import ManagerBasedRLEnv

from isaaclab.envs.mdp import (
    base_ang_vel,
    joint_pos_rel,
    joint_vel_rel,
    projected_gravity,
    last_action,
)

__all__ = [
    "base_ang_vel", "joint_pos_rel", "joint_vel_rel", "projected_gravity", "last_action",
    "sensor_obs", "sensor_obs_all",
    "projectile_position_relative", "projectile_velocity", "projectile_distance_obs",
]


def sensor_obs(env: ManagerBasedRLEnv, sensor_prefix: str, signal_type: str) -> torch.Tensor:
    """Unified sensor observation function.

    Args:
        env: Environment instance.
        sensor_prefix: Prefix to match sensor names (e.g., "field_0", "ray_1").
        signal_type: One of "DIST", "MINDIST", "BIN", "MINBIN", "EVENT", "TRUE_POS".

    Returns:
        Observation tensor of shape (num_envs, features).
    """
    from h12_bullet_time.sensors import RaySensor

    if signal_type == "TRUE_POS":
        return _true_pos_gated(env, sensor_prefix)

    num_envs = env.num_envs
    all_data = []

    if hasattr(env.scene, '_sensors') and isinstance(env.scene._sensors, dict):
        for name, sensor in env.scene._sensors.items():
            if not name.startswith(sensor_prefix):
                continue

            sd = sensor.data

            if signal_type == "DIST":
                data = sd.dist_est_normalized
            elif signal_type == "MINDIST":
                data = sd.dist_est_normalized
                if isinstance(sensor, RaySensor):
                    data = data.min(dim=-1).values
            elif signal_type == "BIN":
                data = sd.binary_detection
            elif signal_type == "MINBIN":
                data = sd.binary_detection
                if isinstance(sensor, RaySensor):
                    data = data.reshape(num_envs, -1).any(dim=-1).float().unsqueeze(-1)
            elif signal_type == "EVENT":
                data = sd.dist_est_change_normalized
            else:
                continue

            all_data.append(data.reshape(num_envs, -1))

    if not all_data:
        return torch.zeros((num_envs, 0), device=env.device)
    return torch.cat(all_data, dim=1)


def _true_pos_gated(env: ManagerBasedRLEnv, sensor_prefix: str) -> torch.Tensor:
    """True position of projectile, gated by detection from sensors matching prefix."""
    num_envs = env.num_envs
    any_detected = torch.zeros(num_envs, dtype=torch.bool, device=env.device)

    if hasattr(env.scene, '_sensors') and isinstance(env.scene._sensors, dict):
        for name, sensor in env.scene._sensors.items():
            if not name.startswith(sensor_prefix):
                continue
            detection = sensor.data.binary_detection
            any_detected |= (detection > 0.5).reshape(num_envs, -1).any(dim=1)

    rel_pos = env.scene["Projectile"].data.root_pos_w - env.scene["robot"].data.root_pos_w
    return rel_pos * any_detected.unsqueeze(-1).float()


def sensor_obs_all(env: ManagerBasedRLEnv, specs: list) -> torch.Tensor:
    """Aggregated observations from all configured sensor specs.

    Args:
        env: Environment instance.
        specs: List of (sensor_prefix, signal_type) tuples.
    """
    all_data = []
    for prefix, signal_type in specs:
        data = sensor_obs(env, prefix, signal_type)
        if data.shape[1] > 0:
            all_data.append(data)
    if not all_data:
        return torch.zeros((env.num_envs, 0), device=env.device)
    return torch.cat(all_data, dim=1)


def projectile_position_relative(env: ManagerBasedRLEnv, projectile_name: str = "Projectile") -> torch.Tensor:
    return env.scene[projectile_name].data.root_pos_w - env.scene["robot"].data.root_pos_w


def projectile_velocity(env: ManagerBasedRLEnv, projectile_name: str = "Projectile") -> torch.Tensor:
    return env.scene[projectile_name].data.root_lin_vel_w


def projectile_distance_obs(env: ManagerBasedRLEnv, projectile_name: str = "Projectile") -> torch.Tensor:
    return torch.norm(
        env.scene[projectile_name].data.root_pos_w - env.scene["robot"].data.root_pos_w,
        dim=1, keepdim=True,
    )
