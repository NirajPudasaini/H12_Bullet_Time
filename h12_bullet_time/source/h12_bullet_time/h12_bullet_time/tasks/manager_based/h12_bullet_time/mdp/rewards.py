"""Minimal reward functions for Phase 1 (standing).

These functions provide simple scalar rewards so the environment can start and
the curriculum logic can operate. Replace with more sophisticated terms later.
"""
from __future__ import annotations

import torch

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg

from isaaclab.envs import ManagerBasedRLEnv

__all__ = [
    "alive_bonus",
    "base_height_l2",
    "base_velocity_reward",
    "projectile_hit_penalty",
    "projectile_proximity_penalty",
    "projectile_distance",
    "torso_pitch_curriculum",
]


def alive_bonus(env: ManagerBasedRLEnv) -> torch.Tensor:

    # Return constant reward per environment (batch)
    return torch.ones(env.num_envs, dtype=torch.float32, device=env.device)



def base_height_l2(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    target_height: float = 1.04,
) -> torch.Tensor:
    """Gaussian reward for maintaining base height at target.
    
    Returns high reward when robot is at target height, decays with Gaussian.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    
    # Get base height
    height = asset.data.root_pos_w[:, 2]  # z-coordinate
    
    # Gaussian penalty: exp(-5 * (height - target)^2)
    error = height - float(target_height)
    reward = torch.exp(-5.0 * error**2)
    
    return reward


def base_velocity_reward(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    scale: float = 10.0,
) -> torch.Tensor:

    asset: Articulation = env.scene[asset_cfg.name]

    lin_vel = asset.data.root_lin_vel_w[:, :2]  # shape: (num_envs, 2)
    vel_norm2 = torch.sum(lin_vel ** 2, dim=1)
    reward = torch.exp(-float(scale) * vel_norm2)

    return reward
 
def projectile_hit_penalty(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    projectile_name: str = "Projectile",
    penalty: float = -10.0,
    threshold: float = 0.5,
) -> torch.Tensor:
    """Penalty for projectile hitting robot (binary collision penalty).
    
    Only triggers when projectile gets within threshold distance of robot body.
    Useful as a hard constraint in Phase 2b after robot learns to dodge.
    
    Args:
        env: RL environment
        asset_cfg: Robot entity config
        projectile_name: Name of projectile entity
        penalty: Reward value when hit (negative, e.g., -10.0)
        threshold: Distance threshold for collision (meters)
    
    Returns:
        Tensor of shape (num_envs,) with penalty if hit, 0 otherwise
    """
    # Get robot
    robot: Articulation = env.scene[asset_cfg.name]
    robot_body_positions = robot.data.body_pos_w  # shape: (num_envs, num_bodies, 3)
    
    # Get projectile
    try:
        projectile = env.scene[projectile_name]
        proj_pos = projectile.data.root_pos_w  # shape: (num_envs, 3)
    except (KeyError, AttributeError):
        # Projectile not found, no penalty
        return torch.zeros(env.num_envs, dtype=torch.float32, device=env.device)
    
    # Compute distance from projectile to each robot body
    # proj_pos: (num_envs, 3) -> (num_envs, 1, 3)
    # robot_body_positions: (num_envs, num_bodies, 3)
    distances = torch.norm(
        robot_body_positions - proj_pos.unsqueeze(1),
        dim=-1
    )
    
    # Find minimum distance to any body for each environment
    min_dist_per_env = distances.min(dim=1)[0]  # shape: (num_envs,)
    
    # Apply penalty if within threshold
    reward = torch.zeros(env.num_envs, dtype=torch.float32, device=env.device)
    hit = min_dist_per_env < float(threshold)
    reward[hit] = float(penalty)
    
    return reward


def projectile_proximity_penalty(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    projectile_name: str = "Projectile",
    max_distance: float = 2.0,
    penalty_scale: float = -1.0,
    approach_gain: float = 2.0,
) -> torch.Tensor:

    # Get robot
    robot: Articulation = env.scene[asset_cfg.name]
    robot_pos = robot.data.root_pos_w[:, :3]  # shape: (num_envs, 3)
    
    # Get projectile
    try:
        projectile = env.scene[projectile_name]
        proj_pos = projectile.data.root_pos_w  # shape: (num_envs, 3)
    except (KeyError, AttributeError):
        # Projectile not found, no penalty
        return torch.zeros(env.num_envs, dtype=torch.float32, device=env.device)
    
    # Compute distance from projectile to robot base
    rel = proj_pos - robot_pos
    distance = torch.norm(rel, dim=-1)  # shape: (num_envs,)

    # Compute approach speed: projection of relative velocity onto relative vector
    # proj_lin_vel: (num_envs, 3)
    proj_lin_vel = projectile.data.root_lin_vel_w
    # robot base linear velocity (world frame)
    try:
        robot_lin_vel = robot.data.root_lin_vel_w
    except Exception:
        robot_lin_vel = torch.zeros_like(proj_lin_vel)

    # Relative velocity of projectile w.r.t robot base
    rel_vel = proj_lin_vel - robot_lin_vel  # (num_envs, 3)

    eps = 1e-6
    # unit direction from robot -> projectile: avoid division by zero
    dir_unit = rel / (distance.unsqueeze(-1) + eps)
    # approach_speed = -dot(rel_vel, dir_unit) so positive when projectile moves toward robot
    approach_speed = -torch.sum(rel_vel * dir_unit, dim=-1)  # (num_envs,)
    approach_speed_clamped = torch.clamp(approach_speed, min=0.0)

    # Base linear penalty: ramps from 0 at max_distance to penalty_scale at distance=0
    base_penalty = float(penalty_scale) * (1.0 - distance / float(max_distance))
    base_penalty = torch.clamp(base_penalty, min=float(penalty_scale), max=0.0)

    # Boost penalty when projectile is approaching: factor = 1 + approach_gain * (approach_speed / (1 + approach_speed))
    # This keeps the boost bounded while growing with approach speed.
    approach_factor = 1.0 + float(approach_gain) * (approach_speed_clamped / (1.0 + approach_speed_clamped))

    penalty = base_penalty * approach_factor

    # For distances beyond max_distance, ensure penalty is zero
    penalty = torch.where(distance >= float(max_distance), torch.zeros_like(penalty), penalty)

    return penalty



def projectile_distance(
    env: ManagerBasedRLEnv,
) -> torch.Tensor:

    try:
        projectile = env.scene["Projectile"]
        proj_pos = projectile.data.root_pos_w  # shape: (num_envs, 3)
    except (KeyError, AttributeError):
        # Projectile not spawned yet, no reward
        return torch.zeros(env.num_envs, dtype=torch.float32, device=env.device)
    
    # Get robot base position
    robot = env.scene["robot"]
    robot_pos = robot.data.root_pos_w  # shape: (num_envs, 3)
    
    # Compute distance
    distance = torch.norm(proj_pos - robot_pos, dim=-1)  # shape: (num_envs,)
    
    # Initialize reward tensor
    reward = torch.zeros(env.num_envs, dtype=torch.float32, device=env.device)
    
    # Hard penalty for distance < 2m (too close!)
    too_close = distance < 1.0
    reward[too_close] = -10.0
    
    # Neutral zone for 2m <= distance <= 3m (safe, but no bonus)
    # (no change needed, already 0)
    
    # Linear reward for distance > 3m (extra distance = bonus)
    # Reward = (distance - 3.0) for each meter beyond 3m
    far = distance >= 1.5
    reward[far] = (distance[far] - 3.0)  # Linear excess distance reward
    
    return reward


def torso_pitch_curriculum(
    env: ManagerBasedRLEnv,
    curriculum_step: int = 500,
    max_pitch_scale: float = 0.5,
) -> torch.Tensor:
    """Curriculum function that returns scaling factor for torso pitch perturbations.
    
    Phase 1 (steps 0-curriculum_step): Returns 0 (no disturbance)
    Phase 2 (steps curriculum_step+): Returns value ramping from 0 to max_pitch_scale
    
    This is a curriculum function that returns a scalar per environment.
    The returned value can be used to scale torso pitch action perturbations.
    
    Args:
        env: The RL environment
        curriculum_step: Training step at which to start perturbations
        max_pitch_scale: Maximum pitch scale to reach (0.5 = 50% of action range)
    
    Returns:
        Tensor of shape (num_envs,) with scaling factors
    """
    # Get current training step
    step = env.common_step_counter
    
    # Phase 1: Before curriculum_step, no perturbation
    if step < curriculum_step:
        return torch.zeros(env.num_envs, dtype=torch.float32, device=env.device)
    
    # Phase 2: Ramp up from 0 to max_pitch_scale
    # Linear ramp over 5000 steps (curriculum_step to curriculum_step + 5000)
    progress = float(step - curriculum_step) / 5000.0
    scale = min(progress, 1.0) * max_pitch_scale  # Clamp to max_pitch_scale
    
    # Return same scale for all environments
    return torch.full((env.num_envs,), scale, dtype=torch.float32, device=env.device)

