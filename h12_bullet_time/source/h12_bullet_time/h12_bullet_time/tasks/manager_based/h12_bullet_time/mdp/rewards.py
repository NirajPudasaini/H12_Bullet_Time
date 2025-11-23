"""Minimal reward functions for Phase 1 (standing).

These functions provide simple scalar rewards so the environment can start and
the curriculum logic can operate. Replace with more sophisticated terms later.
"""
from __future__ import annotations

import torch

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg

from isaaclab.envs import ManagerBasedRLEnv

def base_height_l2(env: ManagerBasedRLEnv, target_height: float, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Reward for maintaining base height close to target (default 1.0 m).
    
    Returns positive reward when at target height, negative penalty when deviating.
    This uses a Gaussian-like reward that peaks at the target height.
    """
    # extract robot asset
    asset: Articulation = env.scene[asset_cfg.name]
    # get base height (z-position of root body)
    base_height = asset.data.body_pos_w[:, 0, 2]
    # compute L2 distance from target
    height_error = base_height - target_height
    # return Gaussian reward: exp(-squared_error) so reward is +1.0 at target, approaches 0 when deviating
    # this is better than negative squared error which gives penalty everywhere
    return torch.exp(-torch.square(height_error) * 5.0)  # scaling factor of 5.0 makes the curve steeper


def alive_bonus(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Bonus reward for staying alive (not falling).
    
    Returns +1.0 for each timestep the robot is still running.
    """
    # Return constant reward per environment (batch)
    return torch.ones(env.num_envs, dtype=torch.float32, device=env.device)


def knee_symmetry(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Reward for keeping left and right knees at similar distance from each other.
    
    Encourages symmetric leg posture by maintaining consistent distance between left and right knee bodies.
    This helps prevent one leg from bending more than the other.
    Returns negative L2 distance from target separation so higher is better.
    """
    # extract robot asset
    asset: Articulation = env.scene[asset_cfg.name]
    
    # Get body indices for left and right knees by name
    body_names = asset.body_names
    left_knee_idx = body_names.index("left_knee_link")
    right_knee_idx = body_names.index("right_knee_link")
    
    # Get left and right knee body positions in world frame
    left_knee_pos = asset.data.body_pos_w[:, left_knee_idx, :]  # shape: (num_envs, 3)
    right_knee_pos = asset.data.body_pos_w[:, right_knee_idx, :]  # shape: (num_envs, 3)
    
    # Compute 3D distance between knees
    knee_distance = torch.norm(left_knee_pos - right_knee_pos, dim=1)  # shape: (num_envs,)
    
    # Target distance is roughly shoulder width (around 0.3-0.4 m for humanoid)
    # We want to penalize deviation from this natural stance width
    target_knee_distance = 0.4  # meters
    
    # Compute error: distance from target
    distance_error = knee_distance - target_knee_distance
    
    # Return negative squared error (so reward increases when knees maintain target distance)
    return -torch.square(distance_error)


def projectile_hit_penalty(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    projectile_name: str = "Projectile",
    penalty: float = -10.0,
    threshold: float = 0.5,
) -> torch.Tensor:

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