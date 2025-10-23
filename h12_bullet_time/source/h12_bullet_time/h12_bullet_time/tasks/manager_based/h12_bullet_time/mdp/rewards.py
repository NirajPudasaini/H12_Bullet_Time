"""Minimal reward functions for Phase 1 (standing).

These functions provide simple scalar rewards so the environment can start and
the curriculum logic can operate. Replace with more sophisticated terms later.
"""
from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def base_height_l2(env: ManagerBasedRLEnv, target_height: float, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Reward for maintaining base height close to target (default 1.0 m).
    
    Returns negative L2 distance from target height so higher is better.
    """
    # extract robot asset
    asset: Articulation = env.scene[asset_cfg.name]
    # get base height (z-position of root body)
    base_height = asset.data.body_pos_w[:, 0, 2]
    # compute L2 distance from target
    height_error = base_height - target_height
    # return negative squared error (so reward decreases as height deviates)
    return -torch.square(height_error)


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
    try:
        left_knee_idx = body_names.index("left_knee_link")
        right_knee_idx = body_names.index("right_knee_link")
    except ValueError:
        # If specific knee link names not found, try alternative names
        try:
            left_knee_idx = body_names.index("left_knee")
            right_knee_idx = body_names.index("right_knee")
        except ValueError:
            # If still not found, return zero reward
            print(f"Warning: Could not find knee bodies. Available bodies: {body_names}")
            return torch.zeros(env.num_envs, dtype=torch.float32, device=env.device)
    
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
# 	"""Reward for being upright. Return +1.0 when standing."""
# 	return 1.0


# def is_fallen(*args: Any, **kwargs: Any) -> float:
# 	"""Return 0.0 when not fallen, 1.0 when fallen (used as penalty elsewhere)."""
# 	return 0.0


# def upright_torso(*args: Any, **kwargs: Any) -> float:
# 	"""Reward for upright torso—simple positive signal."""
# 	return 0.5


# def joint_vel_penalty(*args: Any, **kwargs: Any) -> float:
# 	"""Small penalty for joint velocity magnitude."""
# 	return -0.01


# def height_control(*args: Any, **kwargs: Any) -> float:
# 	"""Stub for height control reward."""
# 	return 0.0


# def lateral_stability(*args: Any, **kwargs: Any) -> float:
# 	"""Stub for lateral stability reward."""
# 	return 0.0


# def dodge_projectile(*args: Any, **kwargs: Any) -> float:
# 	"""Stub for dodge reward (unused in Phase 1)."""
# 	return 0.0


# def projectile_collision_penalty(*args: Any, **kwargs: Any) -> float:
# 	"""Stub penalty for collision (unused in Phase 1)."""
# 	return 0.0


# def movement_reward(*args: Any, **kwargs: Any) -> float:
# 	"""Stub for movement reward."""
# 	return 0.0


# __all__ = [
# 	"is_alive",
# 	"is_fallen",
# 	"upright_torso",
# 	"joint_vel_penalty",
# 	"height_control",
# 	"lateral_stability",
# 	"dodge_projectile",
# 	"projectile_collision_penalty",
# 	"movement_reward",
# ]
