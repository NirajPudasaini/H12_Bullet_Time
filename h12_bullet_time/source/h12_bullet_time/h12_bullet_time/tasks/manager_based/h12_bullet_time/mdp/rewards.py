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

# def is_alive(*args: Any, **kwargs: Any) -> float:
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
