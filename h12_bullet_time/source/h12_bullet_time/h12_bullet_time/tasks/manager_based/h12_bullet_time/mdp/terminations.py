
from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def base_height_below_threshold(
    env: ManagerBasedRLEnv, threshold: float, asset_cfg: SceneEntityCfg
) -> torch.Tensor:
    """Terminate if base height drops below threshold (robot fell down).
    
    Args:
        env: The environment.
        threshold: Height threshold (in meters). Episode terminates if base_height < threshold.
        asset_cfg: Configuration for the asset (robot).
    
    Returns:
        Boolean tensor indicating which environments should terminate.
    """
    # extract robot asset
    asset: Articulation = env.scene[asset_cfg.name]
    # get base height (z-position of root body)
    base_height = asset.data.body_pos_w[:, 0, 2]
    # compute termination condition
    is_terminated = base_height < threshold
    return is_terminated


# def time_out(*args: Any, **kwargs: Any) -> bool:
# 	"""Time-out handler stub. Return False (no timeout by default)."""
# 	return False


# def is_fallen(*args: Any, **kwargs: Any) -> bool:
# 	"""Simple fallen check stub. Always return False for Phase 1."""
# 	return False


# def robot_out_of_bounds(*args: Any, **kwargs: Any) -> bool:
# 	"""Check if robot out of bounds - stub returns False."""
# 	return False


# __all__ = ["time_out", "is_fallen", "robot_out_of_bounds"]
