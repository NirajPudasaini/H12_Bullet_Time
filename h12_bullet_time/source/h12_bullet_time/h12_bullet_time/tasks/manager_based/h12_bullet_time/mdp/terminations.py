
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


def projectile_hit(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    projectile_names: list | None = None,
    threshold: float = 0.3,
) -> torch.Tensor:
    """Terminate episode when any projectile comes within `threshold` of robot base.

    This mirrors `projectile_hit_penalty` in rewards and uses the same name
    matching heuristic when `projectile_names` is None.
    Returns a boolean tensor (shape (num_envs,)) where True indicates termination.
    """
    # robot base position
    asset: Articulation = env.scene[asset_cfg.name]
    base_pos = asset.data.body_pos_w[:, 0, :]

    # candidate projectile names
    scene_names = list(env.scene.keys())
    candidates = [] if projectile_names is None else list(projectile_names)
    if projectile_names is None:
        for n in scene_names:
            if "projectile" in n.lower() or "obstacle" in n.lower():
                candidates.append(n)

    if len(candidates) == 0:
        return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

    min_dists = None
    for name in candidates:
        try:
            obj = env.scene[name]
            pos = obj.data.body_pos_w[:, 0, :]
        except Exception:
            continue
        d = torch.norm(base_pos - pos, dim=1)
        min_dists = d if min_dists is None else torch.minimum(min_dists, d)

    if min_dists is None:
        return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

    return min_dists < float(threshold)