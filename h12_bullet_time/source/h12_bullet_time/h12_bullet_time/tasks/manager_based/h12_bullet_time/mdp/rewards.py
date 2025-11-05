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
    projectile_names: list | None = None,
    penalty: float = -10.0,
    threshold: float = 0.3,
) -> torch.Tensor:
    """Return per-env penalty when a projectile collides (within threshold) with the robot base.

    This mirrors the termination check and is intentionally simple: it computes
    the minimum distance from the robot base to any projectile candidate and
    applies `penalty` if closer than `threshold`.
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
        return torch.zeros(env.num_envs, dtype=torch.float32, device=env.device)

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
        return torch.zeros(env.num_envs, dtype=torch.float32, device=env.device)

    hit = min_dists < float(threshold)
    out = torch.zeros(env.num_envs, dtype=torch.float32, device=env.device)
    out[hit] = float(penalty)
    return out