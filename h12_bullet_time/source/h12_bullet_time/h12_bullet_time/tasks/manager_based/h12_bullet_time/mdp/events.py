from __future__ import annotations

import math
from typing import Optional

import torch

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg

from isaaclab.envs import ManagerBasedRLEnv


def spawn_projectiles_on_reset(env: ManagerBasedRLEnv, kwargs=None) -> None:
    """Event handler to spawn and launch spherical projectiles toward the robot.

    This is intended to be called at episode reset (EventTerm with mode="reset").
    The function uses internal defaults for spawn distance, speed, and elevation
    and is defensive about API differences between IsaacLab versions.
    """
    # defaults (internal)
    projectile_name = "Projectile"
    spawn_distance = 5.0
    min_speed = 4.0
    max_speed = 8.0
    elevation_deg = 10.0

    device = getattr(env, "device", "cpu")

    # find projectile and robot
    if projectile_name not in env.scene:
        return

    proj = env.scene[projectile_name]
    if "robot" not in env.scene:
        return
    robot = env.scene["robot"]

    # Resolve env_ids and global_env_step_count from provided kwargs dict
    env_ids = None
    global_env_step_count = None
    if isinstance(kwargs, dict):
        env_ids = kwargs.get("env_ids", None)
        global_env_step_count = kwargs.get("global_env_step_count", None)

    # If env_ids is None or resets all envs, do a batched spawn. Otherwise
    # handle per-env resets by looping over provided env ids.
    try:
        base_pos_all = robot.data.body_pos_w[:, 0, :]
    except Exception:
        return

    # helper to spawn for a batch of indices — simplified: single spawn direction (+X)
    def _spawn_for_indices(indices: torch.Tensor):
        n = indices.numel()
        # fixed direction: +X with small elevation variation
        az = torch.zeros(n, device=device)
        el = math.radians(float(elevation_deg)) * (2.0 * torch.rand(n, device=device) - 1.0)
        dx = torch.cos(az) * torch.cos(el)
        dy = torch.sin(az) * torch.cos(el)
        dz = torch.sin(el)
        dirs = torch.stack((dx, dy, dz), dim=1)

        base_pos = base_pos_all[indices]
        spawn_pos = base_pos + dirs * float(spawn_distance)

        quats = torch.zeros((n, 4), dtype=torch.float32, device=device)
        quats[:, 0] = 1.0

        speeds = (min_speed + (max_speed - min_speed) * torch.rand(n, device=device)).unsqueeze(1)
        to_base = base_pos - spawn_pos
        to_base_norm = torch.norm(to_base, dim=1, keepdim=True).clamp(min=1e-6)
        lin_vel = (to_base / to_base_norm) * speeds

        # try batched set first
        try:
            proj.set_world_poses(positions=spawn_pos, orientations=quats)
        except Exception:
            # fallback to per-env pose set
            for i in range(n):
                p = spawn_pos[i : i + 1]
                q = quats[i : i + 1]
                try:
                    proj.set_world_poses(positions=p, orientations=q)
                except Exception:
                    try:
                        proj.write_root_pose_to_sim(p)
                    except Exception:
                        pass

        # try writing linear velocities
        try:
            proj.write_root_velocity_to_sim(lin_vel)
        except Exception:
            # try per-env velocity writes
            for i in range(n):
                v = lin_vel[i : i + 1]
                try:
                    proj.write_root_velocity_to_sim(v)
                except Exception:
                    pass

    # decide whether to do batch or per-env
    if env_ids is None:
        indices = torch.arange(env.num_envs, device=device, dtype=torch.long)
        _spawn_for_indices(indices)
    else:
        # env_ids might be a list, numpy array, or torch tensor
        try:
            ids_tensor = torch.as_tensor(env_ids, device=device, dtype=torch.long)
        except Exception:
            # fallback: convert via list
            ids_tensor = torch.tensor(list(env_ids), device=device, dtype=torch.long)
        if ids_tensor.numel() == env.num_envs:
            _spawn_for_indices(ids_tensor)
        else:
            # handle smaller subsets
            _spawn_for_indices(ids_tensor)

    return
