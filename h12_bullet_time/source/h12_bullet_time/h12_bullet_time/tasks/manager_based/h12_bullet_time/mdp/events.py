from __future__ import annotations

import math
import sys
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

from isaaclab.managers import SceneEntityCfg


def launch_projectile(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("Projectile"),
)-> None:

    # Parameters (defaults chosen to spawn relative to torso link)
    spawn_distance = 1.5
    height_offset = 1.0  # 30 cm above the specified robot link
    throw_speed = 5.0
    offset_range_x = 0.2
    offset_range_z = 0.2

    # Get projectile and robot from scene
    proj = env.scene[asset_cfg.name]
    robot = env.scene["robot"]

    # Try to get body link positions; fall back to root if not available
    try:
        body_names = list(robot.body_names)
    except Exception:
        body_names = []

    root_pos = robot.data.root_pos_w  # (num_envs, 3)
    device = root_pos.device

    # Default center positions for the envs being reset
    center = root_pos[env_ids]
    # If torso (or requested link) exists, use that link's world position
    if "torso" in body_names:
        idx = body_names.index("torso")
        link_pos = robot.data.body_pos_w[:, idx, :]
        center = link_pos[env_ids]

    n = env_ids.numel()

    # Random offsets
    x_offset = torch.rand((n,), device=device, dtype=torch.float32) * 2 * offset_range_x - offset_range_x
    z_offset = torch.rand((n,), device=device, dtype=torch.float32) * 2 * offset_range_z - offset_range_z

    # Spawn position: in front of link and height offset above link
    spawn_pos = torch.zeros((n, 3), device=device, dtype=torch.float32)
    spawn_pos[:, 0] = center[:, 0] + spawn_distance + x_offset
    spawn_pos[:, 1] = center[:, 1]
    spawn_pos[:, 2] = center[:, 2] + height_offset + z_offset

    # Identity quaternion
    quats = torch.zeros((n, 4), device=device, dtype=torch.float32)
    quats[:, 0] = 1.0

    # Linear velocity toward robot (-X)
    lin_vel = torch.zeros((n, 3), device=device, dtype=torch.float32)
    lin_vel[:, 0] = -throw_speed

    ang_vel = torch.zeros((n, 3), device=device, dtype=torch.float32)

    pose = torch.cat([spawn_pos, quats], dim=-1)
    velocity = torch.cat([lin_vel, ang_vel], dim=-1)

    proj.write_root_pose_to_sim(pose, env_ids)
    proj.write_root_velocity_to_sim(velocity, env_ids)
    # Debug print: show spawn info for the envs we updated
    # try:
    #     sp = spawn_pos.cpu().numpy()
    #     lv = lin_vel.cpu().numpy()
    #     ids = env_ids.cpu().numpy()
    #     print(f"[launch_projectile] env_ids={ids.tolist()}, spawn_pos={sp.tolist()}, lin_vel={lv.tolist()}")
    # except Exception:
    #     # Fallback safe print if tensors are not CPU-accessible
    #     print(f"[launch_projectile] spawned projectiles for env_ids={env_ids}")


def launch_projectile_curriculum(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("Projectile"),
    curriculum_step: int = 2000,
) -> None:

    # Only launch projectiles after curriculum milestone
    # Note: common_step_counter counts training iterations, not environment steps
    if env.common_step_counter < curriculum_step:
        return
    
    # Call the regular launch_projectile function
    launch_projectile(env, env_ids, asset_cfg)


# def apply_torso_pitch_disturbance(
#     env: ManagerBasedRLEnv,
# ) -> None:

#     # Get the torso pitch curriculum scale (returns 0 before curriculum_step, ramps up after)
#     from . import rewards as mdp_rewards
    
#     scale = mdp_rewards.torso_pitch_curriculum(
#         env,
#         curriculum_step=500,
#         max_pitch_scale=0.5,
#     )  # shape: (num_envs,)
    
#     if scale.max() > 0:  # Only apply if there's non-zero scale
#         # Torso joint is at index 12 in the joint_names list in ActionsCfg
#         # Joint order: left_leg (6) + right_leg (6) + torso (1) = index 12
#         torso_pitch_idx = 12
        
#         num_envs = env.num_envs
        
#         # Generate random pitch perturbations (Gaussian noise)
#         perturbation = torch.randn(
#             (num_envs,),
#             device=env.device,
#             dtype=torch.float32,
#         ) * scale  # Scale by curriculum value
        
#         # Apply to action directly
#         # env.action_manager.action is the processed action tensor
#         # We add perturbation to the torso pitch joint command
#         try:
#             env.action_manager.action[:, torso_pitch_idx] += perturbation
#         except (IndexError, RuntimeError):
#             # If action tensor shape is different, silently skip
#             pass

