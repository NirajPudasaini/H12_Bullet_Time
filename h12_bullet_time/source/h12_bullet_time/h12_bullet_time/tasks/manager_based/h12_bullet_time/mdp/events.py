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
    height_offset = 0.85
    throw_speed = 5.0
    offset_range_x = 1.0
    offset_range_z = 0.1
    # Randomization: allow +/-20% variation on spawn_distance and height_offset per env
    var_frac = 0.2  # 20%

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
    # x_offset only to the right: sample in [0, offset_range_x]
    x_offset = torch.rand((n,), device=device, dtype=torch.float32) * offset_range_x
    z_offset = torch.rand((n,), device=device, dtype=torch.float32) * 2 * offset_range_z - offset_range_z

    # Per-env random scaling in [1-0.2, 1+0.2]
    sd_scale = 1.0 + (torch.rand((n,), device=device, dtype=torch.float32) * 2.0 * var_frac - var_frac)
    ho_scale = 1.0 + (torch.rand((n,), device=device, dtype=torch.float32) * 2.0 * var_frac - var_frac)
    spawn_distance_per_env = spawn_distance * sd_scale
    height_offset_per_env = height_offset * ho_scale

    # Spawn position: in front of link and height offset above link
    spawn_pos = torch.zeros((n, 3), device=device, dtype=torch.float32)
    # apply per-env spawn distance and height offset with random +/-20% variation
    spawn_pos[:, 0] = center[:, 0] + spawn_distance_per_env + x_offset
    spawn_pos[:, 1] = center[:, 1]
    spawn_pos[:, 2] = center[:, 2] + height_offset_per_env + z_offset

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


def print_tof_readings(env: ManagerBasedRLEnv, env_ids: torch.Tensor | None = None) -> None:
    """Debug helper: print a compact summary of TOF sensor readings.

    Prints shapes, number of valid detections and a few sample values for the
    first environment index (or the first index in env_ids if provided).
    """
    try:
        sensor_names = getattr(env.scene, "sensors", None)
    except Exception:
        sensor_names = None

    if not sensor_names:
        print("[TOF DEBUG] No sensors found in scene.")
        return

    # choose environment index to inspect
    env_idx = 0
    if env_ids is not None:
        try:
            if isinstance(env_ids, torch.Tensor) and env_ids.numel() > 0:
                env_idx = int(env_ids.flatten()[0].item())
            elif isinstance(env_ids, (list, tuple)) and len(env_ids) > 0:
                env_idx = int(env_ids[0])
            else:
                env_idx = int(env_ids)
        except Exception:
            env_idx = 0

    # Print header
    print(f"[TOF DEBUG] Env {env_idx}: {len(sensor_names)} sensor(s) present")

    for s_idx, sensor_name in enumerate(sensor_names):
        # Get the actual sensor object from the scene
        try:
            sensor = env.scene[sensor_name]
            data = sensor.data
        except Exception as e:
            print(f"  Sensor[{s_idx}] ({sensor_name}): failed to access data: {e}")
            continue

        if data is None:
            print(f"  Sensor[{s_idx}] ({sensor_name}): no data")
            continue

        # prefer tof_distances, fall back to raw_target_distances or distances
        tof = None
        for attr in ("tof_distances", "raw_target_distances", "distances", "target_distances"):
            if hasattr(data, attr):
                tof = getattr(data, attr)
                break

        if tof is None:
            print(f"  Sensor[{s_idx}] ({sensor_name}): no distance attribute found")
            continue

        try:
            # Ensure tensor on CPU for printing (handle numpy arrays too)
            if hasattr(tof, "cpu"):
                arr = tof.cpu()
            else:
                import numpy as _np

                arr = _np.asarray(tof)

            # arr expected shape: (num_envs, num_sensors, num_targets) or similar
            if hasattr(arr, "numpy"):
                # torch tensor
                vals = arr.numpy()
            else:
                vals = arr

            # Extract env slice
            if vals.ndim == 0:
                print(f"  Sensor[{s_idx}] ({sensor_name}): scalar={vals}")
                continue

            if vals.shape[0] <= env_idx:
                print(f"  Sensor[{s_idx}] ({sensor_name}): env index {env_idx} out of range (shape {vals.shape})")
                continue

            slice_env = vals[env_idx]

            # Flatten and compute stats
            flat = slice_env.flatten()
            import numpy as _np

            nan_count = int(_np.isnan(flat).sum()) if _np.isnan(flat).any() else 0
            valid_count = flat.size - nan_count
            mean_val = float(_np.nanmean(flat)) if valid_count > 0 else float("nan")
            sample_vals = flat[:8]

            print(f"  Sensor[{s_idx}] ({sensor_name}): shape={vals.shape}, valid={valid_count}, nans={nan_count}, mean={mean_val:.3f}")
            print("    samples:", ", ".join([f"{float(x):.3f}" if not _np.isnan(x) else "nan" for x in sample_vals]))
        except Exception as e:
            print(f"  Sensor[{s_idx}] ({sensor_name}): failed to read data: {e}")

    # flush stdout to ensure visibility in logs
    try:
        sys.stdout.flush()
    except Exception:
        pass
