from __future__ import annotations

import torch

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg

from isaaclab.envs import ManagerBasedRLEnv
from h12_bullet_time.sensors.capacitive_sensor import CapacitiveSensor
from h12_bullet_time.sensors.tof_sensor import TofSensor

__all__ = [
    "alive_bonus",
    "base_height_l2",
    "base_velocity_reward",
    "projectile_hit_penalty",
    "projectile_proximity_penalty",
    "projectile_distance",
    "torso_pitch_curriculum",
    "torso_pitch_reward",
    "distances_penalty",
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


# def projectile_contact_penalty(
#     env: ManagerBasedRLEnv,
#     asset_cfg: SceneEntityCfg,
#     projectile_name: str = "Projectile",
#     contact_threshold: float = 0.05,
#     penalty: float = -500.0,
# ) -> torch.Tensor:

#     # Get robot body positions
#     robot: Articulation = env.scene[asset_cfg.name]
#     try:
#         robot_body_positions = robot.data.body_pos_w  # (num_envs, num_bodies, 3)
#     except Exception:
#         # If body positions are not available, return zeros
#         return torch.zeros(env.num_envs, dtype=torch.float32, device=env.device)

#     # Get projectile
#     scene_names = list(env.scene.keys())
#     candidates = [] if projectile_name is None else [projectile_name]
#     if projectile_name is None:
#         for n in scene_names:
#             if "projectile" in n.lower() or "obstacle" in n.lower():
#                 candidates.append(n)

#     if len(candidates) == 0:
#         return torch.zeros(env.num_envs, dtype=torch.float32, device=env.device)

#     penalty_tensor = torch.zeros(env.num_envs, dtype=torch.float32, device=env.device)

#     for name in candidates:
#         obj = env.scene[name]
#         try:
#             proj_pos = obj.data.root_pos_w  # (num_envs, 3)
#         except Exception:
#             try:
#                 proj_pos = obj.data.body_pos_w[:, 0, :]
#             except Exception:
#                 continue

#         # Compute distances to all robot bodies
#         distances = torch.norm(robot_body_positions - proj_pos.unsqueeze(1), dim=-1)  # (num_envs, num_bodies)
#         min_dist = distances.min(dim=1)[0]

#         hit_mask = min_dist < float(contact_threshold)
#         if hit_mask.any():
#             penalty_tensor[hit_mask] = float(penalty)

#     return penalty_tensor

def distances_penalty(
    env: ManagerBasedRLEnv,
    proximity_scale: float = -1.0,
    contact_scale: float = -1000.0,
    contact_threshold: float = 0.1, #% of sensor range
) -> torch.Tensor:
    """Penalize proximity to capacitive sensors. This sums up the proximity to all sensors.
    """

    
    num_envs = env.num_envs
    total_penalty = torch.zeros(num_envs, dtype=torch.float32, device=env.device)
    
    # Get sensors from env.scene._sensors dict
    if hasattr(env.scene, '_sensors') and isinstance(env.scene._sensors, dict):
        for sensor_name, sensor_obj in env.scene._sensors.items():
            if isinstance(sensor_obj, CapacitiveSensor) or isinstance(sensor_obj, TofSensor):
                sensor_data = sensor_obj.data
                if hasattr(sensor_data, "dist_est_normalized"):
                    # Shape: (num_envs, num_sensors, num_targets) or similar
                    normalized_distances = sensor_data.dist_est_normalized
                    if isinstance(sensor_obj, TofSensor):
                        # Take min across pixel dimension (dim=3) to get closest detection per sensor-target
                        # Shape: (N, S, M, P) -> (N, S, M)
                        # .min() returns (values, indices) tuple, so extract .values
                        normalized_distances = normalized_distances.min(dim=3).values
                    # Proximity = 1 - normalized_distance (1 = touching, 0 = far)
                    proximity = 1.0 - normalized_distances
                    # Sum all proximity values per environment
                    proximity_penalty = proximity.reshape(num_envs, -1).sum(dim=1)*proximity_scale
                    contact_mask = normalized_distances < contact_threshold
                    contact_penalty = contact_mask.reshape(num_envs, -1).sum(dim=1).float()*contact_scale
                    total_penalty += proximity_penalty + contact_penalty

    return total_penalty 

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

    # Get projectile
    try:
        projectile = env.scene[projectile_name]
        proj_pos = projectile.data.root_pos_w  # shape: (num_envs, 3)
    except (KeyError, AttributeError):
        # Projectile not found, no penalty
        return torch.zeros(env.num_envs, dtype=torch.float32, device=env.device)

    # Build a list of robot link positions to consider for proximity checks.
    # Include base/root plus important upper-body links so the robot fully avoids obstacles.
    positions_list = []
    # root/base position (index 0)
    try:
        root_pos = robot.data.root_pos_w[:, :3]
        positions_list.append(root_pos.unsqueeze(1))  # (num_envs, 1, 3)
    except Exception:
        # Fallback to zeros if unavailable
        positions_list.append(torch.zeros((env.num_envs, 1, 3), device=env.device, dtype=torch.float32))

    # Preferred link names to check (as requested by user)
    link_names_to_check = [
        "left_elbow_link",
        "right_elbow_link",
        "left_shoulder_yaw_link",
        "right_shoulder_yaw_link",
        "lidar_link",
    ]

    # Robot may expose body/link names via "body_names" attribute
    body_names = []
    try:
        body_names = list(robot.body_names)
    except Exception:
        body_names = []

    # Track which appended index corresponds to which link name so we can
    # apply link-specific amplifications (e.g., for `lidar_link`).
    link_indices: dict = {}

    # For each requested link, if present, append its world position
    for ln in link_names_to_check:
        if ln in body_names:
            idx = body_names.index(ln)
            link_pos = robot.data.body_pos_w[:, idx, :]
            # current index in concatenated positions will be len(positions_list)
            link_indices[ln] = len(positions_list)
            positions_list.append(link_pos.unsqueeze(1))

    # Concatenate positions -> (num_envs, n_links, 3)
    positions = torch.cat(positions_list, dim=1)

    # Compute relative vectors from each considered link to projectile: (num_envs, n_links, 3)
    rel = proj_pos.unsqueeze(1) - positions
    distances = torch.norm(rel, dim=-1)  # (num_envs, n_links)

    # Base penalty per link: purely distance-based (no approach-speed term)
    # Linear ramp from 0 at max_distance to penalty_scale at distance=0
    base_penalty_each = float(penalty_scale) * (1.0 - distances / float(max_distance))
    base_penalty_each = torch.clamp(base_penalty_each, min=float(penalty_scale), max=0.0)

    # Zero out penalties beyond max_distance per link
    penalty_each = torch.where(distances >= float(max_distance), torch.zeros_like(base_penalty_each), base_penalty_each)

    # Amplify penalty for lidar_link if present: we care strongly about the lidar
    # being hit or closely approached (sensor protection). Multiply the per-link
    # penalty by a factor so that close approaches to `lidar_link` are punished
    # more heavily than other links. This only affects that link's penalty;
    # the overall penalty still uses the minimum-distance link.
    try:
        if "lidar_link" in link_indices:
            lidar_idx = link_indices["lidar_link"]
            # Make lidar penalty more severe. Factor selected empirically —
            # increase if you want even stronger protection.
            lidar_factor = 3.0
            lidar_pen = penalty_each[:, lidar_idx] * float(lidar_factor)
            # Clamp to penalty_scale min (can't exceed the configured worst penalty)
            lidar_pen = torch.clamp(lidar_pen, min=float(penalty_scale), max=0.0)
            penalty_each[:, lidar_idx] = lidar_pen
    except Exception:
        # If anything goes wrong with link indexing, fall back silently.
        pass

    # Select the minimum distance link per environment (the one that matters most)
    min_idx = distances.argmin(dim=1)  # (num_envs,)
    batch_idx = torch.arange(env.num_envs, device=env.device)
    penalty = penalty_each[batch_idx, min_idx]

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


def torso_pitch_reward(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    torso_link_name: str = "torso_link",
    head_link_candidates: list | None = None,
    scale: float = 5.0,
    max_pitch: float = 0.8,
) -> torch.Tensor:
    """Encourage torso pitch (bending) by rewarding absolute pitch angle.

    This function attempts to compute a pitch estimate for the torso by
    finding a second link to compute a torso-to-head vector (preferred
    candidates are provided). If a quaternion is available for the torso
    body, it would be preferable, but to keep this robust across different
    Articulation representations we compute pitch from link positions when
    possible.

    Returns a per-environment scalar reward encouraging larger absolute
    torso pitch up to `max_pitch` (radians). The reward is scaled by
    `scale` and clipped.
    """
    robot: Articulation = env.scene[asset_cfg.name]

    # Default candidate head links if none provided
    if head_link_candidates is None:
        head_link_candidates = ["head_link", "neck_link", "lidar_link"]

    # Resolve body names
    try:
        body_names = list(robot.body_names)
    except Exception:
        return torch.zeros(env.num_envs, dtype=torch.float32, device=env.device)

    if torso_link_name not in body_names:
        return torch.zeros(env.num_envs, dtype=torch.float32, device=env.device)

    torso_idx = body_names.index(torso_link_name)

    # Choose the first available candidate link to estimate pitch direction
    other_idx = None
    for cand in head_link_candidates:
        if cand in body_names and cand != torso_link_name:
            other_idx = body_names.index(cand)
            break

    # Need positions for torso and other link
    try:
        torso_pos = robot.data.body_pos_w[:, torso_idx, :]  # (num_envs, 3)
    except Exception:
        return torch.zeros(env.num_envs, dtype=torch.float32, device=env.device)

    if other_idx is None:
        # No secondary link available; cannot estimate pitch reliably
        return torch.zeros(env.num_envs, dtype=torch.float32, device=env.device)

    try:
        other_pos = robot.data.body_pos_w[:, other_idx, :]
    except Exception:
        return torch.zeros(env.num_envs, dtype=torch.float32, device=env.device)

    # Vector from torso to other link
    v = other_pos - torso_pos  # (num_envs, 3)
    horiz = torch.sqrt(v[:, 0] ** 2 + v[:, 1] ** 2) + 1e-8
    pitch = torch.atan2(v[:, 2], horiz)  # radians; positive = up

    # Reward absolute pitch (encourage bending away from upright). Clip to max_pitch
    abs_pitch = torch.clamp(torch.abs(pitch), max=float(max_pitch))

    # Scale to reward range
    reward = float(scale) * (abs_pitch / float(max_pitch))

    return reward.to(device=env.device)
