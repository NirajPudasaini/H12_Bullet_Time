
from __future__ import annotations

import math
import sys
import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor

from h12_bullet_time.sensors.capacitive_sensor import CapacitiveSensor
from h12_bullet_time.sensors.tof_sensor import TofSensor

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def bad_orientation(
    env: ManagerBasedRLEnv, angle_threshold_deg: float, asset_cfg: SceneEntityCfg
) -> torch.Tensor:
    """Terminate if the torso's up vector is within angle_threshold_deg of negative world z.
    
    This detects when the robot has fallen over or flipped upside down.
    
    Args:
        env: The RL environment.
        angle_threshold_deg: Maximum angle (in degrees) from negative world z before termination.
                            e.g., 60 means terminate if torso up-vector points within 60° of downward.
        asset_cfg: Configuration for the robot asset.
    
    Returns:
        Boolean tensor of shape (num_envs,) indicating which environments should terminate.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    
    # Get quaternion orientation of root body: (num_envs, 4) in (w, x, y, z) format
    quat = asset.data.root_quat_w
    
    # Rotate local up vector [0, 0, 1] to world frame using quaternion rotation formula
    # v' = v + 2w(xyz × v) + 2(xyz × (xyz × v))
    w = quat[:, 0:1]
    xyz = quat[:, 1:4]
    
    # Local up vector: (num_envs, 3)
    up_local = torch.zeros(quat.shape[0], 3, device=env.device)
    up_local[:, 2] = 1.0
    
    # Quaternion rotation: v' = v + 2w(xyz × v) + 2(xyz × (xyz × v))
    t = 2.0 * torch.linalg.cross(xyz, up_local)
    up_world = up_local + w * t + torch.linalg.cross(xyz, t)
    
    # Compute cos of angle between up_world and negative world z [0, 0, -1]
    # dot(up_world, [0, 0, -1]) = -up_world_z
    cos_angle = -up_world[:, 2]
    
    # Convert threshold to cosine (cos is decreasing, so smaller angle = larger cosine)
    cos_threshold = math.cos(math.radians(angle_threshold_deg))
    
    # Terminate if angle < threshold (i.e., cos_angle > cos_threshold)
    is_terminated = cos_angle > cos_threshold
    
    return is_terminated
    
def base_height_below_threshold(
    env: ManagerBasedRLEnv, threshold: float, asset_cfg: SceneEntityCfg
) -> torch.Tensor:
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
    threshold: float = 0.1,
) -> torch.Tensor:

    # Get robot and projectiles
    robot: Articulation = env.scene[asset_cfg.name]
    
    # Get all body positions of the robot: shape (num_envs, num_bodies, 3)
    robot_body_positions = robot.data.body_pos_w  # shape: (num_envs, num_bodies, 3)
    
    # Find projectiles
    scene_names = list(env.scene.keys())
    candidates = [] if projectile_names is None else list(projectile_names)
    if projectile_names is None:
        for n in scene_names:
            if "projectile" in n.lower() or "obstacle" in n.lower():
                candidates.append(n)

    if len(candidates) == 0:
        return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

    hit = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    
    # Check distance from projectile to each robot body
    for name in candidates:

        obj = env.scene[name]
        # Get projectile position
        try:
            proj_pos = obj.data.root_pos_w  # shape: (num_envs, 3)
        except AttributeError:
            proj_pos = obj.data.body_pos_w[:, 0, :]  # shape: (num_envs, 3)
        
        # Compute distance from projectile to each robot body
        # proj_pos: (num_envs, 3) -> (num_envs, 1, 3)
        # robot_body_positions: (num_envs, num_bodies, 3)
        # distance: (num_envs, num_bodies)
        distances = torch.norm(
            robot_body_positions - proj_pos.unsqueeze(1),
            dim=-1
        )
        
        # Find minimum distance to any body for each environment
        min_dist_per_env = distances.min(dim=1)[0]  # shape: (num_envs,)
        
        # Update hit mask
        hit = hit | (min_dist_per_env < float(threshold))
            
    return hit


def projectile_hit_after_steps(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    projectile_names: list | None = None,
    threshold: float = 0.1,
    start_step: int = 2000,
) -> torch.Tensor:
    """Return hit mask only after `start_step` training iterations.

    Before `start_step`, this returns all-false so termination is disabled.
    After `start_step`, delegates to `projectile_hit` to compute per-env hits.
    """
    # Use env.common_step_counter (training iterations) to gate termination
    if getattr(env, "common_step_counter", 0) < int(start_step):
        return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

    # Delegate to existing projectile_hit implementation
    return projectile_hit(env, asset_cfg, projectile_names=projectile_names, threshold=threshold)

def contact_termination(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    threshold: float = 1.0,
) -> torch.Tensor:
    """Terminate when any robot body makes physical contact with the projectile.
    
    This uses Isaac Lab's ContactSensor which reports actual physics collision forces
    between surfaces, providing accurate surface-to-surface contact detection regardless
    of link origin positions.
    
    Args:
        env: The RL environment.
        sensor_cfg: Configuration for the contact sensor (should be configured with
                   filter_prim_paths_expr pointing to the projectile).
        threshold: Force threshold (N) above which contact is considered to have occurred.
                  Default is 1.0 N.
    
    Returns:
        Boolean tensor of shape (num_envs,) indicating which environments should terminate.
    """
    # Get the contact sensor from the scene
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    
    # force_matrix_w has shape: (num_envs, num_bodies, num_filter_shapes, 3)
    # where num_filter_shapes corresponds to the projectile(s) we're filtering against
    force_matrix = contact_sensor.data.force_matrix_w
    
    # Compute the magnitude of contact forces for each body-filter pair
    # Shape: (num_envs, num_bodies, num_filter_shapes)
    force_magnitudes = torch.norm(force_matrix, dim=-1)
    
    # Check if any body in any environment has contact force above threshold
    # First reduce over filter shapes and bodies to get per-env max contact force
    max_force_per_env = force_magnitudes.view(env.num_envs, -1).max(dim=1)[0]
    
    # Terminate if max contact force exceeds threshold
    is_terminated = max_force_per_env > threshold
    
    return is_terminated


def multi_contact_termination(
    env: ManagerBasedRLEnv,
    sensor_names: list[str],
    threshold: float = 1.0,
) -> torch.Tensor:
    """Terminate when any of multiple robot bodies makes physical contact with the projectile.
    
    This uses multiple ContactSensors (one per body) to detect actual physics collision forces
    between surfaces. Each sensor monitors a single body filtered against the projectile.
    
    Args:
        env: The RL environment.
        sensor_names: List of contact sensor names to check.
        threshold: Force threshold (N) above which contact is considered to have occurred.
                  Default is 1.0 N.
    
    Returns:
        Boolean tensor of shape (num_envs,) indicating which environments should terminate.
    """
    is_terminated = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    
    for sensor_name in sensor_names:
        # Check if sensor exists
        if sensor_name not in env.scene.sensors:
            continue
            
        contact_sensor: ContactSensor = env.scene.sensors[sensor_name]
        
        # force_matrix_w has shape: (num_envs, num_bodies, num_filter_shapes, 3)
        # For single-body sensors with single filter, shape is (num_envs, 1, 1, 3)
        force_matrix = contact_sensor.data.force_matrix_w
        
        # Compute the magnitude of contact forces
        force_magnitudes = torch.norm(force_matrix, dim=-1)
        
        # Get max force per environment (flatten body and filter dims)
        max_force_per_env = force_magnitudes.view(env.num_envs, -1).max(dim=1)[0]
        
        # Update termination mask
        is_terminated = is_terminated | (max_force_per_env > threshold)
    
    return is_terminated


def sensor_based_contact_termination(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    threshold: float = 0.05,
) -> torch.Tensor:
    """Terminate based on capacitive/TOF sensor readings (legacy implementation).
    
    This uses custom capacitive or TOF sensors to detect proximity/contact.
    Note: This may not detect contacts on robot surfaces without sensors.
    
    Args:
        env: The RL environment.
        asset_cfg: Configuration for the asset (unused, kept for API compatibility).
        threshold: Distance threshold below which contact is considered to have occurred.
    
    Returns:
        Boolean tensor of shape (num_envs,) indicating which environments should terminate.
    """
    num_envs = env.num_envs
    is_terminated = torch.zeros(num_envs, dtype=torch.bool, device=env.device)
    
    # Get sensors from env.scene._sensors dict
    if hasattr(env.scene, '_sensors') and isinstance(env.scene._sensors, dict):
        for sensor_name, sensor_obj in env.scene._sensors.items():
            if isinstance(sensor_obj, CapacitiveSensor) or isinstance(sensor_obj, TofSensor):
                sensor_data = sensor_obj.data
                if hasattr(sensor_data, "raw_target_distances"):
                    # Shape: (num_envs, num_sensors, num_targets) or similar
                    raw_target_distances = sensor_data.raw_target_distances
                    # Check if any sensor in each environment detected contact below threshold
                    # Flatten all sensor dimensions and reduce to per-env boolean
                    contact_mask = (raw_target_distances < threshold).view(num_envs, -1).any(dim=1)
                    is_terminated = is_terminated | contact_mask
    return is_terminated