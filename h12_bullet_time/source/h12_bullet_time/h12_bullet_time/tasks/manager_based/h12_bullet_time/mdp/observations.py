"""Observation functions for H12 Bullet Time environment."""
from __future__ import annotations

import os
import xml.etree.ElementTree as ET

import numpy as np
import torch
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.utils.math import matrix_from_quat

from isaaclab.envs.mdp import (
    base_ang_vel,
    joint_pos_rel,
    joint_vel_rel,
    projected_gravity,
    last_action,
)

__all__ = [
    "base_ang_vel", "joint_pos_rel", "joint_vel_rel", "projected_gravity", "last_action",
    "sensor_obs", "sensor_obs_all",
    "tof_distances_obs", "future_projectile",
    "projectile_position_relative", "projectile_velocity", "projectile_distance_obs",
]


def sensor_obs(env: ManagerBasedRLEnv, sensor_prefix: str, signal_type: str) -> torch.Tensor:
    """Unified sensor observation function.

    Args:
        env: Environment instance.
        sensor_prefix: Prefix to match sensor names (e.g., "field_0", "ray_1").
        signal_type: One of "DIST", "MINDIST", "BIN", "MINBIN", "EVENT", "TRUE_POS", "ORACLE", "ORACLE-CP".

    Returns:
        Observation tensor of shape (num_envs, features).
    """
    from h12_bullet_time.sensors import RaySensor

    if signal_type == "TRUE_POS":
        return _true_pos_gated(env, sensor_prefix)
    if signal_type == "ORACLE":
        return _oracle_gated(env, sensor_prefix)
    if signal_type == "ORACLE-CP":
        return _oracle_cp_gated(env, sensor_prefix)

    num_envs = env.num_envs
    all_data = []

    if hasattr(env.scene, '_sensors') and isinstance(env.scene._sensors, dict):
        for name, sensor in env.scene._sensors.items():
            if not name.startswith(sensor_prefix):
                continue

            sd = sensor.data

            if signal_type == "DIST":
                data = sd.dist_est_normalized
            elif signal_type == "MINDIST":
                data = sd.dist_est_normalized
                if isinstance(sensor, RaySensor):
                    data = data.min(dim=-1).values
            elif signal_type == "BIN":
                data = sd.binary_detection
            elif signal_type == "MINBIN":
                data = sd.binary_detection
                if isinstance(sensor, RaySensor):
                    data = data.reshape(num_envs, -1).any(dim=-1).float().unsqueeze(-1)
            elif signal_type == "EVENT":
                data = sd.dist_est_change_normalized
            else:
                continue

            all_data.append(data.reshape(num_envs, -1))

    if not all_data:
        return torch.zeros((num_envs, 0), device=env.device)
    return torch.cat(all_data, dim=1)


def _any_detected(env: ManagerBasedRLEnv, sensor_prefix: str) -> torch.Tensor:
    """(num_envs,) bool: projectile detected by any sensor matching prefix."""
    num_envs = env.num_envs
    any_detected = torch.zeros(num_envs, dtype=torch.bool, device=env.device)

    if hasattr(env.scene, '_sensors') and isinstance(env.scene._sensors, dict):
        for name, sensor in env.scene._sensors.items():
            if not name.startswith(sensor_prefix):
                continue
            detection = sensor.data.binary_detection
            any_detected |= (detection > 0.5).reshape(num_envs, -1).any(dim=1)
    return any_detected


def _true_pos_gated(env: ManagerBasedRLEnv, sensor_prefix: str) -> torch.Tensor:
    """True position of projectile, gated by detection from sensors matching prefix."""
    rel_pos = env.scene["Projectile"].data.root_pos_w - env.scene["robot"].data.root_pos_w
    return rel_pos * _any_detected(env, sensor_prefix).unsqueeze(-1).float()


def _oracle_gated(env: ManagerBasedRLEnv, sensor_prefix: str) -> torch.Tensor:
    """TRUE_POS concatenated with the launch impact point, both gated by detection.

    Layout per env: [ball_xyz (3), impact_xyz (3)], relative to the robot root.
    """
    robot_pos = env.scene["robot"].data.root_pos_w
    ball_rel = env.scene["Projectile"].data.root_pos_w - robot_pos
    target_w = getattr(env, "_projectile_target_w", None)
    if target_w is None:
        impact_rel = torch.zeros_like(ball_rel)
    else:
        impact_rel = target_w - robot_pos
    gate = _any_detected(env, sensor_prefix).unsqueeze(-1).float()
    return torch.cat([ball_rel, impact_rel], dim=-1) * gate


_COLLISION_PARTS: dict = {}


def _robot_collision_parts(env: ManagerBasedRLEnv, num_points: int = 1024):
    """URDF collision geometry sampled as surface points in each body frame.

    Returns (body_idx (L,), points (L, P, 3), bound_center (L, 3), bound_radius (L,)).
    """
    if "parts" in _COLLISION_PARTS:
        return _COLLISION_PARTS["parts"]
    import trimesh

    robot = env.scene["robot"]
    urdf = robot.cfg.spawn.asset_path
    bodies = list(robot.body_names)
    body_idx, points = [], []
    for link in ET.parse(urdf).getroot().findall("link"):
        if link.get("name") not in bodies:
            continue
        for col in link.findall("collision"):
            geom = col.find("geometry")[0]
            if geom.tag == "mesh":
                mesh = trimesh.load(os.path.join(os.path.dirname(urdf), geom.get("filename")), force="mesh")
                if geom.get("scale"):
                    mesh.apply_scale([float(v) for v in geom.get("scale").split()])
            elif geom.tag == "cylinder":
                mesh = trimesh.creation.cylinder(radius=float(geom.get("radius")), height=float(geom.get("length")))
            elif geom.tag == "box":
                mesh = trimesh.creation.box(extents=[float(v) for v in geom.get("size").split()])
            elif geom.tag == "sphere":
                mesh = trimesh.creation.icosphere(radius=float(geom.get("radius")))
            else:
                continue
            origin = col.find("origin")
            if origin is not None:
                tf = trimesh.transformations.euler_matrix(*[float(v) for v in origin.get("rpy", "0 0 0").split()], axes="sxyz")
                tf[:3, 3] = [float(v) for v in origin.get("xyz", "0 0 0").split()]
                mesh.apply_transform(tf)
            body_idx.append(bodies.index(link.get("name")))
            points.append(trimesh.sample.sample_surface(mesh, num_points, seed=0)[0])
    pts = torch.tensor(np.stack(points), dtype=torch.float32, device=env.device)
    center = (pts.amin(1) + pts.amax(1)) / 2
    radius = (pts - center[:, None]).norm(dim=-1).amax(1)
    _COLLISION_PARTS["parts"] = (torch.tensor(body_idx, device=env.device), pts, center, radius)
    return _COLLISION_PARTS["parts"]


def _robot_occupies(env: ManagerBasedRLEnv, centers_w: torch.Tensor, radius: float, chunk: int = 8192) -> torch.Tensor:
    """Robot collision geometry intersects the sphere (centers_w, radius).

    centers_w (N, 3) -> (N,) bool, or (N, K, 3) -> (N, K) bool.
    """
    body_idx, points, bound_center, bound_radius = _robot_collision_parts(env)
    data = env.scene["robot"].data
    centers = centers_w if centers_w.dim() == 3 else centers_w[:, None]
    rot = matrix_from_quat(data.body_quat_w[:, body_idx])
    local = torch.einsum("nlji,nklj->nkli", rot, centers[:, :, None] - data.body_pos_w[:, None, body_idx])
    env_ids, k_ids, part_ids = ((local - bound_center).norm(dim=-1) < bound_radius + radius).nonzero(as_tuple=True)
    occupied = torch.zeros(centers.shape[:2], dtype=torch.bool, device=env.device)
    for s in range(0, len(env_ids), chunk):
        e, k, p = env_ids[s : s + chunk], k_ids[s : s + chunk], part_ids[s : s + chunk]
        # Surface-only test: valid because every H1-2 link is thinner than the ball diameter.
        hit = (points[p] - local[e, k, p, None]).norm(dim=-1).amin(1) < radius
        occupied[e[hit], k[hit]] = True
    return occupied if centers_w.dim() == 3 else occupied[:, 0]


def _render_kinematic_update(sim) -> None:
    """Redraw after a kinematic write without recomputing articulation poses.

    ``sim.render`` runs ``update_articulations_kinematic``. Physics already rendered the
    solved pose, and that call is a no-op then. After ``write_root_pose_to_sim`` the same
    call rebuilds every link from the joint coordinates and drops the solver's contact
    correction, so the recorded frame shows the feet in the ground.
    """
    view = sim.physics_sim_view
    if view is None:
        sim.render()
        return
    original = view.update_articulations_kinematic
    view.update_articulations_kinematic = lambda *args, **kwargs: None
    try:
        sim.render()
    finally:
        view.update_articulations_kinematic = original


def future_projectile(env: ManagerBasedRLEnv, asset_name: str = "Future_Projectile") -> torch.Tensor:
    """Place Future_Projectile one frame before the projectile would occupy the robot, then refresh sensor futures.

    The projectile is propagated ballistically, matching PhysX's semi-implicit Euler (the extra
    0.5 * g * t * physics_dt term), with z clamped at the ground, in frames of CONTEXT_STRIDE control
    steps for up to R = ORACLE_ROLLOUT_FRAMES frames, with the robot held at its current pose. Occupancy is
    swept at every control step so the ball cannot tunnel through thin links between frames. Without contact
    the ball is placed at frame R. Results are stored on ``env._future_projectile_{pos_w,contact,frame}`` and
    pushed to every sensor exposing ``update_future``. Returns an empty (num_envs, 0) observation.
    """
    ball = env.scene["Projectile"]
    pos, vel = ball.data.root_pos_w, ball.data.root_lin_vel_w
    radius = float(ball.cfg.spawn.radius)
    g = torch.tensor(env.cfg.sim.gravity, dtype=pos.dtype, device=env.device)
    stride = int(os.environ.get("CONTEXT_STRIDE", 6))
    frames = int(os.environ.get("ORACLE_ROLLOUT_FRAMES", 9))
    t = torch.arange(frames * stride + 1, dtype=pos.dtype, device=env.device)[:, None] * env.step_dt
    path = pos[:, None] + vel[:, None] * t + 0.5 * g * t * (t + env.physics_dt)
    path[..., 2] = path[..., 2].clamp(min=radius)
    hit = _robot_occupies(env, path[:, 1:], radius)
    contact = hit.any(dim=1)
    frame = torch.where(contact, hit.int().argmax(dim=1) // stride, frames)
    future = path[torch.arange(env.num_envs, device=env.device), frame * stride]
    env._future_projectile_pos_w, env._future_projectile_contact, env._future_projectile_frame = future, contact, frame
    if asset_name in env.scene.rigid_objects:
        quat = torch.zeros(env.num_envs, 4, device=env.device)
        quat[:, 0] = 1.0
        env.scene[asset_name].write_root_pose_to_sim(torch.cat([future, quat], dim=-1))
        # The physics render happens before observations, and the video frame is captured from
        # the last render. Draw again so the frame includes this pose, without articulation FK.
        if env.render_mode == "rgb_array":
            _render_kinematic_update(env.sim)
    for sensor in env.scene.sensors.values():
        if hasattr(sensor, "update_future"):
            sensor.update_future(future)
    return torch.zeros((env.num_envs, 0), device=env.device)


def _show_oracle_cp_ball(env: ManagerBasedRLEnv, pos_w: torch.Tensor) -> None:
    """Put Future_Projectile on the ORACLE-CP point. SIM-* keeps future_projectile's placement."""
    if "Future_Projectile" not in getattr(env.scene, "rigid_objects", {}):
        return
    terms = getattr(getattr(env, "observation_manager", None), "active_terms", None) or {}
    if any("future_projectile" in names for names in terms.values()):
        return
    if getattr(env, "_oracle_cp_vis_step", None) == env.common_step_counter:
        return
    env._oracle_cp_vis_step = env.common_step_counter
    quat = torch.zeros(env.num_envs, 4, device=env.device)
    quat[:, 0] = 1.0
    env.scene["Future_Projectile"].write_root_pose_to_sim(torch.cat([pos_w, quat], dim=-1))
    if env.render_mode == "rgb_array":
        _render_kinematic_update(env.sim)


def _oracle_cp_gated(env: ManagerBasedRLEnv, sensor_prefix: str) -> torch.Tensor:
    """Ground-truth counterpart of the PWM observation [f_tp(x_t), f_tp(x_k), c_k], gated by detection.

    Layout per env: [ball_xyz (3), future_xyz (3), contact (1)], relative to the robot root.
    The horizon is R = ORACLE_ROLLOUT_FRAMES frames of CONTEXT_STRIDE control steps. If the ball's
    ballistic path reaches its launch target within R while the robot occupies a ball-sized sphere
    there, contact = 1 and future_xyz is the target; otherwise future_xyz is the ball center at R.
    """
    robot_pos = env.scene["robot"].data.root_pos_w
    ball = env.scene["Projectile"]
    pos, vel = ball.data.root_pos_w, ball.data.root_lin_vel_w
    radius = float(ball.cfg.spawn.radius)
    g = torch.tensor(env.cfg.sim.gravity, dtype=pos.dtype, device=env.device)
    horizon = int(os.environ.get("ORACLE_ROLLOUT_FRAMES", 9)) * int(os.environ.get("CONTEXT_STRIDE", 6)) * env.step_dt
    future = pos + vel * horizon + 0.5 * g * horizon**2
    future[:, 2] = future[:, 2].clamp(min=radius)
    contact = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    target = getattr(env, "_projectile_target_w", None)
    if target is not None:
        v_xy = vel[:, :2]
        t_hit = ((target[:, :2] - pos[:, :2]) * v_xy).sum(-1) / v_xy.pow(2).sum(-1).clamp(min=1e-6)
        at_hit = pos + vel * t_hit[:, None] + 0.5 * g * t_hit[:, None] ** 2
        on_course = (t_hit >= 0) & (t_hit <= horizon) & ((at_hit - target).norm(dim=-1) < radius)
        contact = on_course & _robot_occupies(env, target, radius)
        future = torch.where(contact[:, None], target, future)
    _show_oracle_cp_ball(env, future)
    gate = _any_detected(env, sensor_prefix).unsqueeze(-1).float()
    return torch.cat([pos - robot_pos, future - robot_pos, contact.unsqueeze(-1).float()], dim=-1) * gate


def sensor_obs_all(env: ManagerBasedRLEnv, specs: list) -> torch.Tensor:
    """Aggregated observations from all configured sensor specs.

    Args:
        env: Environment instance.
        specs: List of (sensor_prefix, signal_type) tuples.
    """
    all_data = []
    for prefix, signal_type in specs:
        data = sensor_obs(env, prefix, signal_type)
        if data.shape[1] > 0:
            all_data.append(data)
    if not all_data:
        return torch.zeros((env.num_envs, 0), device=env.device)
    return torch.cat(all_data, dim=1)


def tof_distances_obs(
    env: ManagerBasedRLEnv, max_range: float = 4.0, handle_nan: str = "replace_with_max"
) -> torch.Tensor:
    values = []
    for name in sorted(getattr(env.scene, "_sensors", {})):
        data = env.scene._sensors[name].data
        distance = getattr(data, "dist_est", None)
        if distance is not None:
            values.append(distance.reshape(env.num_envs, -1))
    if not values:
        return torch.zeros((env.num_envs, 0), device=env.device)
    output = torch.cat(values, dim=1)
    if handle_nan == "replace_with_max":
        output = torch.nan_to_num(output, nan=max_range, posinf=max_range, neginf=0.0)
    elif handle_nan != "keep":
        raise ValueError("handle_nan must be 'replace_with_max' or 'keep'")
    return output


def projectile_position_relative(env: ManagerBasedRLEnv, projectile_name: str = "Projectile") -> torch.Tensor:
    return env.scene[projectile_name].data.root_pos_w - env.scene["robot"].data.root_pos_w


def projectile_velocity(env: ManagerBasedRLEnv, projectile_name: str = "Projectile") -> torch.Tensor:
    return env.scene[projectile_name].data.root_lin_vel_w


def projectile_distance_obs(env: ManagerBasedRLEnv, projectile_name: str = "Projectile") -> torch.Tensor:
    return torch.norm(
        env.scene[projectile_name].data.root_pos_w - env.scene["robot"].data.root_pos_w,
        dim=1, keepdim=True,
    )
