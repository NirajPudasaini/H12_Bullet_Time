# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import argparse
import math
import xml.etree.ElementTree as ET
from collections.abc import Callable

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Example on using the MultiMeshRayCaster sensor.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to spawn.")
parser.add_argument("-v", "--vis", action="store_true", help="Enable ray-hit visualization markers.")
parser.add_argument("-b", "--ball", action="store_true", help="Detect the projectile sphere.")
parser.add_argument("-s", "--self", action="store_true", dest="self_collision", help="Detect robot body meshes (self-detection).")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import time

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sensors.ray_caster import MultiMeshRayCasterCfg
from isaaclab.sensors.ray_caster.patterns.patterns_cfg import PatternBaseCfg
from isaaclab.utils import configclass

##
# Pre-defined configs
##
from h12_bullet_time.assets.robots.unitree import H12_CFG_HANDLESS
from h12_bullet_time.utils.urdf_tools import extract_sensor_poses_from_urdf


# Red spheres placed at every ray hit point, drawn live in the viewport.
RAY_HIT_MARKER_CFG = VisualizationMarkersCfg(
    prim_path="/Visuals/RayCaster",
    markers={
        "hit": sim_utils.SphereCfg(
            radius=0.02,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0)),
        ),
    },
)


# ------------------------------------------------------------------------------
# Ray pattern: 8x8 square cone (pinhole-style) pointing along the sensor +Z axis
# ------------------------------------------------------------------------------

def square_cone_pattern(cfg: "ConePatternCfg", device: str):
    """Generate an NxN grid of diverging rays forming a square cone about +Z.

    Each axis spans the full ``fov_deg`` field of view (i.e. +/- fov_deg/2), and the
    rays pass through a planar grid at unit distance so the cone is a proper pyramid.
    """
    n = cfg.pixel_count
    half_fov = math.radians(cfg.fov_deg) * 0.5
    angles = torch.linspace(-half_fov, half_fov, n, device=device)
    pitch, yaw = torch.meshgrid(angles, angles, indexing="xy")
    pitch, yaw = pitch.reshape(-1), yaw.reshape(-1)
    directions = torch.stack([torch.tan(yaw), torch.tan(pitch), torch.ones_like(yaw)], dim=-1)
    directions = directions / torch.linalg.norm(directions, dim=-1, keepdim=True)
    return torch.zeros_like(directions), directions


@configclass
class ConePatternCfg(PatternBaseCfg):
    """Configuration for the square-cone ToF-like ray pattern."""

    func: Callable = square_cone_pattern
    fov_deg: float = 45.0
    pixel_count: int = 8


# ------------------------------------------------------------------------------
# URDF parsing helpers
# ------------------------------------------------------------------------------

def extract_rigid_body_links(urdf_path: str) -> list[str]:
    """Return the names of links that survive URDF import as their own rigid-body prims.

    With ``merge_fixed_joints=True`` (the importer default), links attached through fixed
    joints (skins, empty sensor frames, logo, imu, ...) are folded into their parent body.
    The remaining rigid bodies are exactly the links that carry a ``<visual>`` and are not
    the child of a fixed joint. Targeting those prims lets the ray-caster see the whole robot,
    since the merged skin meshes end up underneath them.
    """
    root = ET.parse(urdf_path).getroot()
    fixed_children = {
        j.find("child").get("link")
        for j in root.findall("joint")
        if j.get("type") == "fixed" and j.find("child") is not None
    }
    return [
        link.get("name")
        for link in root.findall("link")
        if link.find("visual") is not None and link.get("name") not in fixed_children
    ]


# ------------------------------------------------------------------------------
# Scene configuration
# ------------------------------------------------------------------------------

def create_scene_config():
    """Create scene configuration with per-sensor ray casters distributed across the robot.

    Returns:
        tuple: (RayCasterSceneCfg class, list of sensor names)
    """
    sensor_library = extract_sensor_poses_from_urdf(H12_CFG_HANDLESS.spawn.asset_path, debug=False)
    projectile_radius = 0.1
    total_sensors = sum(len(poses) for poses in sensor_library.values())
    print(f"[INFO]: Placing {total_sensors} ray-cast sensors across {len(sensor_library)} robot links")

    mesh_targets: list[MultiMeshRayCasterCfg.RaycastTargetCfg] = []
    if args_cli.ball:
        mesh_targets.append(
            MultiMeshRayCasterCfg.RaycastTargetCfg(prim_expr="{ENV_REGEX_NS}/Projectile", track_mesh_transforms=True)
        )
        print("[INFO]: Ball detection enabled")
    if args_cli.self_collision:
        rigid_links = extract_rigid_body_links(H12_CFG_HANDLESS.spawn.asset_path)
        print(f"[INFO]: Self-detection enabled — adding {len(rigid_links)} robot body meshes")
        mesh_targets += [
            MultiMeshRayCasterCfg.RaycastTargetCfg(prim_expr=f"{{ENV_REGEX_NS}}/Robot/{link}", track_mesh_transforms=True)
            for link in rigid_links
        ]

    @configclass
    class RayCasterSceneCfg(InteractiveSceneCfg):
        """Scene configuration with a moving sphere, the H1-2 robot and body-mounted ray casters."""

        ground = AssetBaseCfg(
            prim_path="/World/Ground",
            spawn=sim_utils.GroundPlaneCfg(),
        )

        dome_light = AssetBaseCfg(
            prim_path="/World/Light",
            spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75)),
        )

        robot = H12_CFG_HANDLESS.replace(prim_path="{ENV_REGEX_NS}/Robot")

        moving_cube = AssetBaseCfg(
            prim_path="{ENV_REGEX_NS}/Projectile",
            spawn=sim_utils.SphereCfg(
                radius=projectile_radius,
                mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
                rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            ),
            init_state=AssetBaseCfg.InitialStateCfg(pos=(1.5, 0.0, 1.5)),
        )

    if not mesh_targets:
        print("[WARN]: No detection targets selected. Pass -b (ball) and/or -s (self). No sensors will be created.")
        return RayCasterSceneCfg, []

    # One ray caster per individual sensor pose. Each fires an 8x8, 45 deg square cone
    # along its local +Z (the surface normal), aligned to the body it is mounted on.
    sensor_names = []
    for link_path, sensor_poses in sensor_library.items():
        base_name = link_path.replace("_skin", "").replace("_link", "")
        for idx, pose in enumerate(sensor_poses):
            sensor_name = f"ray_caster_{base_name}_{idx}"
            sensor_names.append(sensor_name)
            sensor_cfg = MultiMeshRayCasterCfg(
                prim_path=f"{{ENV_REGEX_NS}}/Robot/{link_path}",
                offset=MultiMeshRayCasterCfg.OffsetCfg(pos=pose.pos, rot=pose.quat),
                mesh_prim_paths=mesh_targets,
                ray_alignment="base",
                pattern_cfg=ConePatternCfg(fov_deg=45.0, pixel_count=8),
                max_distance=4.0,
                update_period=1.0 / 60.0,  # throttle: don't recompute every physics step
                debug_vis=False,  # drawn via a single shared marker set instead (see run_simulator)
            )
            setattr(RayCasterSceneCfg, sensor_name, sensor_cfg)

    return RayCasterSceneCfg, sensor_names


# ------------------------------------------------------------------------------
# Simulation loop
# ------------------------------------------------------------------------------

def run_simulator(
    sim: sim_utils.SimulationContext,
    scene: InteractiveScene,
    sensor_names: list[str],
    hit_markers: VisualizationMarkers | None,
):
    """Run simulation loop."""
    sim_dt = sim.get_physics_dt()
    sim_time = 0.0
    count = 0
    wall_start = time.monotonic()
    last_wall = wall_start
    last_count = 0

    print("[INFO]: Simulation running... Press Ctrl+C to exit.")

    while simulation_app.is_running():
        # Reset periodically
        if count % 500 == 0:
            count = 0
            root_state = scene["robot"].data.default_root_state.clone()
            root_state[:, :3] += scene.env_origins
            scene["robot"].write_root_pose_to_sim(root_state[:, :7])
            scene["robot"].write_root_velocity_to_sim(root_state[:, 7:])
            joint_pos = scene["robot"].data.default_joint_pos.clone() + torch.rand_like(scene["robot"].data.default_joint_pos) * 0.1
            joint_vel = scene["robot"].data.default_joint_vel.clone()
            scene["robot"].write_joint_state_to_sim(joint_pos, joint_vel)
            scene.reset()
            print("[INFO]: Resetting robot state...")

        # Keep the projectile in a fixed pose in front of the robot
        cube_pos = torch.tensor([[1.0, 0.0, 0.15]], dtype=torch.float32, device=scene.device)
        cube_quat = torch.tensor([[1.0, 0.0, 0.0, 0.0]], dtype=torch.float32, device=scene.device)
        scene["moving_cube"].set_world_poses(positions=cube_pos, orientations=cube_quat)

        # Apply default robot command
        targets = scene["robot"].data.default_joint_pos
        scene["robot"].set_joint_position_target(targets)
        scene.write_data_to_sim()

        # Step simulation
        sim.step()
        sim_time += sim_dt
        count += 1
        scene.update(sim_dt)

        # Aggregate all sensor hits, optionally draw markers, and periodically print stats
        if count % 4 == 0:
            hit_points = []
            valid_chunks = []
            n_rays = 0
            n_hits = 0
            for sensor_name in sensor_names:
                data = scene[sensor_name].data
                hits = data.ray_hits_w  # (N, B, 3)
                finite = torch.isfinite(hits).all(dim=-1)  # (N, B)
                n_rays += hits.shape[0] * hits.shape[1]
                n_hits += int(finite.sum().item())
                finite_hits = hits[finite]  # (M, 3)
                if finite_hits.shape[0] > 0:
                    hit_points.append(finite_hits)
                    origins = data.pos_w.unsqueeze(1).expand(-1, hits.shape[1], -1)[finite]
                    valid_chunks.append(torch.linalg.norm(finite_hits - origins, dim=-1))

            if hit_markers is not None and hit_points:
                hit_markers.visualize(translations=torch.cat(hit_points))

            if count % 20 == 0:
                now = time.monotonic()
                real_elapsed = now - wall_start
                fps = (count - last_count) / max(now - last_wall, 1e-9)
                rtf = sim_time / real_elapsed if real_elapsed > 0 else 0.0
                last_wall = now
                last_count = count

                valid = torch.cat(valid_chunks) if valid_chunks else torch.empty(0)
                print("\n" + "=" * 80)
                print(
                    f"sim {sim_time:.2f}s | real {real_elapsed:.2f}s | "
                    f"RTF {rtf:.3f}x | FPS {fps:.1f}"
                )
                print(f"  sensors        : {len(sensor_names)}")
                print(f"  rays cast      : {n_rays}")
                print(f"  rays with hits : {n_hits} ({100.0 * n_hits / max(n_rays, 1):.1f}%)")
                if valid.numel() > 0:
                    print(f"  hit distance   : min {valid.min():.3f} | mean {valid.mean():.3f} | max {valid.max():.3f} m")
                print("=" * 80)

    print("[INFO]: Simulation finished.")


# ------------------------------------------------------------------------------
# Main entry
# ------------------------------------------------------------------------------

def main():
    """Main function."""
    sim_cfg = sim_utils.SimulationCfg(dt=0.005, device=args_cli.device)
    sim = sim_utils.SimulationContext(sim_cfg)

    # Camera for visualization
    sim.set_camera_view(eye=[3.5, 3.5, 3.5], target=[0.0, 0.0, 0.0])

    # Build scene
    RayCasterSceneCfg, sensor_names = create_scene_config()
    if not sensor_names:
        print("[ERROR]: No sensors found in URDF!")
        return
    scene_cfg = RayCasterSceneCfg(num_envs=args_cli.num_envs, env_spacing=2.0, replicate_physics=False)
    scene = InteractiveScene(scene_cfg)

    # Single shared marker set for all ray hits (only created when vis is enabled)
    hit_markers = VisualizationMarkers(RAY_HIT_MARKER_CFG) if args_cli.vis else None

    sim.reset()
    print("[INFO]: Setup complete. Running simulator...")
    run_simulator(sim, scene, sensor_names, hit_markers)


if __name__ == "__main__":
    main()
    simulation_app.close()
