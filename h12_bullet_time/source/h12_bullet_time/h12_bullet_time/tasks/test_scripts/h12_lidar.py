# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import argparse
import numpy as np

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Example on using the raycaster sensor.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to spawn.")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sensors.ray_caster import RayCasterCfg, patterns
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

##
# Pre-defined configs
##
from h12_bullet_time.assets.robots.unitree import H12_CFG_HANDLESS


# ------------------------------------------------------------------------------
# Scene configuration
# ------------------------------------------------------------------------------

@configclass
class RaycasterSensorSceneCfg(InteractiveSceneCfg):
    """Scene configuration with a moving cube and RayCaster sensor."""

    # Ground plane with collision enabled
    ground = AssetBaseCfg(
        prim_path="/World/Ground",
        spawn=sim_utils.GroundPlaneCfg(),
    )

    # Lighting
    dome_light = AssetBaseCfg(
        prim_path="/World/Light",
        spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75)),
    )

    # Robot
    robot = H12_CFG_HANDLESS.replace(prim_path="{ENV_REGEX_NS}/Robot")

    # RayCaster (LiDAR-like)
    ray_caster = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/torso_link/lidar_link",
        update_period= 1.0,  # update every frame
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 0.5)),
        mesh_prim_paths=["/World"],  # collide with everything in world
        ray_alignment="yaw",
        pattern_cfg=patterns.LidarPatternCfg(
            channels=64,
            vertical_fov_range=[-10, 10],
            horizontal_fov_range=[-90, 90],
            horizontal_res=1.0,
        ),
        debug_vis=not args_cli.headless,
    )

    # Moving cube
    moving_cube = AssetBaseCfg(
        prim_path="/World/MovingCube",
        spawn=sim_utils.CuboidCfg(
            size=(0.3, 0.3, 0.3),
            mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
        ),
    )


# ------------------------------------------------------------------------------
# Simulation loop
# ------------------------------------------------------------------------------

def run_simulator(sim: sim_utils.SimulationContext, scene: InteractiveScene):
    """Run simulation loop."""
    sim_dt = sim.get_physics_dt()
    sim_time = 0.0
    count = 0

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

        # Move cube in sine wave motion
        cube_prim = scene["moving_cube"]
        cube_pos = torch.tensor([[np.sin(sim_time) * 1.0, 0.0, 0.15]], dtype=torch.float32)  # 0.15 = half of cube height
        # Use identity quaternion for orientation
        cube_quat = torch.tensor([[1.0, 0.0, 0.0, 0.0]], dtype=torch.float32)  # [w, x, y, z]
        cube_prim.set_world_poses(positions=cube_pos, orientations=cube_quat)

        # Apply default robot command
        targets = scene["robot"].data.default_joint_pos
        scene["robot"].set_joint_position_target(targets)
        scene.write_data_to_sim()

        # Step simulation
        sim.step()
        sim_time += sim_dt
        count += 1
        scene.update(sim_dt)

        # Read and print LiDAR data
        hits = scene["ray_caster"].data.ray_hits_w
        # Calculate distances from origin (approximate)
        distances = torch.linalg.norm(hits, dim=-1)
        avg_dist = distances.mean().item()

        print(f"Time: {sim_time:.2f}s | Avg LiDAR Distance: {avg_dist:.3f} m")

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
    scene_cfg = RaycasterSensorSceneCfg(num_envs=args_cli.num_envs, env_spacing=2.0)
    scene = InteractiveScene(scene_cfg)

    sim.reset()
    print("[INFO]: Setup complete. Running simulator...")
    run_simulator(sim, scene)


if __name__ == "__main__":
    main()
    simulation_app.close()