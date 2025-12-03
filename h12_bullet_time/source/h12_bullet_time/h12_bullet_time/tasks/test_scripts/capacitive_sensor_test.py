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
import foxglove

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.sensors import FrameTransformer, FrameTransformerCfg

from h12_bullet_time.utils.urdf_tools import extract_sensor_positions_from_urdf

##
# Pre-defined configs
##
from h12_bullet_time.assets.robots.unitree import H12_CFG_HANDLESS
from h12_bullet_time.sensors.capacitive_sensor_cfg import CapacitiveSensorCfg


# ------------------------------------------------------------------------------
# Scene configuration
# ------------------------------------------------------------------------------

def create_scene_config():
    """Create scene configuration with dynamically generated sensor configs.
    
    Returns:
        tuple: (CapacitiveSensorSceneCfg class, list of sensor names)
    """
    # First extract sensor positions from URDF
    sensor_library = extract_sensor_positions_from_urdf(H12_CFG_HANDLESS.spawn.asset_path, debug=False)
    
    print(f"[INFO]: Extracted {len(sensor_library)} links with sensors from URDF")
    total_sensors = sum(len(positions) for positions in sensor_library.values())
    print(f"[INFO]: Total individual sensors: {total_sensors}")
    
    @configclass
    class CapacitiveSensorSceneCfg(InteractiveSceneCfg):
        """Scene configuration with a moving cube and Capacitive sensor."""

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

        # Moving cube
        moving_cube = AssetBaseCfg(
            prim_path="{ENV_REGEX_NS}/Projectile",
            spawn=sim_utils.CuboidCfg(
                size=(0.3, 0.3, 0.3),
                mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
                rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            ),
            init_state=AssetBaseCfg.InitialStateCfg(pos=(1.0, 0.0, 0.15)),
        )
    
    # Dynamically add sensor configs as class attributes
    sensor_names = []
    for idx, (link_path, sensor_positions) in enumerate(sensor_library.items()):
        # Create a valid sensor name (replace problematic characters)
        sensor_name = f"capacitive_sensor_{link_path.replace('_skin', '').replace('_link', '')}"
        sensor_names.append(sensor_name)
        
        # Create the sensor config
        sensor_cfg = CapacitiveSensorCfg(
            prim_path=f"{{ENV_REGEX_NS}}/Robot/{link_path}",
            target_frames=[
                CapacitiveSensorCfg.FrameCfg(prim_path="{ENV_REGEX_NS}/Projectile"),
            ],
            relative_sensor_pos=sensor_positions,
            debug_vis=True,
            max_range=0.15,  # meters
        )
        
        # Add it as a class attribute
        setattr(CapacitiveSensorSceneCfg, sensor_name, sensor_cfg)
    
    # Return both the config class and the sensor names list
    return CapacitiveSensorSceneCfg, sensor_names


# ------------------------------------------------------------------------------
# Simulation loop
# ------------------------------------------------------------------------------

def run_simulator(sim: sim_utils.SimulationContext, scene: InteractiveScene, sensor_names: list[str]):
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
        cube_pos = torch.tensor([[1.0, 0.0, 0.15]], dtype=torch.float32)  # 0.15 = half of cube height
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

        # Debug visualization - log data from sensors
        if count % 10 == 0:
            print("\n" + "="*120)
            print(f"Simulation Time: {sim_time:.3f}s | Step: {count}")
            print("="*120)
            print(f"{'Link Name':<35} {'Num Sensors':<12} {'Min Dist (m)':<15} {'Max Dist (m)':<15} {'Avg Capacitance':<20}")
            print("-"*120)
            
            for sensor_name in sensor_names:
                sensor_data = scene[sensor_name].data
                distances = sensor_data.target_distances.cpu().numpy()
                capacitances = sensor_data.capacitance_values.cpu().numpy()
                
                # Get statistics
                num_sensors = distances.shape[1] if len(distances.shape) > 1 else distances.shape[0]
                min_dist = distances.min()
                max_dist = distances.max()
                avg_cap = capacitances.mean()
                
                # Clean up sensor name for display
                display_name = sensor_name.replace("capacitive_sensor_", "")
                
                print(f"{display_name:<35} {num_sensors:<12} {min_dist:<15.4f} {max_dist:<15.4f} {avg_cap:<20.6f}")
            
            print("="*120 + "\n")
            
            # foxglove.log(
            #     "/capacitive_sensors",
            #     {
            #         "distances": sensor_data.target_distances.cpu().numpy().tolist(),
            #         "capacitance_values": sensor_data.capacitance_values.cpu().numpy().tolist(),
            #     },
            #     log_time=int(sim_time * 1e6),
            # )


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

    # Create scene config with dynamically generated sensors
    CapacitiveSensorSceneCfg, sensor_names = create_scene_config()
    
    print(f"[INFO]: Monitoring {len(sensor_names)} capacitive sensor groups")
    if not sensor_names:
        print("[ERROR]: No capacitive sensors found in scene config!")
        return
    
    # Build scene
    scene_cfg = CapacitiveSensorSceneCfg(num_envs=args_cli.num_envs, env_spacing=2.0)
    scene = InteractiveScene(scene_cfg)

    ## Debugging
    # mcap_file = foxglove.open_mcap("capacitive_sensor_test.mcap", allow_overwrite=True)
    
    sim.reset()
    print("[INFO]: Setup complete. Running simulator...")
    run_simulator(sim, scene, sensor_names)

    # mcap_file.close()


if __name__ == "__main__":
    main()
    simulation_app.close()