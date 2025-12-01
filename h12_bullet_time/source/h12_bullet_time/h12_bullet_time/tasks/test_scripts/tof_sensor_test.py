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

from h12_bullet_time.utils.urdf_tools import extract_sensor_poses_from_urdf

##
# Pre-defined configs
##
from h12_bullet_time.assets.robots.unitree import H12_CFG_HANDLESS
from h12_bullet_time.sensors.tof_sensor_cfg import TofSensorCfg


# ------------------------------------------------------------------------------
# ASCII visualization helpers
# ------------------------------------------------------------------------------

def print_tof_ascii_grid(tof_distances: np.ndarray, pixel_count: int, max_range: float, grids_per_row: int = 10):
    """Print ToF distance data as compact ASCII art grids arranged horizontally.
    
    Args:
        tof_distances: Array of shape (N, S, M, P) where P = pixel_count^2
        pixel_count: Number of pixels per side (e.g., 8 for 8x8 grid)
        max_range: Maximum sensor range for normalization
        grids_per_row: Number of grids to display per row (default 10)
    """
    CHARS = " ·░▒▓█"
    N_CHARS = len(CHARS) - 1
    
    n_envs, n_sensors, n_targets, n_pixels = tof_distances.shape
    content_width = pixel_count * 2  # "X X X X " for each row
    grid_width = content_width + 2   # +2 for │ borders
    
    def build_grid_lines(sensor_idx: int, grid: np.ndarray) -> list[str]:
        """Build ASCII lines for a single sensor grid."""
        lines = []
        # Header with sensor index (padded to grid width)
        header = f"S{sensor_idx}".center(grid_width)
        lines.append(header)
        lines.append("┌" + "─" * content_width + "┐")
        
        for row in range(pixel_count):
            row_chars = []
            for col in range(pixel_count):
                val = grid[row, col]
                if np.isnan(val):
                    row_chars.append(" ")
                else:
                    normalized = 1.0 - np.clip(val / max_range, 0, 1)
                    char_idx = min(int(normalized * N_CHARS), N_CHARS - 1)
                    row_chars.append(CHARS[char_idx + 1])
            lines.append("│" + "".join(f"{c} " for c in row_chars) + "│")
        
        lines.append("└" + "─" * content_width + "┘")
        return lines
    
    for env_idx in range(n_envs):
        # Build all grids for this environment
        all_grids = []
        for sensor_idx in range(n_sensors):
            pixel_data = tof_distances[env_idx, sensor_idx]  # (M, P)
            min_dists = np.nanmin(pixel_data, axis=0)  # (P,)
            grid = min_dists.reshape(pixel_count, pixel_count)
            all_grids.append(build_grid_lines(sensor_idx, grid))
        
        # Print grids in rows of grids_per_row
        for row_start in range(0, n_sensors, grids_per_row):
            row_grids = all_grids[row_start:row_start + grids_per_row]
            n_lines = len(row_grids[0])
            
            for line_idx in range(n_lines):
                print(" ".join(g[line_idx] for g in row_grids))
            print()  # Blank line between rows


# ------------------------------------------------------------------------------
# Scene configuration
# ------------------------------------------------------------------------------

def create_scene_config():
    """Create scene configuration with dynamically generated sensor configs.
    
    Returns:
        tuple: (TofSensorSceneCfg class, list of sensor names)
    """
    # First extract sensor positions from URDF
    sensor_library = extract_sensor_poses_from_urdf(H12_CFG_HANDLESS.spawn.asset_path, debug=False)
    # Moving sphere projectile radius
    projectile_radius = 0.5
    
    print(f"[INFO]: Extracted {len(sensor_library)} links with sensors from URDF")
    total_sensors = sum(len(positions) for positions in sensor_library.values())
    print(f"[INFO]: Total individual sensors: {total_sensors}")
    
    @configclass
    class TofSensorSceneCfg(InteractiveSceneCfg):
        """Scene configuration with a moving sphere and Tof sensor."""

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

        moving_cube = AssetBaseCfg(
            prim_path="{ENV_REGEX_NS}/Projectile",
            spawn=sim_utils.SphereCfg(
                radius=projectile_radius,
                mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
                rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            ),
            init_state=AssetBaseCfg.InitialStateCfg(pos=(1.5, 0.0, 1.5)),
        )
    
    # Dynamically add sensor configs as class attributes
    sensor_names = []
    for idx, (link_path, sensor_poses) in enumerate(sensor_library.items()):
        # Create a valid sensor name (replace problematic characters)
        sensor_name = f"tof_sensor_{link_path.replace('_skin', '').replace('_link', '')}"
        sensor_names.append(sensor_name)
        
        # Extract positions and orientations from Pose3D objects
        sensor_positions = [pose.pos for pose in sensor_poses]
        sensor_orientations = [pose.quat for pose in sensor_poses]
        
        # Create the sensor config
        sensor_cfg = TofSensorCfg(
            prim_path=f"{{ENV_REGEX_NS}}/Robot/{link_path}",
            target_frames=[
                TofSensorCfg.FrameCfg(prim_path="{ENV_REGEX_NS}/Projectile"),
            ],
            relative_sensor_pos=sensor_positions,
            relative_sensor_quat=sensor_orientations,  # Pass orientations
            debug_vis=False,
            max_range=4.0,  # meters
            projectile_radius=projectile_radius,
        )
        
        # Add it as a class attribute
        setattr(TofSensorSceneCfg, sensor_name, sensor_cfg)

    # Debug with a single sensor
    # for idx, (link_path, sensor_poses) in enumerate(sensor_library.items()):
    #     if idx == 0:
    #         # Create a valid sensor name (replace problematic characters)
    #         sensor_name = f"tof_sensor_{link_path.replace('_skin', '').replace('_link', '')}"
    #         sensor_names.append(sensor_name)
            
    #         # Extract positions and orientations from Pose3D objects
    #         sensor_positions = [pose.pos for pose in sensor_poses]
    #         sensor_orientations = [pose.quat for pose in sensor_poses]
            
    #         # Create the sensor config
    #         sensor_cfg = TofSensorCfg(
    #             prim_path=f"{{ENV_REGEX_NS}}/Robot/{link_path}",
    #             target_frames=[
    #                 TofSensorCfg.FrameCfg(prim_path="{ENV_REGEX_NS}/Projectile"),
    #             ],
    #             relative_sensor_pos=sensor_positions,
    #             relative_sensor_quat=sensor_orientations,  # Pass orientations
    #             debug_vis=True,
    #             max_range=4.0,  # meters
    #             projectile_radius=projectile_radius,
    #         )
            
    #         # Add it as a class attribute
    #         setattr(TofSensorSceneCfg, sensor_name, sensor_cfg)

    # Return both the config class and the sensor names list
    return TofSensorSceneCfg, sensor_names


# ------------------------------------------------------------------------------
# Simulation loop
# ------------------------------------------------------------------------------

def run_simulator(sim: sim_utils.SimulationContext, scene: InteractiveScene, sensor_names: list[str]):
    """Run simulation loop."""
    sim_dt = sim.get_physics_dt()
    sim_time = 0.0
    count = 0
    debug=True

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
            print(f"{'Link Name':<35} {'Num Sensors':<12} {'Num Detections':<12} {'Avg ToF Distance (m)':<12} {'TOF Error (m)':<12}")
            print("-"*120)
            
            for sensor_name in sensor_names:
                sensor_data = scene[sensor_name].data
                distances = sensor_data.raw_target_distances.cpu().numpy()
                tof_distances = sensor_data.tof_distances.cpu().numpy()
                
                # Get statistics
                num_sensors = distances.shape[1] if len(distances.shape) > 1 else distances.shape[0]
                num_detections = (~np.isnan(tof_distances)).sum()
                tof_distances_error = np.nanmean(np.abs(tof_distances - distances))
                avg_tof_distance = np.nanmean(tof_distances)
                
                # Clean up sensor name for display
                display_name = sensor_name.replace("tof_sensor_", "")
                
                print(f"{display_name:<35} {num_sensors:<12} {num_detections:<12} {avg_tof_distance:<12.4f} {tof_distances_error:<12.4f}")
                
                # Print ToF grids as ASCII art if debug mode is enabled
                if debug:
                    print_tof_ascii_grid(tof_distances, scene[sensor_name].cfg.pixel_count, 
                                        scene[sensor_name].cfg.max_range)
            
            print("="*120 + "\n")
            
            # foxglove.log(
            #     "/tof_sensors",
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
    TofSensorSceneCfg, sensor_names = create_scene_config()
    
    print(f"[INFO]: Monitoring {len(sensor_names)} tof sensor groups")
    if not sensor_names:
        print("[ERROR]: No tof sensors found in scene config!")
        return
    
    # Build scene
    scene_cfg = TofSensorSceneCfg(num_envs=args_cli.num_envs, env_spacing=2.0)
    scene = InteractiveScene(scene_cfg)

    ## Debugging
    # mcap_file = foxglove.open_mcap("tof_sensor_test.mcap", allow_overwrite=True)
    
    sim.reset()
    print("[INFO]: Setup complete. Running simulator...")
    run_simulator(sim, scene, sensor_names)

    # mcap_file.close()


if __name__ == "__main__":
    main()
    simulation_app.close()