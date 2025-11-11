# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""This script demonstrates how to spawn multiple objects in multiple environments.

.. code-block:: bash

    # Usage
    ./isaaclab.sh -p scripts/demos/multi_asset.py --num_envs 2048

"""

from __future__ import annotations

"""Launch Isaac Sim Simulator first."""


import argparse

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Demo on throwing sphere from random direction.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to spawn.")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import random
import math

from isaacsim.core.utils.stage import get_current_stage
from pxr import Gf, Sdf

import isaaclab.sim as sim_utils
import torch
from isaaclab.assets import (
    AssetBaseCfg,
    RigidObject,
    RigidObjectCfg,
)
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sim import SimulationContext
from isaaclab.utils import Timer, configclass
from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR

##
# Pre-defined Configuration
##


##
# Randomization events.
##


def randomize_shape_color(prim_path_expr: str):
    """Randomize the color of the geometry."""
    # get stage handle
    stage = get_current_stage()
    # resolve prim paths for spawning and cloning
    prim_paths = sim_utils.find_matching_prim_paths(prim_path_expr)
    # manually clone prims if the source prim path is a regex expression
    with Sdf.ChangeBlock():
        for prim_path in prim_paths:
            # spawn single instance
            prim_spec = Sdf.CreatePrimInLayer(stage.GetRootLayer(), prim_path)

            # DO YOUR OWN OTHER KIND OF RANDOMIZATION HERE!
            # Note: Just need to acquire the right attribute about the property you want to set
            # Here is an example on setting color randomly
            color_spec = prim_spec.GetAttributeAtPath(prim_path + "/geometry/material/Shader.inputs:diffuseColor")
            color_spec.default = Gf.Vec3f(random.random(), random.random(), random.random())


##
# Scene Configuration
##


@configclass
class MultiObjectSceneCfg(InteractiveSceneCfg):
    """Configuration for a sphere-throwing demo."""

    # ground plane
    ground = AssetBaseCfg(prim_path="/World/defaultGroundPlane", spawn=sim_utils.GroundPlaneCfg())

    # lights
    dome_light = AssetBaseCfg(
        prim_path="/World/Light", spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
    )

    # spheres to throw at center
    sphere: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Sphere",
        spawn=sim_utils.SphereCfg(
            radius=0.15,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 0.0, 1.0), metallic=0.2),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                solver_position_iteration_count=4, solver_velocity_iteration_count=0
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.5),
            collision_props=sim_utils.CollisionPropertiesCfg(),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 2.0)),
    )


##
# Simulation Loop
##


def run_simulator(sim: SimulationContext, scene: InteractiveScene):
    """Runs the simulation loop."""
    # Extract scene entities
    sphere: RigidObject = scene["sphere"]
    
    # Define simulation stepping
    sim_dt = sim.get_physics_dt()
    count = 0
    num_envs = sphere.data.root_pos_w.shape[0]
    center = torch.tensor([0.0, 0.0, 0.5], device=sim.device)  # Target center position at ground level
    spawn_height = 3.0  # Height from which to throw (3 meters)
    area_size = 5.0  # 5x5 meter area
    
    # Simulation loop
    while simulation_app.is_running():
        # Reset every 250 steps
        if count % 250 == 0:
            # reset counter
            count = 0
            
            # Throw sphere from random position within 5x5 meter area towards center
            root_state = sphere.data.default_root_state.clone()
            
            for env_idx in range(num_envs):
                # Randomly pick position within 5x5 meter area around center
                spawn_x = center[0] - area_size / 2.0 + random.random() * area_size
                spawn_y = center[1] - area_size / 2.0 + random.random() * area_size
                spawn_z = center[2] + spawn_height  # 3 meters above center
                
                # Position
                root_state[env_idx, 0:3] = torch.tensor([spawn_x, spawn_y, spawn_z], device=sim.device)
                
                # Velocity towards center
                spawn_pos = torch.tensor([spawn_x, spawn_y, spawn_z], device=sim.device)
                direction = center - spawn_pos
                direction = direction / (torch.norm(direction) + 1e-6)
                # Apply velocity towards center
                root_state[env_idx, 7:10] = direction * 5.0  # Velocity magnitude towards center
            
            root_state[:, :3] += scene.env_origins
            sphere.write_root_pose_to_sim(root_state[:, :7])
            sphere.write_root_velocity_to_sim(root_state[:, 7:])
            
            # clear internal buffers
            scene.reset()
            print(f"[INFO]: Reset! Throwing sphere from random position (3m height) within 5x5m area towards center!")

        # Write data to sim
        scene.write_data_to_sim()
        # Perform step
        sim.step()
        # Increment counter
        count += 1
        # Update buffers
        scene.update(sim_dt)


def main():
    """Main function."""
    # Load kit helper
    sim_cfg = sim_utils.SimulationCfg(dt=0.005, device=args_cli.device)
    sim = SimulationContext(sim_cfg)
    # Set main camera
    sim.set_camera_view([2.5, 0.0, 4.0], [0.0, 0.0, 2.0])

    # Design scene
    scene_cfg = MultiObjectSceneCfg(num_envs=args_cli.num_envs, env_spacing=2.0, replicate_physics=False)
    with Timer("[INFO] Time to create scene: "):
        scene = InteractiveScene(scene_cfg)

    with Timer("[INFO] Time to randomize scene: "):
        # Randomization for spheres
        randomize_shape_color(scene_cfg.sphere.prim_path)

    # Play the simulator
    sim.reset()
    # Now we are ready!
    print("[INFO]: Setup complete...")
    # Run the simulator
    run_simulator(sim, scene)


if __name__ == "__main__":
    # run the main execution
    main()
    # close sim app
    simulation_app.close()