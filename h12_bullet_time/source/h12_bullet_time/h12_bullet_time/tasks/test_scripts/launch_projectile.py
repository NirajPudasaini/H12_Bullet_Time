# standalone_projectile.py
# Launch a projectile toward scene center every second

import math
import numpy as np
from pxr import Gf, UsdPhysics, PhysxSchema, Usd

import omni
from omni.isaac.core.utils.stage import get_current_stage, close_stage
from omni.isaac.core.utils.prims import create_prim
from omni.isaac.core.physics import get_physx_scene
from omni.isaac.core.simulation_context import SimulationContext
from omni.isaac.core.prims import RigidPrimView
from omni.isaac.core.utils.physics import add_physx_scene


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def look_at_center(pos):
    """Compute unit direction vector from pos → center (0,0,0)."""
    v = -np.array(pos)
    n = np.linalg.norm(v)
    return v / max(n, 1e-6)


# -----------------------------------------------------------------------------
# MAIN
# -----------------------------------------------------------------------------

def main():
    # Initialize simulation context
    sim = SimulationContext(
        stage_units_in_meters=1.0,
        physics_dt=1/60,
        rendering_dt=1/60
    )

    stage = get_current_stage()
    close_stage(load_empty_stage=True)

    # -------------------------------------------------------------------------
    # Add PhysX scene
    # -------------------------------------------------------------------------
    scene = UsdPhysics.Scene.Define(stage, "/World/physicsScene")
    add_physx_scene("/World/physicsScene")

    # -------------------------------------------------------------------------
    # Ground plane
    # -------------------------------------------------------------------------
    create_prim(
        "/World/GroundPlane",
        "Xform",
        attributes={}
    )
    planeGeom = UsdPhysics.CollisionAPI.Apply(stage.GetPrimAtPath("/World/GroundPlane"))
    PhysxSchema.PhysxRigidBodyAPI.Apply(stage.GetPrimAtPath("/World/GroundPlane"))

    # -------------------------------------------------------------------------
    # Create projectile
    # -------------------------------------------------------------------------
    projectile_path = "/World/Projectile"

    create_prim(
        projectile_path,
        "Sphere",
        attributes={"radius": 0.08}
    )

    UsdPhysics.CollisionAPI.Apply(stage.GetPrimAtPath(projectile_path))
    PhysxSchema.PhysxRigidBodyAPI.Apply(stage.GetPrimAtPath(projectile_path))

    projectile = RigidPrimView(
        prim_paths_expr=[projectile_path],
        name="projectile_view"
    )

    sim.reset()

    print("[INFO] Scene ready. Projectile will launch every second.")

    # -------------------------------------------------------------------------
    # Simulation loop
    # -------------------------------------------------------------------------
    next_fire_time = 0.0
    fire_interval = 1.0       # seconds
    launch_speed = 5.0
    spawn_height = 2.0

    while sim.is_running():
        sim.step()

        t = sim.current_time

        if t >= next_fire_time:
            next_fire_time += fire_interval

            # Random spawn location
            spawn_pos = np.array([
                np.random.uniform(-2, 2),
                np.random.uniform(-2, 2),
                spawn_height
            ])

            direction = look_at_center(spawn_pos)

            projectile.set_world_poses(
                positions=np.array([spawn_pos]),
                orientations=np.array([[1, 0, 0, 0]])
            )

            projectile.set_linear_velocities(
                linear_velocities=np.array([direction * launch_speed])
            )

            print(f"[PROJECTILE] Fired from {spawn_pos} -> center")


if __name__ == "__main__":
    main()
