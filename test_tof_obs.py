#!/usr/bin/env python3
"""Quick test to debug TOF observations."""

import torch
from h12_bullet_time.tasks.manager_based.h12_bullet_time.h12_bullet_time_env_cfg_tof import H12BulletTimeEnvCfg
from isaaclab.envs import ManagerBasedRLEnv

# Create environment with small batch size
cfg = H12BulletTimeEnvCfg()
env = ManagerBasedRLEnv(cfg)

print("\n=== Scene Sensors ===")
print(f"env.scene.sensors type: {type(env.scene.sensors)}")
print(f"env.scene.sensors: {env.scene.sensors}")
print(f"Number of sensors: {len(env.scene.sensors)}")

print("\n=== Checking Sensor Access ===")
for sensor_name in env.scene.sensors:
    try:
        sensor = env.scene[sensor_name]
        print(f"\n{sensor_name}:")
        print(f"  sensor type: {type(sensor)}")
        print(f"  sensor.data: {sensor.data}")
        
        if hasattr(sensor.data, "tof_distances"):
            td = sensor.data.tof_distances
            print(f"  tof_distances shape: {td.shape}")
            print(f"  tof_distances dtype: {td.dtype}")
            print(f"  tof_distances device: {td.device}")
        else:
            print(f"  NO tof_distances attribute")
            print(f"  Available attributes: {dir(sensor.data)}")
    except Exception as e:
        print(f"  ERROR accessing {sensor_name}: {e}")

# Try calling the obs function
print("\n=== Testing tof_distances_obs ===")
try:
    from h12_bullet_time.tasks.manager_based.h12_bullet_time.mdp import observations
    obs = observations.tof_distances_obs(env, max_range=4.0, handle_nan="replace_with_max")
    print(f"tof_distances_obs shape: {obs.shape}")
    print(f"tof_distances_obs dtype: {obs.dtype}")
    print(f"tof_distances_obs sample values (env 0): {obs[0, :10]}")
except Exception as e:
    import traceback
    print(f"ERROR: {e}")
    traceback.print_exc()

env.close()
