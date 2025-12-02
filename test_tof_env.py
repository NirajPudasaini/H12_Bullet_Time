#!/usr/bin/env python3
"""Quick test of TOF environment integration."""

import torch
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.utils.dict import class_to_dict

# Test 1: Import and config loading
print("[TEST 1] Loading environment configuration...")
try:
    from h12_bullet_time.tasks.manager_based.h12_bullet_time.h12_bullet_time_env_cfg_tof import (
        H12BulletTimeEnvCfg_TOF,
    )
    print("  ✓ Configuration imported successfully")
except Exception as e:
    print(f"  ✗ Failed to import config: {e}")
    exit(1)

# Test 2: Create environment
print("\n[TEST 2] Creating environment...")
try:
    env_cfg = H12BulletTimeEnvCfg_TOF()
    env = ManagerBasedRLEnv(cfg=env_cfg)
    print(f"  ✓ Environment created with {env.num_envs} parallel environments")
except Exception as e:
    print(f"  ✗ Failed to create environment: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# Test 3: Check sensors
print("\n[TEST 3] Checking sensors in scene...")
try:
    if hasattr(env.scene, "sensors"):
        print(f"  ✓ Scene has {len(env.scene.sensors)} sensors")
        for i, sensor in enumerate(env.scene.sensors):
            print(f"    Sensor {i}: {sensor.__class__.__name__}")
    else:
        print("  ✗ Scene has no sensors attribute")
except Exception as e:
    print(f"  ✗ Failed to access sensors: {e}")

# Test 4: Step environment and check sensor data
print("\n[TEST 4] Stepping environment and accessing sensor data...")
try:
    obs, _ = env.reset()
    print(f"  ✓ Reset successful, obs shape: {obs.shape}")
    
    # Step a few times
    for step in range(5):
        actions = torch.zeros((env.num_envs, env.action_manager.action_dim), device=env.device)
        obs, rewards, dones, truncated, info = env.step(actions)
        
        if step == 0:
            print(f"  ✓ Step {step}: obs shape {obs.shape}")
        
        # Try to access sensor data
        if hasattr(env.scene, "sensors") and len(env.scene.sensors) > 0:
            sensor = env.scene.sensors[0]
            try:
                # Accessing .data property triggers automatic updates
                sensor_data = sensor.data
                if hasattr(sensor_data, "tof_distances"):
                    tof_data = sensor_data.tof_distances
                    print(f"  ✓ Step {step}: TOF data shape {tof_data.shape}, sample value {tof_data[0, 0, 0]:.4f}")
                elif hasattr(sensor_data, "distances"):
                    dist_data = sensor_data.distances
                    print(f"  ✓ Step {step}: distances shape {dist_data.shape}, sample value {dist_data[0, 0, 0]:.4f}")
                else:
                    print(f"  ✗ Step {step}: sensor data has no tof_distances or distances attr")
                    print(f"    Available attrs: {[a for a in dir(sensor_data) if not a.startswith('_')]}")
            except Exception as e:
                print(f"  ✗ Step {step}: Failed to access sensor data: {e}")
                import traceback
                traceback.print_exc()

except Exception as e:
    print(f"  ✗ Failed during stepping: {e}")
    import traceback
    traceback.print_exc()

print("\n[DONE] Test complete")
env.close()
