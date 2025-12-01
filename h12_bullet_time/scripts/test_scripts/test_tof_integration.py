#!/usr/bin/env python3
"""Test script to verify TOF sensor integration in training environment."""

import sys
from pathlib import Path

# Add parent directories to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / "source"))

import torch
from isaaclab.app import AppLauncher
from isaaclab.envs import ManagerBasedRLEnv
import gymnasium as gym

# Create app launcher
app_launcher = AppLauncher(headless=False)
app = app_launcher.app

# Now import the environment config
from h12_bullet_time.tasks.manager_based.h12_bullet_time.h12_bullet_time_env_cfg_tof import (
    H12BulletTimeEnvCfg_TOF,
)

def main():
    """Test TOF sensor readings."""
    
    print("[INFO] Creating environment with TOF sensors...")
    
    # Create environment
    env_cfg = H12BulletTimeEnvCfg_TOF()
    env_cfg.scene.num_envs = 1  # Start with 1 environment for testing
    
    # Create the environment
    try:
        env = gym.make("Template-H12-Bullet-Time-TOF", cfg=env_cfg)
        print("[SUCCESS] Environment created successfully!")
    except Exception as e:
        print(f"[ERROR] Failed to create environment: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Reset environment
    print("\n[INFO] Resetting environment...")
    try:
        obs, info = env.reset()
        print("[SUCCESS] Environment reset successfully!")
    except Exception as e:
        print(f"[ERROR] Failed to reset environment: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Check if sensors exist
    print("\n[INFO] Checking for sensors in scene...")
    if hasattr(env.unwrapped.env_cfg.scene, "sensors"):
        print(f"[INFO] Scene has sensors attribute")
    else:
        print(f"[WARNING] Scene does not have sensors attribute")
    
    # Get scene sensors
    scene = env.unwrapped.scene
    print(f"\n[INFO] Scene object: {scene}")
    
    # Check for sensor data
    if hasattr(scene, "sensors"):
        sensors = scene.sensors
        print(f"[INFO] Number of sensors in scene: {len(sensors)}")
        
        if len(sensors) > 0:
            print("\n[INFO] Sensor names and data:")
            for sensor_name in sensors:
                sensor = scene[sensor_name]
                print(f"\n  Sensor: {sensor_name}")
                if hasattr(sensor, "data"):
                    sensor_data = sensor.data
                    print(f"    - Has data: True")
                    
                    # Check for TOF distances
                    if hasattr(sensor_data, "tof_distances"):
                        tof_dist = sensor_data.tof_distances
                        print(f"    - TOF distances shape: {tof_dist.shape}")
                        print(f"    - TOF distances (first 5): {tof_dist.flatten()[:5]}")
                    else:
                        print(f"    - No tof_distances attribute")
                    
                    # Check for raw distances
                    if hasattr(sensor_data, "raw_target_distances"):
                        raw_dist = sensor_data.raw_target_distances
                        print(f"    - Raw distances shape: {raw_dist.shape}")
                        print(f"    - Raw distances (first 5): {raw_dist.flatten()[:5]}")
        else:
            print("[WARNING] No sensors found in scene despite having sensors attribute")
    else:
        print("[WARNING] Scene does not have sensors attribute - TOF sensors may not be properly registered")
    
    # Run a few steps to get sensor data
    print("\n[INFO] Running simulation steps to collect sensor data...")
    for step in range(5):
        action = env.action_space.sample()
        try:
            obs, reward, terminated, truncated, info = env.step(action)
            print(f"\n  Step {step + 1}:")
            print(f"    - Observation shape: {obs.shape}")
            print(f"    - Reward: {reward}")
            
            # Try to access TOF data
            if hasattr(scene, "sensors") and len(scene.sensors) > 0:
                for sensor_name in scene.sensors:
                    sensor = scene[sensor_name]
                    if hasattr(sensor, "data") and hasattr(sensor.data, "tof_distances"):
                        tof_dist = sensor.data.tof_distances
                        if tof_dist.numel() > 0:
                            valid_count = (~torch.isnan(tof_dist)).sum().item()
                            nan_count = torch.isnan(tof_dist).sum().item()
                            print(f"    - {sensor_name}: {valid_count} valid, {nan_count} NaN readings")
                            if valid_count > 0:
                                print(f"      Valid readings: {tof_dist[~torch.isnan(tof_dist)][:3]}")
        
        except Exception as e:
            print(f"[ERROR] Step {step + 1} failed: {e}")
            import traceback
            traceback.print_exc()
            break
    
    print("\n[INFO] Test completed!")
    env.close()

if __name__ == "__main__":
    main()
    app.close()
