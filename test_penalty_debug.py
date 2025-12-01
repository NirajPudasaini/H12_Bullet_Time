#!/usr/bin/env python3
"""Debug script to check projectile hit penalty calculation."""

import torch
import isaaclab.sim as sim_utils
from isaaclab.app import AppLauncher
from isaaclab.envs import ManagerBasedRLEnv
from h12_bullet_time.tasks.manager_based.h12_bullet_time import agents
from h12_bullet_time.tasks.manager_based.h12_bullet_time.h12_bullet_time_env_cfg_curriculum_phase import (
    H12BulletTimeEnvCfg_CurriculumPhasePlay,
)

# Create app
app_launcher = AppLauncher()
simulation_app = app_launcher.app

# Create environment
env_cfg = H12BulletTimeEnvCfg_CurriculumPhasePlay()
env = ManagerBasedRLEnv(cfg=env_cfg)

print(f"Num envs: {env.num_envs}")
print(f"Projectile in scene: {'Projectile' in env.scene.keys()}")
print(f"Robot in scene: {'robot' in env.scene.keys()}")

# Run for a few steps
obs, _ = env.reset()
print(f"\nInitial observation shape: {obs.shape}")

for step in range(50):
    # Get projectile and robot positions
    proj = env.scene["Projectile"]
    robot = env.scene["robot"]
    
    proj_pos = proj.data.root_pos_w  # (num_envs, 3)
    robot_pos = robot.data.root_pos_w  # (num_envs, 3)
    robot_bodies = robot.data.body_pos_w  # (num_envs, num_bodies, 3)
    
    # Calculate distance
    distance = torch.norm(proj_pos - robot_pos, dim=-1)
    
    # Calculate body distances
    body_distances = torch.norm(
        robot_bodies - proj_pos.unsqueeze(1),
        dim=-1
    )
    min_body_dist = body_distances.min(dim=1)[0]
    
    # Get penalty
    reward = env.reward_manager.compute(dt=env.step_dt)
    penalty_term = reward[:, env.reward_manager._term_names.index("projectile_penalty")]
    
    if step % 10 == 0:
        print(f"\nStep {step}:")
        print(f"  Proj pos (env 0): {proj_pos[0].tolist()}")
        print(f"  Robot pos (env 0): {robot_pos[0].tolist()}")
        print(f"  Distance (env 0): {distance[0]:.3f}m")
        print(f"  Min body distance (env 0): {min_body_dist[0]:.3f}m")
        print(f"  Penalty signal (env 0): {penalty_term[0]:.4f}")
    
    actions = torch.zeros(env.num_envs, env.action_manager.action.shape[-1], device=env.device)
    obs, reward, terminated, truncated, info = env.step(actions)

simulation_app.close()
print("\nDone!")
