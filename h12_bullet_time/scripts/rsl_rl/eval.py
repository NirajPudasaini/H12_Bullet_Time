# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Evaluation script that collects statistics and saves to JSON for ablation studies."""

import argparse
import sys

from isaaclab.app import AppLauncher

import cli_args  # isort: skip

parser = argparse.ArgumentParser(description="Evaluate RL agent and save statistics.")
parser.add_argument("--num_envs", type=int, default=64, help="Number of environments.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--agent", type=str, default="rsl_rl_cfg_entry_point", help="Agent config entry point.")
parser.add_argument("--seed", type=int, default=None, help="Seed for environment.")
parser.add_argument("--max_steps", type=int, default=1000, help="Max evaluation steps.")
parser.add_argument("--output_file", type=str, default="eval_results.json", help="Output JSON file path.")
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import os
import json
import torch
import numpy as np

from rsl_rl.runners import OnPolicyRunner

from isaaclab.envs import DirectMARLEnv, DirectRLEnvCfg, ManagerBasedRLEnvCfg, multi_agent_to_single_agent
from isaaclab.utils.assets import retrieve_file_path

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

import h12_bullet_time.tasks  # noqa: F401
from h12_bullet_time.sensors.capacitive_sensor import CapacitiveSensor
from h12_bullet_time.sensors.tof_sensor import TofSensor


def get_min_sensor_distances(env) -> torch.Tensor:
    """Get minimum normalized distance across all sensors for each environment."""
    unwrapped = env.unwrapped
    num_envs = unwrapped.num_envs
    min_dists = torch.ones(num_envs, device=unwrapped.device)  # Start at 1.0 (max normalized)
    
    if hasattr(unwrapped.scene, '_sensors'):
        for sensor_obj in unwrapped.scene._sensors.values():
            if isinstance(sensor_obj, (CapacitiveSensor, TofSensor)):
                if hasattr(sensor_obj.data, "dist_est_normalized"):
                    dists = sensor_obj.data.dist_est_normalized  # Already normalized [0,1]
                    # Flatten per env and take min
                    per_env = dists.reshape(num_envs, -1).min(dim=1).values
                    min_dists = torch.minimum(min_dists, per_env)
    return min_dists


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg, agent_cfg):
    """Evaluate agent and save statistics to JSON."""
    
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device else env_cfg.sim.device

    # Get checkpoint
    log_root_path = os.path.abspath(os.path.join("logs", "rsl_rl", agent_cfg.experiment_name))
    resume_path = args_cli.checkpoint if args_cli.checkpoint else get_checkpoint_path(
        log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint
    )
    if args_cli.checkpoint:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    
    env_cfg.log_dir = os.path.dirname(resume_path)

    # Create environment
    env = gym.make(args_cli.task, cfg=env_cfg)
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    # Load policy
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(resume_path)
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    # Collect statistics
    num_envs = env_cfg.scene.num_envs
    device = env.unwrapped.device
    episode_rewards = torch.zeros(num_envs, device=device)
    episode_lengths = torch.zeros(num_envs, device=device)
    completed_rewards = []
    completed_lengths = []
    
    # Track minimum distance ever seen per environment
    env_min_distances = torch.ones(num_envs, device=device)  # Start at 1.0 (normalized max)
    all_step_min_distances = []  # For computing median

    obs = env.get_observations()
    
    for step in range(args_cli.max_steps):
        with torch.inference_mode():
            actions = policy(obs)
            obs, rewards, dones, infos = env.step(actions)
        
        # Track sensor distances
        step_min_dists = get_min_sensor_distances(env)
        env_min_distances = torch.minimum(env_min_distances, step_min_dists)
        all_step_min_distances.append(step_min_dists.min().item())
        
        episode_rewards += rewards
        episode_lengths += 1
        
        # Record completed episodes
        done_indices = dones.nonzero(as_tuple=False).squeeze(-1)
        for idx in done_indices:
            completed_rewards.append(episode_rewards[idx].item())
            completed_lengths.append(episode_lengths[idx].item())
            episode_rewards[idx] = 0
            episode_lengths[idx] = 0
        
        if (step + 1) % 100 == 0:
            print(f"[EVAL] Step {step+1}/{args_cli.max_steps}, Episodes: {len(completed_rewards)}, Min dist: {step_min_dists.min().item():.4f}")

    # Include any still-running episodes
    for i in range(num_envs):
        if episode_lengths[i] > 0:
            completed_rewards.append(episode_rewards[i].item())
            completed_lengths.append(episode_lengths[i].item())

    env.close()

    # Compute statistics
    rewards_arr = np.array(completed_rewards) if completed_rewards else np.array([0.0])
    lengths_arr = np.array(completed_lengths) if completed_lengths else np.array([0.0])
    env_min_np = env_min_distances.cpu().numpy()
    
    # Distance thresholds for counting
    thresholds = [0.0001, 0.01, 0.1, 0.2, 0.3, 0.4, 0.5]
    dist_below_threshold = {f"dist_min_below_{t}": int((env_min_np < t).sum()) for t in thresholds}
    
    # Success: fraction of envs that never measured < 0.01
    envs_safe = int((env_min_np >= 0.01).sum())
    success = envs_safe / num_envs
    
    stats = {
        "mean_reward": float(np.mean(rewards_arr)),
        "std_reward": float(np.std(rewards_arr)),
        "min_reward": float(np.min(rewards_arr)),
        "max_reward": float(np.max(rewards_arr)),
        "mean_episode_length": float(np.mean(lengths_arr)),
        "std_episode_length": float(np.std(lengths_arr)),
        "total_episodes": len(completed_rewards),
        "total_steps": args_cli.max_steps,
        "num_envs": num_envs,
        "checkpoint": resume_path,
        # Distance metrics
        "median_closest_distance": float(np.median(env_min_np)),
        "mean_closest_distance": float(np.mean(env_min_np)),
        "min_closest_distance": float(np.min(env_min_np)),
        **dist_below_threshold,
        "envs_safe_count": envs_safe,
        "success": success,
    }

    # Save to JSON
    os.makedirs(os.path.dirname(args_cli.output_file) or ".", exist_ok=True)
    with open(args_cli.output_file, "w") as f:
        json.dump(stats, f, indent=2)
    
    print(f"\n{'='*50}")
    print("EVALUATION COMPLETE")
    print(f"{'='*50}")
    for k, v in stats.items():
        print(f"  {k}: {v}")
    print(f"\nResults saved to: {args_cli.output_file}")


if __name__ == "__main__":
    main()
    simulation_app.close()

