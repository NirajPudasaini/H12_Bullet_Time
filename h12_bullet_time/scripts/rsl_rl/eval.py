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
parser.add_argument("--max_ep_duration", type=int, default=5, help="Max evaluation episode duration (seconds).")
parser.add_argument("--ep_per_env", type=int, default=1, help="Episodes to run per environment before stopping.")
parser.add_argument("--output_file", type=str, default="eval_results.json", help="Output JSON file path.")
parser.add_argument("--contact_threshold", type=float, default=0.01, help="Contact threshold for success.")
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
    min_dists = torch.inf * torch.ones(num_envs, device=unwrapped.device)  # Start at inf
    
    if hasattr(unwrapped.scene, '_sensors'):
        for sensor_obj in unwrapped.scene._sensors.values():
            if isinstance(sensor_obj, (CapacitiveSensor, TofSensor)):
                if hasattr(sensor_obj.data, "dist_est"):
                    raw_target_distances = sensor_obj.data.raw_target_distances
                    # Flatten per env and take min
                    per_env = raw_target_distances.reshape(num_envs, -1).min(dim=1).values
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
    ep_per_env = args_cli.ep_per_env
    target_episodes = num_envs * ep_per_env
    
    episode_rewards = torch.zeros(num_envs, device=device)
    episode_lengths = torch.zeros(num_envs, device=device)
    episode_min_dists = torch.inf * torch.ones(num_envs, device=device)
    env_ep_count = torch.zeros(num_envs, dtype=torch.int, device=device)  # Episodes completed per env
    env_active = torch.ones(num_envs, dtype=torch.bool, device=device)  # Which envs are still collecting
    
    completed_rewards = []
    completed_lengths = []
    completed_min_dists = []
    completed_stayed_alive = []  # True if episode didn't hit terminal state
    completed_alive_rewards = []  # Per-episode alive_bonus reward
    completed_proximity_penalties = []  # Per-episode distances_penalty reward

    # Track per-episode reward contributions for specific reward terms
    reward_manager = env.unwrapped.reward_manager
    term_names = reward_manager.active_terms
    alive_term_idx = term_names.index("alive_bonus") if "alive_bonus" in term_names else None
    proximity_term_idx = term_names.index("distances_penalty") if "distances_penalty" in term_names else None
    alive_episode_rewards = torch.zeros(num_envs, device=device)
    proximity_episode_rewards = torch.zeros(num_envs, device=device)
    dt = env.unwrapped.step_dt
    
    obs = env.get_observations()
    step = 0
    
    while len(completed_min_dists) < target_episodes:
        with torch.inference_mode():
            actions = policy(obs)
            obs, rewards, dones, infos = env.step(actions)
        
        # Track sensor distances (only for active envs)
        step_min_dists = get_min_sensor_distances(env)
        episode_min_dists = torch.where(env_active, torch.minimum(episode_min_dists, step_min_dists), episode_min_dists)
        
        episode_rewards += rewards * env_active
        episode_lengths += env_active.float()

        # Accumulate per-term rewards for alive_bonus and distances_penalty (only for active envs)
        step_term_rewards = env.unwrapped.reward_manager._step_reward  # shape: (num_envs, n_terms)
        active_mask = env_active.float()
        if alive_term_idx is not None:
            alive_episode_rewards += step_term_rewards[:, alive_term_idx] * dt * active_mask
        if proximity_term_idx is not None:
            proximity_episode_rewards += step_term_rewards[:, proximity_term_idx] * dt * active_mask
        
        # Record completed episodes (only from active envs)
        done_indices = (dones & env_active).nonzero(as_tuple=False).squeeze(-1)
        # Time-outs come from the RslRlVecEnvWrapper extras for infinite-horizon tasks
        time_outs = infos.get("time_outs", None)
        for idx in done_indices:
            ep_len = episode_lengths[idx].item()
            completed_rewards.append(episode_rewards[idx].item())
            completed_lengths.append(ep_len)
            completed_min_dists.append(episode_min_dists[idx].item())
            if alive_term_idx is not None:
                completed_alive_rewards.append(alive_episode_rewards[idx].item())
            if proximity_term_idx is not None:
                completed_proximity_penalties.append(proximity_episode_rewards[idx].item())
            # Stayed alive = episode ended due to time-out (did not hit fall/terminal state)
            if time_outs is not None:
                stayed_alive = bool(time_outs[idx].item())
            else:
                # Fallback: if time-out info is unavailable, mark as not stayed-alive
                stayed_alive = False
            completed_stayed_alive.append(stayed_alive)
            env_ep_count[idx] += 1
            # Reset for next episode
            episode_rewards[idx] = 0
            episode_lengths[idx] = 0
            episode_min_dists[idx] = torch.inf
            alive_episode_rewards[idx] = 0.0
            proximity_episode_rewards[idx] = 0.0
            # Deactivate env if it hit quota
            if env_ep_count[idx] >= ep_per_env:
                env_active[idx] = False
        
        step += 1
        if step % 100 == 0:
            print(f"[EVAL] Step {step}, Episodes: {len(completed_min_dists)}/{target_episodes}, Active envs: {env_active.sum().item()}")

    env.close()

    # Compute statistics
    rewards_arr = np.array(completed_rewards) if completed_rewards else np.array([0.0])
    lengths_arr = np.array(completed_lengths) if completed_lengths else np.array([0.0])
    min_dists_arr = np.array(completed_min_dists) if completed_min_dists else np.array([np.inf])
    alive_rewards_arr = np.array(completed_alive_rewards) if completed_alive_rewards else np.array([0.0])
    proximity_penalties_arr = np.array(completed_proximity_penalties) if completed_proximity_penalties else np.array([0.0])
    num_episodes = len(completed_min_dists)
    
    # Distance thresholds for counting (per episode)
    thresholds = [0.0001, 0.01, 0.1, 0.2, 0.3, 0.4, 0.5]
    dist_below_threshold = {f"dist_min_below_{t}": int((min_dists_arr < t).sum()) for t in thresholds}
    
    # Success: fraction of episodes that never measured < 0.01
    eps_safe = int((min_dists_arr > args_cli.contact_threshold + 0.01).sum())
    # success = eps_safe / max(num_episodes, 1)
    
    # Stayed alive: episodes that didn't hit terminal state (ran full duration)
    stayed_alive_count = sum(completed_stayed_alive)
    stayed_alive_rate = stayed_alive_count / max(num_episodes, 1)
    success = stayed_alive_rate
    median_staying_alive_reward = float(np.median(alive_rewards_arr))
    median_proximity_penalty_reward = float(np.median(proximity_penalties_arr))
    
    stats = {
        "mean_reward": float(np.mean(rewards_arr)),
        "std_reward": float(np.std(rewards_arr)),
        "min_reward": float(np.min(rewards_arr)),
        "max_reward": float(np.max(rewards_arr)),
        "mean_episode_length": float(np.mean(lengths_arr)),
        "std_episode_length": float(np.std(lengths_arr)),
        "total_episodes": num_episodes,
        "total_steps": step,
        "num_envs": num_envs,
        "ep_per_env": ep_per_env,
        "checkpoint": resume_path,
        # Distance metrics (per episode)
        "median_staying_alive_reward": median_staying_alive_reward,
        "median_proximity_penalty_reward": median_proximity_penalty_reward,
        "median_closest_distance": float(np.median(min_dists_arr)),
        "mean_closest_distance": float(np.mean(min_dists_arr)),
        "min_closest_distance": float(np.min(min_dists_arr)),
        **dist_below_threshold,
        "episodes_safe_count": eps_safe,
        "stayed_alive_count": stayed_alive_count,
        "stayed_alive_rate": stayed_alive_rate,
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

