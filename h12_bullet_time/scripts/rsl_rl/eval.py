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
parser.add_argument("--throw_log_file", type=str, default="", help="Path to binned throw heatmap JSON (merged across runs). Empty to disable.")
parser.add_argument("--throw_bins", type=int, default=20, help="Number of bins per axis for throw heatmaps.")
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

import math
from itertools import combinations as _combinations

import h12_bullet_time.tasks  # noqa: F401
from h12_bullet_time.sensors.capacitive_sensor import CapacitiveSensor
from h12_bullet_time.sensors.tof_sensor import TofSensor

import h12_bullet_time.tasks.manager_based.h12_bullet_time.mdp.events as _throw_events
if args_cli.throw_log_file:
    _throw_events._throw_logging_enabled = True

# ── Throw-parameter binning for heatmap accumulation ─────────────────────────
# (name, min, max) — ranges chosen wider than physical limits to avoid clipping
_THROW_BIN_PARAMS = [
    ("azimuth_deg", 0.0, 360.0),
    ("spawn_distance", 0.5, 2.5),
    ("target_z", 0.3, 2.0),
    ("speed", 1.0, 10.0),
    ("elevation_deg", 10.0, 80.0),
]
_THROW_PARAM_PAIRS = list(_combinations(range(len(_THROW_BIN_PARAMS)), 2))


def _throw_vals(tp):
    return {
        "azimuth_deg": math.degrees(tp["azimuth_rad"]),
        "spawn_distance": tp["spawn_distance"],
        "target_z": tp["target_z"],
        "speed": tp["speed"],
        "elevation_deg": math.degrees(tp["elevation_rad"]),
    }


def _bin_idx(value, lo, hi, n):
    return max(0, min(n - 1, int((value - lo) / (hi - lo) * n)))


def _init_throw_bins(n):
    hm = {}
    for i, j in _THROW_PARAM_PAIRS:
        key = f"{_THROW_BIN_PARAMS[i][0]}__{_THROW_BIN_PARAMS[j][0]}"
        hm[key] = {"successes": [[0] * n for _ in range(n)], "totals": [[0] * n for _ in range(n)]}
    mg = {}
    for name, _, _ in _THROW_BIN_PARAMS:
        mg[name] = {"successes": [0] * n, "totals": [0] * n}
    return hm, mg


def _record_throw(tp, survived, hm, mg, n):
    vals = _throw_vals(tp)
    bi = {}
    for name, lo, hi in _THROW_BIN_PARAMS:
        bi[name] = _bin_idx(vals[name], lo, hi, n)
    s = int(survived)
    for name, _, _ in _THROW_BIN_PARAMS:
        b = bi[name]
        mg[name]["totals"][b] += 1
        mg[name]["successes"][b] += s
    for i, j in _THROW_PARAM_PAIRS:
        key = f"{_THROW_BIN_PARAMS[i][0]}__{_THROW_BIN_PARAMS[j][0]}"
        ci, cj = bi[_THROW_BIN_PARAMS[i][0]], bi[_THROW_BIN_PARAMS[j][0]]
        hm[key]["totals"][cj][ci] += 1
        hm[key]["successes"][cj][ci] += s


def _merge_bins(dst_hm, dst_mg, src_hm, src_mg):
    for k in src_hm:
        if k not in dst_hm:
            dst_hm[k] = src_hm[k]
        else:
            for r in range(len(src_hm[k]["totals"])):
                for c in range(len(src_hm[k]["totals"][r])):
                    dst_hm[k]["totals"][r][c] += src_hm[k]["totals"][r][c]
                    dst_hm[k]["successes"][r][c] += src_hm[k]["successes"][r][c]
    for k in src_mg:
        if k not in dst_mg:
            dst_mg[k] = src_mg[k]
        else:
            for b in range(len(src_mg[k]["totals"])):
                dst_mg[k]["totals"][b] += src_mg[k]["totals"][b]
                dst_mg[k]["successes"][b] += src_mg[k]["successes"][b]


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

    n_bins = args_cli.throw_bins
    throw_hm, throw_mg = _init_throw_bins(n_bins) if args_cli.throw_log_file else (None, None)

    while len(completed_min_dists) < target_episodes:
        throw_snapshot = None
        if throw_hm is not None:
            uw = env.unwrapped
            if hasattr(uw, "_current_throws"):
                throw_snapshot = [len(uw._current_throws[i]) for i in range(num_envs)]
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
            if throw_hm is not None and throw_snapshot is not None:
                eid = int(idx.item())
                uw = env.unwrapped
                if hasattr(uw, "_current_throws"):
                    cut = throw_snapshot[eid]
                    for tp in uw._current_throws[eid][:cut]:
                        _record_throw(tp, stayed_alive, throw_hm, throw_mg, n_bins)
                    uw._current_throws[eid] = uw._current_throws[eid][cut:]
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

    # ── Save binned throw heatmap data (merge with existing file) ────────────
    if args_cli.throw_log_file and throw_hm is not None:
        tl_path = args_cli.throw_log_file
        os.makedirs(os.path.dirname(tl_path) or ".", exist_ok=True)
        config_key = (
            f"{os.environ.get('ABLATION_SENSORS', 'unknown')}"
            f"|{os.environ.get('ABLATION_MAX_RANGE', 'unknown')}"
        )
        bin_cfg = {p[0]: {"min": p[1], "max": p[2], "n_bins": n_bins} for p in _THROW_BIN_PARAMS}
        new_entry = {
            "ablation_sensors": os.environ.get("ABLATION_SENSORS", ""),
            "ablation_max_range": os.environ.get("ABLATION_MAX_RANGE", ""),
            "heatmaps": throw_hm,
            "marginals": throw_mg,
        }
        existing = {"bin_config": bin_cfg, "configs": {}}
        if os.path.exists(tl_path):
            try:
                with open(tl_path) as f:
                    existing = json.load(f)
            except (json.JSONDecodeError, IOError):
                pass
        if config_key in existing.get("configs", {}):
            _merge_bins(
                existing["configs"][config_key]["heatmaps"],
                existing["configs"][config_key]["marginals"],
                throw_hm, throw_mg,
            )
        else:
            existing.setdefault("configs", {})[config_key] = new_entry
        existing["bin_config"] = bin_cfg
        with open(tl_path, "w") as f:
            json.dump(existing, f)
        total_t = sum(sum(r) for r in next(iter(throw_hm.values()))["totals"])
        print(f"[EVAL] Saved {total_t} binned throws to {tl_path}")


if __name__ == "__main__":
    main()
    simulation_app.close()

