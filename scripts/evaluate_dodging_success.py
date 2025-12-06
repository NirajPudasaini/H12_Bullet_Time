#!/usr/bin/env python3
"""Evaluate projectile-dodging success rate.

Runs multiple reset batches (parallel envs) and records whether the projectile
hit the robot during each episode. Success = projectile did NOT hit the robot
for the duration of the episode.

Usage:
    python scripts/evaluate_dodging_success.py --num-envs 16 --batches 25

This will evaluate 16*25 = 400 episodes by default.
"""
from __future__ import annotations

import argparse
import math
import time

import torch
try:
    from isaaclab.app import AppLauncher
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.managers import SceneEntityCfg
except ModuleNotFoundError as e:
    import sys
    print('\nERROR: failed to import Isaac/Omniverse Python modules:', e)
    print('Common fixes:')
    print(' - Run this script using the Isaac Sim "kit/python" runtime (recommended).')
    print(' - Or ensure your PYTHONPATH includes the IsaacSim extscache omni dirs and that')
    print('   LD_LIBRARY_PATH includes the omni package root so native libs (libcarb.so) are found.')
    print('\nExample (bash):')
    print("  export PYTHONPATH=$(python - <<'PY'\nimport glob,os\nbase='/home/niraj/miniconda3/envs/env_isaaclab/lib/python3.11/site-packages/isaacsim/extscache'\nprint(':'.join([os.path.join(d,'omni') for d in glob.glob(os.path.join(base,'*')) if os.path.isdir(os.path.join(d,'omni'))]))\nPY\n)")
    print('  export LD_LIBRARY_PATH=/home/niraj/miniconda3/envs/env_isaaclab/lib/python3.11/site-packages/omni:$LD_LIBRARY_PATH')
    print('  python scripts/evaluate_dodging_success.py')
    print('\nIf you installed IsaacSim to a separate location, replace the paths above with your isaacsim path.')
    sys.exit(1)

from h12_bullet_time.tasks.manager_based.h12_bullet_time.h12_bullet_time_env_cfg_curriculum_phase import (
    H12BulletTimeEnvCfg_Curriculum_Phase,
    H12BulletTimeSceneCfg_Curriculum_Phase,
)
from h12_bullet_time.tasks.manager_based.h12_bullet_time import mdp as local_mdp


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--num-envs", type=int, default=16, help="Number of parallel environments per batch")
    p.add_argument("--batches", type=int, default=25, help="Number of reset batches to run")
    p.add_argument("--steps", type=int, default=None, help="Steps per episode (overrides env.episode_length_s)")
    p.add_argument("--threshold", type=float, default=0.12, help="Distance threshold for projectile hit (m)")
    p.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def main():
    args = parse_args()

    # Launch minimal Isaac app
    app_launcher = AppLauncher()
    app = app_launcher.app

    # Build environment config with requested parallelism
    scene_cfg = H12BulletTimeSceneCfg_Curriculum_Phase(num_envs=args.num_envs, env_spacing=3.0)
    env_cfg = H12BulletTimeEnvCfg_Curriculum_Phase(scene=scene_cfg)

    # Create environment
    env = ManagerBasedRLEnv(cfg=env_cfg)

    # Determine steps per episode
    if args.steps is not None:
        steps_per_episode = int(args.steps)
    else:
        # Use episode length and env step dt
        steps_per_episode = int(math.ceil(env.episode_length_s / max(1e-8, float(env.step_dt))))

    num_envs = env.num_envs
    batches = args.batches
    threshold = float(args.threshold)

    total_episodes = num_envs * batches
    successes = 0

    print(f"Evaluating {total_episodes} episodes ({batches} batches x {num_envs} envs)...")
    start_time = time.time()

    # Run batches: each batch is one reset of all parallel envs
    for b in range(batches):
        # Reset: events with mode="reset" (including launch_projectile) will trigger
        obs, _ = env.reset()

        # Track whether each env got hit this episode
        hit_record = torch.zeros(num_envs, dtype=torch.bool, device=env.device)

        for step in range(steps_per_episode):
            # Zero actions (no motion) — we only evaluate passive dodging if any
            try:
                action_dim = env.action_manager.action.shape[-1]
                actions = torch.zeros((num_envs, action_dim), device=env.device, dtype=torch.float32)
            except Exception:
                # Fallback: some env variants may not expose action_manager like this
                actions = torch.zeros((num_envs, 19), device=env.device, dtype=torch.float32)

            obs, reward, terminated, truncated, info = env.step(actions)

            # Compute hit mask using termination helper
            hit_mask = local_mdp.projectile_hit(env, SceneEntityCfg("robot"), threshold=threshold)
            hit_record = hit_record | hit_mask

            # Early exit if all envs have been hit already
            if hit_record.all():
                break

        # Count successes in this batch
        batch_success = (~hit_record).sum().item()
        successes += int(batch_success)

        print(f"Batch {b+1}/{batches}: successes {int(batch_success)}/{num_envs}")

    elapsed = time.time() - start_time
    success_rate = float(successes) / float(total_episodes) if total_episodes > 0 else 0.0

    print("\nEvaluation complete")
    print(f"Total episodes: {total_episodes}")
    print(f"Total successes: {successes}")
    print(f"Success rate: {success_rate*100:.2f}%")
    print(f"Elapsed time: {elapsed:.1f}s")

    # Close the simulation app cleanly
    app.close()


if __name__ == "__main__":
    main()
