# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Curriculum learning scheduler for H12 Bullet Time task.

The curriculum progresses through three stages:
1. Stage 1 (Standing & Balance): Robot learns to stand upright and maintain balance
2. Stage 2 (Height Control): Robot learns to adjust height and improve agility
3. Stage 3 (Dodging): Robot learns to dodge incoming projectiles
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch


@dataclass
class CurriculumStage:
    """Configuration for a curriculum stage."""
    
    name: str
    description: str
    
    # Reward weights for this stage
    reward_weights: dict[str, float]
    
    # Projectile parameters for this stage
    projectile_frequency: float  # How often to spawn projectiles (0.0 = never, 1.0 = always)
    projectile_speed_range: tuple[float, float]  # Min and max projectile speeds
    projectile_spawn_distance: tuple[float, float]  # Min and max spawn distances
    
    # Success criteria for progression
    success_threshold: float  # Average reward threshold to move to next stage
    success_window: int  # Number of episodes to average over
    
    # Curriculum modifications
    gravity_scale: float = 1.0  # Scale gravity for difficulty
    episode_length_s: float = 10.0  # Episode length in seconds


class H12BulletTimeCurriculum:
    """Curriculum scheduler for H12 Bullet Time task."""
    
    def __init__(self, num_envs: int = 4096, device: str = "cuda"):
        """
        Initialize curriculum scheduler.
        
        Args:
            num_envs: Number of parallel environments
            device: Device to use for computations
        """
        self.num_envs = num_envs
        self.device = device
        self.current_stage = 0
        self.total_steps = 0
        self.stage_steps = 0
        
        # Tracking for stage progression
        self.episode_rewards = torch.zeros(num_envs, device=device)
        self.episode_lengths = torch.ones(num_envs, device=device)
        self.reward_history = []
        self.stage_start_step = 0
        
        # Define curriculum stages
        self._initialize_stages()
    
    def _initialize_stages(self) -> None:
        """Initialize the three curriculum stages."""
        
        # Stage 1: Standing and Balance
        self.stage_1 = CurriculumStage(
            name="Standing & Balance",
            description="Learn to stand upright and maintain balance",
            reward_weights={
                "is_alive": 1.0,
                "fallen": -5.0,
                "upright_torso": 2.0,
                "smooth_joints": -0.01,
                "height_control": 0.0,
                "lateral_stability": 0.0,
                "dodge_reward": 0.0,
                "projectile_collision": 0.0,
                "movement_reward": 0.0,
            },
            projectile_frequency=0.0,  # No projectiles in stage 1
            projectile_speed_range=(0.0, 0.0),
            projectile_spawn_distance=(0.0, 0.0),
            success_threshold=20.0,  # Average reward of 20 per episode
            success_window=100,  # Over 100 episodes
            episode_length_s=10.0,
        )
        
        # Stage 2: Height Control & Agility
        self.stage_2 = CurriculumStage(
            name="Height Control & Agility",
            description="Learn to adjust height and improve agility",
            reward_weights={
                "is_alive": 1.0,
                "fallen": -5.0,
                "upright_torso": 1.0,
                "smooth_joints": -0.01,
                "height_control": 0.5,
                "lateral_stability": 0.5,
                "dodge_reward": 0.0,
                "projectile_collision": 0.0,
                "movement_reward": 0.0,
            },
            projectile_frequency=0.0,  # Still no projectiles
            projectile_speed_range=(0.0, 0.0),
            projectile_spawn_distance=(0.0, 0.0),
            success_threshold=25.0,
            success_window=100,
            episode_length_s=10.0,
        )
        
        # Stage 3: Dodging
        self.stage_3 = CurriculumStage(
            name="Projectile Dodging",
            description="Learn to dodge incoming projectiles",
            reward_weights={
                "is_alive": 1.0,
                "fallen": -5.0,
                "upright_torso": 0.5,
                "smooth_joints": -0.01,
                "height_control": 0.3,
                "lateral_stability": 0.3,
                "dodge_reward": 1.0,
                "projectile_collision": -10.0,
                "movement_reward": 0.1,
            },
            projectile_frequency=0.5,  # Spawn projectiles 50% of the time
            projectile_speed_range=(5.0, 15.0),  # 5-15 m/s
            projectile_spawn_distance=(3.0, 5.0),  # 3-5 meters away
            success_threshold=30.0,
            success_window=100,
            episode_length_s=10.0,
        )
        
        self.stages = [self.stage_1, self.stage_2, self.stage_3]
    
    def get_current_stage(self) -> CurriculumStage:
        """Get the current curriculum stage."""
        return self.stages[self.current_stage]
    
    def get_current_reward_weights(self) -> dict[str, float]:
        """Get reward weights for current stage."""
        return self.get_current_stage().reward_weights
    
    def get_projectile_config(self) -> dict:
        """Get projectile configuration for current stage."""
        stage = self.get_current_stage()
        return {
            "frequency": stage.projectile_frequency,
            "speed_range": stage.projectile_speed_range,
            "spawn_distance": stage.projectile_spawn_distance,
        }
    
    def update(self, episode_rewards: torch.Tensor, done: torch.Tensor) -> bool:
        """
        Update curriculum based on performance.
        
        Args:
            episode_rewards: Rewards for current episode per environment
            done: Done flags per environment
            
        Returns:
            True if curriculum stage changed, False otherwise
        """
        self.total_steps += 1
        self.stage_steps += 1
        
        # Track episode rewards
        self.episode_rewards += episode_rewards
        
        # When episodes finish, record their rewards and reset tracking
        finished = done.nonzero(as_tuple=True)[0]
        if len(finished) > 0:
            avg_episode_reward = self.episode_rewards[finished] / self.episode_lengths[finished]
            self.reward_history.extend(avg_episode_reward.cpu().numpy().tolist())
            
            # Reset tracking for finished episodes
            self.episode_rewards[finished] = 0.0
            self.episode_lengths[finished] = 1.0
        
        # Update episode lengths for all
        self.episode_lengths += 1.0
        
        # Check if we should progress to next stage
        if self._should_progress_stage():
            return self._progress_to_next_stage()
        
        return False
    
    def _should_progress_stage(self) -> bool:
        """Check if we should progress to the next stage."""
        if self.current_stage >= len(self.stages) - 1:
            # Already in final stage
            return False
        
        stage = self.get_current_stage()
        
        # Need enough episodes recorded
        if len(self.reward_history) < stage.success_window:
            return False
        
        # Check average reward in recent history
        recent_rewards = self.reward_history[-stage.success_window:]
        avg_reward = sum(recent_rewards) / len(recent_rewards)
        
        return avg_reward >= stage.success_threshold
    
    def _progress_to_next_stage(self) -> bool:
        """Progress to the next curriculum stage."""
        self.current_stage = min(self.current_stage + 1, len(self.stages) - 1)
        self.stage_steps = 0
        self.stage_start_step = self.total_steps
        
        stage = self.get_current_stage()
        print(f"\n{'='*60}")
        print(f"🎓 CURRICULUM PROGRESSION")
        print(f"{'='*60}")
        print(f"Stage: {self.current_stage + 1}/{len(self.stages)}")
        print(f"Name: {stage.name}")
        print(f"Description: {stage.description}")
        print(f"Total steps: {self.total_steps}")
        print(f"{'='*60}\n")
        
        return True
    
    def get_debug_info(self) -> dict:
        """Get debug information about curriculum state."""
        stage = self.get_current_stage()
        
        recent_rewards = self.reward_history[-100:] if self.reward_history else []
        avg_recent = sum(recent_rewards) / len(recent_rewards) if recent_rewards else 0.0
        
        return {
            "stage": self.current_stage + 1,
            "stage_name": stage.name,
            "total_steps": self.total_steps,
            "stage_steps": self.stage_steps,
            "avg_recent_reward": avg_recent,
            "num_episodes": len(self.reward_history),
            "success_threshold": stage.success_threshold,
            "success_window": stage.success_window,
            "projectile_frequency": stage.projectile_frequency,
        }


# Example usage and testing
if __name__ == "__main__":
    # Create curriculum
    curriculum = H12BulletTimeCurriculum(num_envs=4096)
    
    # Print initial state
    print("Curriculum Scheduler Initialized")
    print(f"Current Stage: {curriculum.get_current_stage().name}")
    print(f"Reward Weights: {curriculum.get_current_reward_weights()}")
    
    # Simulate training loop
    print("\nSimulating training progression...")
    for step in range(1000):
        # Simulate rewards and done signals
        episode_rewards = torch.randn(4096) + 20.0  # Random rewards around 20
        done = torch.rand(4096) < 0.01  # 1% chance of episode done
        
        curriculum.update(episode_rewards, done)
        
        if step % 100 == 0:
            debug_info = curriculum.get_debug_info()
            print(f"Step {step}: {debug_info['stage_name']} - Avg Reward: {debug_info['avg_recent_reward']:.2f}")
