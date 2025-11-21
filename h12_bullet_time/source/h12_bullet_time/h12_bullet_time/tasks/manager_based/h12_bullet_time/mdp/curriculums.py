# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Curriculum learning utilities for H12 Bullet Time."""

from typing import Sequence

import torch

from isaaclab.envs import ManagerBasedRLEnv


def modify_reward_weight(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    term_name: str,
    weight: float,
    num_steps: int,
) -> float:

    # Check if milestone has been reached
    if env.common_step_counter > num_steps:
        # Get and update term configuration
        term_cfg = env.reward_manager.get_term_cfg(term_name)
        term_cfg.weight = weight
        env.reward_manager.set_term_cfg(term_name, term_cfg)
        
        return weight
    
    # Return current weight before milestone
    term_cfg = env.reward_manager.get_term_cfg(term_name)
    return term_cfg.weight
