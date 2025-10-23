"""Observation functions - import from Isaac Lab instead of using stubs."""
from __future__ import annotations

# Import the real observation functions from Isaac Lab
from isaaclab.envs.mdp import (
    base_lin_vel,
    base_ang_vel,
    joint_pos_rel,
    joint_vel_rel,
)

__all__ = [
    "base_lin_vel",
    "base_ang_vel",
    "joint_pos_rel",
    "joint_vel_rel",
]
