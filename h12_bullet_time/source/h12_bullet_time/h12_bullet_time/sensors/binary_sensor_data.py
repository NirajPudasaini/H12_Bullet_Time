import torch
from dataclasses import dataclass


@dataclass
class BinarySensorData:
    """Data container for the binary proximity sensor."""

    target_frame_names: list[str] = None
    target_pos_source: torch.Tensor = None
    """Position of target frame(s) relative to source. Shape: (N, M, 3)."""
    target_quat_source: torch.Tensor = None
    """Orientation of target frame(s) relative to source. Shape: (N, M, 4)."""
    target_pos_w: torch.Tensor = None
    """Position of target frame(s) in world frame. Shape: (N, M, 3)."""
    target_quat_w: torch.Tensor = None
    """Orientation of target frame(s) in world frame. Shape: (N, M, 4)."""
    source_pos_w: torch.Tensor = None
    """Position of source frame in world frame. Shape: (N, 3)."""
    source_quat_w: torch.Tensor = None
    """Orientation of source frame in world frame. Shape: (N, 4)."""
    raw_target_distances: torch.Tensor = None
    """Raw distances from each sensor to each target. Shape: (N, S, M)."""
    target_pos_sensor: torch.Tensor = None
    """Position of targets relative to each sensor. Shape: (N, S, M, 3)."""
    binary_detection: torch.Tensor = None
    """Binary in-range detection: 1.0 if target within max_range, else 0.0. Shape: (N, S, M)."""
    binary_detection_change: torch.Tensor = None
    """Change in binary detection from previous step. Shape: (N, S, M)."""
