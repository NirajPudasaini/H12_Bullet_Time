# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from dataclasses import MISSING

import isaaclab.sim as sim_utils
from isaaclab.markers.config import FRAME_MARKER_CFG, VisualizationMarkersCfg
from isaaclab.utils import configclass

from isaaclab.sensors import FrameTransformerCfg, SensorBaseCfg
from .capacitive_sensor import CapacitiveSensor


# Line-only marker configuration for capacitive sensor visualization (no frame axes)
CAPACITIVE_LINE_MARKER_CFG = VisualizationMarkersCfg(
    prim_path="/Visuals/CapacitiveSensor",
    markers={
        # Placeholder for index 0 (invisible, since we only use index 1 for lines)
        "placeholder": sim_utils.SphereCfg(
            radius=0.001,
            visible=False,
        ),
        # Index 1: connecting line (shown when in range)
        "connecting_line": sim_utils.CylinderCfg(
            radius=0.003,
            height=1.0,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.5), roughness=1.0),
        ),
    }
)


@configclass
class OffsetCfg:
    """The offset pose of one frame relative to another frame."""

    pos: tuple[float, float, float] = (0.0, 0.0, 0.0)
    """Translation w.r.t. the parent frame. Defaults to (0.0, 0.0, 0.0)."""
    rot: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)
    """Quaternion rotation (w, x, y, z) w.r.t. the parent frame. Defaults to (1.0, 0.0, 0.0, 0.0)."""


@configclass
class CapacitiveSensorCfg(FrameTransformerCfg):
    """Configuration for the frame transformer sensor."""

    @configclass
    class FrameCfg:
        """Information specific to a coordinate frame."""

        prim_path: str = MISSING
        """The prim path corresponding to a rigid body.

        This can be a regex pattern to match multiple prims. For example, "/Robot/.*" will match all prims under "/Robot".

        This means that if the source :attr:`CapacitiveSensorCfg.prim_path` is "/Robot/base", and the target :attr:`CapacitiveSensorCfg.FrameCfg.prim_path` is "/Robot/.*",
        then the frame transformer will track the poses of all the prims under "/Robot",
        including "/Robot/base" (even though this will result in an identity pose w.r.t. the source frame).
        """

        name: str | None = None
        """User-defined name for the new coordinate frame. Defaults to None.

        If None, then the name is extracted from the leaf of the prim path.
        """

        offset: OffsetCfg = OffsetCfg()
        """The pose offset from the parent prim frame."""

    class_type: type = CapacitiveSensor

    prim_path: str = MISSING
    """The prim path of the body to transform from (source frame)."""

    source_frame_offset: OffsetCfg = OffsetCfg()
    """The pose offset from the source prim frame."""

    target_frames: list[FrameCfg] = MISSING
    """A list of the target frames.

    This allows a single CapacitiveSensor to handle multiple target prims. For example, in a quadruped,
    we can use a single CapacitiveSensor to track each foot's position and orientation in the body
    frame using four frame offsets.
    """

    relative_sensor_pos: list[tuple[float, float, float]] = [(0.0, 0.0, 0.0)]
    """The relative positions of sensors per link. 0.0, 0.0, 0.0 means the sensor is at the origin of the link.

    When a distance to a target is computed, the distances will be computed from the relative positions of the sensors to the target in the world frame.
    """

    max_range: float = 0.15 # meters. This is the point where SNR becomes <= 3.5
    max_SNR: float = 100.0 # dB
    k_factor: float = 1.0 # emperically derived parallel plate capacitor constant
    projectile_radius: float = 0.05 # meters
    """Radius of the projectile. This is subtracted from the distance to the target to compute the closest point.
    """

    """
    Parameter to adjust the sensitivity of the sensor. C = k/d.
    """


    visualizer_cfg: VisualizationMarkersCfg = CAPACITIVE_LINE_MARKER_CFG
    """The configuration object for the visualization markers. Defaults to CAPACITIVE_LINE_MARKER_CFG.

    Note:
        This attribute is only used when debug visualization is enabled.
        Only shows lines for sensor-target pairs within max_range (no frame axes).
    """