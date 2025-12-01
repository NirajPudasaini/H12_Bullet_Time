# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from dataclasses import MISSING

from isaaclab.markers.config import FRAME_MARKER_CFG, VisualizationMarkersCfg
from isaaclab.utils import configclass

from isaaclab.sensors import FrameTransformerCfg, SensorBaseCfg
from .tof_sensor import TofSensor


@configclass
class OffsetCfg:
    """The offset pose of one frame relative to another frame."""

    pos: tuple[float, float, float] = (0.0, 0.0, 0.0)
    """Translation w.r.t. the parent frame. Defaults to (0.0, 0.0, 0.0)."""
    rot: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)
    """Quaternion rotation (w, x, y, z) w.r.t. the parent frame. Defaults to (1.0, 0.0, 0.0, 0.0)."""


@configclass
class TofSensorCfg(FrameTransformerCfg):
    """Configuration for the frame transformer sensor."""

    @configclass
    class FrameCfg:
        """Information specific to a coordinate frame."""

        prim_path: str = MISSING
        """The prim path corresponding to a rigid body.

        This can be a regex pattern to match multiple prims. For example, "/Robot/.*" will match all prims under "/Robot".

        This means that if the source :attr:`TofSensorCfg.prim_path` is "/Robot/base", and the target :attr:`TofSensorCfg.FrameCfg.prim_path` is "/Robot/.*",
        then the frame transformer will track the poses of all the prims under "/Robot",
        including "/Robot/base" (even though this will result in an identity pose w.r.t. the source frame).
        """

        name: str | None = None
        """User-defined name for the new coordinate frame. Defaults to None.

        If None, then the name is extracted from the leaf of the prim path.
        """

        offset: OffsetCfg = OffsetCfg()
        """The pose offset from the parent prim frame."""

    class_type: type = TofSensor

    prim_path: str = MISSING
    """The prim path of the body to transform from (source frame)."""

    source_frame_offset: OffsetCfg = OffsetCfg()
    """The pose offset from the source prim frame."""

    target_frames: list[FrameCfg] = MISSING
    """A list of the target frames.

    This allows a single TofSensor to handle multiple target prims. For example, in a quadruped,
    we can use a single TofSensor to track each foot's position and orientation in the body
    frame using four frame offsets.
    """

    relative_sensor_pos: list[tuple[float, float, float]] = [(0.0, 0.0, 0.0)]
    """The relative positions of sensors per link. 0.0, 0.0, 0.0 means the sensor is at the origin of the link.

    When a distance to a target is computed, the distances will be computed from the relative positions of the sensors to the target in the world frame.
    """

    relative_sensor_quat: list[tuple[float, float, float, float]] = [(1.0, 0.0, 0.0, 0.0)]
    """The relative orientations (quaternions w,x,y,z) of sensors per link. 
    
    (1.0, 0.0, 0.0, 0.0) is the identity quaternion meaning the sensor is aligned with the link frame.
    The sensor's forward direction is along the local Z-axis.
    """

    max_range: float = 4.0  # meters
    """Maximum detection range of the sensor in meters."""

    # projectile_radius: float = 0.5  # meters
    """Radius of the target projectile/sphere in meters."""

    sensor_fov_radius: float = 0.05  # meters
    """Field of view radius (beam width) in meters.
    
    This defines the cylindrical beam width of the ToF sensor. A target is only detected
    if its perpendicular distance from the sensor's axis (XY plane in sensor frame) is 
    less than or equal to this radius.
    """

    visualizer_cfg: VisualizationMarkersCfg = FRAME_MARKER_CFG.replace(prim_path="/Visuals/TofSensor")
    """The configuration object for the visualization markers. Defaults to FRAME_MARKER_CFG.

    Note:
        This attribute is only used when debug visualization is enabled.
    """