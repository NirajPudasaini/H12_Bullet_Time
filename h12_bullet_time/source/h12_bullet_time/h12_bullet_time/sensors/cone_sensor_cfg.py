from dataclasses import MISSING

import isaaclab.sim as sim_utils
from isaaclab.markers.config import VisualizationMarkersCfg
from isaaclab.utils import configclass

from isaaclab.sensors import FrameTransformerCfg
from .cone_sensor import ConeSensor


CONE_LINE_MARKER_CFG = VisualizationMarkersCfg(
    prim_path="/Visuals/ConeSensor",
    markers={
        "placeholder": sim_utils.SphereCfg(radius=0.001, visible=False),
        "connecting_line": sim_utils.CylinderCfg(
            radius=0.003,
            height=1.0,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.5, 0.0, 1.0), roughness=1.0),
        ),
    }
)


@configclass
class OffsetCfg:
    pos: tuple[float, float, float] = (0.0, 0.0, 0.0)
    rot: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)


@configclass
class ConeSensorCfg(FrameTransformerCfg):

    @configclass
    class FrameCfg:
        prim_path: str = MISSING
        name: str | None = None
        offset: OffsetCfg = OffsetCfg()

    class_type: type = ConeSensor

    prim_path: str = MISSING
    source_frame_offset: OffsetCfg = OffsetCfg()
    target_frames: list[FrameCfg] = MISSING
    relative_sensor_pos: list[tuple[float, float, float]] = [(0.0, 0.0, 0.0)]
    relative_sensor_quat: list[tuple[float, float, float, float]] = [(1.0, 0.0, 0.0, 0.0)]
    max_range: float = 4.0
    max_SNR: float = 100.0
    k_factor: float = 1.0
    projectile_radius: float = 0.05
    cone_half_angle_deg: float = 30.0

    visualizer_cfg: VisualizationMarkersCfg = CONE_LINE_MARKER_CFG
