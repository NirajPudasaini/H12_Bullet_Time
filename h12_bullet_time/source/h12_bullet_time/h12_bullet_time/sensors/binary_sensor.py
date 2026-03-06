from __future__ import annotations

import logging
import re
import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

from isaacsim.core.simulation_manager import SimulationManager
from pxr import UsdPhysics

import isaaclab.sim as sim_utils
import isaaclab.utils.string as string_utils
from isaaclab.markers import VisualizationMarkers
from isaaclab.utils.math import (
    combine_frame_transforms,
    convert_quat,
    is_identity_pose,
    normalize,
    quat_from_angle_axis,
    subtract_frame_transforms,
)

from isaaclab.sensors import SensorBase
from .binary_sensor_data import BinarySensorData

if TYPE_CHECKING:
    from .binary_sensor_cfg import BinarySensorCfg

logger = logging.getLogger(__name__)


class BinarySensor(SensorBase):
    """Binary proximity sensor that reports 1.0 when a target is within max_range, 0.0 otherwise."""

    cfg: BinarySensorCfg

    def __init__(self, cfg: BinarySensorCfg):
        super().__init__(cfg)
        self._data: BinarySensorData = BinarySensorData()

    def __str__(self) -> str:
        return (
            f"BinarySensor @ '{self.cfg.prim_path}': \n"
            f"\ttracked body frames: {[self._source_frame_body_name] + self._target_frame_body_names} \n"
            f"\tnumber of envs: {self._num_envs}\n"
            f"\tsource body frame: {self._source_frame_body_name}\n"
            f"\ttarget frames (count: {len(self._target_frame_names)}): {self._target_frame_names}\n"
        )

    @property
    def data(self) -> BinarySensorData:
        self._update_outdated_buffers()
        return self._data

    @property
    def num_bodies(self) -> int:
        return len(self._target_frame_body_names)

    @property
    def body_names(self) -> list[str]:
        return self._target_frame_body_names

    def reset(self, env_ids: Sequence[int] | None = None):
        super().reset(env_ids)
        if env_ids is None:
            env_ids = ...

    def find_bodies(self, name_keys: str | Sequence[str], preserve_order: bool = False) -> tuple[list[int], list[str]]:
        return string_utils.resolve_matching_names(name_keys, self._target_frame_names, preserve_order)

    def _initialize_impl(self):
        super()._initialize_impl()

        source_frame_offset_pos = torch.tensor(self.cfg.source_frame_offset.pos, device=self.device)
        source_frame_offset_quat = torch.tensor(self.cfg.source_frame_offset.rot, device=self.device)
        self._apply_source_frame_offset = True
        if is_identity_pose(source_frame_offset_pos, source_frame_offset_quat):
            self._apply_source_frame_offset = False
        else:
            self._source_frame_offset_pos = source_frame_offset_pos.unsqueeze(0).repeat(self._num_envs, 1)
            self._source_frame_offset_quat = source_frame_offset_quat.unsqueeze(0).repeat(self._num_envs, 1)

        body_names_to_frames: dict[str, dict[str, set[str] | str]] = {}
        target_offsets: dict[str, dict[str, torch.Tensor]] = {}
        non_identity_offset_frames: list[str] = []
        self._apply_target_frame_offset = False
        self._source_is_also_target_frame = False

        frames = [None] + [target_frame.name for target_frame in self.cfg.target_frames]
        frame_prim_paths = [self.cfg.prim_path] + [target_frame.prim_path for target_frame in self.cfg.target_frames]
        frame_offsets = [None] + [target_frame.offset for target_frame in self.cfg.target_frames]
        frame_types = ["source"] + ["target"] * len(self.cfg.target_frames)

        for frame, prim_path, offset, frame_type in zip(frames, frame_prim_paths, frame_offsets, frame_types):
            matching_prims = sim_utils.find_matching_prims(prim_path)
            if len(matching_prims) == 0:
                raise ValueError(
                    f"Failed to create BinarySensor for frame '{frame}' with path '{prim_path}'."
                    " No matching prims were found."
                )
            for prim in matching_prims:
                matching_prim_path = prim.GetPath().pathString
                if not prim.HasAPI(UsdPhysics.RigidBodyAPI):
                    raise ValueError(
                        f"While resolving '{prim_path}' found '{matching_prim_path}' which is not a rigid body."
                    )
                body_name = matching_prim_path.rsplit("/", 1)[-1]
                frame_name = frame if frame is not None else body_name

                if body_name in body_names_to_frames:
                    body_names_to_frames[body_name]["frames"].add(frame_name)
                    if body_names_to_frames[body_name]["type"] == "source" and frame_type == "target":
                        self._source_is_also_target_frame = True
                else:
                    body_names_to_frames[body_name] = {
                        "frames": {frame_name},
                        "prim_path": matching_prim_path,
                        "type": frame_type,
                    }

                if offset is not None:
                    offset_pos = torch.tensor(offset.pos, device=self.device)
                    offset_quat = torch.tensor(offset.rot, device=self.device)
                    if not is_identity_pose(offset_pos, offset_quat):
                        non_identity_offset_frames.append(frame_name)
                        self._apply_target_frame_offset = True
                    target_offsets[frame_name] = {"pos": offset_pos, "quat": offset_quat}

        tracked_prim_paths = [body_names_to_frames[bn]["prim_path"] for bn in body_names_to_frames.keys()]
        tracked_body_names = list(body_names_to_frames.keys())
        body_names_regex = [p.replace("env_0", "env_*") for p in tracked_prim_paths]

        self._physics_sim_view = SimulationManager.get_physics_sim_view()
        self._frame_physx_view = self._physics_sim_view.create_rigid_body_view(body_names_regex)

        all_prim_paths = self._frame_physx_view.prim_paths

        if "env_" in all_prim_paths[0]:
            def extract_env_num_and_prim_path(item: str) -> tuple[int, str]:
                match = re.search(r"env_(\d+)(.*)", item)
                return (int(match.group(1)), match.group(2))

            self._per_env_indices = [
                index for index, _ in sorted(
                    list(enumerate(all_prim_paths)), key=lambda x: extract_env_num_and_prim_path(x[1])
                )
            ]
            sorted_prim_paths = [
                all_prim_paths[index] for index in self._per_env_indices if "env_0" in all_prim_paths[index]
            ]
        else:
            self._per_env_indices = [index for index, _ in sorted(enumerate(all_prim_paths), key=lambda x: x[1])]
            sorted_prim_paths = [all_prim_paths[index] for index in self._per_env_indices]

        self._target_frame_body_names = [p.split("/")[-1] for p in sorted_prim_paths]
        self._source_frame_body_name = self.cfg.prim_path.split("/")[-1]
        source_frame_index = self._target_frame_body_names.index(self._source_frame_body_name)

        if not self._source_is_also_target_frame:
            self._target_frame_body_names.remove(self._source_frame_body_name)

        all_ids = torch.arange(self._num_envs * len(tracked_body_names))
        self._source_frame_body_ids = torch.arange(self._num_envs) * len(tracked_body_names) + source_frame_index

        if self._source_is_also_target_frame:
            self._target_frame_body_ids = all_ids
        else:
            self._target_frame_body_ids = all_ids[~torch.isin(all_ids, self._source_frame_body_ids)]

        self._target_frame_names: list[str] = []
        target_frame_offset_pos = []
        target_frame_offset_quat = []
        duplicate_frame_indices = []

        for i, body_name in enumerate(self._target_frame_body_names):
            for frame in body_names_to_frames[body_name]["frames"]:
                if frame in target_offsets:
                    target_frame_offset_pos.append(target_offsets[frame]["pos"])
                    target_frame_offset_quat.append(target_offsets[frame]["quat"])
                    self._target_frame_names.append(frame)
                    duplicate_frame_indices.append(i)

        duplicate_frame_indices = torch.tensor(duplicate_frame_indices, device=self.device)
        if self._source_is_also_target_frame:
            num_target_body_frames = len(tracked_body_names)
        else:
            num_target_body_frames = len(tracked_body_names) - 1

        self._duplicate_frame_indices = torch.cat(
            [duplicate_frame_indices + num_target_body_frames * env_num for env_num in range(self._num_envs)]
        )

        if self._apply_target_frame_offset:
            self._target_frame_offset_pos = torch.stack(target_frame_offset_pos).repeat(self._num_envs, 1)
            self._target_frame_offset_quat = torch.stack(target_frame_offset_quat).repeat(self._num_envs, 1)

        self._relative_sensor_pos = torch.tensor(self.cfg.relative_sensor_pos, device=self.device)
        self._num_sensors = len(self.cfg.relative_sensor_pos)

        n_dup = len(duplicate_frame_indices)
        self._data.target_frame_names = self._target_frame_names
        self._data.source_pos_w = torch.zeros(self._num_envs, 3, device=self._device)
        self._data.source_quat_w = torch.zeros(self._num_envs, 4, device=self._device)
        self._data.target_pos_w = torch.zeros(self._num_envs, n_dup, 3, device=self._device)
        self._data.target_quat_w = torch.zeros(self._num_envs, n_dup, 4, device=self._device)
        self._data.target_pos_source = torch.zeros_like(self._data.target_pos_w)
        self._data.target_quat_source = torch.zeros_like(self._data.target_quat_w)
        self._data.raw_target_distances = torch.zeros(self._num_envs, self._num_sensors, n_dup, device=self._device)
        self._data.target_pos_sensor = torch.zeros(self._num_envs, self._num_sensors, n_dup, 3, device=self._device)
        self._data.binary_detection = torch.zeros(self._num_envs, self._num_sensors, n_dup, device=self._device)
        self._data.binary_detection_change = torch.zeros(self._num_envs, self._num_sensors, n_dup, device=self._device)
        self._last_binary_detection = torch.zeros(self._num_envs, self._num_sensors, n_dup, device=self._device)

    def _update_buffers_impl(self, env_ids: Sequence[int]):
        if len(env_ids) == self._num_envs:
            env_ids = ...

        transforms = self._frame_physx_view.get_transforms()
        transforms = transforms[self._per_env_indices]
        transforms[:, 3:] = convert_quat(transforms[:, 3:], to="wxyz")

        source_frames = transforms[self._source_frame_body_ids]
        if self._apply_source_frame_offset:
            source_pos_w, source_quat_w = combine_frame_transforms(
                source_frames[:, :3], source_frames[:, 3:],
                self._source_frame_offset_pos, self._source_frame_offset_quat,
            )
        else:
            source_pos_w = source_frames[:, :3]
            source_quat_w = source_frames[:, 3:]

        target_frames = transforms[self._target_frame_body_ids]
        duplicated_target_frame_pos_w = target_frames[self._duplicate_frame_indices, :3]
        duplicated_target_frame_quat_w = target_frames[self._duplicate_frame_indices, 3:]

        if self._apply_target_frame_offset:
            target_pos_w, target_quat_w = combine_frame_transforms(
                duplicated_target_frame_pos_w, duplicated_target_frame_quat_w,
                self._target_frame_offset_pos, self._target_frame_offset_quat,
            )
        else:
            target_pos_w = duplicated_target_frame_pos_w
            target_quat_w = duplicated_target_frame_quat_w

        total_num_frames = len(self._target_frame_names)
        target_pos_source, target_quat_source = subtract_frame_transforms(
            source_pos_w.unsqueeze(1).expand(-1, total_num_frames, -1).reshape(-1, 3),
            source_quat_w.unsqueeze(1).expand(-1, total_num_frames, -1).reshape(-1, 4),
            target_pos_w, target_quat_w,
        )

        target_pos_sensor = target_pos_source.view(-1, total_num_frames, 3).unsqueeze(1) - self._relative_sensor_pos.view(
            1, self._num_sensors, 1, 3
        )
        raw_target_distances = torch.linalg.norm(target_pos_sensor, dim=-1) - self.cfg.projectile_radius

        # Binary thresholding: 1.0 if within max_range, 0.0 otherwise
        binary_detection = (raw_target_distances <= self.cfg.max_range).float()
        binary_detection_change = binary_detection - self._last_binary_detection

        self._data.source_pos_w[:] = source_pos_w.view(-1, 3)
        self._data.source_quat_w[:] = source_quat_w.view(-1, 4)
        self._data.target_pos_w[:] = target_pos_w.view(-1, total_num_frames, 3)
        self._data.target_quat_w[:] = target_quat_w.view(-1, total_num_frames, 4)
        self._data.target_pos_source[:] = target_pos_source.view(-1, total_num_frames, 3)
        self._data.target_quat_source[:] = target_quat_source.view(-1, total_num_frames, 4)
        self._data.raw_target_distances[:] = raw_target_distances
        self._data.target_pos_sensor[:] = target_pos_sensor
        self._data.binary_detection[:] = binary_detection
        self._data.binary_detection_change[:] = binary_detection_change
        self._last_binary_detection[:] = binary_detection

    def _set_debug_vis_impl(self, debug_vis: bool):
        if debug_vis:
            if not hasattr(self, "frame_visualizer"):
                self.frame_visualizer = VisualizationMarkers(self.cfg.visualizer_cfg)
            self.frame_visualizer.set_visibility(True)
        else:
            if hasattr(self, "frame_visualizer"):
                self.frame_visualizer.set_visibility(False)

    def _debug_vis_callback(self, event):
        source_pos_expanded = self._data.source_pos_w.repeat_interleave(self._num_sensors, dim=0)
        source_quat_expanded = self._data.source_quat_w.repeat_interleave(self._num_sensors, dim=0)
        sensor_rel_expanded = self._relative_sensor_pos.repeat(self._num_envs, 1)
        sensor_quat_rel = torch.tensor(
            [1.0, 0.0, 0.0, 0.0], device=self.device
        ).repeat(self._num_envs * self._num_sensors, 1)

        sensor_pos_w, _ = combine_frame_transforms(
            source_pos_expanded, source_quat_expanded, sensor_rel_expanded, sensor_quat_rel
        )

        num_targets = self._data.target_pos_w.shape[1]
        start_pos = sensor_pos_w.view(self._num_envs, self._num_sensors, 3).unsqueeze(2)
        start_pos = start_pos.expand(-1, -1, num_targets, -1).reshape(-1, 3)

        end_pos = self._data.target_pos_w.unsqueeze(1).expand(-1, self._num_sensors, -1, -1).reshape(-1, 3)

        in_range_mask = self._data.binary_detection.reshape(-1) > 0.5

        if in_range_mask.any():
            lines_pos, lines_quat, lines_length = self._get_connecting_lines(
                start_pos=start_pos[in_range_mask], end_pos=end_pos[in_range_mask],
            )
            marker_scales = torch.ones(lines_pos.size(0), 3, device=self.device)
            marker_indices = torch.ones(lines_pos.size(0), device=self.device)
            marker_scales[:, -1] = lines_length
            self.frame_visualizer.visualize(
                translations=lines_pos, orientations=lines_quat,
                scales=marker_scales, marker_indices=marker_indices,
            )
        else:
            self.frame_visualizer.visualize(
                translations=torch.zeros(0, 3, device=self.device),
                orientations=torch.zeros(0, 4, device=self.device),
                scales=torch.zeros(0, 3, device=self.device),
                marker_indices=torch.zeros(0, device=self.device),
            )

    def _invalidate_initialize_callback(self, event):
        super()._invalidate_initialize_callback(event)
        self._frame_physx_view = None

    def _get_connecting_lines(
        self, start_pos: torch.Tensor, end_pos: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        direction = end_pos - start_pos
        lengths = torch.norm(direction, dim=-1)
        positions = (start_pos + end_pos) / 2
        default_direction = torch.tensor([0.0, 0.0, 1.0], device=self.device).expand(start_pos.size(0), -1)
        direction_norm = normalize(direction)
        rotation_axis = torch.linalg.cross(default_direction, direction_norm)
        rotation_axis_norm = torch.norm(rotation_axis, dim=-1)
        mask = rotation_axis_norm > 1e-6
        rotation_axis = torch.where(
            mask.unsqueeze(-1), normalize(rotation_axis),
            torch.tensor([1.0, 0.0, 0.0], device=self.device).expand(start_pos.size(0), -1),
        )
        cos_angle = torch.sum(default_direction * direction_norm, dim=-1)
        cos_angle = torch.clamp(cos_angle, -1.0, 1.0)
        angle = torch.acos(cos_angle)
        orientations = quat_from_angle_axis(angle, rotation_axis)
        return positions, orientations, lengths
