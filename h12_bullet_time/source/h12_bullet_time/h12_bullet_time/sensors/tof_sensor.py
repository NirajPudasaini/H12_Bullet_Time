# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

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
from .tof_sensor_data import TofSensorData

if TYPE_CHECKING:
    from .tof_sensor_cfg import TofSensorCfg

# import logger
logger = logging.getLogger(__name__)


class TofSensor(SensorBase):
    """
    Modification of the FrameTransformer
    This is only intended for spherical projectile detection.

    """

    cfg: TofSensorCfg
    """The configuration parameters."""

    def __init__(self, cfg: TofSensorCfg):
        """Initializes the frame transformer object.

        Args:
            cfg: The configuration parameters.
        """
        # initialize base class
        super().__init__(cfg)
        # Create empty variables for storing output data
        self._data: TofSensorData = TofSensorData()

    def __str__(self) -> str:
        """Returns: A string containing information about the instance."""
        return (
            f"TofSensor @ '{self.cfg.prim_path}': \n"
            f"\ttracked body frames: {[self._source_frame_body_name] + self._target_frame_body_names} \n"
            f"\tnumber of envs: {self._num_envs}\n"
            f"\tsource body frame: {self._source_frame_body_name}\n"
            f"\ttarget frames (count: {self._target_frame_names}): {len(self._target_frame_names)}\n"
        )

    """
    Properties
    """

    @property
    def data(self) -> TofSensorData:
        # update sensors if needed
        self._update_outdated_buffers()
        # return the data
        return self._data

    @property
    def num_bodies(self) -> int:
        """Returns the number of target bodies being tracked.

        Note:
            This is an alias used for consistency with other sensors. Otherwise, we recommend using
            :attr:`len(data.target_frame_names)` to access the number of target frames.
        """
        return len(self._target_frame_body_names)

    @property
    def body_names(self) -> list[str]:
        """Returns the names of the target bodies being tracked.

        Note:
            This is an alias used for consistency with other sensors. Otherwise, we recommend using
            :attr:`data.target_frame_names` to access the target frame names.
        """
        return self._target_frame_body_names

    """
    Operations
    """

    def reset(self, env_ids: Sequence[int] | None = None):
        # reset the timers and counters
        super().reset(env_ids)
        # resolve None
        if env_ids is None:
            env_ids = ...

    def find_bodies(self, name_keys: str | Sequence[str], preserve_order: bool = False) -> tuple[list[int], list[str]]:
        """Find bodies in the articulation based on the name keys.

        Args:
            name_keys: A regular expression or a list of regular expressions to match the body names.
            preserve_order: Whether to preserve the order of the name keys in the output. Defaults to False.

        Returns:
            A tuple of lists containing the body indices and names.
        """
        return string_utils.resolve_matching_names(name_keys, self._target_frame_names, preserve_order)

    """
    Implementation.
    """

    def _initialize_impl(self):
        super()._initialize_impl()

        # resolve source frame offset
        source_frame_offset_pos = torch.tensor(self.cfg.source_frame_offset.pos, device=self.device)
        source_frame_offset_quat = torch.tensor(self.cfg.source_frame_offset.rot, device=self.device)
        # Only need to perform offsetting of source frame if the position offsets is non-zero and rotation offset is
        # not the identity quaternion for efficiency in _update_buffer_impl
        self._apply_source_frame_offset = True
        # Handle source frame offsets
        if is_identity_pose(source_frame_offset_pos, source_frame_offset_quat):
            logger.debug(f"No offset application needed for source frame as it is identity: {self.cfg.prim_path}")
            self._apply_source_frame_offset = False
        else:
            logger.debug(f"Applying offset to source frame as it is not identity: {self.cfg.prim_path}")
            # Store offsets as tensors (duplicating each env's offsets for ease of multiplication later)
            self._source_frame_offset_pos = source_frame_offset_pos.unsqueeze(0).repeat(self._num_envs, 1)
            self._source_frame_offset_quat = source_frame_offset_quat.unsqueeze(0).repeat(self._num_envs, 1)

        # Keep track of mapping from the rigid body name to the desired frames and prim path, as there may be multiple frames
        # based upon the same body name and we don't want to create unnecessary views
        body_names_to_frames: dict[str, dict[str, set[str] | str]] = {}
        # The offsets associated with each target frame
        target_offsets: dict[str, dict[str, torch.Tensor]] = {}
        # The frames whose offsets are not identity
        non_identity_offset_frames: list[str] = []

        # Only need to perform offsetting of target frame if any of the position offsets are non-zero or any of the
        # rotation offsets are not the identity quaternion for efficiency in _update_buffer_impl
        self._apply_target_frame_offset = False

        # Need to keep track of whether the source frame is also a target frame
        self._source_is_also_target_frame = False

        # Collect all target frames, their associated body prim paths and their offsets so that we can extract
        # the prim, check that it has the appropriate rigid body API in a single loop.
        # First element is None because user can't specify source frame name
        frames = [None] + [target_frame.name for target_frame in self.cfg.target_frames]
        frame_prim_paths = [self.cfg.prim_path] + [target_frame.prim_path for target_frame in self.cfg.target_frames]
        # First element is None because source frame offset is handled separately
        frame_offsets = [None] + [target_frame.offset for target_frame in self.cfg.target_frames]
        frame_types = ["source"] + ["target"] * len(self.cfg.target_frames)
        for frame, prim_path, offset, frame_type in zip(frames, frame_prim_paths, frame_offsets, frame_types):
            # Find correct prim
            matching_prims = sim_utils.find_matching_prims(prim_path)
            if len(matching_prims) == 0:
                raise ValueError(
                    f"Failed to create frame transformer for frame '{frame}' with path '{prim_path}'."
                    " No matching prims were found."
                )
            for prim in matching_prims:
                # Get the prim path of the matching prim
                matching_prim_path = prim.GetPath().pathString
                # Check if it is a rigid prim
                if not prim.HasAPI(UsdPhysics.RigidBodyAPI):
                    raise ValueError(
                        f"While resolving expression '{prim_path}' found a prim '{matching_prim_path}' which is not a"
                        " rigid body. The class only supports transformations between rigid bodies."
                    )

                # Get the name of the body
                body_name = matching_prim_path.rsplit("/", 1)[-1]
                # Use body name if frame isn't specified by user
                frame_name = frame if frame is not None else body_name

                # Keep track of which frames are associated with which bodies
                if body_name in body_names_to_frames:
                    body_names_to_frames[body_name]["frames"].add(frame_name)

                    # This is a corner case where the source frame is also a target frame
                    if body_names_to_frames[body_name]["type"] == "source" and frame_type == "target":
                        self._source_is_also_target_frame = True

                else:
                    # Store the first matching prim path and the type of frame
                    body_names_to_frames[body_name] = {
                        "frames": {frame_name},
                        "prim_path": matching_prim_path,
                        "type": frame_type,
                    }

                if offset is not None:
                    offset_pos = torch.tensor(offset.pos, device=self.device)
                    offset_quat = torch.tensor(offset.rot, device=self.device)
                    # Check if we need to apply offsets (optimized code path in _update_buffer_impl)
                    if not is_identity_pose(offset_pos, offset_quat):
                        non_identity_offset_frames.append(frame_name)
                        self._apply_target_frame_offset = True
                    target_offsets[frame_name] = {"pos": offset_pos, "quat": offset_quat}

        if not self._apply_target_frame_offset:
            logger.info(
                f"No offsets application needed from '{self.cfg.prim_path}' to target frames as all"
                f" are identity: {frames[1:]}"
            )
        else:
            logger.info(
                f"Offsets application needed from '{self.cfg.prim_path}' to the following target frames:"
                f" {non_identity_offset_frames}"
            )

        # The names of bodies that RigidPrim will be tracking to later extract transforms from
        tracked_prim_paths = [body_names_to_frames[body_name]["prim_path"] for body_name in body_names_to_frames.keys()]
        tracked_body_names = [body_name for body_name in body_names_to_frames.keys()]

        body_names_regex = [tracked_prim_path.replace("env_0", "env_*") for tracked_prim_path in tracked_prim_paths]

        # obtain global simulation view
        self._physics_sim_view = SimulationManager.get_physics_sim_view()
        # Create a prim view for all frames and initialize it
        # order of transforms coming out of view will be source frame followed by target frame(s)
        self._frame_physx_view = self._physics_sim_view.create_rigid_body_view(body_names_regex)

        # Determine the order in which regex evaluated body names so we can later index into frame transforms
        # by frame name correctly
        all_prim_paths = self._frame_physx_view.prim_paths

        if "env_" in all_prim_paths[0]:

            def extract_env_num_and_prim_path(item: str) -> tuple[int, str]:
                """Separates the environment number and prim_path from the item.

                Args:
                    item: The item to extract the environment number from. Assumes item is of the form
                        `/World/envs/env_1/blah` or `/World/envs/env_11/blah`.
                Returns:
                    The environment number and the prim_path.
                """
                match = re.search(r"env_(\d+)(.*)", item)
                return (int(match.group(1)), match.group(2))

            # Find the indices that would reorganize output to be per environment. We want `env_1/blah` to come before `env_11/blah`
            # and env_1/Robot/base to come before env_1/Robot/foot so we need to use custom key function
            self._per_env_indices = [
                index
                for index, _ in sorted(
                    list(enumerate(all_prim_paths)), key=lambda x: extract_env_num_and_prim_path(x[1])
                )
            ]

            # Only need 0th env as the names and their ordering are the same across environments
            sorted_prim_paths = [
                all_prim_paths[index] for index in self._per_env_indices if "env_0" in all_prim_paths[index]
            ]

        else:
            # If no environment is present, then the order of the body names is the same as the order of the prim paths sorted alphabetically
            self._per_env_indices = [index for index, _ in sorted(enumerate(all_prim_paths), key=lambda x: x[1])]
            sorted_prim_paths = [all_prim_paths[index] for index in self._per_env_indices]

        # -- target frames
        self._target_frame_body_names = [prim_path.split("/")[-1] for prim_path in sorted_prim_paths]

        # -- source frame
        self._source_frame_body_name = self.cfg.prim_path.split("/")[-1]
        source_frame_index = self._target_frame_body_names.index(self._source_frame_body_name)

        # Only remove source frame from tracked bodies if it is not also a target frame
        if not self._source_is_also_target_frame:
            self._target_frame_body_names.remove(self._source_frame_body_name)

        # Determine indices into all tracked body frames for both source and target frames
        all_ids = torch.arange(self._num_envs * len(tracked_body_names))
        self._source_frame_body_ids = torch.arange(self._num_envs) * len(tracked_body_names) + source_frame_index

        # If source frame is also a target frame, then the target frame body ids are the same as the source frame body ids
        if self._source_is_also_target_frame:
            self._target_frame_body_ids = all_ids
        else:
            self._target_frame_body_ids = all_ids[~torch.isin(all_ids, self._source_frame_body_ids)]

        # The name of each of the target frame(s) - either user specified or defaulted to the body name
        self._target_frame_names: list[str] = []
        # The position and rotation components of target frame offsets
        target_frame_offset_pos = []
        target_frame_offset_quat = []
        # Stores the indices of bodies that need to be duplicated. For instance, if body "LF_SHANK" is needed
        # for 2 frames, this list enables us to duplicate the body to both frames when doing the calculations
        # when updating sensor in _update_buffers_impl
        duplicate_frame_indices = []

        self.make_grid()

        # Go through each body name and determine the number of duplicates we need for that frame
        # and extract the offsets. This is all done to handle the case where multiple frames
        # reference the same body, but have different names and/or offsets
        for i, body_name in enumerate(self._target_frame_body_names):
            for frame in body_names_to_frames[body_name]["frames"]:
                # Only need to handle target frames here as source frame is handled separately
                if frame in target_offsets:
                    target_frame_offset_pos.append(target_offsets[frame]["pos"])
                    target_frame_offset_quat.append(target_offsets[frame]["quat"])
                    self._target_frame_names.append(frame)
                    duplicate_frame_indices.append(i)

        # To handle multiple environments, need to expand so [0, 1, 1, 2] with 2 environments becomes
        # [0, 1, 1, 2, 3, 4, 4, 5]. Again, this is a optimization to make _update_buffer_impl more efficient
        duplicate_frame_indices = torch.tensor(duplicate_frame_indices, device=self.device)
        if self._source_is_also_target_frame:
            num_target_body_frames = len(tracked_body_names)
        else:
            num_target_body_frames = len(tracked_body_names) - 1

        self._duplicate_frame_indices = torch.cat(
            [duplicate_frame_indices + num_target_body_frames * env_num for env_num in range(self._num_envs)]
        )

        # Target frame offsets are only applied if at least one of the offsets are non-identity
        if self._apply_target_frame_offset:
            # Stack up all the frame offsets for shape (num_envs, num_frames, 3) and (num_envs, num_frames, 4)
            self._target_frame_offset_pos = torch.stack(target_frame_offset_pos).repeat(self._num_envs, 1)
            self._target_frame_offset_quat = torch.stack(target_frame_offset_quat).repeat(self._num_envs, 1)

        # Append the relative sensor positions and orientations to the source frame offset tensor
        self._relative_sensor_pos = torch.tensor(self.cfg.relative_sensor_pos, device=self.device)
        self._num_sensors = len(self.cfg.relative_sensor_pos)
        
        # Handle sensor orientations - expand to match number of sensors if only default provided
        if len(self.cfg.relative_sensor_quat) == 1 and self._num_sensors > 1:
            # Repeat the single quaternion for all sensors
            self._relative_sensor_quat = torch.tensor(
                self.cfg.relative_sensor_quat * self._num_sensors, device=self.device
            ).view(self._num_sensors, 4)
        else:
            self._relative_sensor_quat = torch.tensor(self.cfg.relative_sensor_quat, device=self.device)

        # fill the data buffer
        self._data.target_frame_names = self._target_frame_names
        self._data.source_pos_w = torch.zeros(self._num_envs, 3, device=self._device)
        self._data.source_quat_w = torch.zeros(self._num_envs, 4, device=self._device)
        self._data.target_pos_w = torch.zeros(self._num_envs, len(duplicate_frame_indices), 3, device=self._device)
        self._data.target_quat_w = torch.zeros(self._num_envs, len(duplicate_frame_indices), 4, device=self._device)
        self._data.target_pos_source = torch.zeros_like(self._data.target_pos_w)
        self._data.target_quat_source = torch.zeros_like(self._data.target_quat_w)
        self._data.raw_target_distances = torch.zeros(
            self._num_envs, self._num_sensors, len(duplicate_frame_indices), device=self._device
        )
        self._data.target_pos_sensor = torch.zeros(
            self._num_envs, self._num_sensors, len(duplicate_frame_indices), 3, device=self._device
        )
        self._data.tof_distances = torch.zeros(
            self._num_envs, self._num_sensors, len(duplicate_frame_indices), self.cfg.pixel_count**2,
            dtype=torch.float32, device=self._device
        )

    def _update_buffers_impl(self, env_ids: Sequence[int]):
        """Fills the buffers of the sensor data."""
        # default to all sensors
        if len(env_ids) == self._num_envs:
            env_ids = ...

        # Extract transforms from view - shape is:
        # (the total number of source and target body frames being tracked * self._num_envs, 7)
        transforms = self._frame_physx_view.get_transforms()

        # Reorder the transforms to be per environment as is expected of SensorData
        transforms = transforms[self._per_env_indices]

        # Convert quaternions as PhysX uses xyzw form
        transforms[:, 3:] = convert_quat(transforms[:, 3:], to="wxyz")

        # Process source frame transform
        source_frames = transforms[self._source_frame_body_ids]
        # Only apply offset if the offsets will result in a coordinate frame transform
        if self._apply_source_frame_offset:
            source_pos_w, source_quat_w = combine_frame_transforms(
                source_frames[:, :3],
                source_frames[:, 3:],
                self._source_frame_offset_pos,
                self._source_frame_offset_quat,
            )
        else:
            source_pos_w = source_frames[:, :3]
            source_quat_w = source_frames[:, 3:]

        # Process target frame transforms
        target_frames = transforms[self._target_frame_body_ids]
        duplicated_target_frame_pos_w = target_frames[self._duplicate_frame_indices, :3]
        duplicated_target_frame_quat_w = target_frames[self._duplicate_frame_indices, 3:]

        # Only apply offset if the offsets will result in a coordinate frame transform
        if self._apply_target_frame_offset:
            target_pos_w, target_quat_w = combine_frame_transforms(
                duplicated_target_frame_pos_w,
                duplicated_target_frame_quat_w,
                self._target_frame_offset_pos,
                self._target_frame_offset_quat,
            )
        else:
            target_pos_w = duplicated_target_frame_pos_w
            target_quat_w = duplicated_target_frame_quat_w

        # Compute the transform of the target frame with respect to the source frame
        total_num_frames = len(self._target_frame_names)
        target_pos_source, target_quat_source = subtract_frame_transforms(
            source_pos_w.unsqueeze(1).expand(-1, total_num_frames, -1).reshape(-1, 3),
            source_quat_w.unsqueeze(1).expand(-1, total_num_frames, -1).reshape(-1, 4),
            target_pos_w,
            target_quat_w,
        )

        # Compute the normalized distances of the target frame(s) relative to the source frame
        # Target pos in source frame: (N, M, 3) -> (N, 1, M, 3)
        # Relative sensor pos: (S, 3) -> (1, S, 1, 3)
        target_pos_sensor = target_pos_source.unsqueeze(1) - self._relative_sensor_pos.view(
            1, self._num_sensors, 1, 3
        )
        normalized_distances = torch.linalg.norm(target_pos_sensor, dim=-1)

        ############### TOF simulation ###############
        # Multi-pixel ToF: each sensor has a pixel_count x pixel_count grid of rays
        # Shapes: N=envs, S=sensors, M=targets, P=pixel_count^2
        
        P = self.cfg.pixel_count ** 2
        
        # Get base sensor forward directions: (S, 3)
        z_axis = torch.tensor([0.0, 0.0, 1.0], device=self.device)
        sensor_forward_base = self._quat_rotate_vec(self._relative_sensor_quat, z_axis)
        
        # Apply grid rotations to get ray directions for each pixel: (S, P, 3)
        # _grid_quats: (P, 4) -> broadcast with sensor_forward_base: (S, 1, 3)
        sensor_forward_base_exp = sensor_forward_base.unsqueeze(1).expand(-1, P, -1)  # (S, P, 3)
        grid_quats_exp = self._grid_quats.unsqueeze(0).expand(self._num_sensors, -1, -1)  # (S, P, 4)
        ray_dirs = self._quat_rotate_vec(grid_quats_exp, sensor_forward_base_exp)  # (S, P, 3)
        
        # Expand target_pos_sensor for pixel dimension: (N, S, M, 3) -> (N, S, M, 1, 3)
        target_pos_exp = target_pos_sensor.unsqueeze(-2)  # (N, S, M, 1, 3)
        
        # Expand ray_dirs for batch/target dims: (S, P, 3) -> (1, S, 1, P, 3)
        ray_dirs_exp = ray_dirs.unsqueeze(0).unsqueeze(2)  # (1, S, 1, P, 3)
        
        # Projection of target onto each ray direction (depth along ray): (N, S, M, P)
        proj_z = torch.sum(ray_dirs_exp * target_pos_exp, dim=-1)
        
        # Perpendicular distance from each ray axis: (N, S, M, P)
        proj_vec = proj_z.unsqueeze(-1) * ray_dirs_exp  # (N, S, M, P, 3)
        perpendicular_vec = target_pos_exp - proj_vec    # (N, S, M, P, 3)
        perpendicular_dist = torch.linalg.norm(perpendicular_vec, dim=-1)  # (N, S, M, P)
        
        # Expand normalized_distances for pixel dimension: (N, S, M) -> (N, S, M, P)
        sphere_offset = self.cfg.projectile_radius*torch.sin(torch.acos(perpendicular_dist/self.cfg.projectile_radius)) # offset = r*sin(theta), where theta = acos(perpendicular_dist/r)
        normalized_distances_exp = normalized_distances.unsqueeze(-1).expand(-1, -1, -1, P)
        final_distances = normalized_distances_exp - sphere_offset
        
        # Detection conditions per pixel
        in_front = proj_z > 0
        within_fov = perpendicular_dist <= self.cfg.projectile_radius
        within_range = normalized_distances_exp <= self.cfg.max_range
        
        # ToF distance per pixel: (N, S, M, P)
        tof_distances = torch.where(
            in_front & within_fov & within_range,
            final_distances,
            torch.full_like(normalized_distances_exp, torch.nan)
        )

        ######################################################

        # Update buffers
        # note: The frame names / ordering don't change so no need to update them after initialization
        self._data.source_pos_w[:] = source_pos_w.view(-1, 3)
        self._data.source_quat_w[:] = source_quat_w.view(-1, 4)
        self._data.target_pos_w[:] = target_pos_w.view(-1, total_num_frames, 3)
        self._data.target_quat_w[:] = target_quat_w.view(-1, total_num_frames, 4)
        self._data.target_pos_source[:] = target_pos_source.view(-1, total_num_frames, 3)
        self._data.target_quat_source[:] = target_quat_source.view(-1, total_num_frames, 4)
        self._data.raw_target_distances[:] = normalized_distances
        self._data.target_pos_sensor[:] = target_pos_sensor
        self._data.tof_distances[:] = tof_distances


    def make_grid(self):
        """Create quaternions representing ray directions for each pixel in the sensor grid.
        
        Output shape: (pixel_count^2, 4) - flattened grid of rotation quaternions.
        Each quaternion rotates the center ray to a pixel's ray direction.
        """
        fov_angle = self.cfg.fov_deg * (torch.pi / 180.0)
        half_fov = fov_angle / 2.0
        P = self.cfg.pixel_count
        
        # Create grid of angles (X=yaw around Y-axis, Y=pitch around X-axis)
        angles_x = torch.linspace(-half_fov, half_fov, P, device=self.device)
        angles_y = torch.linspace(half_fov, -half_fov, P, device=self.device)
        grid_X, grid_Y = torch.meshgrid(angles_x, angles_y, indexing='xy')
        
        # Flatten to (P^2,)
        ax = grid_X.reshape(-1)
        ay = grid_Y.reshape(-1)
        
        # Build rotation quaternions: yaw (around Y), then pitch (around X)
        x_axis = torch.tensor([1.0, 0.0, 0.0], device=self.device).expand(P * P, -1)
        y_axis = torch.tensor([0.0, 1.0, 0.0], device=self.device).expand(P * P, -1)
        
        quat_yaw = quat_from_angle_axis(ax, y_axis)    # (P^2, 4)
        quat_pitch = quat_from_angle_axis(ay, x_axis)  # (P^2, 4)
        
        # Combined rotation: pitch after yaw -> q_pitch * q_yaw
        self._grid_quats = self._quat_multiply(quat_pitch, quat_yaw)  # (P^2, 4)

    
    def _set_debug_vis_impl(self, debug_vis: bool):
        # set visibility of markers
        # note: parent only deals with callbacks. not their visibility
        if debug_vis:
            if not hasattr(self, "frame_visualizer"):
                self.frame_visualizer = VisualizationMarkers(self.cfg.visualizer_cfg)

            # set their visibility to true
            self.frame_visualizer.set_visibility(True)
        else:
            if hasattr(self, "frame_visualizer"):
                self.frame_visualizer.set_visibility(False)

    def _debug_vis_callback(self, event):
        """Visualize sensor ray directions as lines for each pixel in the grid."""
        
        N, S, P = self._num_envs, self._num_sensors, self.cfg.pixel_count ** 2
        
        # Calculate sensor world positions and orientations: (N, S, 3/4)
        source_pos = self._data.source_pos_w.unsqueeze(1)  # (N, 1, 3)
        source_quat = self._data.source_quat_w.unsqueeze(1)  # (N, 1, 4)
        sensor_rel_pos = self._relative_sensor_pos.unsqueeze(0)  # (1, S, 3)
        sensor_rel_quat = self._relative_sensor_quat.unsqueeze(0)  # (1, S, 4)
        
        # Compute sensor world positions: (N, S, 3)
        sensor_pos_w, sensor_quat_w = combine_frame_transforms(
            source_pos.expand(-1, S, -1).reshape(-1, 3),
            source_quat.expand(-1, S, -1).reshape(-1, 4),
            sensor_rel_pos.expand(N, -1, -1).reshape(-1, 3),
            sensor_rel_quat.expand(N, -1, -1).reshape(-1, 4),
        )
        sensor_pos_w = sensor_pos_w.view(N, S, 3)
        sensor_quat_w = sensor_quat_w.view(N, S, 4)
        
        # Get ray directions for each pixel in world frame
        # Base forward direction scaled by max_range
        forward_local = torch.tensor([0.0, 0.0, self.cfg.max_range], device=self.device)
        
        # Apply grid rotations to get local ray directions: (P, 3)
        ray_dirs_local = self._quat_rotate_vec(self._grid_quats, forward_local)
        
        # Transform to world frame for each sensor: (N, S, P, 3)
        # sensor_quat_w: (N, S, 4) -> (N, S, 1, 4)
        sensor_quat_exp = sensor_quat_w.unsqueeze(2).expand(-1, -1, P, -1)  # (N, S, P, 4)
        ray_dirs_local_exp = ray_dirs_local.unsqueeze(0).unsqueeze(0).expand(N, S, -1, -1)  # (N, S, P, 3)
        ray_dirs_w = self._quat_rotate_vec(sensor_quat_exp, ray_dirs_local_exp)  # (N, S, P, 3)
        
        # Compute start and end positions for all rays
        sensor_pos_exp = sensor_pos_w.unsqueeze(2).expand(-1, -1, P, -1)  # (N, S, P, 3)
        start_pos = sensor_pos_exp.reshape(-1, 3)  # (N*S*P, 3)
        end_pos = (sensor_pos_exp + ray_dirs_w).reshape(-1, 3)  # (N*S*P, 3)
        
        # Get line visualization
        lines_pos, lines_quat, lines_length = self._get_connecting_lines(start_pos, end_pos)
        
        num_lines = lines_pos.size(0)
        marker_scales = torch.ones(num_lines, 3, device=self.device)
        marker_scales[:, 2] = lines_length
        marker_indices = torch.ones(num_lines, device=self.device, dtype=torch.int32)

        self.frame_visualizer.visualize(
            translations=lines_pos,
            orientations=lines_quat,
            scales=marker_scales,
            marker_indices=marker_indices,
        )

    """
    Internal simulation callbacks.
    """

    def _invalidate_initialize_callback(self, event):
        """Invalidates the scene elements."""
        # call parent
        super()._invalidate_initialize_callback(event)
        # set all existing views to None to invalidate them
        self._frame_physx_view = None

    """
    Internal helpers.
    """

    def _quat_rotate_vec(self, quat: torch.Tensor, vec: torch.Tensor) -> torch.Tensor:
        """Rotate a vector by a quaternion.
        
        Args:
            quat: Quaternion(s) in (w, x, y, z) format. Shape (..., 4)
            vec: Vector(s) to rotate. Shape (3,) or (..., 3)
            
        Returns:
            Rotated vector(s). Shape (..., 3)
        """
        w = quat[..., 0:1]
        xyz = quat[..., 1:4]
        
        if vec.dim() == 1:
            vec = vec.expand(quat.shape[:-1] + (3,))
        
        t = 2.0 * torch.linalg.cross(xyz, vec)
        return vec + w * t + torch.linalg.cross(xyz, t)

    def _quat_multiply(self, q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
        """Multiply two quaternions (wxyz format): q1 * q2.
        
        Args:
            q1, q2: Quaternions in (w, x, y, z) format. Shape (..., 4)
            
        Returns:
            Product quaternion. Shape (..., 4)
        """
        w1, x1, y1, z1 = q1[..., 0], q1[..., 1], q1[..., 2], q1[..., 3]
        w2, x2, y2, z2 = q2[..., 0], q2[..., 1], q2[..., 2], q2[..., 3]
        
        return torch.stack([
            w1*w2 - x1*x2 - y1*y2 - z1*z2,
            w1*x2 + x1*w2 + y1*z2 - z1*y2,
            w1*y2 - x1*z2 + y1*w2 + z1*x2,
            w1*z2 + x1*y2 - y1*x2 + z1*w2,
        ], dim=-1)

    def _get_connecting_lines(
        self, start_pos: torch.Tensor, end_pos: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Given start and end points, compute the positions (mid-point), orientations, and lengths of the connecting lines.

        Args:
            start_pos: The start positions of the connecting lines. Shape is (N, 3).
            end_pos: The end positions of the connecting lines. Shape is (N, 3).

        Returns:
            positions: The position of each connecting line. Shape is (N, 3).
            orientations: The orientation of each connecting line in quaternion. Shape is (N, 4).
            lengths: The length of each connecting line. Shape is (N,).
        """
        direction = end_pos - start_pos
        lengths = torch.norm(direction, dim=-1)
        positions = (start_pos + end_pos) / 2

        # Get default direction (along z-axis)
        default_direction = torch.tensor([0.0, 0.0, 1.0], device=self.device).expand(start_pos.size(0), -1)

        # Normalize direction vector
        direction_norm = normalize(direction)

        # Calculate rotation from default direction to target direction
        rotation_axis = torch.linalg.cross(default_direction, direction_norm)
        rotation_axis_norm = torch.norm(rotation_axis, dim=-1)

        # Handle case where vectors are parallel
        mask = rotation_axis_norm > 1e-6
        rotation_axis = torch.where(
            mask.unsqueeze(-1),
            normalize(rotation_axis),
            torch.tensor([1.0, 0.0, 0.0], device=self.device).expand(start_pos.size(0), -1),
        )

        # Calculate rotation angle
        cos_angle = torch.sum(default_direction * direction_norm, dim=-1)
        cos_angle = torch.clamp(cos_angle, -1.0, 1.0)
        angle = torch.acos(cos_angle)
        orientations = quat_from_angle_axis(angle, rotation_axis)

        return positions, orientations, lengths
