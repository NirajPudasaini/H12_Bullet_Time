# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Environment config with TOF and CAP sensor readings integrated for RL training.

Supports ablation studies via environment variables:
    ABLATION_MAX_RANGE: Override max_range for TOF and CAP sensors (default: 0.15)
    ABLATION_PROJECTILE_RADIUS: Override projectile radius (default: 0.15)
"""

import math
import os

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import CurriculumTermCfg as CurTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise
from isaaclab.envs import mdp 

from . import mdp as local_mdp
from h12_bullet_time.assets.robots.unitree import H12_CFG_HANDLESS
from h12_bullet_time.sensors.capacitive_sensor_cfg import CapacitiveSensorCfg
from h12_bullet_time.sensors.tof_sensor_cfg import TofSensorCfg
from h12_bullet_time.utils.urdf_tools import extract_sensor_poses_from_urdf


# Default parameter values (can be overridden via environment variables for ablation studies)
_DEFAULT_PROJECTILE_RADIUS = 0.15
_DEFAULT_MAX_RANGE = 0.1
_DEFAULT_DEBUG_VIS = True
_DEFAULT_SENSOR_TYPE = "CAP"
_DEFAULT_PROXIMITY_SCALE = -0.01
_DEFAULT_CONTACT_SCALE = -0.1
_DEFAULT_CONTACT_THRESHOLD = 0.03 # (%) of sensor range
_DEFAULT_PROJECTILE_MASS = 0.1
_DEFAULT_CONTACT_TERMINATION = True
_DEFAULT_TERMINATION_ANGLE_THRESHOLD_DEG = 60
_DEFAULT_TERMINATION_HEIGHT_THRESHOLD = 0.4
_DEFAULT_PROJECTILE_MIN_SPAWN_DIST = 2.0
_DEFAULT_PROJECTILE_MAX_SPAWN_DIST = 3.0
_DEFAULT_PROJECTILE_MIN_HEIGHT = 1.0
_DEFAULT_PROJECTILE_MAX_HEIGHT = 3.0
_DEFAULT_PROJECTILE_MIN_SPEED = 4.0
_DEFAULT_PROJECTILE_MAX_SPEED = 6.0
# Read ablation overrides from environment variables
_projectile_radius = float(os.environ.get("ABLATION_PROJECTILE_RADIUS", _DEFAULT_PROJECTILE_RADIUS))
_max_range = float(os.environ.get("ABLATION_MAX_RANGE", _DEFAULT_MAX_RANGE))
_debug_vis = bool(os.environ.get("ABLATION_DEBUG_VIS", _DEFAULT_DEBUG_VIS))
_sensor_type = os.environ.get("ABLATION_SENSOR_TYPE", _DEFAULT_SENSOR_TYPE)
_proximity_scale = float(os.environ.get("ABLATION_PROXIMITY_SCALE", _DEFAULT_PROXIMITY_SCALE))
_contact_scale = float(os.environ.get("ABLATION_CONTACT_SCALE", _DEFAULT_CONTACT_SCALE))
_contact_threshold = float(os.environ.get("ABLATION_CONTACT_THRESHOLD", _DEFAULT_CONTACT_THRESHOLD))
_projectile_mass = float(os.environ.get("ABLATION_PROJECTILE_MASS", _DEFAULT_PROJECTILE_MASS))
_contact_termination = bool(os.environ.get("ABLATION_CONTACT_TERMINATION", _DEFAULT_CONTACT_TERMINATION))
_track_contact_points = os.environ.get("ABLATION_TRACK_CONTACT_POINTS", "").lower() in ("1", "true", "yes")
_termination_angle_threshold_deg = float(os.environ.get("ABLATION_TERMINATION_ANGLE_THRESHOLD_DEG", _DEFAULT_TERMINATION_ANGLE_THRESHOLD_DEG))
_termination_height_threshold = float(os.environ.get("ABLATION_TERMINATION_HEIGHT_THRESHOLD", _DEFAULT_TERMINATION_HEIGHT_THRESHOLD))
_projectile_min_spawn_dist = float(os.environ.get("ABLATION_PROJECTILE_MIN_SPAWN_DIST", _DEFAULT_PROJECTILE_MIN_SPAWN_DIST))
_projectile_max_spawn_dist = float(os.environ.get("ABLATION_PROJECTILE_MAX_SPAWN_DIST", _DEFAULT_PROJECTILE_MAX_SPAWN_DIST))
_projectile_min_height = float(os.environ.get("ABLATION_PROJECTILE_MIN_HEIGHT", _DEFAULT_PROJECTILE_MIN_HEIGHT))
_projectile_max_height = float(os.environ.get("ABLATION_PROJECTILE_MAX_HEIGHT", _DEFAULT_PROJECTILE_MAX_HEIGHT))
_projectile_min_speed = float(os.environ.get("ABLATION_PROJECTILE_MIN_SPEED", _DEFAULT_PROJECTILE_MIN_SPEED))
_projectile_max_speed = float(os.environ.get("ABLATION_PROJECTILE_MAX_SPEED", _DEFAULT_PROJECTILE_MAX_SPEED))
# Log ablation configuration if any overrides are present
if any(key.startswith("ABLATION_") for key in os.environ):
    print(f"[{_sensor_type.upper()} CONFIG] Ablation parameters detected:")
    print(f"  - max_range: {_max_range} (default: {_DEFAULT_MAX_RANGE})")
    print(f"  - projectile_radius: {_projectile_radius} (default: {_DEFAULT_PROJECTILE_RADIUS})")
    print(f"  - debug_vis: {_debug_vis} (default: {_DEFAULT_DEBUG_VIS})")
    print(f"  - sensor_type: {_sensor_type} (default: {_DEFAULT_SENSOR_TYPE})")
    print(f"  - proximity_scale: {_proximity_scale} (default: {_DEFAULT_PROXIMITY_SCALE})")
    print(f"  - contact_scale: {_contact_scale} (default: {_DEFAULT_CONTACT_SCALE})")
    print(f"  - contact_threshold: {_contact_threshold} (default: {_DEFAULT_CONTACT_THRESHOLD})")
    print(f"  - projectile_mass: {_projectile_mass} (default: {_DEFAULT_PROJECTILE_MASS})")
    print(f"  - contact_termination: {_contact_termination} (default: {_DEFAULT_CONTACT_TERMINATION})")
    print(f"  - termination_angle_threshold_deg: {_termination_angle_threshold_deg} (default: {_DEFAULT_TERMINATION_ANGLE_THRESHOLD_DEG})")
    print(f"  - termination_height_threshold: {_termination_height_threshold} (default: {_DEFAULT_TERMINATION_HEIGHT_THRESHOLD})")
    print(f"  - projectile_min_spawn_dist: {_projectile_min_spawn_dist} (default: {_DEFAULT_PROJECTILE_MIN_SPAWN_DIST})")
    print(f"  - projectile_max_spawn_dist: {_projectile_max_spawn_dist} (default: {_DEFAULT_PROJECTILE_MAX_SPAWN_DIST})")
    print(f"  - projectile_min_height: {_projectile_min_height} (default: {_DEFAULT_PROJECTILE_MIN_HEIGHT})")
    print(f"  - projectile_max_height: {_projectile_max_height} (default: {_DEFAULT_PROJECTILE_MAX_HEIGHT})")
    print(f"  - projectile_min_speed: {_projectile_min_speed} (default: {_DEFAULT_PROJECTILE_MIN_SPEED})")
    print(f"  - projectile_max_speed: {_projectile_max_speed} (default: {_DEFAULT_PROJECTILE_MAX_SPEED})")
# Extract sensor poses from URDF
_sensor_library = extract_sensor_poses_from_urdf(H12_CFG_HANDLESS.spawn.asset_path, debug=False)

# Debug: print how many sensors were found
if not _sensor_library:
    import warnings
    warnings.warn(
        f"[{_sensor_type.upper()} CONFIG] No {_sensor_type} sensors found in URDF at {H12_CFG_HANDLESS.spawn.asset_path}\n"
        "Sensors will not be added to scene. Check URDF for {_sensor_type} marker elements."
    )
else:
    print(f"[{_sensor_type.upper()} CONFIG] Found {len(_sensor_library)} sensor locations in URDF")
    for link_path, poses in _sensor_library.items():
        print(f"  - {link_path}: {len(poses)} sensor poses")


# Build sensor configs dictionary BEFORE class definition so they can be added to the class
if ("CAP" in _sensor_type) and ("TOF" in _sensor_type) and (_max_range > 0.01):
    _cap_max_range = _max_range
    _tof_max_range = 4.0 # default TOF max range to the realistic value
else:
    _cap_max_range = _max_range
    _tof_max_range = _max_range

_sensor_configs = {}
for idx, (link_path, sensor_poses) in enumerate(_sensor_library.items()):
    # Create a valid sensor name
    sensor_name = f"{_sensor_type.lower()}_sensor_{link_path.replace('_skin', '').replace('_link', '')}"
    
    # Extract positions and orientations from Pose3D objects
    sensor_positions = [pose.pos for pose in sensor_poses]
    sensor_orientations = [pose.quat for pose in sensor_poses]

    # Create the sensor config
    if "CAP" in _sensor_type:
        cap_sensor_name = f"cap_sensor_{link_path.replace('_skin', '').replace('_link', '')}"
        cap_sensor_cfg = CapacitiveSensorCfg(
            prim_path=f"{{ENV_REGEX_NS}}/Robot/{link_path}",
            target_frames=[
                CapacitiveSensorCfg.FrameCfg(prim_path="{ENV_REGEX_NS}/Projectile"),
            ],
            relative_sensor_pos=sensor_positions,
            debug_vis=_debug_vis,
            max_range=_cap_max_range,
            projectile_radius=_projectile_radius,
        )
        _sensor_configs[cap_sensor_name] = cap_sensor_cfg
        print(f"[CAP CONFIG] Added sensor: {cap_sensor_name} with {len(sensor_poses)} points")
    
    # Create TOF sensor if enabled
    if "TOF" in _sensor_type:
        tof_sensor_name = f"tof_sensor_{link_path.replace('_skin', '').replace('_link', '')}"
        tof_sensor_cfg = TofSensorCfg(
            prim_path=f"{{ENV_REGEX_NS}}/Robot/{link_path}",
            target_frames=[
                TofSensorCfg.FrameCfg(prim_path="{ENV_REGEX_NS}/Projectile"),
            ],
            relative_sensor_pos=sensor_positions,
            relative_sensor_quat=sensor_orientations,
            debug_vis=_debug_vis,
            max_range=_tof_max_range,
            projectile_radius=_projectile_radius,
        )
        _sensor_configs[tof_sensor_name] = tof_sensor_cfg
        print(f"[TOF CONFIG] Added sensor: {tof_sensor_name} with {len(sensor_poses)} points")


@configclass
class H12BulletTimeSceneCfg_HYBRID_ALL_CONTACT(InteractiveSceneCfg):
    f"""Configuration for H12 Bullet Time with {_sensor_type.upper()} sensors."""
    # ground plane
    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(size=(100.0, 100.0)),
    )

    # robot - H12 humanoid
    robot: ArticulationCfg = H12_CFG_HANDLESS.replace(prim_path="{ENV_REGEX_NS}/Robot")
   
    # lights
    dome_light = AssetBaseCfg(
        prim_path="/World/DomeLight",
        spawn=sim_utils.DomeLightCfg(color=(0.9, 0.9, 0.9), intensity=500.0),
    )

    # Projectile (always in scene, but only spawned after curriculum milestone)
    Projectile = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Projectile",
        spawn=sim_utils.SphereCfg(
            radius=_projectile_radius,  
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.0, 0.0, 0.2),  # Blue
                metallic=0.2,
            ),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                solver_position_iteration_count=4,
                solver_velocity_iteration_count=0,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=_projectile_mass),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(-1.0, -1.0, 0.0),
            rot=(1.0, 0.0, 0.0, 0.0),
            lin_vel=(0.0, 0.0, 0.0),
            ang_vel=(0.0, 0.0, 0.0),
        ),
    )


# List of robot links with collision geometry for contact detection
# NOTE: Only links with <collision> elements in the URDF are included
_CONTACT_DETECTION_LINKS = [
    # Torso
    "torso_link",
    "pelvis",
    # Left leg
    "left_hip_pitch_link",
    "left_hip_roll_link",
    "left_knee_link",
    "left_ankle_pitch_link",
    "left_ankle_roll_link",
    # Right leg
    "right_hip_pitch_link",
    "right_hip_roll_link",
    "right_knee_link",
    "right_ankle_pitch_link",
    "right_ankle_roll_link",
    # Left arm
    "left_shoulder_pitch_link",
    "left_shoulder_roll_link",
    "left_shoulder_yaw_link",
    "left_elbow_link",
    "left_wrist_roll_link",
    "left_wrist_pitch_link",
    # Right arm
    "right_shoulder_pitch_link",
    "right_shoulder_roll_link",
    "right_shoulder_yaw_link",
    "right_elbow_link",
    "right_wrist_roll_link",
    "right_wrist_pitch_link",
]

# Build contact sensor configs dictionary
_contact_sensor_configs = {}
_contact_sensor_names = []  # Store names for termination function

for link_name in _CONTACT_DETECTION_LINKS:
    # Create sensor name from link name (e.g., "torso_link" -> "contact_torso_link")
    sensor_name = f"contact_{link_name}"
    _contact_sensor_names.append(sensor_name)
    
    # Create the contact sensor config
    _contact_sensor_configs[sensor_name] = ContactSensorCfg(
        prim_path=f"{{ENV_REGEX_NS}}/Robot/{link_name}",
        filter_prim_paths_expr=["{ENV_REGEX_NS}/Projectile"],
        update_period=0.0,
        history_length=1,
        force_threshold=1.0,
        track_contact_points=_track_contact_points,
        max_contact_data_count_per_prim=4,
    )

print(f"[CONTACT CONFIG] Created {len(_contact_sensor_configs)} contact sensors for links: {_CONTACT_DETECTION_LINKS}")
print(f"[CONTACT CONFIG] Sensor names: {_contact_sensor_names}")


# Now add all sensor configs as class attributes after class is defined
# Add proximity sensors (CAP/TOF)
print(f"[{_sensor_type.upper()} CONFIG] Adding {len(_sensor_configs)} proximity sensors to scene class...")
for sensor_name, sensor_cfg in _sensor_configs.items():
    setattr(H12BulletTimeSceneCfg_HYBRID_ALL_CONTACT, sensor_name, sensor_cfg)
    print(f"[{_sensor_type.upper()} CONFIG] Successfully added: {sensor_name}")

# Add contact sensors for projectile collision detection
print(f"[CONTACT CONFIG] Adding {len(_contact_sensor_configs)} contact sensors to scene class...")
for sensor_name, sensor_cfg in _contact_sensor_configs.items():
    setattr(H12BulletTimeSceneCfg_HYBRID_ALL_CONTACT, sensor_name, sensor_cfg)
print(f"[CONTACT CONFIG] Successfully added contact sensors: {_contact_sensor_names}")

##
# MDP settings
##

@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    joint_effort = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=[
            # Left leg
            "left_hip_yaw_joint",
            "left_hip_roll_joint",
            "left_hip_pitch_joint",
            "left_knee_joint",        #6      
            "left_ankle_pitch_joint", #0
            "left_ankle_roll_joint",  #1

            # Right leg
            "right_hip_yaw_joint",   #5
            "right_hip_roll_joint",  #4
            "right_hip_pitch_joint", #3
            "right_knee_joint",
            "right_ankle_pitch_joint",
            "right_ankle_roll_joint",

            #torso
            "torso_joint",

            #wrists are ignored !! yeah dodging with upper body only no wrist movement
            #shoulder yaw also ignored
            #Left arm
            "left_shoulder_pitch_joint", #7
            "left_shoulder_roll_joint",  #8
            "left_shoulder_yaw_joint",  #8

            "left_elbow_joint",   #2

            # Right arm
            "right_shoulder_pitch_joint",   
            "right_shoulder_roll_joint",   
            "right_shoulder_yaw_joint",   
            "right_elbow_joint",
        ],
        scale= 0.25, # change this scaling to make it 
        # scale= 1.0,
    )

@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group (actor)."""

        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, scale=0.2, noise=Unoise(n_min=-0.2, n_max=0.2))
        projected_gravity = ObsTerm(func=mdp.projected_gravity, noise=Unoise(n_min=-0.05, n_max=0.05))
        joint_pos_rel = ObsTerm(func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.2, n_max=0.2))
        joint_vel_rel = ObsTerm(func=mdp.joint_vel_rel, noise=Unoise(n_min=-0.01, n_max=0.01))
        last_action = ObsTerm(func=mdp.last_action)

        # distances_obs = ObsTerm(
        #     func=local_mdp.distances_obs,
        #     scale=0.25,
        # )
        min_distances_obs = ObsTerm(
            func=local_mdp.min_distances_obs,
            scale=0.25,
        )
        
        def __post_init__(self) -> None:
            self.enable_corruption = True
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()
    
    @configclass
    class CriticCfg(ObsGroup):
        """Observations for critic group (value function, privileged access)."""
        
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, scale=0.2, noise=Unoise(n_min=-0.2, n_max=0.2))
        projected_gravity = ObsTerm(func=mdp.projected_gravity, noise=Unoise(n_min=-0.05, n_max=0.05))
        joint_pos_rel = ObsTerm(func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.2, n_max=0.2))
        joint_vel_rel = ObsTerm(func=mdp.joint_vel_rel, noise=Unoise(n_min=-0.01, n_max=0.01))
        last_action = ObsTerm(func=mdp.last_action)
        
        # Privileged info: linear velocity
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel, scale=0.1)
        

        distances_obs = ObsTerm(
            func=local_mdp.distances_obs,
            scale=0.25,
        )
        
        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = True

    critic: CriticCfg = CriticCfg()


@configclass
class RewardsCfg:

    # Phase 1 rewards (always active): Stand and balance
    base_height = RewTerm(
        func=local_mdp.base_height_l2,
        weight=10.0,
        params={"asset_cfg": SceneEntityCfg("robot"), "target_height": 1.04},
    )

    alive_bonus = RewTerm(
        func=local_mdp.alive_bonus,
        weight=5.0,
        params={},
    )

    flat_orientation_l2 = RewTerm(func=mdp.flat_orientation_l2, weight=-1.0)
    joint_acc = RewTerm(func=mdp.joint_acc_l2, weight=-2.5e-7)
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.005)
    dof_pos_limits = RewTerm(func=mdp.joint_pos_limits, weight=-3.0)

    # Standing still reward
    base_velocity_reward = RewTerm(
        func=local_mdp.base_velocity_reward,
        weight=10,
        params={"asset_cfg": SceneEntityCfg("robot"), "scale": 100.0},
    )

    distances_penalty = RewTerm(
        func=local_mdp.distances_penalty,
        weight=5.0,
        params={"proximity_scale": _proximity_scale, "contact_scale": _contact_scale, "contact_threshold": _contact_threshold},
    )

@configclass
class EventCfg:
    """Configuration for events."""

    # Reset base position and velocity
    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.0, 0.0), "y": (-0.0, 0.0), "yaw": (0.0, 0.0)},
            "velocity_range": {
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (0.0, 0.0),
            },
        },
    )

    # Reset robot joints
    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={
            "position_range": (1.0, 1.0),
            "velocity_range": (-1.0, 1.0),
        },
    )

    # Launch projectiles on reset with varied positions and angles
    # launch_projectile = EventTerm(
    #     func=local_mdp.launch_projectile_radial,
    #     mode="reset",
    #     params={
    #         "asset_cfg": SceneEntityCfg("Projectile"),
    #     },
    # )
    launch_projectile = EventTerm(
        func=local_mdp.launch_projectile_target_sampling,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("Projectile"),
            "min_spawn_dist": _projectile_min_spawn_dist,
            "max_spawn_dist": _projectile_max_spawn_dist,
            "min_height": _projectile_min_height,
            "max_height": _projectile_max_height,
            "min_speed": _projectile_min_speed,
            "max_speed": _projectile_max_speed,
        },
    )

    # Debug: log TOF readings at reset to verify sensors (DISABLED for multi-env compatibility)
    # log_tof = EventTerm(
    #     func=local_mdp.print_tof_readings,
    #     mode="reset",
    #     params={},
    # )

@configclass
class CurriculumCfg:
    """Curriculum manager configuration (empty: projectile penalty active from start)."""
    pass


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    # Time out
    time_out = DoneTerm(func=mdp.time_out, time_out=True)

    # Base height too low (fell down)
    base_height_low = DoneTerm(
        func=local_mdp.base_height_below_threshold,
        params={"asset_cfg": SceneEntityCfg("robot"), "threshold": _termination_height_threshold},
    )
    bad_orientation = DoneTerm(
        func=local_mdp.bad_orientation,
        params={"asset_cfg": SceneEntityCfg("robot"), "angle_threshold_deg": _termination_angle_threshold_deg},
    )

    # Contact termination using physics-based contact detection
    # This uses multiple ContactSensors to detect actual surface-to-surface collisions
    # between robot bodies and the projectile
    if _contact_termination:
        contact_termination = DoneTerm(
            func=local_mdp.multi_contact_termination,
            params={
                "sensor_names": _contact_sensor_names,
                "threshold": _contact_threshold,  # Force threshold in Newtons
            },
        )
    else:
        contact_termination = None

##
# Environment configuration
##

@configclass
class H12BulletTimeEnvCfg_HYBRID_ALL_CONTACT(ManagerBasedRLEnvCfg):
    """RL environment config with {_sensor_type.upper()} sensor integration."""
    
    # Scene settings
    scene: H12BulletTimeSceneCfg_HYBRID_ALL_CONTACT = H12BulletTimeSceneCfg_HYBRID_ALL_CONTACT(num_envs=4096, env_spacing=4.0)
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    events: EventCfg = EventCfg()
    # MDP settings
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    # Curriculum settings
    curriculum: CurriculumCfg = CurriculumCfg()

    # Post initialization
    def __post_init__(self) -> None:
        """Post initialization."""
        # general settings
        self.decimation = 2
        self.episode_length_s = 3  # 5 second episodes
        # viewer settings
        self.viewer.eye = (8.0, 0.0, 5.0)
        # simulation settings
        self.sim.dt = 1 / 120
        self.sim.render_interval = self.decimation