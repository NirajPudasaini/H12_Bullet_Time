# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Environment config with CAP sensor readings integrated for RL training.

Supports ablation studies via environment variables:
    ABLATION_MAX_RANGE: Override max_range for capacitive sensors (default: 0.15)
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
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise
from isaaclab.envs import mdp 

from . import mdp as local_mdp
from h12_bullet_time.assets.robots.unitree import H12_CFG_HANDLESS
from h12_bullet_time.sensors.capacitive_sensor_cfg import CapacitiveSensorCfg
from h12_bullet_time.utils.urdf_tools import extract_sensor_poses_from_urdf


# Default parameter values (can be overridden via environment variables for ablation studies)
_DEFAULT_PROJECTILE_RADIUS = 0.15
_DEFAULT_MAX_RANGE = 0.2
_DEFAULT_DEBUG_VIS = True

# Read ablation overrides from environment variables
_projectile_radius = float(os.environ.get("ABLATION_PROJECTILE_RADIUS", _DEFAULT_PROJECTILE_RADIUS))
_max_range = float(os.environ.get("ABLATION_MAX_RANGE", _DEFAULT_MAX_RANGE))
_debug_vis = bool(os.environ.get("ABLATION_DEBUG_VIS", _DEFAULT_DEBUG_VIS))

# Log ablation configuration if any overrides are present
if any(key.startswith("ABLATION_") for key in os.environ):
    print(f"[CAP CONFIG] Ablation parameters detected:")
    print(f"  - max_range: {_max_range} (default: {_DEFAULT_MAX_RANGE})")
    print(f"  - projectile_radius: {_projectile_radius} (default: {_DEFAULT_PROJECTILE_RADIUS})")
    print(f"  - debug_vis: {_debug_vis} (default: {_DEFAULT_DEBUG_VIS})")


# Extract sensor poses from URDF
_sensor_library = extract_sensor_poses_from_urdf(H12_CFG_HANDLESS.spawn.asset_path, debug=False)

# Debug: print how many sensors were found
if not _sensor_library:
    import warnings
    warnings.warn(
        f"[CAP CONFIG] No CAP sensors found in URDF at {H12_CFG_HANDLESS.spawn.asset_path}\n"
        "Sensors will not be added to scene. Check URDF for CAP marker elements."
    )
else:
    print(f"[CAP CONFIG] Found {len(_sensor_library)} sensor locations in URDF")
    for link_path, poses in _sensor_library.items():
        print(f"  - {link_path}: {len(poses)} sensor poses")


# Build sensor configs dictionary BEFORE class definition so they can be added to the class
_sensor_configs = {}
for idx, (link_path, sensor_poses) in enumerate(_sensor_library.items()):
    # Create a valid sensor name
    sensor_name = f"cap_sensor_{link_path.replace('_skin', '').replace('_link', '')}"
    
    # Extract positions and orientations from Pose3D objects
    sensor_positions = [pose.pos for pose in sensor_poses]
    
    # Create the sensor config
    sensor_cfg = CapacitiveSensorCfg(
        prim_path=f"{{ENV_REGEX_NS}}/Robot/{link_path}",
        target_frames=[
            CapacitiveSensorCfg.FrameCfg(prim_path="{ENV_REGEX_NS}/Projectile"),
        ],
        relative_sensor_pos=sensor_positions,
        debug_vis=_debug_vis,
        max_range=_max_range,  # meters 
        projectile_radius=_projectile_radius,
    )
    
    _sensor_configs[sensor_name] = sensor_cfg
    print(f"[CAP CONFIG] Added sensor: {sensor_name} with {len(sensor_poses)} measurement points")


@configclass
class H12BulletTimeSceneCfg_CAP(InteractiveSceneCfg):
    """Configuration for H12 Bullet Time with CAP sensors."""
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
            mass_props=sim_utils.MassPropertiesCfg(mass=50.0),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(-1.0, -1.0, 0.0),
            rot=(1.0, 0.0, 0.0, 0.0),
            lin_vel=(0.0, 0.0, 0.0),
            ang_vel=(0.0, 0.0, 0.0),
        ),
    )


# Now add all sensor configs as class attributes after class is defined
print(f"[CAP CONFIG] Adding {len(_sensor_configs)} sensors to scene class...")
for sensor_name, sensor_cfg in _sensor_configs.items():
    setattr(H12BulletTimeSceneCfg_CAP, sensor_name, sensor_cfg)
    print(f"[CAP CONFIG] Successfully added: {sensor_name}")

# Debug: verify sensors were actually added
print(f"[CAP CONFIG] Scene class now has these attributes:")
for attr_name in dir(H12BulletTimeSceneCfg_CAP):
    if 'cap' in attr_name.lower() or 'sensor' in attr_name.lower():
        print(f"  - {attr_name}")

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
        
        # Projectile observations
        # projectile_pos_rel = ObsTerm(
        #     func=local_mdp.projectile_position_relative,
        #     scale=0.25,
        # )
        # projectile_vel = ObsTerm(
        #     func=local_mdp.projectile_velocity,
        #     scale=0.1,
        # )
        # projectile_dist = ObsTerm(
        #     func=local_mdp.projectile_distance_obs,
        #     scale=0.5,
        # )
        
        # CAP sensor readings: distance measurements from each sensor
        cap_distances_obs = ObsTerm(
            func=local_mdp.cap_distances_obs,
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
        
        # Projectile observations
        # projectile_pos_rel = ObsTerm(
        #     func=local_mdp.projectile_position_relative,
        #     scale=0.25,
        # )
        # projectile_vel = ObsTerm(
        #     func=local_mdp.projectile_velocity,
        #     scale=0.1,
        # )
        # projectile_dist = ObsTerm(
        #     func=local_mdp.projectile_distance_obs,
        #     scale=0.5,
        # )
        
        # CAP sensor readings: distance measurements from each sensor
        cap_distances_obs = ObsTerm(
            func=local_mdp.cap_distances_obs,
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

    # Phase 2 reward: Projectile avoidance penalty
    # Starts at weight=0.0, transitions to 1.0 at curriculum milestone
    # projectile_penalty = RewTerm(
    #     func=local_mdp.projectile_proximity_penalty,
    #     weight=1.0,
    #     params={
    #         "asset_cfg": SceneEntityCfg("robot"),
    #         "projectile_name": "Projectile",
    #         "max_distance": 3.0,
    #         "penalty_scale": -30.0,
    #     },
    # )

    cap_distances_penalty = RewTerm(
        func=local_mdp.cap_distances_penalty,
        weight=5.0,
        params={"proximity_scale": -0.01, "contact_scale": -1.0, "contact_threshold": 0.1},
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
    launch_projectile = EventTerm(
        func=local_mdp.launch_projectile_radial,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("Projectile"),
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
        params={"asset_cfg": SceneEntityCfg("robot"), "threshold": 0.4},
    )


##
# Environment configuration
##

@configclass
class H12BulletTimeEnvCfg_CAP(ManagerBasedRLEnvCfg):
    """RL environment config with CAP sensor integration."""
    
    # Scene settings
    scene: H12BulletTimeSceneCfg_CAP = H12BulletTimeSceneCfg_CAP(num_envs=4096, env_spacing=4.0)
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