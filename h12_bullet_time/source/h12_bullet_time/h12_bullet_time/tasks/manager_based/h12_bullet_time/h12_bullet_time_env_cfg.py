# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import math

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass

from . import mdp
from h12_bullet_time.assets.robots.unitree import H12_CFG_HANDLESS

##
# Scene definition
##


@configclass
class H12BulletTimeSceneCfg(InteractiveSceneCfg):
    """Configuration for H12 Bullet Time scene with projectile dodging."""

    # ground plane
    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(size=(100.0, 100.0)),
    )

    # robot - H12 humanoid
    robot: ArticulationCfg = H12_CFG_HANDLESS.replace(prim_path="{ENV_REGEX_NS}/Robot")
    
    # projectile objects (spheres for simplicity)
    projectile: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Projectile",
        spawn=sim_utils.SphereCfg(
            radius=0.1,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                # mass=2.0,
                disable_gravity=False,
                max_linear_velocity=100.0,
                max_angular_velocity=100.0,
            ),
            material_props=sim_utils.RigidBodyMaterialCfg(
                static_friction=0.5,
                dynamic_friction=0.5,
                restitution=0.8,
            ),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(10.0, 0.0, 1.5),
            vel=(0.0, 0.0, 0.0),
        ),
    )

    # lights
    dome_light = AssetBaseCfg(
        prim_path="/World/DomeLight",
        spawn=sim_utils.DomeLightCfg(color=(0.9, 0.9, 0.9), intensity=500.0),
    )


##
# MDP settings
##


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    # H12 has 19 DOF: 6 per leg + 3 per arm (shoulder pitch/roll, elbow) + torso too ! 
    # We control: hip yaw/pitch/roll, knee, ankle pitch/roll for each leg
    # and shoulder pitch/roll, elbow for each arm and torso joint

    joint_effort = mdp.JointEffortActionCfg(
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

            # Left arm
            "left_shoulder_pitch_joint", #7
            "left_shoulder_roll_joint",  #8
            "left_elbow_joint",   #2

            # Right arm
            "right_shoulder_pitch_joint",   
            "right_shoulder_roll_joint",   
            "right_elbow_joint",
        ],
        scale= 0.25, #unitree rl lab uses 0.25 ? !! might need to change later
    )


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        # robot state observations
        robot_base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
        robot_base_ang_vel = ObsTerm(func=mdp.base_ang_vel)
        robot_base_euler = ObsTerm(func=mdp.base_euler_angles)
        
        # joint observations
        joint_pos_rel = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel_rel = ObsTerm(func=mdp.joint_vel_rel)
        
        # center of mass observations
        robot_com_height = ObsTerm(func=mdp.com_height)
        
        # projectile relative observations
        projectile_relative_pos = ObsTerm(func=mdp.projectile_relative_pos)
        projectile_relative_vel = ObsTerm(func=mdp.projectile_relative_vel)

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    """Configuration for events."""

    # reset robot joints
    reset_robot_joints = EventTerm(
        func=mdp.reset_robot_to_standing,
        mode="reset",
        params={},
    )
    
    # spawn projectile randomly
    spawn_projectile = EventTerm(
        func=mdp.spawn_projectile_randomly,
        mode="startup",
        params={
            "projectile_cfg": SceneEntityCfg("projectile"),
            "robot_cfg": SceneEntityCfg("robot"),
        },
    )


@configclass
class RewardsCfg:
    """Reward terms for the MDP - Curriculum based."""

    # ============ STAGE 1: Standing and Balance ============
    # Reward for maintaining upright posture
    is_alive = RewTerm(
        func=mdp.is_alive,
        weight=1.0,
        params={"threshold": 0.65},  # 60cm COM height threshold
    )
    
    # Penalize falling
    fallen = RewTerm(
        func=mdp.is_fallen,
        weight=-5.0,
        params={"threshold": 0.50},  # 50cm COM height means fallen
    )
    
    # Reward for upright base orientation (Stage 1)
    upright_torso = RewTerm(
        func=mdp.upright_torso,
        weight=2.0,
    )
    
    # Penalize excessive joint velocities (Stage 1)
    smooth_joints = RewTerm(
        func=mdp.joint_vel_penalty,
        weight=-0.01,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )

    # ============ STAGE 2: Height Control & Agility ============
    # Reward for height adjustment (crouching and standing)
    height_control = RewTerm(
        func=mdp.height_control,
        weight=0.5,
        params={"target_height": 1.03},
    )
    
    # Reward lateral stability
    lateral_stability = RewTerm(
        func=mdp.lateral_stability,
        weight=0.5,
    )

    # ============ STAGE 3: Obstacle Dodging ============
    # Reward for staying away from projectile
    dodge_reward = RewTerm(
        func=mdp.dodge_projectile,
        weight=1.0,
        params={
            "projectile_cfg": SceneEntityCfg("projectile"),
            "robot_cfg": SceneEntityCfg("robot"),
            "safe_distance": 0.3,  # 30cm safe distance
        },
    )
    
    # Penalty for collision with projectile
    projectile_collision = RewTerm(
        func=mdp.projectile_collision_penalty,
        weight=-10.0,
        params={
            "projectile_cfg": SceneEntityCfg("projectile"),
            "robot_cfg": SceneEntityCfg("robot"),
            "collision_distance": 0.15,  # 15cm collision threshold
        },
    )
    
    # Reward movement (displacement) for dodging
    movement_reward = RewTerm(
        func=mdp.movement_reward,
        weight=0.5,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    # Time out
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    
    # Robot fell down
    robot_fallen = DoneTerm(
        func=mdp.is_fallen,
        params={"threshold": 0.3},
    )
    
    # Robot went out of bounds
    robot_out_of_bounds = DoneTerm(
        func=mdp.robot_out_of_bounds,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "bounds": (-2.0, 2.0, -2.0, 2.0),  # x_min, x_max, y_min, y_max
        },
    )


##
# Environment configuration
##


@configclass
class H12BulletTimeEnvCfg(ManagerBasedRLEnvCfg):
    # Scene settings
    scene: H12BulletTimeSceneCfg = H12BulletTimeSceneCfg(num_envs=4096, env_spacing=4.0)
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    events: EventCfg = EventCfg()
    # MDP settings
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()

    # Post initialization
    def __post_init__(self) -> None:
        """Post initialization."""
        # general settings
        self.decimation = 2
        self.episode_length_s = 10  # 10 second episodes
        # viewer settings
        self.viewer.eye = (8.0, 0.0, 5.0)
        # simulation settings
        self.sim.dt = 1 / 120
        self.sim.render_interval = self.decimation


##
# Curriculum Configuration
##


@configclass
class CurriculumCfg:
    """Curriculum learning configuration."""
    
    # Stage durations (in environment steps)
    stage_1_steps = 1_000_000  # Standing and balance
    stage_2_steps = 2_000_000  # Height control and agility
    stage_3_steps = 3_000_000  # Projectile dodging
    
    # Environment configuration variants for each stage
    stage_1_cfg = H12BulletTimeEnvCfg()
    stage_2_cfg = H12BulletTimeEnvCfg()
    stage_3_cfg = H12BulletTimeEnvCfg()