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


from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise


from . import mdp
from h12_bullet_time.assets.robots.unitree import H12_CFG_HANDLESS
# print(H12_CFG_HANDLESS.spawn.usd_path)
# exit()

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

    # H12 has 19 DOF: 6 per leg + 3 per arm (shoulder pitch/roll, elbow) + torso too ! ~ ignored wrist and shoulder yaw
    # We control: hip yaw/pitch/roll, knee, ankle pitch/roll for each leg
    # and shoulder pitch/roll, elbow for each arm and torso joint

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
            "left_elbow_joint",   #2

            # Right arm
            "right_shoulder_pitch_joint",   
            "right_shoulder_roll_joint",   
            "right_elbow_joint",
        ],
        scale= 0.25,  
    )


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        # observation terms (order preserved)
        # currently no noise added? and no scaling ?
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, scale = 0.2, noise=Unoise(n_min=-0.2, n_max=0.2))
        projected_gravity = ObsTerm(func=mdp.projected_gravity, noise=Unoise(n_min=-0.05, n_max=0.05))
        joint_pos_rel = ObsTerm(func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.2, n_max=0.2))
        joint_vel_rel = ObsTerm(func=mdp.joint_vel_rel, noise=Unoise(n_min=-0.01, n_max=0.01))
        last_action = ObsTerm(func=mdp.last_action)
        
        def __post_init__(self) -> None:
            # self.history_length = 5
            self.enable_corruption = True
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()
    
    @configclass
    class CriticCfg(ObsGroup):
        """Observations for critic group."""
        pass

    # NEED TO FIX THIS LATER ~ when adding camera depths    

    #     # observation terms (order preserved)
    #     # currently no noise added? and no scaling ?
    #     base_ang_vel = ObsTerm(func=mdp.base_ang_vel, scale = 0.2, noise=Unoise(n_min=-0.2, n_max=0.2))
    #     projected_gravity = ObsTerm(func=mdp.projected_gravity, noise=Unoise(n_min=-0.05, n_max=0.05))
    #     joint_pos_rel = ObsTerm(func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.2, n_max=0.2))
    #     joint_vel_rel = ObsTerm(func=mdp.joint_vel_rel, noise=Unoise(n_min=-0.01, n_max=0.01))
    #     last_action = ObsTerm(func=mdp.last_action)
        
    #     def __post_init__(self) -> None:
    #         self.history_length = 5
    #         self.enable_corruption = False
    #         self.concatenate_terms = True
    # # privileged observations
    # critic: CriticCfg = CriticCfg()


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    # Minimal reward: maintain base height at 1.04 m
    base_height = RewTerm(
        func=mdp.base_height_l2,
        weight= -10.0,
        params={"asset_cfg": SceneEntityCfg("robot"), "target_height": 1.04},
    )

    # Alive bonus: reward for staying alive (not falling)
    alive_bonus = RewTerm(
        func=mdp.alive_bonus,
        weight= 5.0,
        params={},
    )

    # # Knee symmetry: encourage left and right knees to maintain similar angles
    # knee_symmetry = RewTerm(
    #     func=mdp.knee_symmetry,
    #     weight= 0.2,
    #     params={"asset_cfg": SceneEntityCfg("robot")},
    # )

    # # Penalty when projectile hits the robot (useful for simple dodge training)
    # projectile_penalty = RewTerm(
    #     func=mdp.projectile_hit_penalty,
    #     weight=1.0,
    #     params={"asset_cfg": SceneEntityCfg("robot"), "penalty": -10.0, "threshold": 0.25},
    # )


@configclass
class EventCfg:
    """Configuration for events."""

    # Reset base position and velocity on episode reset/termination
    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},
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

    # Reset robot joints to default positions on episode reset/termination
    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={
            "position_range": (1.0, 1.0),
            "velocity_range": (-1.0, 1.0),
        },
    )

@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    # (1) Time out
    time_out = DoneTerm(func=mdp.time_out, time_out=True)

    # (2) Base height too low (fell down)
    base_height_low = DoneTerm(
        func=mdp.base_height_below_threshold,
        params={"asset_cfg": SceneEntityCfg("robot"), "threshold": 0.4},
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
