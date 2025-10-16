# Copyright (c) 2025, Unitree Robotics Co., Ltd. All Rights Reserved.
# License: Apache License, Version 2.0  
"""Configuration for Unitree robots."""


#for armatures: https://github.com/correlllab/h12-lab-docs/blob/main/docs/specs.md

import os

project_root = "/home/niraj/isaac_projects/H12_Obstacle_Aware_Locomotion/h12_obstacle_aware_locomotion/source/h12_obstacle_aware_locomotion/h12_obstacle_aware_locomotion"

import isaaclab.sim as sim_utils
from isaaclab.actuators import ActuatorNetMLPCfg, DCMotorCfg, ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg
#from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR
# from unitree_actuators import (
#     UnitreeActuatorCfg_N7520_14p3,
#     UnitreeActuatorCfg_N7520_22p5,
#     UnitreeActuatorCfg_N5020_16,
#     UnitreeActuatorCfg_M107_15,
#     UnitreeActuatorCfg_M107_24,
# )

#SET ARMATURES ~ can be found in unitree_actuators.py
UnitreeActuatorCfg_N7520_14p3 = 0.01017752
UnitreeActuatorCfg_N7520_22p5 = 0.025101925
UnitreeActuatorCfg_N5020_16 = 0.003609725
UnitreeActuatorCfg_M107_15 = 0.063259741
UnitreeActuatorCfg_M107_24 = 0.160478022

import math
NATURAL_FREQ = 10 * 2.0 * math.pi  # 10Hz
DAMPING_RATIO = 2.0

# Calculate Stiffness (Kp) for each motor type
STIFFNESS_M107_24 = UnitreeActuatorCfg_M107_24 * (NATURAL_FREQ**2)
STIFFNESS_M107_15 = UnitreeActuatorCfg_M107_15 * (NATURAL_FREQ**2)
STIFFNESS_N7520_22p5 = UnitreeActuatorCfg_N7520_22p5 * (NATURAL_FREQ**2)
STIFFNESS_N5020_16 = UnitreeActuatorCfg_N5020_16 * (NATURAL_FREQ**2)
STIFFNESS_N7520_14p3 = UnitreeActuatorCfg_N7520_14p3 * (NATURAL_FREQ**2)

# Calculate Damping (Kd) for each motor type
DAMPING_M107_24 = 2.0 * DAMPING_RATIO * UnitreeActuatorCfg_M107_24 * NATURAL_FREQ
DAMPING_M107_15 = 2.0 * DAMPING_RATIO * UnitreeActuatorCfg_M107_15 * NATURAL_FREQ
DAMPING_N7520_22p5 = 2.0 * DAMPING_RATIO * UnitreeActuatorCfg_N7520_22p5 * NATURAL_FREQ
DAMPING_N5020_16 = 2.0 * DAMPING_RATIO * UnitreeActuatorCfg_N5020_16 * NATURAL_FREQ
DAMPING_N7520_14p3 = 2.0 * DAMPING_RATIO * UnitreeActuatorCfg_N7520_14p3 * NATURAL_FREQ

print(f"STIFFNESS_M107_24: {STIFFNESS_M107_24}, DAMPING_M107_24: {DAMPING_M107_24}")
print(f"STIFFNESS_M107_15: {STIFFNESS_M107_15}, DAMPING_M107_15: {DAMPING_M107_15}")
print(f"STIFFNESS_N7520_22p5: {STIFFNESS_N7520_22p5}, DAMPING_N7520_22p5: {DAMPING_N7520_22p5}")
print(f"STIFFNESS_N5020_16: {STIFFNESS_N5020_16}, DAMPING_N5020_16: {DAMPING_N5020_16}")
print(f"STIFFNESS_N7520_14p3: {STIFFNESS_N7520_14p3}, DAMPING_N7520_14p3: {DAMPING_N7520_14p3}")

exit()

H12_CFG_HANDLESS = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{project_root}/assets/robots/unitree_model/H1-2/h1_2_handless/h1_2_handless.usd",
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False, 
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=4
        ),

    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.05),               #same as gym
        joint_pos={
            # legs joints
            "left_hip_yaw_joint": 0.0,
            "left_hip_roll_joint": 0.0,
            "left_hip_pitch_joint": -0.16, #same as gym
            "left_knee_joint": 0.36,    #same as gym
            "left_ankle_pitch_joint": -0.15, #gym is -0.2,reduced a bit to avoid foot collision with ground
            "left_ankle_roll_joint": 0.0,
            
            "right_hip_yaw_joint": 0.0,
            "right_hip_roll_joint": 0.0,
            "right_hip_pitch_joint": -0.16,
            "right_knee_joint": 0.36,
            "right_ankle_pitch_joint": -0.15,
            "right_ankle_roll_joint": 0.0,
            
            
            # arms joints
            "left_shoulder_pitch_joint": 0.4, # ~ 23 degrees, same as gym
            "left_shoulder_roll_joint": 0.0,
            "left_shoulder_yaw_joint": 0.0,
            "left_elbow_joint": 0.3,           # ~ 14.1 degrees, same as gym
            "left_wrist_roll_joint": 0.0,
            "left_wrist_pitch_joint": 0.0,
            "left_wrist_yaw_joint": 0.0,
            
            "right_shoulder_pitch_joint": 0.4,  # ~ 23 degrees, same as gym
            "right_shoulder_roll_joint": 0.0,
            "right_shoulder_yaw_joint": 0.0,
            "right_elbow_joint": 0.3,        # ~ 14.1 degrees, same as gym
            "right_wrist_roll_joint": 0.0,
            "right_wrist_pitch_joint": 0.0,
            "right_wrist_yaw_joint": 0.0,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,


    # actuators={
    #     "legs": ImplicitActuatorCfg(
    #         joint_names_expr=[
    #             ".*_hip_yaw_joint", 
    #             ".*_hip_roll_joint",
    #             ".*_hip_pitch_joint", 
    #             ".*_knee_joint",
    #         ],
    #         effort_limit=300,
    #         velocity_limit=100,
    #         stiffness=None,
    #         damping=None,
    #         armature=None,
    #     ),
    #     "feet": ImplicitActuatorCfg(
    #         effort_limit=None,
    #         joint_names_expr=[".*_ankle_pitch_joint", ".*_ankle_roll_joint"],
    #         stiffness=None,
    #         damping=None,
    #         # armature=0.001,
    #     ),
    #     "arms": ImplicitActuatorCfg(
    #         joint_names_expr=[
    #             ".*_shoulder_.*_joint",
    #             ".*_elbow_joint",
    #             ".*_wrist_.*_joint"
    #         ],
    #         effort_limit=None,
    #         velocity_limit=None,
    #          stiffness={  # increase the stiffness (kp)
    #              ".*_shoulder_.*_joint": 25.0,
    #              ".*_elbow_joint": 50.0,
    #              ".*_wrist_.*_joint": 40.0,
    #         },
    #          damping={    # increase the damping (kd)
    #              ".*_shoulder_.*_joint": 2.0,
    #              ".*_elbow_joint": 2.0,
    #              ".*_wrist_.*_joint": 2.0,
    #          },
    #         armature=None,
    #     ),

actuators={
    # Motor: M107-24, Torque: 300 Nm
    # From your original "legs" group
    "hip_pitch_roll_knee": ImplicitActuatorCfg(
        joint_names_expr=[
            ".*_hip_pitch_joint", 
            ".*_hip_roll_joint",
            ".*_knee_joint"
        ],
        effort_limit=300,
        velocity_limit=100,
        stiffness={
            ".*_hip_pitch_joint": 200.0,
            ".*_hip_roll_joint": 200.0,
            ".*_knee_joint": 300.0,
        },
        damping={
            ".*_hip_pitch_joint": 2.5,
            ".*_hip_roll_joint": 2.5,
            ".*_knee_joint": 4.0,},
        armature=UnitreeActuatorCfg_M107_24.armature,
    ),
    # Motor: M107-15, Torque: 200 Nm
    # From your original "legs" group
    "hip_yaw": ImplicitActuatorCfg(
        joint_names_expr=[".*_hip_yaw_joint"],
        effort_limit=200,
        velocity_limit=100,
        stiffness={"^.*_hip_yaw_joint$": 200.0,},
        damping={"^.*_hip_yaw_joint$": 2.5,},
        armature=UnitreeActuatorCfg_M107_15.armature,
    ),
    # Motor: N7520-22.5, Torque: 120 Nm
    # From your original "arms" group
    "shoulder_pitch_roll_and_elbow": ImplicitActuatorCfg(
        joint_names_expr=[
            ".*_shoulder_pitch_joint",
            ".*_shoulder_roll_joint",
            ".*_elbow_joint"
        ],
        effort_limit= 120,
        velocity_limit= 100,
        stiffness={
            ".*_shoulder_.*_joint":80.0,
            ".*_elbow_joint": 40.0,
        },
        damping={
            ".*_shoulder_.*_joint": 2.0,
            ".*_elbow_joint": 1.0,
        },
        armature=UnitreeActuatorCfg_N7520_22p5.armature,
    ),
    # Motor: N5020-16, Torque: 25 Nm
    # From your original "arms" group
    "wrists": ImplicitActuatorCfg(
        joint_names_expr=[
            ".*_wrist_.*_joint"
        ],
        effort_limit=25.0,
        velocity_limit=100,
        stiffness={
            ".*_wrist_.*_joint": 60.0,
        },
        damping={
            ".*_wrist_.*_joint": 0.5,
        },
        armature=UnitreeActuatorCfg_N5020_16.armature,
    ),
    # Motor: N7520-14.3, Torque: 75 Nm
    # Combined from your "feet" and "arms" groups
    "ankles_and_shoulder_yaw": ImplicitActuatorCfg(
        joint_names_expr=[
            ".*_ankle_pitch_joint",
            ".*_ankle_roll_joint",
            ".*_shoulder_yaw_joint"
        ],
        effort_limit=75,
        velocity_limit=100,
        stiffness={
            ".*_ankle_pitch_joint": 60.0,      # From original "feet" config
            ".*_ankle_roll_joint": 40.0,      # From original "feet" config
            ".*_shoulder_yaw_joint": 40.0, # From original "arms" config
        },
        damping={
            ".*_ankle_pitch_joint": 1.0,       # From original "feet" config
            ".*_ankle_roll_joint": 0.3,       # From original "feet" config
            ".*_shoulder_yaw_joint": 1.0,   # From original "arms" config
        },
        armature=UnitreeActuatorCfg_N7520_14p3.armature,
    ),
    },
)


# #THE FOLLOWING NEEDS EDITING AND TUNING
# H12_CFG_WITH_INSPIRE_HAND = ArticulationCfg(
#     spawn=sim_utils.UsdFileCfg(
#         usd_path=f"{project_root}/assets/robots/h1_2-26dof-inspire-base-fix-usd/h1_2_26dof_with_inspire_rev_1_0.usd",
#         activate_contact_sensors=True,
#         rigid_props=sim_utils.RigidBodyPropertiesCfg(
#             disable_gravity=False,
#             retain_accelerations=False,
#             linear_damping=0.0,
#             angular_damping=0.0,
#             max_linear_velocity=1000.0,
#             max_angular_velocity=1000.0,
#             max_depenetration_velocity=1.0,
#         ),
#         articulation_props=sim_utils.ArticulationRootPropertiesCfg(
#             enabled_self_collisions=False, 
#             solver_position_iteration_count=8,
#             solver_velocity_iteration_count=4
#         ),

#     ),
#     init_state=ArticulationCfg.InitialStateCfg(
#         pos=(0.0, 0.0, 0.75),
#         joint_pos={
#             # legs joints
#             "left_hip_yaw_joint": 0.0,
#             "left_hip_roll_joint": 0.0,
#             "left_hip_pitch_joint": -0.05,
#             "left_knee_joint": 0.2,
#             "left_ankle_pitch_joint": -0.15,
#             "left_ankle_roll_joint": 0.0,
            
#             "right_hip_yaw_joint": 0.0,
#             "right_hip_roll_joint": 0.0,
#             "right_hip_pitch_joint": -0.05,
#             "right_knee_joint": 0.2,
#             "right_ankle_pitch_joint": -0.15,
#             "right_ankle_roll_joint": 0.0,
            
            
#             # arms joints
#             "left_shoulder_pitch_joint": 0.0,
#             "left_shoulder_roll_joint": 0.0,
#             "left_shoulder_yaw_joint": 0.0,
#             "left_elbow_joint": 0.0,
#             "left_wrist_roll_joint": 0.0,
#             "left_wrist_pitch_joint": 0.0,
#             "left_wrist_yaw_joint": 0.0,
            
#             "right_shoulder_pitch_joint": 0.0,
#             "right_shoulder_roll_joint": 0.0,
#             "right_shoulder_yaw_joint": 0.0,
#             "right_elbow_joint": 0.0,
#             "right_wrist_roll_joint": 0.0,
#             "right_wrist_pitch_joint": 0.0,
#             "right_wrist_yaw_joint": 0.0,
            
#             # fingers joints
#             "L_index_proximal_joint": 0.0,
#             "L_index_intermediate_joint": 0.0,
#             "L_middle_proximal_joint": 0.0,
#             "L_middle_intermediate_joint": 0.0,
#             "L_pinky_proximal_joint":0.0,
#             "L_pinky_intermediate_joint":0.0,
#             "L_ring_proximal_joint":0.0,
#             "L_ring_intermediate_joint":0.0,
#             "L_thumb_proximal_yaw_joint":0.0,
#             "L_thumb_proximal_pitch_joint":0.0,
#             "L_thumb_intermediate_joint":0.0,
#             "L_thumb_distal_joint":0.0,

#             "R_index_proximal_joint": 0.0,
#             "R_index_intermediate_joint": 0.0,
#             "R_middle_proximal_joint": 0.0,
#             "R_middle_intermediate_joint": 0.0,
#             "R_pinky_proximal_joint":0.0,
#             "R_pinky_intermediate_joint":0.0,
#             "R_ring_proximal_joint":0.0,
#             "R_ring_intermediate_joint":0.0,
#             "R_thumb_proximal_yaw_joint":0.0,
#             "R_thumb_proximal_pitch_joint":0.0,
#             "R_thumb_intermediate_joint":0.0,
#             "R_thumb_distal_joint":0.0,
#         },
#         joint_vel={".*": 0.0},
#     ),
#     soft_joint_pos_limit_factor=0.9,
#     actuators={
#         "legs": ImplicitActuatorCfg(
#             joint_names_expr=[
#                 ".*_hip_yaw_joint", 
#                 ".*_hip_roll_joint",
#                 ".*_hip_pitch_joint", 
#                 ".*_knee_joint",
#             ],
#             effort_limit=None,
#             velocity_limit=None,
#             stiffness=None,
#             damping=None,
#             armature=None,
#         ),
#         "feet": ImplicitActuatorCfg(
#             effort_limit=None,
#             joint_names_expr=[".*_ankle_pitch_joint", ".*_ankle_roll_joint"],
#             stiffness=None,
#             damping=None,
#             # armature=0.001,
#         ),
#         "arms": ImplicitActuatorCfg(
#             joint_names_expr=[
#                 ".*_shoulder_.*_joint",
#                 ".*_elbow_joint",
#                 ".*_wrist_.*_joint"
#             ],
#             effort_limit=None,
#             velocity_limit=None,
#              stiffness={  # increase the stiffness (kp)
#                  ".*_shoulder_.*_joint": 25.0,
#                  ".*_elbow_joint": 50.0,
#                  ".*_wrist_.*_joint": 40.0,
#             },
#              damping={    # increase the damping (kd)
#                  ".*_shoulder_.*_joint": 2.0,
#                  ".*_elbow_joint": 2.0,
#                  ".*_wrist_.*_joint": 2.0,
#              },
#             armature=None,
#         ),
#         "hands": ImplicitActuatorCfg(
#             joint_names_expr=[
#                 ".*_index_proximal_joint",
#                 ".*_index_intermediate_joint",
#                 ".*_middle_proximal_joint",
#                 ".*_middle_intermediate_joint",
#                 ".*_pinky_proximal_joint",
#                 ".*_pinky_intermediate_joint",
#                 ".*_ring_proximal_joint",
#                 ".*_ring_intermediate_joint",
#                 ".*_thumb_proximal_yaw_joint",
#                 ".*_thumb_proximal_pitch_joint",
#                 ".*_thumb_intermediate_joint",
#                 ".*_thumb_distal_joint",
#             ],
#             effort_limit=100.0,
#             velocity_limit=50,
#             stiffness={
#                 ".*_index_proximal_joint":1000.0,
#                 ".*_index_intermediate_joint":1000.0,
#                 ".*_middle_proximal_joint":1000.0,
#                 ".*_middle_intermediate_joint":1000.0,
#                 ".*_pinky_proximal_joint":1000.0,
#                 ".*_pinky_intermediate_joint":1000.0,
#                 ".*_ring_proximal_joint":1000.0,
#                 ".*_ring_intermediate_joint":1000.0,
#                 ".*_thumb_proximal_yaw_joint":1000.0,
#                 ".*_thumb_proximal_pitch_joint":1000.0,
#                 ".*_thumb_intermediate_joint":1000.0,
#                 ".*_thumb_distal_joint":1000.0,
#             },
#             damping={
#                 ".*_index_proximal_joint":15,
#                 ".*_index_intermediate_joint":15,
#                 ".*_middle_proximal_joint":15,
#                 ".*_middle_intermediate_joint":15,
#                 ".*_pinky_proximal_joint":15,
#                 ".*_pinky_intermediate_joint":15,
#                 ".*_ring_proximal_joint":15,
#                 ".*_ring_intermediate_joint":15,
#                 ".*_thumb_proximal_yaw_joint":15,
#                 ".*_thumb_proximal_pitch_joint":15,
#                 ".*_thumb_intermediate_joint":15,
#                 ".*_thumb_distal_joint":15,
#             },
#             armature={
#                 ".*": 0.0
#             },
#         ),

#     },
# )