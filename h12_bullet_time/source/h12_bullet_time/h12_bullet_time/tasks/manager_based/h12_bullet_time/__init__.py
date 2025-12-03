# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import gymnasium as gym

from . import agents

##
# Register Gym environments.
##



gym.register(
    id="Template-H12-Bullet-Time-Curriculum-Phase",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={    
        "env_cfg_entry_point": f"{__name__}.h12_bullet_time_env_cfg_curriculum_phase:H12BulletTimeEnvCfg_Curriculum_Phase",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:PPORunnerCfg",
    },
)   


gym.register(
    id="Template-H12-Bullet-Time-Curriculum-Phase-Play",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={    
        "env_cfg_entry_point": f"{__name__}.h12_bullet_time_env_cfg_curriculum_phase:H12BulletTimeEnvCfg_CurriculumPhasePlay",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:PPORunnerCfg",
    },
)


gym.register(
    id="Template-H12-Bullet-Time-Curriculum",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={    
        "env_cfg_entry_point": f"{__name__}.h12_bullet_time_env_cfg_curriculum:H12BulletTimeEnvCfg_Curriculum",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:PPORunnerCfg",
    },
)   


gym.register(
    id="Template-H12-Bullet-Time-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.h12_bullet_time_env_cfg:H12BulletTimeEnvCfg"        ,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:PPORunnerCfg",
    },
)

gym.register(
    id="Template-H12-Bullet-Time-Minimal-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.h12_bullet_time_env_cfg_minimal:MinimalH12EnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:PPORunnerCfg",
    },
)

gym.register(
    id="Template-H12-Bullet-Time-TOF",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.h12_bullet_time_env_cfg_tof:H12BulletTimeEnvCfg_TOF",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:PPORunnerCfg",
    },
)

gym.register(
    id="Template-H12-Bullet-Time-CAP",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.h12_bullet_time_env_cfg_cap:H12BulletTimeEnvCfg_CAP",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:PPORunnerCfg",
    },
)

gym.register(
    id="Template-H12-Bullet-Time-HYBRID",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.h12_bullet_time_env_cfg_hybrid:H12BulletTimeEnvCfg_HYBRID",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:PPORunnerCfg",
    },
)