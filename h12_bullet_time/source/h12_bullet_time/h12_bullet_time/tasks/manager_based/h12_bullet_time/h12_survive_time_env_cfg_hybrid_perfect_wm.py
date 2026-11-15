"""WM environment plus a simulated Future_Projectile for the SIM-* future observations.

Each step the projectile is propagated up to R = ORACLE_ROLLOUT_FRAMES frames of CONTEXT_STRIDE control steps
and Future_Projectile is placed one frame before it would occupy the robot (see ``mdp.future_projectile``).
Every ray/field sensor then exposes ``future_dist_est(_normalized)`` for that pose.
"""

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.utils import configclass

from . import h12_survive_time_env_cfg_hybrid as base
from . import h12_survive_time_env_cfg_wm as wm
from . import mdp as local_mdp


@configclass
class H12SurviveTimeSceneCfg_PerfectWM(wm.H12SurviveTimeSceneCfg_WM):
    Future_Projectile = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Future_Projectile",
        spawn=sim_utils.SphereCfg(
            radius=base._projectile_radius,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0), opacity=1.0),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True, disable_gravity=True),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=False),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(-1.0, -1.0, 0.0)),
    )


@configclass
class H12SurviveTimeEnvCfg_PerfectWM(wm.H12SurviveTimeEnvCfg_WM):
    scene: H12SurviveTimeSceneCfg_PerfectWM = H12SurviveTimeSceneCfg_PerfectWM(num_envs=4096, env_spacing=4.0)

    def __post_init__(self):
        super().__post_init__()
        self.observations.policy.future_projectile = ObsTerm(func=local_mdp.future_projectile)
        self.sim.render.enable_translucency = False
