from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.utils import configclass

from h12_bullet_time.assets.robots.unitree import H12_CFG_HANDLESS
from h12_bullet_time.sensors.tof_sensor_cfg import TofSensorCfg
from h12_bullet_time.utils.urdf_tools import extract_sensor_poses_from_urdf

from . import h12_survive_time_env_cfg_hybrid as base
from . import mdp as local_mdp


@configclass
class H12SurviveTimeSceneCfg_WM(base.H12SurviveTimeSceneCfg_HYBRID):
    pass


for name in base._sensor_configs:
    setattr(H12SurviveTimeSceneCfg_WM, name, None)

for link_path, poses in extract_sensor_poses_from_urdf(
    H12_CFG_HANDLESS.spawn.asset_path, debug=False
).items():
    name = f"tof_sensor_{link_path.replace('_skin', '').replace('_link', '')}"
    setattr(
        H12SurviveTimeSceneCfg_WM,
        name,
        TofSensorCfg(
            prim_path=f"{{ENV_REGEX_NS}}/Robot/{link_path}",
            target_frames=[TofSensorCfg.FrameCfg(prim_path="{ENV_REGEX_NS}/Projectile")],
            relative_sensor_pos=[pose.pos for pose in poses],
            relative_sensor_quat=[pose.quat for pose in poses],
            max_range=base._default_max_range,
            projectile_radius=base._projectile_radius,
        ),
    )


@configclass
class H12SurviveTimeEnvCfg_WM(base.H12SurviveTimeEnvCfg_HYBRID):
    scene: H12SurviveTimeSceneCfg_WM = H12SurviveTimeSceneCfg_WM(
        num_envs=4096, env_spacing=4.0
    )

    def __post_init__(self):
        super().__post_init__()
        for name in base._sensor_configs:
            setattr(self.scene, name, None)
        self.observations.policy.sensor_observations = None
        self.observations.critic.sensor_observations = ObsTerm(
            func=local_mdp.tof_distances_obs,
            scale=0.25,
            params={"max_range": base._default_max_range, "handle_nan": "replace_with_max"},
        )
