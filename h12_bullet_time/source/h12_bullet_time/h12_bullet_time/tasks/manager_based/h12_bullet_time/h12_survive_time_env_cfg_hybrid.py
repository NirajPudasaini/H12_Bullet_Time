"""
Environment config with sensor readings integrated for RL training.

Supports ablation studies via environment variables:
    ABLATION_SENSORS: Semicolon-separated list of shape:signal:max_range specs
                      e.g. "FIELD:DIST:4.0" or "FIELD:MINDIST:4.0;RAY:DIST:4.0"
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
from h12_bullet_time.sensors import FieldSensorCfg, RaySensorCfg, ConeSensorCfg
from h12_bullet_time.utils.urdf_tools import extract_sensor_poses_from_urdf


# ── Default parameter values ─────────────────────────────────────────────────
_DEFAULT_PROJECTILE_RADIUS = 0.15
_DEFAULT_PROJECTILE_MASS = 0.1
_DEFAULT_PROJECTILE_MIN_SPAWN_DIST = 4.0
_DEFAULT_PROJECTILE_MAX_SPAWN_DIST = 6.0
_DEFAULT_PROJECTILE_MIN_HEIGHT = 2.0
_DEFAULT_PROJECTILE_MAX_HEIGHT = 3.0
_DEFAULT_PROJECTILE_MIN_SPEED = 4.0
_DEFAULT_PROJECTILE_MAX_SPEED = 8.0
_DEFAULT_PROJECTILE_SPAWN_INTERVAL_RANGE = (1.0, 2.0)
_DEFAULT_DEBUG_VIS = False
_DEFAULT_SENSORS = "FIELD:MINDIST:4.0"
_DEFAULT_PROXIMITY_SCALE = -0.01
_DEFAULT_CONTACT_SCALE = -0.1
_DEFAULT_CONTACT_THRESHOLD = 0.03
_DEFAULT_CONTACT_TERMINATION = True
_DEFAULT_TERMINATION_ANGLE_THRESHOLD_DEG = 60
_DEFAULT_TERMINATION_HEIGHT_THRESHOLD = 0.4

# ── Read ablation overrides from environment variables ────────────────────────
_projectile_radius = float(os.environ.get("ABLATION_PROJECTILE_RADIUS", _DEFAULT_PROJECTILE_RADIUS))
_projectile_min_spawn_dist = float(os.environ.get("ABLATION_PROJECTILE_MIN_SPAWN_DIST", _DEFAULT_PROJECTILE_MIN_SPAWN_DIST))
_projectile_max_spawn_dist = float(os.environ.get("ABLATION_PROJECTILE_MAX_SPAWN_DIST", _DEFAULT_PROJECTILE_MAX_SPAWN_DIST))
_projectile_min_height = float(os.environ.get("ABLATION_PROJECTILE_MIN_HEIGHT", _DEFAULT_PROJECTILE_MIN_HEIGHT))
_projectile_max_height = float(os.environ.get("ABLATION_PROJECTILE_MAX_HEIGHT", _DEFAULT_PROJECTILE_MAX_HEIGHT))
_projectile_min_speed = float(os.environ.get("ABLATION_PROJECTILE_MIN_SPEED", _DEFAULT_PROJECTILE_MIN_SPEED))
_projectile_max_speed = float(os.environ.get("ABLATION_PROJECTILE_MAX_SPEED", _DEFAULT_PROJECTILE_MAX_SPEED))
_spawn_interval_str = os.environ.get("ABLATION_PROJECTILE_SPAWN_INTERVAL_RANGE", None)
if _spawn_interval_str:
    try:
        parts = _spawn_interval_str.split(",")
        _projectile_spawn_interval_range = (float(parts[0].strip()), float(parts[1].strip()))
    except (ValueError, IndexError):
        print(f"[WARNING] Invalid ABLATION_PROJECTILE_SPAWN_INTERVAL_RANGE: {_spawn_interval_str}, using default")
        _projectile_spawn_interval_range = _DEFAULT_PROJECTILE_SPAWN_INTERVAL_RANGE
else:
    _projectile_spawn_interval_range = _DEFAULT_PROJECTILE_SPAWN_INTERVAL_RANGE
_debug_vis = bool(os.environ.get("ABLATION_DEBUG_VIS", _DEFAULT_DEBUG_VIS))
_proximity_scale = float(os.environ.get("ABLATION_PROXIMITY_SCALE", _DEFAULT_PROXIMITY_SCALE))
_contact_scale = float(os.environ.get("ABLATION_CONTACT_SCALE", _DEFAULT_CONTACT_SCALE))
_contact_threshold = float(os.environ.get("ABLATION_CONTACT_THRESHOLD", _DEFAULT_CONTACT_THRESHOLD))
_projectile_mass = float(os.environ.get("ABLATION_PROJECTILE_MASS", _DEFAULT_PROJECTILE_MASS))
_contact_termination = bool(os.environ.get("ABLATION_CONTACT_TERMINATION", _DEFAULT_CONTACT_TERMINATION))
_termination_angle_threshold_deg = float(os.environ.get("ABLATION_TERMINATION_ANGLE_THRESHOLD_DEG", _DEFAULT_TERMINATION_ANGLE_THRESHOLD_DEG))
_termination_height_threshold = float(os.environ.get("ABLATION_TERMINATION_HEIGHT_THRESHOLD", _DEFAULT_TERMINATION_HEIGHT_THRESHOLD))

# ── Parse sensor specifications ───────────────────────────────────────────────
# Format: "SHAPE:SIGNAL:MAX_RANGE;SHAPE:SIGNAL:MAX_RANGE;..."
# Use "X" as max_range to inherit from ABLATION_MAX_RANGE env var.
# Examples:
#   "FIELD:DIST:4.0"                        single field sensor, raw distance
#   "FIELD:DIST:X"                           field sensor, range from ABLATION_MAX_RANGE
#   "FIELD:MINDIST:2.0;RAY:BIN:4.0"         combo of field + ray
#   "CONE:EVENT:X;RAY:MINDIST:4.0"          cone uses ABLATION_MAX_RANGE, ray fixed at 4.0
_default_max_range = float(os.environ.get("ABLATION_MAX_RANGE", 4.0))
_sensors_str = os.environ.get("ABLATION_SENSORS", _DEFAULT_SENSORS)
_sensor_specs = []
for _spec_str in _sensors_str.split(";"):
    _parts = _spec_str.strip().split(":")
    if len(_parts) >= 3:
        _range_str = _parts[2].strip()
        _range_val = _default_max_range if _range_str.upper() == "X" else float(_range_str)
        _sensor_specs.append({"shape": _parts[0].upper(), "signal": _parts[1].upper(), "max_range": _range_val})
    elif len(_parts) == 2:
        _sensor_specs.append({"shape": _parts[0].upper(), "signal": _parts[1].upper(), "max_range": _default_max_range})

if any(key.startswith("ABLATION_") for key in os.environ):
    print(f"[SENSOR CONFIG] Ablation parameters detected:")
    print(f"  - sensors: {_sensors_str}")
    print(f"  - parsed specs: {_sensor_specs}")
    print(f"  - projectile_radius: {_projectile_radius}")
    print(f"  - projectile_mass: {_projectile_mass}")
    print(f"  - debug_vis: {_debug_vis}")
    print(f"  - contact_termination: {_contact_termination}")

# ── Extract sensor poses from URDF ───────────────────────────────────────────
_sensor_library = extract_sensor_poses_from_urdf(H12_CFG_HANDLESS.spawn.asset_path, debug=False)

if not _sensor_library:
    import warnings
    warnings.warn(
        f"[SENSOR CONFIG] No sensor locations found in URDF at {H12_CFG_HANDLESS.spawn.asset_path}\n"
        "Sensors will not be added to scene. Check URDF for sensor marker elements."
    )
else:
    print(f"[SENSOR CONFIG] Found {len(_sensor_library)} sensor locations in URDF")

# ── Build proximity sensor configs ───────────────────────────────────────────
_SENSOR_CFG_BUILDERS = {
    "FIELD": lambda link_path, positions, orientations, max_range: FieldSensorCfg(
        prim_path=f"{{ENV_REGEX_NS}}/Robot/{link_path}",
        target_frames=[FieldSensorCfg.FrameCfg(prim_path="{ENV_REGEX_NS}/Projectile")],
        relative_sensor_pos=positions,
        debug_vis=_debug_vis,
        max_range=max_range,
        projectile_radius=_projectile_radius,
    ),
    "RAY": lambda link_path, positions, orientations, max_range: RaySensorCfg(
        prim_path=f"{{ENV_REGEX_NS}}/Robot/{link_path}",
        target_frames=[RaySensorCfg.FrameCfg(prim_path="{ENV_REGEX_NS}/Projectile")],
        relative_sensor_pos=positions,
        relative_sensor_quat=orientations,
        debug_vis=_debug_vis,
        max_range=max_range,
        projectile_radius=_projectile_radius,
    ),
    "CONE": lambda link_path, positions, orientations, max_range: ConeSensorCfg(
        prim_path=f"{{ENV_REGEX_NS}}/Robot/{link_path}",
        target_frames=[ConeSensorCfg.FrameCfg(prim_path="{ENV_REGEX_NS}/Projectile")],
        relative_sensor_pos=positions,
        relative_sensor_quat=orientations,
        debug_vis=_debug_vis,
        max_range=max_range,
        projectile_radius=_projectile_radius,
    ),
}

# ── Group specs by (shape, max_range) to deduplicate sensor instances ─────────
# Multiple signal types from the same shape+range share one set of sensors.
_sensor_groups = {}   # (shape, max_range) -> group_id
_group_counter = 0
_sensor_obs_specs = []  # [(prefix, signal_type), ...] for observation function

for _spec in _sensor_specs:
    _key = (_spec["shape"], _spec["max_range"])
    if _key not in _sensor_groups:
        _sensor_groups[_key] = _group_counter
        _group_counter += 1
    _gid = _sensor_groups[_key]
    _prefix = f"{_spec['shape'].lower()}_{_gid}"
    _sensor_obs_specs.append((_prefix, _spec["signal"]))

_sensor_configs = {}
for (_shape, _max_range), _gid in _sensor_groups.items():
    _builder = _SENSOR_CFG_BUILDERS.get(_shape)
    if _builder is None:
        print(f"[SENSOR CONFIG] WARNING: Unknown sensor shape '{_shape}', skipping")
        continue
    for _link_path, _sensor_poses in _sensor_library.items():
        _positions = [pose.pos for pose in _sensor_poses]
        _orientations = [pose.quat for pose in _sensor_poses]
        _link_short = _link_path.replace('_skin', '').replace('_link', '')
        _prefix = f"{_shape.lower()}_{_gid}"
        _name = f"{_prefix}_{_link_short}"
        _sensor_configs[_name] = _builder(_link_path, _positions, _orientations, _max_range)
    print(f"[SENSOR CONFIG] Added {_shape} group {_gid} (range={_max_range})")

print(f"[SENSOR CONFIG] Observation specs: {_sensor_obs_specs}")


@configclass
class H12SurviveTimeSceneCfg_HYBRID(InteractiveSceneCfg):
    """Configuration for H12 Survive Time with proximity sensors."""
    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(size=(100.0, 100.0)),
    )

    robot: ArticulationCfg = H12_CFG_HANDLESS.replace(prim_path="{ENV_REGEX_NS}/Robot")
   
    dome_light = AssetBaseCfg(
        prim_path="/World/DomeLight",
        spawn=sim_utils.DomeLightCfg(color=(0.9, 0.9, 0.9), intensity=500.0),
    )

    Projectile = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Projectile",
        spawn=sim_utils.SphereCfg(
            radius=_projectile_radius,  
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.0, 0.0, 0.2),
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


# ── Contact detection ────────────────────────────────────────────────────────
_CONTACT_DETECTION_LINKS = [
    "torso_link", "pelvis",
    "left_hip_pitch_link", "left_hip_roll_link", "left_knee_link",
    "left_ankle_pitch_link", "left_ankle_roll_link",
    "right_hip_pitch_link", "right_hip_roll_link", "right_knee_link",
    "right_ankle_pitch_link", "right_ankle_roll_link",
    "left_shoulder_pitch_link", "left_shoulder_roll_link",
    "left_shoulder_yaw_link", "left_elbow_link",
    "left_wrist_roll_link", "left_wrist_pitch_link",
    "right_shoulder_pitch_link", "right_shoulder_roll_link",
    "right_shoulder_yaw_link", "right_elbow_link",
    "right_wrist_roll_link", "right_wrist_pitch_link",
]

_contact_sensor_configs = {}
_contact_sensor_names = []

for link_name in _CONTACT_DETECTION_LINKS:
    sensor_name = f"contact_{link_name}"
    _contact_sensor_names.append(sensor_name)
    _contact_sensor_configs[sensor_name] = ContactSensorCfg(
        prim_path=f"{{ENV_REGEX_NS}}/Robot/{link_name}",
        filter_prim_paths_expr=["{ENV_REGEX_NS}/Projectile"],
        update_period=0.0,
        history_length=1,
        force_threshold=1.0,
    )

# ── Attach sensors to scene class ────────────────────────────────────────────
for sensor_name, sensor_cfg in _sensor_configs.items():
    setattr(H12SurviveTimeSceneCfg_HYBRID, sensor_name, sensor_cfg)

for sensor_name, sensor_cfg in _contact_sensor_configs.items():
    setattr(H12SurviveTimeSceneCfg_HYBRID, sensor_name, sensor_cfg)

##
# MDP settings
##

@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    joint_effort = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=[
            "left_hip_yaw_joint", "left_hip_roll_joint", "left_hip_pitch_joint",
            "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
            "right_hip_yaw_joint", "right_hip_roll_joint", "right_hip_pitch_joint",
            "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
            "torso_joint",
            "left_shoulder_pitch_joint", "left_shoulder_roll_joint",
            "left_shoulder_yaw_joint", "left_elbow_joint",
            "right_shoulder_pitch_joint", "right_shoulder_roll_joint",
            "right_shoulder_yaw_joint", "right_elbow_joint",
        ],
        scale=0.25,
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
        sensor_observations = ObsTerm(
            func=local_mdp.sensor_obs_all,
            params={"specs": _sensor_obs_specs},
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
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel, scale=0.1)
        sensor_observations = ObsTerm(
            func=local_mdp.sensor_obs_all,
            params={"specs": _sensor_obs_specs},
            scale=0.25,
        )
        
        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = True

    critic: CriticCfg = CriticCfg()


@configclass
class RewardsCfg:

    base_height = RewTerm(
        func=local_mdp.base_height_l2,
        weight=10.0,
        params={"asset_cfg": SceneEntityCfg("robot"), "target_height": 1.04},
    )

    energy_penalty = RewTerm(
        func=local_mdp.energy_penalty,
        weight=0.01,
    )

    pos_drift_penalty = RewTerm(
        func=local_mdp.pos_drift_penalty,
        weight=1.0,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )

    alive_bonus = RewTerm(
        func=local_mdp.alive_bonus,
        weight=2.0,
        params={},
    )

@configclass
class EventCfg:
    """Configuration for events."""

    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.0, 0.0), "y": (-0.0, 0.0), "yaw": (0.0, 0.0)},
            "velocity_range": {
                "x": (0.0, 0.0), "y": (0.0, 0.0), "z": (0.0, 0.0),
                "roll": (0.0, 0.0), "pitch": (0.0, 0.0), "yaw": (0.0, 0.0),
            },
        },
    )

    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={
            "position_range": (1.0, 1.0),
            "velocity_range": (-1.0, 1.0),
        },
    )

    launch_projectile_reset = EventTerm(
        func=local_mdp.launch_projectile_radial,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("Projectile"),
        },
    )
    launch_projectile_interval = EventTerm(
        func=local_mdp.launch_projectile_radial,
        mode="interval",
        interval_range_s=_projectile_spawn_interval_range,
        params={
            "asset_cfg": SceneEntityCfg("Projectile"),
        },
    )

@configclass
class CurriculumCfg:
    """Curriculum manager configuration (empty: projectile penalty active from start)."""
    pass


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)

    base_height_low = DoneTerm(
        func=local_mdp.base_height_below_threshold,
        params={"asset_cfg": SceneEntityCfg("robot"), "threshold": _termination_height_threshold},
    )
    bad_orientation = DoneTerm(
        func=local_mdp.bad_orientation,
        params={"asset_cfg": SceneEntityCfg("robot"), "angle_threshold_deg": _termination_angle_threshold_deg},
    )

    if _contact_termination:
        direct_contact_termination = DoneTerm(
            func=local_mdp.multi_contact_termination,
            params={
                "sensor_names": _contact_sensor_names,
                "threshold": _contact_threshold,
            },
        )
        sensor_based_contact_termination = DoneTerm(
            func=local_mdp.sensor_based_contact_termination,
            params={"asset_cfg": SceneEntityCfg("Projectile"), "threshold": _contact_threshold},
        )
    else:
        direct_contact_termination = None
        sensor_based_contact_termination = None

##
# Environment configuration
##

@configclass
class H12SurviveTimeEnvCfg_HYBRID(ManagerBasedRLEnvCfg):
    """RL environment config with sensor integration."""
    
    scene: H12SurviveTimeSceneCfg_HYBRID = H12SurviveTimeSceneCfg_HYBRID(num_envs=4096, env_spacing=4.0)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    events: EventCfg = EventCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self) -> None:
        """Post initialization."""
        self.decimation = 2
        self.episode_length_s = 3
        self.viewer.eye = (8.0, 0.0, 5.0)
        self.sim.dt = 1 / 120
        self.sim.render_interval = self.decimation
