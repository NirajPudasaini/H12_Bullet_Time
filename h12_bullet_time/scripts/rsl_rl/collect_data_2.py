"""Script to collect sensor data using a trained RL agent into H5 files."""

"""Launch Isaac Sim Simulator first."""

import argparse
import os
import re
import sys
from pathlib import Path

from isaaclab.app import AppLauncher
from importlib.metadata import version as pkg_version

import cli_args  # isort: skip

parser = argparse.ArgumentParser(description="Collect sensor data with a trained RL agent.")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=64, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--agent", type=str, default="rsl_rl_cfg_entry_point", help="Name of the RL agent configuration entry point."
)
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument("--use_pretrained_checkpoint", action="store_true")
parser.add_argument("--num_trajectories", type=int, default=100, help="Total trajectories to collect.")
parser.add_argument("--max_traj_length", type=int, default=5000, help="Max timesteps per trajectory.")
parser.add_argument("--min_traj_length", type=int, default=10, help="Discard trajectories shorter than this.")
parser.add_argument("--output_dir", type=str, default="collected_data", help="Output directory for H5 files.")
parser.add_argument("--trajs_per_file", type=int, default=1000, help="Trajectories per H5 file.")
parser.add_argument(
    "--sensors", type=str, default=None,
    help="Sensor spec 'SHAPE:SIGNAL:MAX_RANGE[;...]' with SHAPE in {FIELD, RAY, CONE}, e.g. 'RAY:DIST:X'. "
         "Must match the spec used during training (sets ABLATION_SENSORS).",
)
parser.add_argument("--max_range", type=float, default=None, help="Sensor max range (sets ABLATION_MAX_RANGE).")
parser.add_argument("--static", action="store_true", default=False, help="Lock all robot joints; robot will not move or fall.")
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

_RUN_TS = re.compile(r"^\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}")


def _latest_run_dir(log_root):
    root = Path(log_root)
    if not root.is_dir():
        return None
    runs = [p for p in root.iterdir() if p.is_dir()]
    return max(runs, key=lambda p: p.stat().st_mtime) if runs else None


def _sensors_from_run_name(name):
    m = _RUN_TS.match(name)
    if not m:
        return None
    tag = name[m.end():].lstrip("_")
    return tag.replace("_", ";").replace("-", ":") if tag else None


# Sensor spec must be set before hydra imports the env cfg. Infer it from the latest
# trained run unless the user passed --sensors / --load_run / --checkpoint.
_log_root = os.path.abspath(os.path.join("logs", "rsl_rl", "h12-bullet-time-ppo"))
if not args_cli.checkpoint and args_cli.load_run is None:
    _latest = _latest_run_dir(_log_root)
    if _latest is not None:
        args_cli.load_run = _latest.name
        print(f"[INFO] Using latest run: {_latest.name}")
        if not args_cli.sensors:
            inferred = _sensors_from_run_name(_latest.name)
            if inferred:
                args_cli.sensors = inferred
                print(f"[INFO] Inferred --sensors {inferred}")
if args_cli.sensors:
    os.environ["ABLATION_SENSORS"] = args_cli.sensors
if args_cli.max_range is not None:
    os.environ["ABLATION_MAX_RANGE"] = str(args_cli.max_range)
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import torch
import numpy as np
import h5py
from datetime import datetime
from collections import defaultdict

from rsl_rl.runners import OnPolicyRunner, DistillationRunner

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.sensors import ContactSensor
from isaaclab.utils.assets import retrieve_file_path
from isaaclab_rl.utils.pretrained_checkpoint import get_published_pretrained_checkpoint

from isaaclab_rl.rsl_rl import (
    RslRlBaseRunnerCfg,
    RslRlVecEnvWrapper,
    handle_deprecated_rsl_rl_cfg,
    handle_deprecated_rsl_rl_checkpoint,
)

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

import h12_bullet_time.tasks  # noqa: F401
from h12_bullet_time.sensors import ConeSensor, FieldSensor, RaySensor

_SENSOR_SHAPES = ("field", "ray", "cone")


def link_from_sensor_name(name):
    """'ray_0_left_elbow' -> 'left_elbow'; the hybrid env names sensors '{shape}_{group}_{link}'."""
    parts = name.split("_")
    if len(parts) > 2 and parts[0] in _SENSOR_SHAPES and parts[1].isdigit():
        return "_".join(parts[2:])
    return name


_TOF_RAW_MAX = 4000


def dist_m_to_tof_raw(dist_m):
    return np.clip(np.rint(np.asarray(dist_m) * 1000.0), 0, _TOF_RAW_MAX).astype(np.int32)


def cap_to_raw(values):
    return np.rint(np.asarray(values)).astype(np.int32)


class TrajectoryBuffer:
    __slots__ = ["data", "length"]

    def __init__(self):
        self.data = defaultdict(list)
        self.length = 0

    def append(self, key, value):
        self.data[key].append(value)

    def step_done(self):
        self.length += 1

    def to_numpy(self):
        return {k: np.stack(v) for k, v in self.data.items()}


def _write_ds(group, name, arr, dtype, compression="gzip"):
    group.create_dataset(name, data=np.asarray(arr, dtype=dtype), compression=compression)


def save_trajectories(trajs, filepath, traj_offset, metadata=None, probe_radius=None,
                      sensor_static=None, fps=None, joint_names=None):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with h5py.File(filepath, "w") as f:
        if metadata:
            for k, v in metadata.items():
                f.attrs[k] = v
        traj_grp = f.create_group("traj_data")
        for i, traj in enumerate(trajs):
            traj_key = f"traj_{traj_offset + i + 1:06d}"
            tg = traj_grp.create_group(traj_key)
            if fps is not None:
                tg.attrs["fps"] = float(fps)
            obs_grp = tg.create_group("observations")
            data = traj.to_numpy()

            for key, arr in data.items():
                if key.endswith(("_link_pos_w", "_link_quat_w")):
                    continue
                if key.startswith("tof_sensor_"):
                    sg = obs_grp.require_group("tof").create_group(key)
                    _write_ds(sg, "tof_data_raw", arr, np.int32)
                elif key.startswith("cap_sensor_"):
                    sg = obs_grp.require_group("cap").create_group(key)
                    _write_ds(sg, "cap_data_raw", arr, np.int32)

            if "actions" in data:
                _write_ds(tg, "actions", data["actions"], np.float32)

            state_grp = tg.create_group("robot_state")
            if joint_names is not None:
                state_grp.attrs["joint_names"] = list(joint_names)
            for key in ("joint_pos", "joint_vel", "base_pos", "base_quat", "base_lin_vel", "base_ang_vel"):
                if key in data:
                    _write_ds(state_grp, key, data[key], np.float32)

            if "probe_pos" in data:
                probe_grp = tg.create_group("probe")
                _write_ds(probe_grp, "position", data["probe_pos"], np.float32)
                if probe_radius is not None:
                    probe_grp.attrs["radius"] = float(probe_radius)
                if "probe_in_contact" in data:
                    in_c = data["probe_in_contact"]
                    if in_c.ndim == 1:
                        in_c = in_c[:, None]
                    _write_ds(probe_grp, "in_contact", in_c, np.bool_)

            if sensor_static:
                st_grp = tg.create_group("sensor_transforms")
                for sname, sinfo in sensor_static.items():
                    sg = st_grp.create_group(sname)
                    sg.create_dataset("relative_pos", data=sinfo["relative_pos"].astype(np.float32))
                    sg.create_dataset("relative_quat", data=sinfo["relative_quat"].astype(np.float32))
                    lp_key, lq_key = f"{sname}_link_pos_w", f"{sname}_link_quat_w"
                    if lp_key in data:
                        _write_ds(sg, "link_pos_w", data[lp_key], np.float32)
                        _write_ds(sg, "link_quat_w", data[lq_key], np.float32)

    print(f"[INFO] Saved {len(trajs)} trajectories to {filepath}")


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    """Collect sensor data with a trained RSL-RL agent."""
    task_name = args_cli.task.split(":")[-1]
    train_task_name = task_name.replace("-Play", "")

    agent_cfg: RslRlBaseRunnerCfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    _rsl_rl_version = pkg_version("rsl-rl-lib")
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, _rsl_rl_version)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    log_root_path = os.path.abspath(os.path.join("logs", "rsl_rl", agent_cfg.experiment_name))
    print(f"[INFO] Loading experiment from directory: {log_root_path}")

    if args_cli.use_pretrained_checkpoint:
        resume_path = get_published_pretrained_checkpoint("rsl_rl", train_task_name)
        if not resume_path:
            print("[INFO] No pre-trained checkpoint available.")
            return
    elif args_cli.checkpoint:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    env_cfg.log_dir = os.path.dirname(resume_path)

    env = gym.make(args_cli.task, cfg=env_cfg)
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    resume_path = handle_deprecated_rsl_rl_checkpoint(resume_path, _rsl_rl_version)
    print(f"[INFO] Loading model checkpoint from: {resume_path}")
    if agent_cfg.class_name == "OnPolicyRunner":
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    elif agent_cfg.class_name == "DistillationRunner":
        runner = DistillationRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    else:
        raise ValueError(f"Unsupported runner class: {agent_cfg.class_name}")
    runner.load(resume_path)
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    # Access unwrapped env for direct sensor/robot data
    unwrapped = env.unwrapped
    num_envs = unwrapped.num_envs
    robot = unwrapped.scene["robot"]
    env_origins = unwrapped.scene.env_origins.cpu().numpy()

    # Discover probe (Projectile)
    projectile = unwrapped.scene["Projectile"]
    projectile_radius = projectile.cfg.spawn.radius

    # Discover sensors. RAY sensors report a pixel grid per element; FIELD/CONE report one
    # scalar per element. ConeSensor subclasses FieldSensor, so it is matched by the same branch.
    ray_sensors, field_sensors, ray_pixel_counts, contact_sensors = {}, {}, {}, {}
    if hasattr(unwrapped.scene, "_sensors"):
        for name, sensor in unwrapped.scene._sensors.items():
            if isinstance(sensor, RaySensor):
                ray_sensors[name] = sensor
                ray_pixel_counts[name] = sensor.cfg.pixel_count
            elif isinstance(sensor, (FieldSensor, ConeSensor)):
                field_sensors[name] = sensor
            elif isinstance(sensor, ContactSensor):
                contact_sensors[name] = sensor

    # DATA.md names: tof_sensor_{link}_{index}, cap_sensor_{link}_{index}
    h5_base = {}
    for name in list(ray_sensors) + list(field_sensors):
        kind = "tof" if name in ray_sensors else "cap"
        base = f"{kind}_sensor_{link_from_sensor_name(name)}"
        if base in h5_base.values():
            base = f"{kind}_sensor_{name}"
            print(f"[WARN] Sensor '{name}' collides with another group; storing it as '{base}'")
        h5_base[name] = base

    sensors_spec = os.environ.get("ABLATION_SENSORS", "FIELD:MINDIST:4.0")
    print(f"[INFO] sensors={sensors_spec} | {len(ray_sensors)} ray, {len(field_sensors)} field/cone | "
          f"{len(contact_sensors)} contact | {num_envs} envs")
    if not ray_sensors and not field_sensors:
        print("[WARN] No sensors found! Verify --sensors matches your environment config.")

    # Gather static sensor transforms (relative to parent link)
    sensor_static_info = {}
    all_sensors = {**field_sensors, **ray_sensors}
    for name, sensor in all_sensors.items():
        rel_pos = sensor._relative_sensor_pos.cpu().numpy()
        if name in field_sensors:
            rel_quat = np.tile(np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32), (rel_pos.shape[0], 1))
        elif hasattr(sensor, "_relative_sensor_quat"):
            rel_quat = sensor._relative_sensor_quat.cpu().numpy()
        else:
            rel_quat = np.tile(np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32), (rel_pos.shape[0], 1))
        sensor_static_info[h5_base[name]] = {"relative_pos": rel_pos, "relative_quat": rel_quat}

    # Output setup
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_subdir = os.path.join(args_cli.output_dir, f"roboset_{timestamp}")
    step_dt = getattr(unwrapped, "step_dt", None) or (unwrapped.cfg.sim.dt * unwrapped.cfg.decimation)
    fps = 1.0 / float(step_dt)
    metadata = {"task": args_cli.task, "robot": "h12", "timestamp": timestamp}
    save_kw = dict(metadata=metadata, probe_radius=projectile_radius, sensor_static=sensor_static_info,
                   fps=fps, joint_names=robot.joint_names)

    # Collection loop
    buffers = [TrajectoryBuffer() for _ in range(num_envs)]
    completed, traj_counter, file_counter = [], 0, 0

    obs = env.get_observations()

    if args_cli.static:
        _dev = robot.data.joint_pos.device
        _static_root_state = robot.data.root_state_w.clone()
        _static_root_state[:, 7:] = 0.0
        _static_jpos = robot.data.joint_pos.clone()
        _static_jvel = torch.zeros_like(robot.data.joint_vel)

    while traj_counter < args_cli.num_trajectories and simulation_app.is_running():
        with torch.inference_mode():
            # Batch GPU -> CPU transfers (one per sensor + robot state)
            # Reduce over targets so each physical sensor writes one stream (min dist / max cap).
            ray_snaps = {n: s.data.dist_est.cpu().numpy().min(axis=2) for n, s in ray_sensors.items()}
            field_snaps = {n: s.data.capacitance_values.cpu().numpy().max(axis=2) for n, s in field_sensors.items()}
            link_tf = {
                h5_base[n]: (s.data.source_pos_w.cpu().numpy(), s.data.source_quat_w.cpu().numpy())
                for n, s in all_sensors.items()
            }
            jp = robot.data.joint_pos.cpu().numpy()
            jv = robot.data.joint_vel.cpu().numpy()
            bp = robot.data.root_pos_w.cpu().numpy()
            bq = robot.data.root_quat_w.cpu().numpy()
            blv = robot.data.root_lin_vel_w.cpu().numpy()
            bav = robot.data.root_ang_vel_w.cpu().numpy()
            pp = projectile.data.root_pos_w.cpu().numpy()

            in_contact_np = None
            if contact_sensors:
                in_contact_np = np.zeros(num_envs, dtype=bool)
                for s in contact_sensors.values():
                    nf = s.data.net_forces_w
                    if nf is None:
                        continue
                    f = nf.cpu().numpy().reshape(num_envs, -1, 3)
                    in_contact_np |= (np.linalg.norm(f, axis=-1) > 1e-6).any(axis=1)

            bp -= env_origins
            pp -= env_origins
            for _n in link_tf:
                link_tf[_n] = (link_tf[_n][0] - env_origins, link_tf[_n][1])

            actions = policy(obs)
            act_np = actions.cpu().numpy()

            # Distribute batched data into per-env buffers
            for ei in range(num_envs):
                buf = buffers[ei]
                for name, snap in ray_snaps.items():
                    pc = ray_pixel_counts[name]
                    base = h5_base[name]
                    for si in range(snap.shape[1]):
                        buf.append(f"{base}_{si}", dist_m_to_tof_raw(snap[ei, si]).reshape(pc, pc))
                for name, snap in field_snaps.items():
                    buf.append(f"{h5_base[name]}_0", cap_to_raw(snap[ei]))
                buf.append("joint_pos", jp[ei])
                buf.append("joint_vel", jv[ei])
                buf.append("base_pos", bp[ei])
                buf.append("base_quat", bq[ei])
                buf.append("base_lin_vel", blv[ei])
                buf.append("base_ang_vel", bav[ei])
                buf.append("probe_pos", pp[ei])
                if in_contact_np is not None:
                    buf.append("probe_in_contact", np.array([in_contact_np[ei]]))
                for sname, (lp, lq) in link_tf.items():
                    buf.append(f"{sname}_link_pos_w", lp[ei])
                    buf.append(f"{sname}_link_quat_w", lq[ei])
                buf.append("actions", act_np[ei])
                buf.step_done()

            # Step environment
            obs, _, dones, _ = env.step(actions)
            dones_np = dones.cpu().numpy() if isinstance(dones, torch.Tensor) else np.asarray(dones)

            if args_cli.static:
                reset_ids = torch.where(torch.from_numpy(dones_np).to(_dev))[0]
                if len(reset_ids) > 0:
                    _static_root_state[reset_ids] = robot.data.root_state_w[reset_ids].clone()
                    _static_root_state[reset_ids, 7:] = 0.0
                    _static_jpos[reset_ids] = robot.data.joint_pos[reset_ids].clone()
                robot.write_root_state_to_sim(_static_root_state)
                robot.write_joint_state_to_sim(_static_jpos, _static_jvel)

            # Handle episode ends and max-length cutoffs
            for ei in range(num_envs):
                if dones_np[ei] or buffers[ei].length >= args_cli.max_traj_length:
                    if buffers[ei].length >= args_cli.min_traj_length:
                        completed.append(buffers[ei])
                        traj_counter += 1
                        if traj_counter % 10 == 0:
                            print(f"[INFO] {traj_counter}/{args_cli.num_trajectories} trajectories collected")
                        if len(completed) >= args_cli.trajs_per_file:
                            fp = os.path.join(
                                output_subdir, f"roboset_{timestamp}_part{file_counter:03d}.h5"
                            )
                            save_trajectories(completed, fp, file_counter * args_cli.trajs_per_file, **save_kw)
                            completed = []
                            file_counter += 1
                        if traj_counter >= args_cli.num_trajectories:
                            break
                    buffers[ei] = TrajectoryBuffer()

    if completed:
        fp = os.path.join(output_subdir, f"roboset_{timestamp}_part{file_counter:03d}.h5")
        save_trajectories(completed, fp, file_counter * args_cli.trajs_per_file, **save_kw)

    print(f"[INFO] Complete: {traj_counter} trajectories saved to {output_subdir}")
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
