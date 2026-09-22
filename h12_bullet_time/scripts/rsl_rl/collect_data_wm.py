"""Collect DATA.md-compatible H5 trajectories from a WM-conditioned PPO policy."""

import argparse
import os
import sys
from collections import defaultdict
from datetime import datetime
from importlib.metadata import version as pkg_version
from pathlib import Path

from isaaclab.app import AppLauncher

import cli_args


parser = argparse.ArgumentParser(description="Collect trajectories from a WM PPO policy.")
parser.add_argument("--task", type=str, default="Template-H12-Survive-Time-WM")
parser.add_argument("--agent", type=str, default="rsl_rl_cfg_entry_point")
parser.add_argument("--num_envs", type=int, default=64)
parser.add_argument("--seed", type=int, default=None)
parser.add_argument("--num_trajectories", type=int, default=100)
parser.add_argument("--max_traj_length", type=int, default=5000)
parser.add_argument("--min_traj_length", type=int, default=10)
parser.add_argument(
    "--output_dir",
    type=str,
    default="/home/carson/GenTact/trybrid_skin_project/data/h5/h12_wm_rollouts",
)
parser.add_argument("--trajs_per_file", type=int, default=1000)
parser.add_argument("--wm_checkpoint", type=str, required=True)
parser.add_argument("--wm_config", type=str, default=None)
parser.add_argument("--wm_contact_threshold", type=float, default=0.5)
parser.add_argument("--wm_batch_size", type=int, default=256)
parser.add_argument("--wm_reduction", choices=("mean", "flatten"), default="mean")
parser.add_argument("--wm_ode_steps", type=int, default=None)
parser.add_argument("--wm_stochastic", action="store_true")
parser.add_argument("--wm_no_amp", action="store_true")
parser.add_argument("--inference_frames", type=int, default=1)
parser.add_argument("--context_stride", type=int, default=1)
parser.add_argument("--wm_current_obs_type", default="latent")
parser.add_argument("--wm_future_obs_type", default="latent")
parser.add_argument("--wm_contact_pred", default="true")
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import h5py
import numpy as np
import torch
from rsl_rl.runners import DistillationRunner, OnPolicyRunner

from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent
from isaaclab.sensors import ContactSensor
from isaaclab.utils.assets import retrieve_file_path
from isaaclab_rl.rsl_rl import (
    RslRlVecEnvWrapper,
    handle_deprecated_rsl_rl_cfg,
    handle_deprecated_rsl_rl_checkpoint,
)
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

import isaaclab_tasks  # noqa: F401
import h12_bullet_time.tasks  # noqa: F401
from h12_bullet_time.sensors import TofSensor
from trybrid_skin import TOFWorldModel


def _latest_run(log_root):
    runs = [path for path in Path(log_root).iterdir() if path.is_dir()]
    if not runs:
        raise FileNotFoundError(f"No PPO runs found in {log_root}")
    return max(runs, key=lambda path: path.stat().st_mtime)


def _contact_terms(env):
    names, threshold = [], 0.03
    manager = getattr(env, "termination_manager", None)
    if manager is not None:
        for name in manager.active_terms:
            cfg = manager.get_term_cfg(name)
            if getattr(cfg.func, "__name__", "") in ("multi_contact_termination", "contact_termination"):
                names.append(name)
                threshold = float(cfg.params.get("threshold", threshold))
    return names, threshold


def _in_contact(sensors, num_envs, threshold):
    force = None
    for sensor in sensors.values():
        matrix = sensor.data.force_matrix_w
        if matrix is None:
            continue
        value = torch.linalg.norm(matrix.reshape(num_envs, -1, 3), dim=-1).max(1).values
        force = value if force is None else torch.maximum(force, value)
    return np.zeros(num_envs, dtype=bool) if force is None else (force > threshold).cpu().numpy()


class TrajectoryBuffer:
    def __init__(self):
        self.data = defaultdict(list)
        self.length = 0

    def append(self, key, value):
        self.data[key].append(value)

    def finish_step(self):
        self.length += 1

    def arrays(self):
        return {key: np.stack(value) for key, value in self.data.items()}


def _dataset(group, name, value, dtype):
    group.create_dataset(name, data=np.asarray(value, dtype=dtype), compression="gzip")


def save_trajectories(trajectories, path, offset, metadata, sensor_static, fps, robot):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with h5py.File(path, "w") as file:
        for key, value in metadata.items():
            file.attrs[key] = value
        root = file.create_group("traj_data")
        for index, trajectory in enumerate(trajectories):
            data = trajectory.arrays()
            group = root.create_group(f"traj_{offset + index + 1:06d}")
            group.attrs["fps"] = float(fps)
            tof = group.create_group("observations").create_group("tof")
            for key, value in data.items():
                if key.startswith("tof_sensor_") and not key.endswith(("_link_pos_w", "_link_quat_w")):
                    _dataset(tof.create_group(key), "tof_data_raw", value, np.int32)
            _dataset(group, "actions", data["actions"], np.float32)

            state = group.create_group("robot_state")
            state.attrs["joint_names"] = robot.joint_names
            for key in ("joint_pos", "joint_vel", "base_pos", "base_quat", "base_lin_vel", "base_ang_vel"):
                _dataset(state, key, data[key], np.float32)

            probe = group.create_group("probe")
            probe.attrs["radius"] = float(metadata["probe_radius"])
            _dataset(probe, "position", data["probe_pos"], np.float32)
            _dataset(probe, "in_contact", data["probe_in_contact"], np.bool_)

            transforms = group.create_group("sensor_transforms")
            for name, values in sensor_static.items():
                sensor = transforms.create_group(name)
                sensor.create_dataset("relative_pos", data=values["relative_pos"])
                sensor.create_dataset("relative_quat", data=values["relative_quat"])
                _dataset(sensor, "link_pos_w", data[f"{name}_link_pos_w"], np.float32)
                _dataset(sensor, "link_quat_w", data[f"{name}_link_quat_w"], np.float32)

            wm = group.create_group("world_model")
            _dataset(wm, "current_latent", data["wm_current_latent"], np.float16)
            _dataset(wm, "future_latent", data["wm_future_latent"], np.float16)
            _dataset(wm, "selected_horizon", data["wm_selected_horizon"], np.int16)
            _dataset(wm, "contact_probabilities", data["wm_contact_probabilities"], np.float32)
            _dataset(wm, "refreshed", data["wm_refreshed"], np.bool_)
    print(f"[INFO] Saved {len(trajectories)} trajectories to {path}")


def _tof_frame(env):
    sensors = []
    for name, sensor in env.unwrapped.scene._sensors.items():
        if not isinstance(sensor, TofSensor):
            continue
        pixels = int(sensor.cfg.pixel_count)
        values = sensor.data.dist_est_normalized.min(dim=2).values
        values = values.reshape(values.shape[0], values.shape[1], pixels, pixels)
        for index in range(values.shape[1]):
            sensors.append((f"{name}_{index}", values[:, index]))
    if not sensors:
        raise RuntimeError("No spatial ToF sensors found")
    sensors.sort(key=lambda item: item[0])
    names, frames = zip(*sensors)
    return torch.nan_to_num(torch.stack(frames, -1), nan=1.0).unsqueeze(1), list(names)


class WMVecEnv(RslRlVecEnvWrapper):
    def __init__(self, env, clip_actions, world_model, inference_frames):
        self.world_model = world_model
        self.inference_frames = int(inference_frames)
        self.step_count = 0
        self.last_refreshed = True
        super().__init__(env, clip_actions=clip_actions)
        if self.num_actions != world_model.action_dim:
            raise ValueError(f"Environment has {self.num_actions} actions, WM expects {world_model.action_dim}")
        self.frame, self.sensor_names = _tof_frame(self)
        robot = self.unwrapped.scene["robot"]
        self.features = world_model.initialize(
            self.frame, joint_pos=robot.data.joint_pos, joint_names=list(robot.joint_names),
            sensor_names=self.sensor_names)

    def _augment(self, observations):
        for key in ("policy", "critic"):
            if key in observations:
                observations[key] = torch.cat((observations[key], self.features), -1)
        return observations

    def get_observations(self):
        return self._augment(super().get_observations())

    def step(self, actions):
        applied = actions if self.clip_actions is None else actions.clamp(-self.clip_actions, self.clip_actions)
        observations, rewards, dones, extras = super().step(actions)
        self.frame, names = _tof_frame(self)
        if names != self.sensor_names:
            raise RuntimeError("ToF sensor ordering changed")
        self.step_count += 1
        self.last_refreshed = self.step_count % self.inference_frames == 0
        robot = self.unwrapped.scene["robot"]
        self.features = self.world_model.advance(
            self.frame, applied, dones.bool(), refresh=self.last_refreshed,
            joint_pos=robot.data.joint_pos, joint_names=list(robot.joint_names),
            sensor_names=self.sensor_names)
        return self._augment(observations), rewards, dones, extras


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg, agent_cfg):
    agent_cfg = handle_deprecated_rsl_rl_cfg(
        cli_args.update_rsl_rl_cfg(agent_cfg, args_cli), pkg_version("rsl-rl-lib"))
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device or env_cfg.sim.device
    if min(args_cli.num_trajectories, args_cli.min_traj_length, args_cli.max_traj_length,
           args_cli.trajs_per_file, args_cli.inference_frames, args_cli.context_stride) < 1:
        raise ValueError("Trajectory lengths, counts, inference_frames, and context_stride must be positive")

    log_root = os.path.abspath(os.path.join("logs", "rsl_rl", agent_cfg.experiment_name))
    if args_cli.checkpoint:
        policy_checkpoint = retrieve_file_path(args_cli.checkpoint)
    else:
        load_run = args_cli.load_run
        if load_run is None:
            load_run = _latest_run(log_root).name
            print(f"[INFO] Using latest PPO run: {load_run}")
        policy_checkpoint = get_checkpoint_path(log_root, load_run, agent_cfg.load_checkpoint)
    policy_checkpoint = handle_deprecated_rsl_rl_checkpoint(
        policy_checkpoint, pkg_version("rsl-rl-lib"))
    env_cfg.log_dir = os.path.dirname(policy_checkpoint)

    env = gym.make(args_cli.task, cfg=env_cfg)
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)
    world_model = TOFWorldModel(
        checkpoint=args_cli.wm_checkpoint,
        config=args_cli.wm_config,
        device=agent_cfg.device,
        inference_batch_size=args_cli.wm_batch_size,
        reduction=args_cli.wm_reduction,
        deterministic=not args_cli.wm_stochastic,
        seed=agent_cfg.seed,
        ode_steps=args_cli.wm_ode_steps,
        amp=not args_cli.wm_no_amp,
        contact_threshold=args_cli.wm_contact_threshold,
        context_stride=args_cli.context_stride,
        current_obs_type=str(args_cli.wm_current_obs_type).lower().replace("_", "-"),
        future_obs_type=str(args_cli.wm_future_obs_type).lower().replace("_", "-"),
        include_contact=str(args_cli.wm_contact_pred).lower() in ("1", "true", "yes"),
    )
    env = WMVecEnv(env, agent_cfg.clip_actions, world_model, args_cli.inference_frames)
    if agent_cfg.class_name == "OnPolicyRunner":
        runner_class = OnPolicyRunner
    elif agent_cfg.class_name == "DistillationRunner":
        runner_class = DistillationRunner
    else:
        raise ValueError(f"Unsupported runner class: {agent_cfg.class_name}")
    runner = runner_class(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    print(f"[INFO] Loading PPO checkpoint: {policy_checkpoint}")
    runner.load(policy_checkpoint)
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    unwrapped = env.unwrapped
    robot = unwrapped.scene["robot"]
    projectile = unwrapped.scene["Projectile"]
    origins = unwrapped.scene.env_origins.cpu().numpy()
    tof_sensors = {
        name: sensor for name, sensor in unwrapped.scene._sensors.items()
        if isinstance(sensor, TofSensor)
    }
    contact_sensors = {
        name: sensor for name, sensor in unwrapped.scene._sensors.items()
        if isinstance(sensor, ContactSensor)
    }
    contact_names, contact_threshold = _contact_terms(unwrapped)
    static = {
        name: {
            "relative_pos": sensor._relative_sensor_pos.cpu().numpy().astype(np.float32),
            "relative_quat": sensor._relative_sensor_quat.cpu().numpy().astype(np.float32),
        }
        for name, sensor in tof_sensors.items()
    }
    step_dt = getattr(unwrapped, "step_dt", unwrapped.cfg.sim.dt * unwrapped.cfg.decimation)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    metadata = {
        "task": args_cli.task,
        "robot": "h12",
        "timestamp": timestamp,
        "probe_radius": float(projectile.cfg.spawn.radius),
        "policy_checkpoint": policy_checkpoint,
        "wm_checkpoint": os.path.abspath(args_cli.wm_checkpoint),
        "inference_frames": args_cli.inference_frames,
        "context_stride": args_cli.context_stride,
    }

    buffers = [TrajectoryBuffer() for _ in range(unwrapped.num_envs)]
    completed, count, part = [], 0, 0
    obs = env.get_observations()
    while count < args_cli.num_trajectories and simulation_app.is_running():
        with torch.inference_mode():
            tof = {
                name: torch.nan_to_num(
                    sensor.data.dist_est_normalized.min(dim=2).values, nan=1.0).cpu().numpy()
                for name, sensor in tof_sensors.items()
            }
            link_tf = {
                name: (sensor.data.source_pos_w.cpu().numpy() - origins,
                       sensor.data.source_quat_w.cpu().numpy())
                for name, sensor in tof_sensors.items()
            }
            joint_pos = robot.data.joint_pos.cpu().numpy()
            joint_vel = robot.data.joint_vel.cpu().numpy()
            base_pos = robot.data.root_pos_w.cpu().numpy() - origins
            base_quat = robot.data.root_quat_w.cpu().numpy()
            base_lin_vel = robot.data.root_lin_vel_w.cpu().numpy()
            base_ang_vel = robot.data.root_ang_vel_w.cpu().numpy()
            probe_pos = projectile.data.root_pos_w.cpu().numpy() - origins
            contact = _in_contact(contact_sensors, unwrapped.num_envs, contact_threshold)
            actions = policy(obs)
            action_np = actions.cpu().numpy()
            current = world_model.latent_history[:, -1].cpu().numpy()
            future = world_model.future_latent.cpu().numpy()
            horizon = world_model.future_index.cpu().numpy()
            probabilities = world_model.contact_probabilities.cpu().numpy()

            for env_index, buffer in enumerate(buffers):
                for name, values in tof.items():
                    pixels = int(tof_sensors[name].cfg.pixel_count)
                    for sensor_index in range(values.shape[1]):
                        raw = np.rint(values[env_index, sensor_index] * 4000).clip(0, 4000)
                        buffer.append(f"{name}_{sensor_index}", raw.reshape(pixels, pixels))
                for key, value in (
                    ("joint_pos", joint_pos), ("joint_vel", joint_vel),
                    ("base_pos", base_pos), ("base_quat", base_quat),
                    ("base_lin_vel", base_lin_vel), ("base_ang_vel", base_ang_vel),
                    ("probe_pos", probe_pos),
                ):
                    buffer.append(key, value[env_index])
                buffer.append("probe_in_contact", np.array([contact[env_index]]))
                buffer.append("actions", action_np[env_index])
                buffer.append("wm_current_latent", current[env_index])
                buffer.append("wm_future_latent", future[env_index])
                buffer.append("wm_selected_horizon", np.array([horizon[env_index]]))
                buffer.append("wm_contact_probabilities", probabilities[env_index])
                buffer.append("wm_refreshed", np.array([env.last_refreshed]))
                for name, (position, quaternion) in link_tf.items():
                    buffer.append(f"{name}_link_pos_w", position[env_index])
                    buffer.append(f"{name}_link_quat_w", quaternion[env_index])
                buffer.finish_step()

            obs, _, dones, _ = env.step(actions)
            done = dones.cpu().numpy()
            manager = getattr(unwrapped, "termination_manager", None)
            if manager is not None:
                contact_done = torch.zeros(unwrapped.num_envs, dtype=torch.bool, device=unwrapped.device)
                for name in contact_names:
                    contact_done |= manager.get_term(name)
                for env_index in np.flatnonzero(contact_done.cpu().numpy()):
                    buffers[env_index].data["probe_in_contact"][-1] = np.array([True])

            for env_index in range(unwrapped.num_envs):
                if done[env_index] or buffers[env_index].length >= args_cli.max_traj_length:
                    if buffers[env_index].length >= args_cli.min_traj_length and count < args_cli.num_trajectories:
                        completed.append(buffers[env_index])
                        count += 1
                        if count % 10 == 0:
                            print(f"[INFO] {count}/{args_cli.num_trajectories} trajectories collected")
                        if len(completed) >= args_cli.trajs_per_file:
                            path = os.path.join(
                                args_cli.output_dir, f"roboset_{timestamp}_part{part:03d}.h5")
                            save_trajectories(
                                completed, path, part * args_cli.trajs_per_file,
                                metadata, static, 1.0 / step_dt, robot)
                            completed, part = [], part + 1
                    buffers[env_index] = TrajectoryBuffer()

    if completed:
        path = os.path.join(args_cli.output_dir, f"roboset_{timestamp}_part{part:03d}.h5")
        save_trajectories(
            completed, path, part * args_cli.trajs_per_file,
            metadata, static, 1.0 / step_dt, robot)
    print(f"[INFO] Complete: {count} trajectories saved to {args_cli.output_dir}")
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
