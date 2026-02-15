"""Script to collect sensor data using a trained RL agent into H5 files."""

"""Launch Isaac Sim Simulator first."""

import argparse
import os
import sys

from isaaclab.app import AppLauncher

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
parser.add_argument("--max_traj_length", type=int, default=500, help="Max timesteps per trajectory.")
parser.add_argument("--min_traj_length", type=int, default=10, help="Discard trajectories shorter than this.")
parser.add_argument("--output_dir", type=str, default="collected_data", help="Output directory for H5 files.")
parser.add_argument("--trajs_per_file", type=int, default=50, help="Trajectories per H5 file.")
parser.add_argument(
    "--sensor_type", type=str, default=None, choices=["CAP", "TOF", "CAP_TOF"],
    help="Sensor type — must match the config used during training (sets ABLATION_SENSOR_TYPE).",
)
parser.add_argument("--max_range", type=float, default=None, help="Sensor max range (sets ABLATION_MAX_RANGE).")
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

# Propagate ablation env vars BEFORE environment creation so the config picks them up
if args_cli.sensor_type:
    os.environ["ABLATION_SENSOR_TYPE"] = args_cli.sensor_type
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
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.pretrained_checkpoint import get_published_pretrained_checkpoint

from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

import h12_bullet_time.tasks  # noqa: F401
from h12_bullet_time.sensors.capacitive_sensor import CapacitiveSensor
from h12_bullet_time.sensors.tof_sensor import TofSensor


class TrajectoryBuffer:
    __slots__ = ["data", "length"]

    def __init__(self):
        self.data = defaultdict(list)
        self.length = 0

    def append(self, key, value):
        self.data[key].append(value)

    def step_done(self):
        self.length += 1

    def reset(self):
        self.data = defaultdict(list)
        self.length = 0

    def to_numpy(self):
        return {k: np.stack(v) for k, v in self.data.items()}


def save_trajectories(trajs, filepath, traj_offset, metadata=None):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with h5py.File(filepath, "w") as f:
        if metadata:
            for k, v in metadata.items():
                f.attrs[k] = v
        traj_grp = f.create_group("traj_data")
        for i, traj in enumerate(trajs):
            traj_key = f"traj_{traj_offset + i + 1:06d}"
            tg = traj_grp.create_group(traj_key)
            obs_grp = tg.create_group("observations")
            data = traj.to_numpy()

            for key, arr in data.items():
                if key.startswith("depth_sensor_"):
                    sg = obs_grp.create_group(key)
                    sg.create_dataset(
                        "depth_to_camera_normalized",
                        data=arr[:, np.newaxis, :, :, np.newaxis].astype(np.float32),
                        compression="gzip",
                    )
                elif key.startswith("cap_sensor_"):
                    sg = obs_grp.create_group(key)
                    sg.create_dataset(
                        "capacitance_normalized",
                        data=arr.astype(np.float32),
                        compression="gzip",
                    )

            if "actions" in data:
                tg.create_dataset("actions", data=data["actions"].astype(np.float32), compression="gzip")

            state_grp = tg.create_group("robot_state")
            for key in ("joint_pos", "joint_vel", "base_pos", "base_quat", "base_lin_vel", "base_ang_vel"):
                if key in data:
                    state_grp.create_dataset(key, data=data[key].astype(np.float32), compression="gzip")

    print(f"[INFO] Saved {len(trajs)} trajectories to {filepath}")


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    """Collect sensor data with a trained RSL-RL agent."""
    task_name = args_cli.task.split(":")[-1]
    train_task_name = task_name.replace("-Play", "")

    agent_cfg: RslRlBaseRunnerCfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
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

    # Discover sensors
    tof_sensors, cap_sensors, tof_pixel_counts = {}, {}, {}
    if hasattr(unwrapped.scene, "_sensors"):
        for name, sensor in unwrapped.scene._sensors.items():
            if isinstance(sensor, TofSensor):
                tof_sensors[name] = sensor
                tof_pixel_counts[name] = sensor.cfg.pixel_count
            elif isinstance(sensor, CapacitiveSensor):
                cap_sensors[name] = sensor

    sensor_type = os.environ.get("ABLATION_SENSOR_TYPE", "CAP")
    print(f"[INFO] sensor_type={sensor_type} | {len(tof_sensors)} ToF, {len(cap_sensors)} cap | {num_envs} envs")
    if not tof_sensors and not cap_sensors:
        print("[WARN] No sensors found! Verify --sensor_type matches your environment config.")

    # Output setup
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_subdir = os.path.join(args_cli.output_dir, f"roboset_{timestamp}")
    metadata = {"task": args_cli.task, "num_envs": num_envs, "timestamp": timestamp, "sensor_type": sensor_type}

    # Collection loop
    buffers = [TrajectoryBuffer() for _ in range(num_envs)]
    completed, traj_counter, file_counter = [], 0, 0

    obs = env.get_observations()

    while traj_counter < args_cli.num_trajectories and simulation_app.is_running():
        with torch.inference_mode():
            # Batch GPU -> CPU transfers (one per sensor + robot state)
            tof_snaps = {n: s.data.dist_est_normalized.cpu().numpy() for n, s in tof_sensors.items()}
            cap_snaps = {n: s.data.dist_est_normalized.cpu().numpy() for n, s in cap_sensors.items()}
            jp = robot.data.joint_pos.cpu().numpy()
            jv = robot.data.joint_vel.cpu().numpy()
            bp = robot.data.root_pos_w.cpu().numpy()
            bq = robot.data.root_quat_w.cpu().numpy()
            blv = robot.data.root_lin_vel_w.cpu().numpy()
            bav = robot.data.root_ang_vel_w.cpu().numpy()

            actions = policy(obs)
            act_np = actions.cpu().numpy()

            # Distribute batched data into per-env buffers
            for ei in range(num_envs):
                buf = buffers[ei]
                # ToF -> depth images per sensor position
                for name, snap in tof_snaps.items():
                    pc = tof_pixel_counts[name]
                    link = name.replace("tof_sensor_", "")
                    n_targets = snap.shape[2]
                    for si in range(snap.shape[1]):
                        for mi in range(n_targets):
                            suffix = f"_t{mi}" if n_targets > 1 else ""
                            buf.append(
                                f"depth_sensor_{link}_{si}{suffix}",
                                snap[ei, si, mi].reshape(pc, pc),
                            )
                # Cap -> 1D vector per sensor
                for name, snap in cap_snaps.items():
                    n_targets = snap.shape[2]
                    for mi in range(n_targets):
                        suffix = f"_t{mi}" if n_targets > 1 else ""
                        buf.append(f"{name}{suffix}", snap[ei, :, mi])
                # Robot state
                buf.append("joint_pos", jp[ei])
                buf.append("joint_vel", jv[ei])
                buf.append("base_pos", bp[ei])
                buf.append("base_quat", bq[ei])
                buf.append("base_lin_vel", blv[ei])
                buf.append("base_ang_vel", bav[ei])
                buf.append("actions", act_np[ei])
                buf.step_done()

            # Step environment
            obs, _, dones, _ = env.step(actions)
            dones_np = dones.cpu().numpy() if isinstance(dones, torch.Tensor) else np.asarray(dones)

            # Handle episode ends and max-length cutoffs
            for ei in range(num_envs):
                if dones_np[ei] or buffers[ei].length >= args_cli.max_traj_length:
                    if buffers[ei].length >= args_cli.min_traj_length:
                        completed.append(buffers[ei])
                        traj_counter += 1
                        if traj_counter % 10 == 0:
                            print(f"[INFO] {traj_counter}/{args_cli.num_trajectories} trajectories collected")
                        # Flush to disk periodically
                        if len(completed) >= args_cli.trajs_per_file:
                            fp = os.path.join(
                                output_subdir, f"roboset_{timestamp}_part{file_counter:03d}.h5"
                            )
                            save_trajectories(
                                completed, fp, file_counter * args_cli.trajs_per_file, metadata
                            )
                            completed = []
                            file_counter += 1
                        if traj_counter >= args_cli.num_trajectories:
                            break
                    buffers[ei] = TrajectoryBuffer()

    # Save remaining trajectories
    if completed:
        fp = os.path.join(output_subdir, f"roboset_{timestamp}_part{file_counter:03d}.h5")
        save_trajectories(completed, fp, file_counter * args_cli.trajs_per_file, metadata)

    print(f"[INFO] Complete: {traj_counter} trajectories saved to {output_subdir}")
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
