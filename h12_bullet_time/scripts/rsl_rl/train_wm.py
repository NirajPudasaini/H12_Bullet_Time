# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to train RL agent with RSL-RL."""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument("--video_interval", type=int, default=2000, help="Interval between video recordings (in steps).")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--agent", type=str, default="rsl_rl_cfg_entry_point", help="Name of the RL agent configuration entry point."
)
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument("--max_iterations", type=int, default=None, help="RL Policy training iterations.")
parser.add_argument(
    "--distributed", action="store_true", default=False, help="Run training with multiple GPUs or nodes."
)
parser.add_argument("--export_io_descriptors", action="store_true", default=False, help="Export IO descriptors.")
parser.add_argument("--wm_checkpoint", type=str, required=True, help="Full TOFWM checkpoint.")
parser.add_argument("--wm_output_checkpoint", type=str, default=None, help="Adapted checkpoint output path.")
parser.add_argument("--wm_config", type=str, default=None, help="Config for checkpoints without embedded config.")
parser.add_argument("--wm_mode", choices=("frozen", "alternating"), default="frozen")
parser.add_argument("--wm_contact_threshold", type=float, default=0.5)
parser.add_argument("--inference_frames", type=int, default=1)
parser.add_argument("--context_stride", type=int, default=1)
parser.add_argument(
    "--wm_rollout_frame", type=int, default=-1,
    help="1-based rollout frame passed as the future observation. "
         "2 is the second predicted frame. -1 uses the first predicted contact, else the last frame.",
)
parser.add_argument("--wm_batch_size", type=int, default=256, help="Inference micro-batch size.")
parser.add_argument("--wm_reduction", choices=("mean", "flatten"), default="mean")
parser.add_argument("--wm_ode_steps", type=int, default=None)
parser.add_argument("--wm_stochastic", action="store_true", help="Sample fresh flow noise at each inference.")
parser.add_argument("--wm_no_amp", action="store_true")
parser.add_argument("--wm_precision", type=str.lower, choices=("fp32", "fp16", "bf16"), default="fp16")
parser.add_argument("--wm_trajectories_per_cycle", type=int, default=4096, help="X trajectories.")
parser.add_argument("--wm_epochs_per_cycle", type=int, default=1, help="Y epochs.")
parser.add_argument("--wm_total_trajectories", type=int, default=40960, help="Z trajectories.")
parser.add_argument("--wm_data_dir", type=str, default="wm_robot_data")
parser.add_argument("--wm_replay_cycles", type=int, default=1)
parser.add_argument("--wm_train_encoder", action="store_true")
parser.add_argument("--wm_no_train_dynamics", action="store_true")
parser.add_argument("--wm_diag_interval", type=int, default=50, help="Print WM inference speed every N steps.")
parser.add_argument(
    "--wm_current_obs_type", type=str.lower, choices=("raw", "latent", "none"), default="latent",
    help="Current observation: raw sensor vector, encoder latent, or none. "
         "RAW follows ABLATION_SENSORS (DIST image or MINDIST per sensor). "
         "The world model still encodes the full distance image.",
)
parser.add_argument(
    "--wm_future_obs_type",
    type=lambda s: str(s).lower().replace("_", "-"),
    choices=("latent", "decoded", "min-decoded", "closest-point", "none"),
    default="latent",
    help="Selected future observation. DECODED is the full decoded distance image. "
         "MIN-DECODED is that image reduced to one closest distance per sensor.",
)
parser.add_argument(
    "--wm_raw_signals",
    default=None,
    help="Comma-separated DIST/MINDIST reductions for RAW. "
         "Default: ray signals from ABLATION_SENSORS.",
)
parser.add_argument(
    "--wm_contact_pred",
    type=lambda v: str(v).lower() in ("1", "true", "yes"),
    default=True,
    help="Append the WM contact-prediction flag to policy observations.",
)
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import json
import os
import time
import torch
from collections import deque
from datetime import datetime
from importlib.metadata import version as pkg_version

import omni
from rsl_rl.runners import DistillationRunner, OnPolicyRunner

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.dict import print_dict
from isaaclab.utils.io import dump_yaml

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

try:
    from trybrid_skin import TOFTrajectoryCollector, TOFWorldModel, resolve_raw_signals
except ImportError as exc:
    raise ImportError(
        "Install trybrid_skin_project into the Isaac Lab environment with "
        "`python -m pip install -e /path/to/trybrid_skin_project`."
    ) from exc

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False


_BLUE = "\033[94m"
_RESET = "\033[0m"


def _tof_name(name):
    if name.startswith("tof_sensor_"):
        return name
    parts = name.split("_")
    if len(parts) > 2 and parts[0] == "ray" and parts[1].isdigit():
        name = "_".join(parts[2:])
    return f"tof_sensor_{name}"


def _tof_frame(env):
    sensors = []
    for name, sensor in getattr(env.unwrapped.scene, "_sensors", {}).items():
        data = sensor.data
        if getattr(data, "dist_est_normalized", None) is None:
            continue
        pixels = getattr(sensor.cfg, "pixel_count", None)
        if pixels is None:
            continue
        pixels = int(pixels)
        values = data.dist_est_normalized.min(dim=2).values
        values = values.reshape(values.shape[0], values.shape[1], pixels, pixels)
        for index in range(values.shape[1]):
            sensors.append((f"{_tof_name(name)}_{index}", values[:, index]))
    if not sensors:
        raise RuntimeError("No ToF sensors with dist_est_normalized were found in the scene")
    sensors.sort(key=lambda item: item[0])
    names, frames = zip(*sensors)
    return torch.nan_to_num(torch.stack(frames, dim=-1), nan=1.0).unsqueeze(1), list(names)


def _joint_state(env):
    robot = env.unwrapped.scene["robot"]
    return robot.data.joint_pos, list(robot.joint_names)


class WorldModelVecEnvWrapper(RslRlVecEnvWrapper):
    def __init__(self, env, clip_actions, world_model, mode, data_dir, trajectories_per_cycle,
                 total_trajectories, epochs_per_cycle, replay_cycles, train_encoder, train_dynamics,
                 inference_frames=1, diag_interval=50):
        self.world_model = world_model
        self.mode = mode
        self.epochs_per_cycle = epochs_per_cycle
        self.train_encoder = train_encoder
        self.train_dynamics = train_dynamics
        self.inference_frames = int(inference_frames)
        if self.inference_frames < 1:
            raise ValueError("inference_frames must be >= 1")
        self.diag_interval = max(int(diag_interval), 0)
        self.replay = deque(maxlen=replay_cycles)
        self.collector = None
        self._infer_ms = 0.0
        self._infer_calls = 0
        self._infer_samples = 0
        self._infer_envs = 0
        self._step_count = 0
        super().__init__(env, clip_actions=clip_actions)
        if self.num_actions != self.world_model.action_dim:
            raise ValueError(
                f"Environment has {self.num_actions} actions, world model expects "
                f"{self.world_model.action_dim}"
            )
        self.frame, self.sensor_names = _tof_frame(self)
        joint_pos, joint_names = _joint_state(self)
        self.features = self._timed_infer(
            lambda: self.world_model.initialize(
                self.frame, joint_pos=joint_pos, joint_names=joint_names,
                sensor_names=self.sensor_names))
        if mode == "alternating":
            step_dt = getattr(self.unwrapped, "step_dt", None) or (
                self.unwrapped.cfg.sim.dt * self.unwrapped.cfg.decimation
            )
            self.collector = TOFTrajectoryCollector(
                data_dir,
                self.sensor_names,
                trajectories_per_cycle,
                total_trajectories,
                self.world_model.context_frames + self.world_model.rollout_frames,
                1.0 / float(step_dt),
            )

    def _augment(self, observations):
        for key in ("policy", "critic"):
            if key in observations:
                observations[key] = torch.cat((observations[key], self.features), dim=-1)
        return observations

    def get_observations(self):
        return self._augment(super().get_observations())

    def step(self, actions):
        previous = self.frame
        applied_actions = actions if self.clip_actions is None else torch.clamp(
            actions, -self.clip_actions, self.clip_actions)
        observations, rewards, dones, extras = super().step(actions)
        path = self.collector.add(previous, dones) if self.collector is not None else None
        if path:
            self.replay.append(path)
            print(
                f"[INFO] Training world model on cycle {self.collector.cycle}: "
                f"{self.collector.completed}/{self.collector.total_trajectories} trajectories"
            )
            self.world_model.train_on_h5(
                list(self.replay),
                self.epochs_per_cycle,
                train_encoder=self.train_encoder,
                train_dynamics=self.train_dynamics,
            )
        self.frame, names = _tof_frame(self)
        if names != self.sensor_names:
            raise RuntimeError("ToF sensor ordering changed during training")
        self._step_count += 1
        refresh = self._step_count % self.inference_frames == 0
        joint_pos, joint_names = _joint_state(self)
        infer = lambda: self.world_model.advance(
            self.frame, applied_actions, dones.bool(), refresh=refresh,
            joint_pos=joint_pos, joint_names=joint_names, sensor_names=self.sensor_names)
        self.features = self._timed_infer(infer) if refresh else infer()
        return self._augment(observations), rewards, dones, extras

    def _timed_infer(self, fn):
        measure = (
            self.diag_interval and self._infer_calls >= 5
            and (self._infer_calls - 5) % self.diag_interval == 0
        )
        if measure and self.world_model.device.type == "cuda":
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record()
            features = fn()
            end.record()
            end.synchronize()
            elapsed_ms = start.elapsed_time(end)
        elif measure:
            start = time.perf_counter()
            features = fn()
            elapsed_ms = (time.perf_counter() - start) * 1000.0
        else:
            features = fn()
        if measure:
            self._infer_ms += elapsed_ms
            self._infer_envs += features.shape[0]
            self._infer_samples += 1
        self._infer_calls += 1
        if measure:
            avg_ms = self._infer_ms / self._infer_samples
            hz = 1000.0 / avg_ms
            envs_per_s = self._infer_envs / (self._infer_ms / 1000.0)
            print(
                f"{_BLUE}[WM] sampled inference {avg_ms:.2f} ms/call "
                f"({hz:.1f} Hz, {envs_per_s:.0f} envs/s, {self._infer_calls} calls){_RESET}"
            )
        return features

    def inference_stats(self):
        avg_ms = self._infer_ms / self._infer_samples if self._infer_samples else None
        return {
            "avg_inference_ms": avg_ms,
            "avg_inference_hz": (1000.0 / avg_ms) if avg_ms else None,
            "envs_per_s": (self._infer_envs / (self._infer_ms / 1000.0)) if self._infer_ms else None,
            "infer_steps": self._infer_calls,
        }


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg, agent_cfg):
    """Train with RSL-RL agent."""
    # override configurations with non-hydra CLI arguments
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, pkg_version("rsl-rl-lib"))
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    agent_cfg.max_iterations = (
        args_cli.max_iterations if args_cli.max_iterations is not None else agent_cfg.max_iterations
    )

    # set the environment seed
    # note: certain randomizations occur in the environment initialization so we set the seed here
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    if args_cli.wm_mode == "alternating":
        raise ValueError("Alternating WM training does not yet collect aligned action/contact labels")
    if min(args_cli.wm_trajectories_per_cycle, args_cli.wm_total_trajectories,
           args_cli.wm_epochs_per_cycle, args_cli.wm_replay_cycles) < 1:
        raise ValueError("World-model cycle, total, epoch, and replay values must be positive")
    if args_cli.wm_diag_interval < 0:
        raise ValueError("--wm_diag_interval must be >= 0")
    if args_cli.inference_frames < 1:
        raise ValueError("--inference_frames must be >= 1")
    if args_cli.context_stride < 1:
        raise ValueError("--context_stride must be >= 1")
    if args_cli.wm_rollout_frame == 0 or args_cli.wm_rollout_frame < -1:
        raise ValueError("--wm_rollout_frame must be -1 or a positive 1-based frame")

    # multi-gpu training configuration
    if args_cli.distributed:
        env_cfg.sim.device = f"cuda:{app_launcher.local_rank}"
        agent_cfg.device = f"cuda:{app_launcher.local_rank}"

        # set seed to have diversity in different threads
        seed = agent_cfg.seed + app_launcher.local_rank
        env_cfg.seed = seed
        agent_cfg.seed = seed

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Logging experiment in directory: {log_root_path}")
    # specify directory for logging runs: {time-stamp}_{run_name}
    log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    # The Ray Tune workflow extracts experiment name using the logging line below, hence, do not change it (see PR #2346, comment-2819298849)
    print(f"Exact experiment name requested from command line: {log_dir}")
    if agent_cfg.run_name:
        log_dir += f"_{agent_cfg.run_name}"
    log_dir = os.path.join(log_root_path, log_dir)
    print(f"[INFO] Run log directory: {log_dir}")

    # set the IO descriptors export flag if requested
    if isinstance(env_cfg, ManagerBasedRLEnvCfg):
        env_cfg.export_io_descriptors = args_cli.export_io_descriptors
    else:
        omni.log.warn(
            "IO descriptors are only supported for manager based RL environments. No IO descriptors will be exported."
        )

    # set the log directory for the environment (works for all environment types)
    env_cfg.log_dir = log_dir

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # save resume path before creating a new log_dir
    if agent_cfg.resume:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
        resume_path = handle_deprecated_rsl_rl_checkpoint(resume_path, pkg_version("rsl-rl-lib"))

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "train"),
            "step_trigger": lambda step: step % args_cli.video_interval == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    if args_cli.wm_raw_signals is None:
        args_cli.wm_raw_signals = ",".join(
            resolve_raw_signals(spec=os.environ.get("ABLATION_SENSORS", ""))
        )
    raw_signals = resolve_raw_signals(args_cli.wm_raw_signals)
    world_model = TOFWorldModel(
        checkpoint=args_cli.wm_checkpoint,
        output_checkpoint=args_cli.wm_output_checkpoint
        or f"{os.path.splitext(args_cli.wm_checkpoint)[0]}_robot.pt",
        config=args_cli.wm_config,
        device=agent_cfg.device,
        inference_batch_size=args_cli.wm_batch_size,
        reduction=args_cli.wm_reduction,
        deterministic=not args_cli.wm_stochastic,
        seed=agent_cfg.seed,
        ode_steps=args_cli.wm_ode_steps,
        amp=not args_cli.wm_no_amp,
        precision="fp32" if args_cli.wm_no_amp else args_cli.wm_precision,
        contact_threshold=args_cli.wm_contact_threshold,
        context_stride=args_cli.context_stride,
        current_obs_type=args_cli.wm_current_obs_type,
        future_obs_type=args_cli.wm_future_obs_type,
        include_contact=args_cli.wm_contact_pred,
        raw_signals=raw_signals,
        rollout_frame=args_cli.wm_rollout_frame,
    )
    env = WorldModelVecEnvWrapper(
        env,
        agent_cfg.clip_actions,
        world_model,
        args_cli.wm_mode,
        os.path.join(log_dir, args_cli.wm_data_dir),
        args_cli.wm_trajectories_per_cycle,
        args_cli.wm_total_trajectories,
        args_cli.wm_epochs_per_cycle,
        args_cli.wm_replay_cycles,
        args_cli.wm_train_encoder,
        not args_cli.wm_no_train_dynamics,
        args_cli.inference_frames,
        args_cli.wm_diag_interval,
    )
    wm_name = os.path.splitext(os.path.basename(args_cli.wm_checkpoint))[0]
    wm_size = sum(p.numel() for p in world_model.model.parameters())
    print(
        f"[INFO] Added {world_model.feature_dim} world-model features "
        f"(current={args_cli.wm_current_obs_type}, future={args_cli.wm_future_obs_type}, "
        f"rollout_frame={args_cli.wm_rollout_frame}, contact={args_cli.wm_contact_pred})"
    )
    print(f"[INFO] World model {wm_name}: {wm_size} parameters")

    # create runner from rsl-rl
    if agent_cfg.class_name == "OnPolicyRunner":
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)
    elif agent_cfg.class_name == "DistillationRunner":
        runner = DistillationRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)
    else:
        raise ValueError(f"Unsupported runner class: {agent_cfg.class_name}")
    # write git state to logs
    runner.add_git_repo_to_log(__file__)
    # load the checkpoint if resuming
    if agent_cfg.resume:
        print(f"[INFO]: Loading model checkpoint from: {resume_path}")
        runner.load(resume_path)

    # dump the configuration into log-directory
    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)
    dump_yaml(
        os.path.join(log_dir, "params", "world_model.yaml"),
        {key: value for key, value in vars(args_cli).items()
         if key.startswith("wm_") or key in ("inference_frames", "context_stride")},
    )

    try:
        runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)
        stats = {
            "wm_name": wm_name,
            "wm_size": wm_size,
            **env.inference_stats(),
        }
        print(f"[WM_STATS] {json.dumps(stats)}")
    finally:
        env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
