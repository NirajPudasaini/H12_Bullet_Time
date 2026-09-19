"""Play a WM-conditioned PPO policy, saving sim video + WM-observation GIFs.

Sim video comes from ``gym.wrappers.RecordVideo``. WM GIFs visualize exactly what
the robot's policy saw from the world model on env 0 over the recorded steps:

- ``wm_current_decoded.gif``: current latents decoded through the loaded WM
  encoder (``decode_from_latents``), stitched along the sensor axis.
- ``wm_future_decoded.gif``: selected future latents decoded the same way.
- ``wm_contact.gif``: solid green (no predicted contact in rollout) / red
  (``contact_flag == 1``) square per step, sized to match the decoded frames.

Also writes ``wm_obs_diagnostics.json`` proving the WM features actually fed the
policy (obs dims, feature drift, refresh rate, contact rate, latency).
"""

import argparse
import json
import math
import os
import sys
import time
from datetime import datetime
from importlib.metadata import version as pkg_version
from pathlib import Path

from isaaclab.app import AppLauncher

import cli_args  # isort: skip

parser = argparse.ArgumentParser(description="Record sim video + WM observation GIFs.")
parser.add_argument("--task", type=str, default="Template-H12-Survive-Time-WM")
parser.add_argument("--agent", type=str, default="rsl_rl_cfg_entry_point")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--seed", type=int, default=None)
parser.add_argument("--video_length", type=int, default=300)
parser.add_argument("--video_folder", type=str, default=None)
parser.add_argument("--video_name_prefix", type=str, default="wm-record")
parser.add_argument("--out_dir", type=str, default=None, help="GIF + diagnostics dir (default: video folder).")
parser.add_argument("--gif_fps", type=float, default=20.0)
parser.add_argument("--contact_gif_size", type=int, default=0, help="0 = match decoded frame size.")
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
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import numpy as np
import torch
from rsl_rl.runners import DistillationRunner, OnPolicyRunner

from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent
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

try:
    from trybrid_skin import TOFWorldModel
except ImportError as exc:
    raise ImportError(
        "Install trybrid_skin_project with `python -m pip install -e /path/to/trybrid_skin_project`."
    ) from exc


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


class WMRecordVecEnv(RslRlVecEnvWrapper):
    def __init__(self, env, clip_actions, world_model, inference_frames):
        self.world_model = world_model
        self.inference_frames = int(inference_frames)
        self.step_count = 0
        self.last_refreshed = True
        super().__init__(env, clip_actions=clip_actions)
        if self.num_actions != world_model.action_dim:
            raise ValueError(f"Env has {self.num_actions} actions, WM expects {world_model.action_dim}")
        self.frame, self.sensor_names = _tof_frame(self)
        self.base_obs_dim = super().get_observations()["policy"].shape[-1]
        self.features = world_model.initialize(self.frame)
        self.aug_obs_dim = self.base_obs_dim + world_model.feature_dim

    def _augment(self, observations):
        observations["policy"] = torch.cat((observations["policy"], self.features), -1)
        return observations

    def get_observations(self):
        return self._augment(super().get_observations())

    def step(self, actions):
        applied = actions if self.clip_actions is None else actions.clamp(-self.clip_actions, self.clip_actions)
        observations, rewards, dones, extras = super().step(actions)
        self.frame, names = _tof_frame(self)
        if names != self.sensor_names:
            raise RuntimeError("ToF sensor ordering changed during rollout")
        self.step_count += 1
        self.last_refreshed = self.step_count % self.inference_frames == 0
        self.features = self.world_model.advance(self.frame, applied, dones.bool(), refresh=self.last_refreshed)
        return self._augment(observations), rewards, dones, extras


def _stitch_sensor_grid(frames):
    """(T, C, H, W, S) -> (T, C, H', W') tiled along sensor axis."""
    T, C, H, W, S = frames.shape
    n = math.ceil(math.sqrt(S))
    canvas = np.ones((T, C, n * H, n * W), dtype=np.float32)
    for s in range(S):
        r, c = divmod(s, n)
        canvas[:, :, r * H : (r + 1) * H, c * W : (c + 1) * W] = frames[:, :, :, :, s].numpy()
    return torch.from_numpy(canvas)


def _to_uint8_grayscale(stitched):
    frames = stitched[:, 0].clamp(0, 1).mul(255).round().byte().cpu().numpy()
    return [np.stack([f] * 3, -1) for f in frames]


def _save_gif(frames_rgb, path, fps):
    path = str(path)
    try:
        import imageio.v2 as imageio

        imageio.mimsave(path, frames_rgb, fps=float(fps))
        return
    except ImportError:
        pass
    from PIL import Image

    imgs = [Image.fromarray(f) for f in frames_rgb]
    imgs[0].save(path, save_all=True, append_images=imgs[1:], duration=int(1000 / fps), loop=0)


@torch.inference_mode()
def _decode_latents(world_model, latents, spatial_shape, batch=32):
    encoder = world_model.model.encoder
    out = []
    lat = torch.as_tensor(np.stack(latents), dtype=torch.float32, device=world_model.device)
    for start in range(0, len(lat), batch):
        out.append(encoder.decode_from_latents(lat[start : start + batch], spatial_shape).cpu())
    return torch.cat(out)


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg, agent_cfg):
    agent_cfg = handle_deprecated_rsl_rl_cfg(cli_args.update_rsl_rl_cfg(agent_cfg, args_cli), pkg_version("rsl-rl-lib"))
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device or env_cfg.sim.device

    log_root = os.path.abspath(os.path.join("logs", "rsl_rl", agent_cfg.experiment_name))
    if args_cli.checkpoint:
        policy_checkpoint = retrieve_file_path(args_cli.checkpoint)
    else:
        if args_cli.load_run is None:
            runs = [p for p in Path(log_root).iterdir() if p.is_dir()]
            args_cli.load_run = max(runs, key=lambda p: p.stat().st_mtime).name
        policy_checkpoint = get_checkpoint_path(log_root, args_cli.load_run, agent_cfg.load_checkpoint)
    policy_checkpoint = handle_deprecated_rsl_rl_checkpoint(policy_checkpoint, pkg_version("rsl-rl-lib"))
    policy_dir = os.path.dirname(policy_checkpoint)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    video_folder = args_cli.video_folder or os.path.join(policy_dir, "videos", f"wm_record_{timestamp}")
    out_dir = Path(args_cli.out_dir or video_folder)
    out_dir.mkdir(parents=True, exist_ok=True)

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array")
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)
    env = gym.wrappers.RecordVideo(
        env,
        video_folder=str(video_folder),
        step_trigger=lambda step: step == 0,
        video_length=args_cli.video_length,
        disable_logger=True,
        name_prefix=args_cli.video_name_prefix,
    )

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
    )
    env = WMRecordVecEnv(env, agent_cfg.clip_actions, world_model, args_cli.inference_frames)
    print(f"[WM_VERIFY] base_obs_dim={env.base_obs_dim} wm_feature_dim={world_model.feature_dim} "
          f"aug_obs_dim={env.aug_obs_dim} (policy sees base + WM)")
    assert env.aug_obs_dim == env.base_obs_dim + world_model.feature_dim

    if agent_cfg.class_name == "OnPolicyRunner":
        runner_cls = OnPolicyRunner
    elif agent_cfg.class_name == "DistillationRunner":
        runner_cls = DistillationRunner
    else:
        raise ValueError(f"Unsupported runner class: {agent_cfg.class_name}")
    runner = runner_cls(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(policy_checkpoint)
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    with torch.no_grad():
        shape_frame = env.world_model.history[:, -1] if env.world_model.history is not None else None
    _, spatial_shape = world_model.model.encoder.encode(shape_frame[:1].to(world_model.device))
    print(f"[WM_VERIFY] spatial_shape={tuple(spatial_shape)} rollout_frames={world_model.rollout_frames} "
          f"context_frames={world_model.context_frames} threshold={world_model.contact_threshold}")

    current, future, flags, probed, refreshed = [], [], [], [], []
    feat_norms, infer_ms = [], []
    obs = env.get_observations()
    for _ in range(args_cli.video_length):
        if not simulation_app.is_running():
            break
        with torch.inference_mode():
            start = time.perf_counter()
            actions = policy(obs)
            obs, _, _, _ = env.step(actions)
            infer_ms.append((time.perf_counter() - start) * 1000.0)
            current.append(world_model.latent_history[0, -1].cpu().numpy())
            future.append(world_model.future_latent[0].cpu().numpy())
            flags.append(float(world_model.contact_flag[0, 0].cpu()))
            probed.append(world_model.contact_probabilities[0].cpu().numpy())
            refreshed.append(bool(env.last_refreshed))
            feat_norms.append(float(world_model.future_latent[0].float().norm().cpu()))

    T = len(current)
    current = np.stack(current)
    future = np.stack(future)
    flags = np.asarray(flags, dtype=np.float32)
    probed = np.stack(probed)
    cur_dec = _decode_latents(world_model, current, spatial_shape)
    fut_dec = _decode_latents(world_model, future, spatial_shape)
    cur_stitched = _stitch_sensor_grid(cur_dec)
    fut_stitched = _stitch_sensor_grid(fut_dec)

    _save_gif(_to_uint8_grayscale(cur_stitched), out_dir / "wm_current_decoded.gif", args_cli.gif_fps)
    _save_gif(_to_uint8_grayscale(fut_stitched), out_dir / "wm_future_decoded.gif", args_cli.gif_fps)
    h, w = cur_stitched.shape[-2:]
    if args_cli.contact_gif_size > 0:
        h = w = int(args_cli.contact_gif_size)
    contact_frames = [
        np.full((h, w, 3), (220, 30, 30) if f >= 0.5 else (40, 180, 60), dtype=np.uint8) for f in flags
    ]
    _save_gif(contact_frames, out_dir / "wm_contact.gif", args_cli.gif_fps)

    cur_fut_dist = float(np.linalg.norm((current - future).reshape(T, -1), axis=1).mean())
    diagnostics = {
        "policy_checkpoint": policy_checkpoint,
        "wm_checkpoint": os.path.abspath(args_cli.wm_checkpoint),
        "task": args_cli.task,
        "steps": T,
        "base_obs_dim": env.base_obs_dim,
        "wm_feature_dim": world_model.feature_dim,
        "aug_obs_dim": env.aug_obs_dim,
        "tokens_per_frame": world_model.model.dynamics.tokens_per_frame,
        "embedding_dim": world_model.model.dynamics.e_dim,
        "rollout_frames": world_model.rollout_frames,
        "contact_threshold": world_model.contact_threshold,
        "contact_flag_rate": float((flags >= 0.5).mean()) if T else 0.0,
        "refresh_rate": float(np.mean(refreshed)) if T else 0.0,
        "mean_future_norm": float(np.mean(feat_norms)) if T else 0.0,
        "std_future_norm": float(np.std(feat_norms)) if T else 0.0,
        "mean_current_future_dist": cur_fut_dist,
        "decoded_shape": list(cur_dec.shape),
        "video_folder": str(video_folder),
        "files": ["wm_current_decoded.gif", "wm_future_decoded.gif", "wm_contact.gif"],
    }
    with open(out_dir / "wm_obs_diagnostics.json", "w") as f:
        json.dump(diagnostics, f, indent=2)
    print(f"[WM_VERIFY] steps={T} flag_rate={diagnostics['contact_flag_rate']:.3f} "
          f"refresh_rate={diagnostics['refresh_rate']:.3f} future_norm={diagnostics['mean_future_norm']:.3f} "
          f"+/-{diagnostics['std_future_norm']:.3f} cur_fut_dist={cur_fut_dist:.3f}")
    print(f"[WM_VERIFY] {'PASS' if diagnostics['std_future_norm'] > 0 and cur_fut_dist > 0 else 'FAIL'}: "
          f"WM features {'vary over time and differ current vs future' if diagnostics['std_future_norm'] > 0 else 'look static!'}. "
          f"Video: {video_folder} GIFs: {out_dir}")
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
