"""Play a WM-conditioned PPO policy, saving sim video + one WM-observation figure.

What env 0's policy saw is written as timestamp-aligned figures:

- ``wm_observations.png``: next true contact frame, current decode, plus selected future decode and contact_flag when present
- ``wm_rollout.png``: all R rollout steps stacked vertically above current, when a future rollout exists


Also writes ``wm_obs_diagnostics.json``.
"""

import argparse
import json
import math
import os
import sys
import time
from contextlib import nullcontext
from datetime import datetime
from importlib.metadata import version as pkg_version
from pathlib import Path

from isaaclab.app import AppLauncher

import cli_args  # isort: skip

parser = argparse.ArgumentParser(description="Record sim video + WM observation figure.")
parser.add_argument("--task", type=str, default="Template-H12-Survive-Time-WM")
parser.add_argument("--agent", type=str, default="rsl_rl_cfg_entry_point")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--seed", type=int, default=None)
parser.add_argument("--video_length", type=int, default=300)
parser.add_argument("--video_folder", type=str, default=None)
parser.add_argument("--video_name_prefix", type=str, default="wm-record")
parser.add_argument("--out_dir", type=str, default=None, help="Figure + diagnostics dir (default: video folder).")
parser.add_argument("--wm_checkpoint", type=str, default=None, help="Defaults to the run's params/world_model.yaml.")
parser.add_argument("--wm_config", type=str, default=None)
parser.add_argument("--wm_contact_threshold", type=float, default=None, help="Defaults to the run log.")
parser.add_argument("--wm_batch_size", type=int, default=None)
parser.add_argument("--wm_reduction", choices=("mean", "flatten"), default=None)
parser.add_argument("--wm_ode_steps", type=int, default=None)
parser.add_argument("--wm_stochastic", action="store_true")
parser.add_argument("--wm_no_amp", action="store_true")
parser.add_argument("--wm_precision", type=str.lower, choices=("fp32", "fp16", "bf16"), default=None)
parser.add_argument("--inference_frames", type=int, default=None, help="Defaults to the run log.")
parser.add_argument("--context_stride", type=int, default=None, help="Defaults to the run log.")
parser.add_argument("--wm_raw_signals", default=None, help="Defaults to the run log.")
parser.add_argument("--wm_current_obs_type", default=None, help="Defaults to the run log.")
parser.add_argument("--wm_future_obs_type", default=None, help="Defaults to the run log.")
parser.add_argument("--wm_contact_pred", default=None, help="Defaults to the run log.")
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
# rgb_array / RecordVideo needs the offscreen render pipeline, which Isaac Lab
# only enables when cameras AND headless are both set (see AppLauncher).
args_cli.enable_cameras = True
args_cli.headless = True
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import torch
from rsl_rl.runners import DistillationRunner, OnPolicyRunner

from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.io import load_yaml
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
    from trybrid_skin import TOFWorldModel, resolve_raw_signals
except ImportError as exc:
    raise ImportError(
        "Install trybrid_skin_project with `python -m pip install -e /path/to/trybrid_skin_project`."
    ) from exc


_CONTACT_TERM_FUNCS = (
    "multi_contact_termination",
    "contact_termination",
    "sensor_based_contact_termination",
)


def _env0_in_contact(env) -> bool:
    manager = getattr(env, "termination_manager", None)
    if manager is None:
        return False
    for name in manager.active_terms:
        cfg = manager.get_term_cfg(name)
        if getattr(cfg.func, "__name__", "") not in _CONTACT_TERM_FUNCS:
            continue
        if bool(manager.get_term(name)[0].item()):
            return True
    return False


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
        robot = self.unwrapped.scene["robot"]
        self.features = world_model.initialize(
            self.frame, joint_pos=robot.data.joint_pos, joint_names=list(robot.joint_names),
            sensor_names=self.sensor_names)
        self.aug_obs_dim = self.base_obs_dim + world_model.feature_dim
        self.contact_frame = None
        self._install_contact_snapshot()

    def _install_contact_snapshot(self):
        """Save env 0's ToF image at a contact reset, before the scene reset clears it."""
        env = self.unwrapped
        original = env._reset_idx
        wrapper = self

        def _reset_idx(env_ids):
            ids = torch.as_tensor(env_ids, device=env.device).reshape(-1)
            if ids.numel() and bool((ids == 0).any().item()) and _env0_in_contact(env):
                frame, _ = _tof_frame(wrapper)
                wrapper.contact_frame = frame[0].detach().cpu().clone()
            original(env_ids)

        env._reset_idx = _reset_idx

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
            raise RuntimeError("ToF sensor ordering changed during rollout")
        self.step_count += 1
        self.last_refreshed = self.step_count % self.inference_frames == 0
        robot = self.unwrapped.scene["robot"]
        self.features = self.world_model.advance(
            self.frame, applied, dones.bool(), refresh=self.last_refreshed,
            joint_pos=robot.data.joint_pos, joint_names=list(robot.joint_names),
            sensor_names=self.sensor_names)
        return self._augment(observations), rewards, dones, extras


def _to_numpy(x):
    if torch.is_tensor(x):
        return x.detach().float().cpu().numpy()
    return np.asarray(x, dtype=np.float32)


def _stitch_sensor_grid(frames):
    """(T, C, H, W, S) -> (T, C, H', W') tiled along sensor axis."""
    frames = torch.as_tensor(_to_numpy(frames))
    T, C, H, W, S = frames.shape
    n = math.ceil(math.sqrt(S))
    canvas = np.ones((T, C, n * H, n * W), dtype=np.float32)
    for s in range(S):
        r, c = divmod(s, n)
        canvas[:, :, r * H : (r + 1) * H, c * W : (c + 1) * W] = frames[:, :, :, :, s].numpy()
    return torch.from_numpy(canvas)


def _depth_strip(stitched):
    """(T, C, H, W) -> (H, T*W) grayscale strip, frames left-to-right."""
    frames = _to_numpy(stitched[:, 0].clamp(0, 1))
    return np.concatenate(list(frames), axis=1)


def _contact_strip(flags, height, width):
    """(T,) flags -> (H, T*W, 3) green/red blocks aligned with decoded frames."""
    blocks = [
        np.full((height, width, 3), (0.86, 0.12, 0.12) if f >= 0.5 else (0.16, 0.71, 0.24), dtype=np.float32)
        for f in flags
    ]
    return np.concatenate(blocks, axis=1) if blocks else np.zeros((height, 1, 3), dtype=np.float32)


def _rollout_strip(stitched, T, R):
    """(T*R, C, H, W) in (t, r) order -> (R*H, T*W), rows = rollout step."""
    _, _, H, W = stitched.shape
    frames = _to_numpy(stitched[:, 0].clamp(0, 1)).reshape(T, R, H, W)
    return np.concatenate(
        [np.concatenate([frames[t, r] for t in range(T)], axis=1) for r in range(R)],
        axis=0,
    )


def _time_markers(axes, T, W):
    xmax = T * W
    for ax in axes:
        ax.set_xlim(0, xmax)
        for t in range(T + 1):
            ax.axvline(t * W, color="0.25", ls=":", lw=0.5, zorder=3)
    tick_idx = np.arange(T)
    axes[-1].set_xticks((tick_idx + 0.5) * W)
    axes[-1].set_xticklabels(tick_idx)
    for ax in axes[:-1]:
        ax.tick_params(labelbottom=False)
    axes[-1].set_xlabel("Timestep")


def _hold_contact_frames(contact_stitched, contact_steps, dones, shape):
    """Repeat each contact image from the previous reset through that contact step."""
    canvas = torch.ones(shape, dtype=torch.float32)
    if contact_stitched is None or len(contact_steps) == 0:
        return canvas
    if tuple(contact_stitched.shape[1:]) != tuple(shape[1:]):
        raise RuntimeError(
            f"Contact frame stitch {tuple(contact_stitched.shape[1:])} "
            f"does not match decoded frames {tuple(shape[1:])}"
        )
    frames = contact_stitched.detach().float().cpu()
    start = 0
    cursor = 0
    for t, done in enumerate(dones):
        if not done:
            continue
        if cursor < len(contact_steps) and contact_steps[cursor] == t:
            canvas[start : t + 1] = frames[cursor]
            cursor += 1
        start = t + 1
    return canvas


def _save_obs_figure(cur_stitched, path, fut_stitched=None, flags=None, true_contact=None):
    T, _, H, W = cur_stitched.shape
    strips = []
    heights = []
    if true_contact is not None:
        strips.append(("next true contact frame", _depth_strip(true_contact)))
        heights.append(true_contact.shape[-2])
    strips.append(("Current", _depth_strip(cur_stitched)))
    heights.append(H)
    if fut_stitched is not None:
        strips.append(("Future", _depth_strip(fut_stitched)))
        heights.append(H)
    if flags is not None:
        contact_h = max(H // 4, 8)
        strips.append(("Contact", _contact_strip(flags, contact_h, W)))
        heights.append(contact_h)
    dpi = 100
    left_in = 1.9 if true_contact is not None else 0.7
    bottom_in, top_in = 0.4, 0.12
    fig_w = strips[0][1].shape[1] / dpi + left_in
    fig_h = sum(heights) / dpi + bottom_in + top_in
    fig, axes = plt.subplots(
        len(strips), 1, figsize=(fig_w, fig_h), dpi=dpi, sharex=True, squeeze=False,
        gridspec_kw={"height_ratios": heights, "hspace": 0.04},
    )
    axes = axes[:, 0]
    fig.subplots_adjust(
        left=left_in / fig_w, right=1.0, bottom=bottom_in / fig_h, top=1.0 - top_in / fig_h,
    )
    for ax, (label, image) in zip(axes, strips):
        if image.ndim == 3:
            ax.imshow(image, interpolation="nearest", aspect="equal")
        else:
            ax.imshow(image, cmap="gray", vmin=0, vmax=1, interpolation="nearest", aspect="equal")
        if len(label) > 10:
            ax.set_ylabel(label, rotation=0, ha="right", va="center", labelpad=6)
        else:
            ax.set_ylabel(label)
        ax.set_yticks([])
    _time_markers(axes, T, W)
    fig.savefig(path, dpi=dpi, pad_inches=0.02)
    plt.close(fig)


def _save_rollout_figure(roll_stitched, cur_stitched, T, R, path):
    H, W = cur_stitched.shape[-2:]
    rollout = _rollout_strip(roll_stitched, T, R)
    current = _depth_strip(cur_stitched)
    dpi = 100
    left_in, bottom_in, top_in = 0.85, 0.4, 0.12
    fig_w = current.shape[1] / dpi + left_in
    fig_h = (R * H + H) / dpi + bottom_in + top_in
    fig, axes = plt.subplots(
        2, 1, figsize=(fig_w, fig_h), dpi=dpi, sharex=True,
        gridspec_kw={"height_ratios": [R * H, H], "hspace": 0.04},
    )
    fig.subplots_adjust(
        left=left_in / fig_w, right=1.0, bottom=bottom_in / fig_h, top=1.0 - top_in / fig_h,
    )
    axes[0].imshow(rollout, cmap="gray", vmin=0, vmax=1, interpolation="nearest", aspect="equal")
    axes[1].imshow(current, cmap="gray", vmin=0, vmax=1, interpolation="nearest", aspect="equal")
    axes[0].set_ylabel("Rollout")
    axes[1].set_ylabel("Current")
    axes[0].set_yticks([(r + 0.5) * H for r in range(R)])
    axes[0].set_yticklabels([str(r) for r in range(R)])
    axes[1].set_yticks([])
    for r in range(R + 1):
        axes[0].axhline(r * H, color="0.25", ls=":", lw=0.5, zorder=3)
    _time_markers(axes, T, W)
    fig.savefig(path, dpi=dpi, pad_inches=0.02)
    plt.close(fig)


@torch.inference_mode()
def _decode_latents(world_model, latents, spatial_shape, batch=32):
    encoder = world_model.model.encoder
    out = []
    lat = torch.as_tensor(np.stack(latents), device=world_model.device).to(world_model.storage_dtype)
    amp = torch.autocast("cuda", dtype=world_model.amp_dtype) if world_model.amp else nullcontext()
    for start in range(0, len(lat), batch):
        with amp:
            out.append(encoder.decode_from_latents(lat[start : start + batch], spatial_shape).float().cpu())
    return torch.cat(out)


def _latest_run(log_root):
    root = Path(log_root)
    if not root.is_dir():
        raise FileNotFoundError(
            f"No PPO logs at {root}. Train first, or pass --checkpoint to a model.pt."
        )
    runs = [path for path in root.iterdir() if path.is_dir()]
    if not runs:
        raise FileNotFoundError(f"No PPO runs in {root}. Train first, or pass --checkpoint.")
    return max(runs, key=lambda path: path.stat().st_mtime)


def _resolve_policy_checkpoint(agent_cfg):
    log_root = os.path.abspath(os.path.join("logs", "rsl_rl", agent_cfg.experiment_name))
    if args_cli.checkpoint:
        path = Path(args_cli.checkpoint)
        if path.is_file():
            return str(path.resolve())
        if path.is_dir():
            models = sorted(path.glob("model*.pt"), key=lambda p: f"{p.name:0>15}")
            if models:
                return str(models[-1].resolve())
        if "/" not in args_cli.checkpoint.replace("\\", "/"):
            load_run = args_cli.load_run or _latest_run(log_root).name
            return get_checkpoint_path(log_root, load_run, args_cli.checkpoint)
        try:
            return retrieve_file_path(args_cli.checkpoint)
        except FileNotFoundError as exc:
            raise FileNotFoundError(
                f"Unable to find checkpoint {args_cli.checkpoint!r}. "
                f"Omit --checkpoint to use the latest run under {log_root}."
            ) from exc
    load_run = args_cli.load_run
    if load_run is None:
        load_run = _latest_run(log_root).name
        print(f"[INFO] Using latest PPO run: {load_run}")
    return get_checkpoint_path(log_root, load_run, agent_cfg.load_checkpoint)


def _apply_saved_wm_args(policy_dir):
    yaml_path = Path(policy_dir) / "params" / "world_model.yaml"
    saved = load_yaml(str(yaml_path)) if yaml_path.is_file() else {}
    if yaml_path.is_file():
        print(f"[INFO] Loaded WM settings from {yaml_path}")
    fills = {
        "wm_checkpoint": saved.get("wm_checkpoint"),
        "wm_config": saved.get("wm_config"),
        "wm_contact_threshold": saved.get("wm_contact_threshold"),
        "inference_frames": saved.get("inference_frames"),
        "context_stride": saved.get("context_stride"),
        "wm_batch_size": saved.get("wm_batch_size"),
        "wm_reduction": saved.get("wm_reduction"),
        "wm_ode_steps": saved.get("wm_ode_steps"),
        "wm_precision": saved.get("wm_precision"),
        "wm_raw_signals": saved.get("wm_raw_signals"),
        "wm_current_obs_type": saved.get("wm_current_obs_type"),
        "wm_future_obs_type": saved.get("wm_future_obs_type"),
        "wm_contact_pred": saved.get("wm_contact_pred"),
    }
    defaults = {
        "wm_contact_threshold": 0.5,
        "inference_frames": 1,
        "context_stride": 1,
        "wm_batch_size": 256,
        "wm_reduction": "mean",
        "wm_precision": "fp16",
        "wm_current_obs_type": "latent",
        "wm_future_obs_type": "latent",
        "wm_contact_pred": True,
    }
    for name, value in fills.items():
        if getattr(args_cli, name) is None and value not in (None, ""):
            setattr(args_cli, name, value)
    for name, value in defaults.items():
        if getattr(args_cli, name) is None:
            setattr(args_cli, name, value)
    for flag in ("wm_stochastic", "wm_no_amp"):
        if not getattr(args_cli, flag) and saved.get(flag) in (True, "1", "true", "yes"):
            setattr(args_cli, flag, True)
    if not args_cli.wm_checkpoint:
        raise FileNotFoundError(
            f"No WM checkpoint on the CLI or in {yaml_path}. Pass --wm_checkpoint."
        )
    args_cli.inference_frames = int(args_cli.inference_frames)
    args_cli.context_stride = int(args_cli.context_stride)
    args_cli.wm_contact_threshold = float(args_cli.wm_contact_threshold)
    args_cli.wm_batch_size = int(args_cli.wm_batch_size)
    args_cli.wm_current_obs_type = str(args_cli.wm_current_obs_type).lower().replace("_", "-")
    args_cli.wm_future_obs_type = str(args_cli.wm_future_obs_type).lower().replace("_", "-")
    args_cli.wm_contact_pred = (
        args_cli.wm_contact_pred if isinstance(args_cli.wm_contact_pred, bool)
        else str(args_cli.wm_contact_pred).lower() in ("1", "true", "yes")
    )
    args_cli.wm_precision = str(args_cli.wm_precision).lower()
    if args_cli.wm_no_amp:
        args_cli.wm_precision = "fp32"
    print(
        f"[INFO] WM play: checkpoint={args_cli.wm_checkpoint} "
        f"inference_frames={args_cli.inference_frames} context_stride={args_cli.context_stride} "
        f"contact_threshold={args_cli.wm_contact_threshold} precision={args_cli.wm_precision} "
        f"ode_steps={args_cli.wm_ode_steps} stochastic={args_cli.wm_stochastic}"
    )


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg, agent_cfg):
    agent_cfg = handle_deprecated_rsl_rl_cfg(cli_args.update_rsl_rl_cfg(agent_cfg, args_cli), pkg_version("rsl-rl-lib"))
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device or env_cfg.sim.device

    policy_checkpoint = handle_deprecated_rsl_rl_checkpoint(
        _resolve_policy_checkpoint(agent_cfg), pkg_version("rsl-rl-lib"))
    policy_dir = os.path.dirname(policy_checkpoint)
    print(f"[INFO] Loading model checkpoint from: {policy_checkpoint}")
    _apply_saved_wm_args(policy_dir)

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
        precision=args_cli.wm_precision,
        contact_threshold=args_cli.wm_contact_threshold,
        context_stride=args_cli.context_stride,
        current_obs_type=args_cli.wm_current_obs_type,
        future_obs_type=args_cli.wm_future_obs_type,
        include_contact=args_cli.wm_contact_pred,
        raw_signals=resolve_raw_signals(args_cli.wm_raw_signals),
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
    amp = torch.autocast("cuda", dtype=world_model.amp_dtype) if world_model.amp else nullcontext()
    with amp:
        _, spatial_shape = world_model.model.encoder.encode(shape_frame[:1].to(world_model.device))
    record_future = world_model.future_latent is not None
    record_rollout = world_model.future_rollout is not None
    record_contact = world_model.include_contact and world_model.contact_flag is not None
    print(f"[WM_VERIFY] spatial_shape={tuple(spatial_shape)} rollout_frames={world_model.rollout_frames} "
          f"context_frames={world_model.context_frames} threshold={world_model.contact_threshold} "
          f"future={record_future} contact={record_contact}")

    current, refreshed, dones = [], [], []
    future = [] if record_future else None
    flags = [] if record_contact else None
    probed = [] if record_contact else None
    rollouts = [] if record_rollout else None
    contact_frames, contact_steps = [], []
    feat_norms, infer_ms = [], []
    env.contact_frame = None
    obs = env.get_observations()
    for _ in range(args_cli.video_length):
        if not simulation_app.is_running():
            break
        with torch.inference_mode():
            start = time.perf_counter()
            actions = policy(obs)
            obs, _, done, _ = env.step(actions)
            infer_ms.append((time.perf_counter() - start) * 1000.0)
            current.append(_to_numpy(world_model.latent_history[0, -1]))
            refreshed.append(bool(env.last_refreshed))
            dones.append(bool(done[0].item()))
            if env.contact_frame is not None:
                contact_frames.append(env.contact_frame)
                contact_steps.append(len(dones) - 1)
                env.contact_frame = None
            if record_future:
                future.append(_to_numpy(world_model.future_latent[0]))
                feat_norms.append(float(world_model.future_latent[0].float().norm().cpu()))
            if record_contact:
                flags.append(float(world_model.contact_flag[0, 0].float().cpu()))
                probed.append(_to_numpy(world_model.contact_probabilities[0]))
            if record_rollout:
                rollouts.append(_to_numpy(world_model.future_rollout[0]))

    T = len(current)
    current = np.stack(current)
    cur_dec = _decode_latents(world_model, current, spatial_shape)
    cur_stitched = _stitch_sensor_grid(cur_dec)
    fut_stitched = None
    if record_future:
        future = np.stack(future)
        fut_stitched = _stitch_sensor_grid(_decode_latents(world_model, future, spatial_shape))
    if record_contact:
        flags = np.asarray(flags, dtype=np.float32)
        probed = np.stack(probed)
    contact_stitched = _stitch_sensor_grid(torch.stack(contact_frames)) if contact_frames else None
    true_contact = _hold_contact_frames(contact_stitched, contact_steps, dones, cur_stitched.shape)
    figure_path = out_dir / "wm_observations.png"
    _save_obs_figure(cur_stitched, figure_path, fut_stitched, flags, true_contact)
    files = ["wm_observations.png"]
    roll_path = None
    if record_rollout:
        roll = np.stack(rollouts)
        T_roll, R, tpf, e_dim = roll.shape
        roll_dec = _decode_latents(world_model, roll.reshape(T_roll * R, tpf, e_dim), spatial_shape)
        roll_path = out_dir / "wm_rollout.png"
        _save_rollout_figure(_stitch_sensor_grid(roll_dec), cur_stitched, T_roll, R, roll_path)
        files.append("wm_rollout.png")

    cur_fut_dist = (
        float(np.linalg.norm((current - future).reshape(T, -1), axis=1).mean())
        if record_future and T else None
    )
    current_std = float(current.reshape(T, -1).std()) if T else 0.0
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
        "precision": world_model.precision,
        "storage_dtype": str(world_model.storage_dtype),
        "ode_steps": world_model.ode_steps,
        "inference_frames": args_cli.inference_frames,
        "context_stride": world_model.context_stride,
        "stochastic": bool(args_cli.wm_stochastic),
        "include_contact": bool(world_model.include_contact),
        "future_obs_type": world_model.future_obs_type,
        "contact_flag_rate": float((flags >= 0.5).mean()) if record_contact and T else None,
        "refresh_rate": float(np.mean(refreshed)) if T else 0.0,
        "mean_future_norm": float(np.mean(feat_norms)) if record_future and T else None,
        "std_future_norm": float(np.std(feat_norms)) if record_future and T else None,
        "std_current_latent": current_std,
        "mean_current_future_dist": cur_fut_dist,
        "decoded_shape": list(cur_dec.shape),
        "video_folder": str(video_folder),
        "files": files,
    }
    with open(out_dir / "wm_obs_diagnostics.json", "w") as f:
        json.dump(diagnostics, f, indent=2)
    summary = [f"steps={T}", f"refresh_rate={diagnostics['refresh_rate']:.3f}"]
    if diagnostics["contact_flag_rate"] is not None:
        summary.append(f"flag_rate={diagnostics['contact_flag_rate']:.3f}")
    if diagnostics["mean_future_norm"] is not None:
        summary.append(
            f"future_norm={diagnostics['mean_future_norm']:.3f} +/-{diagnostics['std_future_norm']:.3f}"
        )
    if cur_fut_dist is not None:
        summary.append(f"cur_fut_dist={cur_fut_dist:.3f}")
    else:
        summary.append(f"current_std={current_std:.3f}")
    print("[WM_VERIFY] " + " ".join(summary))
    if record_future:
        varying = diagnostics["std_future_norm"] > 0 and cur_fut_dist > 0
        detail = (
            "WM features vary over time and differ current vs future"
            if varying else "WM future features look static"
        )
    else:
        varying = current_std > 0
        detail = "current WM latents vary over time" if varying else "current WM latents look static"
    figure_list = str(figure_path) if roll_path is None else f"{figure_path}, {roll_path}"
    print(f"[WM_VERIFY] {'PASS' if varying else 'FAIL'}: {detail}. Video: {video_folder} figures: {figure_list}")
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
