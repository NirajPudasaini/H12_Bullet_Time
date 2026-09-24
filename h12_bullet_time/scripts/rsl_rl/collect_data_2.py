"""Script to collect sensor data using a trained RL agent into H5 files."""

"""Launch Isaac Sim Simulator first."""

import argparse
import json
import os
import re
import secrets
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
parser.add_argument(
    "--seed", type=int, default=None,
    help="Environment seed. Omit to sample a new random seed for each checkpoint.",
)
parser.add_argument(
    "--random_seed",
    action="store_true",
    default=False,
    help="Sample a new random seed for each checkpoint (overrides --seed).",
)
parser.add_argument("--use_pretrained_checkpoint", action="store_true")
parser.add_argument(
    "--ablation_results", type=str, default=None,
    help="Ablation-results JSON. Collect --num_trajectories from each recorded run whose "
         "ABLATION_SENSORS and range match --sensors and --max_range.",
)
parser.add_argument(
    "--ckpt", type=int, action="append", default=None,
    help="Training iteration to collect, repeatable: --ckpt 500 --ckpt 1000. "
         "With --ablation_results, each matching run is collected at each listed iteration "
         "(model_<N>.pt). Omit to use the run's recorded/latest checkpoint.",
)
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


_MODEL_CKPT = re.compile(r"^model_(\d+)\.pt$")
_SENSOR_SPLIT = re.compile(r"_(?=(?:FIELD|RAY|CONE)-)")


def _sensors_from_run_name(name):
    """Inverse of the ablator run tag: ':' -> '-', ';' -> '_', optional '-WM-...' suffix."""
    m = _RUN_TS.match(name)
    if not m:
        return None
    tag = name[m.end():].lstrip("_")
    wm = tag.find("-WM-")
    if wm >= 0:
        tag = tag[:wm]
    if not tag:
        return None
    return ";".join(part.replace("-", ":") for part in _SENSOR_SPLIT.split(tag))


def _largest_checkpoint(run_dir):
    best, best_n = None, -1
    for path in Path(run_dir).iterdir():
        if not path.is_file():
            continue
        match = _MODEL_CKPT.match(path.name)
        if match and (step := int(match.group(1))) > best_n:
            best, best_n = path, step
    return best


def _parse_sensor_specs(sensors, default_range):
    """'RAY:DIST:X' + max range 2 -> ((RAY, DIST, 2.0),). 'X' takes default_range."""
    specs = []
    for raw in str(sensors).split(";"):
        parts = [part.strip() for part in raw.split(":") if part.strip()]
        if len(parts) < 2:
            raise ValueError(f"Bad sensor spec {raw!r}")
        shape, signal = parts[0].upper(), parts[1].upper()
        if len(parts) < 3 or parts[2].upper() == "X":
            if default_range is None:
                raise ValueError(f"Sensor spec {raw!r} needs --max_range")
            rng = float(default_range)
        else:
            rng = float(parts[2])
        specs.append((shape, signal, round(rng, 6)))
    return tuple(specs)


def _with_logs_archive(path):
    needle = f"{os.sep}logs{os.sep}"
    text = str(path)
    if needle not in text:
        return None
    alt = Path(text.replace(needle, f"{os.sep}logs_archive{os.sep}", 1))
    return alt if alt.exists() else None


def _as_dir(path):
    if not path:
        return None
    path = Path(path)
    return path.parent if path.suffix == ".pt" else path


def _run_dir_candidates(recorded, train_log_dir):
    found = []
    for raw in (recorded, train_log_dir):
        folder = _as_dir(raw)
        if folder is None:
            continue
        for option in (folder, _with_logs_archive(folder)):
            if option is not None and option.is_dir() and option not in found:
                found.append(option)
    return found


def _model_in_dirs(run_dirs, step):
    name = f"model_{step}.pt"
    for folder in run_dirs:
        path = folder / name
        if path.is_file():
            return path
    return None


def _resolve_recorded_checkpoint(recorded, train_log_dir):
    """Prefer the JSON checkpoint path, then the same file under logs_archive."""
    if not recorded and not train_log_dir:
        return None
    if recorded:
        path = Path(recorded)
        if path.is_file():
            return path
        archived = _with_logs_archive(path)
        if archived is not None and archived.is_file():
            return archived
        for folder in (path.parent, _with_logs_archive(path.parent)):
            if folder is not None and (folder / path.name).is_file():
                return folder / path.name
    for folder in _run_dir_candidates(recorded, train_log_dir):
        found = _largest_checkpoint(folder)
        if found is not None:
            return found
    return None


def _ablation_jobs(results_path, sensors, max_range, ckpt_iters=None):
    wanted = _parse_sensor_specs(sensors, max_range)
    rows = json.loads(Path(results_path).expanduser().read_text())
    if isinstance(rows, dict):
        rows = rows.get("results", [rows])
    steps, seen_steps = [], set()
    for step in ckpt_iters or []:
        if step not in seen_steps:
            seen_steps.add(step)
            steps.append(step)
    jobs, seen = [], set()
    for row in rows:
        params = row.get("params") or {}
        try:
            got = _parse_sensor_specs(params.get("ABLATION_SENSORS", ""), params.get("ABLATION_MAX_RANGE"))
        except ValueError:
            continue
        if got != wanted:
            continue
        recorded = (row.get("test_metrics") or {}).get("checkpoint")
        run_name = Path(recorded).parent.name if recorded else Path(str(row.get("train_log_dir") or "run")).name
        run_dirs = _run_dir_candidates(recorded, row.get("train_log_dir"))
        if steps:
            targets = [(f"{run_name}/model_{step}", _model_in_dirs(run_dirs, step), step) for step in steps]
        else:
            checkpoint = _resolve_recorded_checkpoint(recorded, row.get("train_log_dir"))
            targets = [(run_name, checkpoint, None)]
        for label, checkpoint, step in targets:
            if checkpoint is None:
                suffix = f"model_{step}.pt" if step is not None else recorded
                print(f"[INFO] Skipping {label}: checkpoint not found ({suffix})")
                continue
            resolved = str(checkpoint.resolve())
            if resolved in seen:
                continue
            seen.add(resolved)
            jobs.append((label, resolved))
    return jobs


# Sensor spec must be set before hydra imports the env cfg.
_log_root = os.path.abspath(os.path.join("logs", "rsl_rl", "h12-bullet-time-ppo"))
if args_cli.checkpoint and args_cli.ablation_results:
    raise SystemExit("[ERROR] Pass only one of --checkpoint and --ablation_results.")
if args_cli.ablation_results:
    if not args_cli.sensors:
        raise SystemExit("[ERROR] --ablation_results requires --sensors (for example RAY:DIST:X).")
    try:
        _wanted = _parse_sensor_specs(args_cli.sensors, args_cli.max_range)
    except ValueError as exc:
        raise SystemExit(f"[ERROR] {exc}") from exc
    if args_cli.max_range is None and len({rng for _, _, rng in _wanted}) == 1:
        args_cli.max_range = _wanted[0][2]
    if args_cli.ckpt and not args_cli.ablation_results:
        raise SystemExit("[ERROR] --ckpt requires --ablation_results.")
    args_cli.ablation_jobs = _ablation_jobs(
        args_cli.ablation_results, args_cli.sensors, args_cli.max_range, args_cli.ckpt
    )
    if not args_cli.ablation_jobs:
        raise SystemExit(
            f"[ERROR] No checkpoints in {args_cli.ablation_results} match "
            f"{args_cli.sensors} with range {args_cli.max_range}."
        )
    print(f"[INFO] {len(args_cli.ablation_jobs)} collections match {args_cli.sensors} range {args_cli.max_range}")
    for _name, _checkpoint in args_cli.ablation_jobs:
        print(f"[INFO] Sweep: {_name} -> {os.path.basename(_checkpoint)}")
elif not args_cli.checkpoint and args_cli.load_run is None:
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
from tqdm import tqdm

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


_CONTACT_TERM_FUNCS = ("multi_contact_termination", "contact_termination")


def _contact_term_names(unwrapped):
    tm = getattr(unwrapped, "termination_manager", None)
    if tm is None:
        return [], float(os.environ.get("ABLATION_CONTACT_THRESHOLD", 0.03))
    names, threshold = [], float(os.environ.get("ABLATION_CONTACT_THRESHOLD", 0.03))
    for name in tm.active_terms:
        cfg = tm.get_term_cfg(name)
        if getattr(cfg.func, "__name__", "") in _CONTACT_TERM_FUNCS:
            names.append(name)
            threshold = float(cfg.params.get("threshold", threshold))
    return names, threshold


def _projectile_in_contact(contact_sensors, num_envs, threshold):
    """Same condition as mdp.multi_contact_termination: max |force_matrix_w| vs projectile."""
    hit = None
    for s in contact_sensors.values():
        fm = s.data.force_matrix_w
        if fm is None:
            continue
        mag = torch.linalg.norm(fm.reshape(num_envs, -1, 3), dim=-1).max(dim=1).values
        hit = mag if hit is None else torch.maximum(hit, mag)
    if hit is None:
        return np.zeros(num_envs, dtype=bool)
    return (hit > threshold).cpu().numpy()


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

    if args_cli.ablation_results:
        jobs = args_cli.ablation_jobs
    elif args_cli.use_pretrained_checkpoint:
        resume_path = get_published_pretrained_checkpoint("rsl_rl", train_task_name)
        if not resume_path:
            print("[INFO] No pre-trained checkpoint available.")
            return
        jobs = [(None, resume_path)]
    elif args_cli.checkpoint:
        jobs = [(None, retrieve_file_path(args_cli.checkpoint))]
    else:
        jobs = [(None, get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint))]

    env_cfg.log_dir = os.path.dirname(jobs[0][1])

    env = gym.make(args_cli.task, cfg=env_cfg)
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    if agent_cfg.class_name == "OnPolicyRunner":
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    elif agent_cfg.class_name == "DistillationRunner":
        runner = DistillationRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    else:
        raise ValueError(f"Unsupported runner class: {agent_cfg.class_name}")

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

    contact_term_names, contact_threshold = _contact_term_names(unwrapped)
    sensors_spec = os.environ.get("ABLATION_SENSORS", "FIELD:MINDIST:4.0")
    print(f"[INFO] sensors={sensors_spec} | {len(ray_sensors)} ray, {len(field_sensors)} field/cone | "
          f"{len(contact_sensors)} contact | {num_envs} envs | seed={env_cfg.seed}")
    print(f"[INFO] probe/in_contact: force_matrix_w > {contact_threshold} N, terms={contact_term_names}")
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

    step_dt = getattr(unwrapped, "step_dt", None) or (unwrapped.cfg.sim.dt * unwrapped.cfg.decimation)
    fps = 1.0 / float(step_dt)
    # Inference only. Pre-5.0 optimizer state is ordered for the combined actor-critic and cannot load.
    infer_load = {"actor": True, "critic": True, "optimizer": False, "iteration": False}

    def _collect_one(policy, run_name, checkpoint, pbar):
        if args_cli.seed is None or args_cli.random_seed:
            seed = secrets.randbits(31)
        else:
            seed = args_cli.seed
        env.seed(seed)
        env_cfg.seed = seed
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        folder = os.path.join(args_cli.output_dir, run_name) if run_name else args_cli.output_dir
        output_subdir = os.path.join(folder, f"roboset_{timestamp}")
        metadata = {
            "task": args_cli.task, "robot": "h12", "timestamp": timestamp,
            "seed": int(seed), "checkpoint": checkpoint,
        }
        if run_name:
            metadata["run"] = run_name
        save_kw = dict(metadata=metadata, probe_radius=projectile_radius, sensor_static=sensor_static_info,
                       fps=fps, joint_names=robot.joint_names)
        label = run_name or os.path.basename(checkpoint)
        pbar.set_postfix_str(label, refresh=False)
        tqdm.write(f"[INFO] {label}: seed {seed}")

        buffers = [TrajectoryBuffer() for _ in range(num_envs)]
        completed, traj_counter, file_counter = [], 0, 0
        obs, _ = env.reset()

        if args_cli.static:
            _dev = robot.data.joint_pos.device
            _static_root_state = robot.data.root_state_w.clone()
            _static_root_state[:, 7:] = 0.0
            _static_jpos = robot.data.joint_pos.clone()
            _static_jvel = torch.zeros_like(robot.data.joint_vel)

        while traj_counter < args_cli.num_trajectories and simulation_app.is_running():
            # Do not wrap env.step in inference_mode: that freezes sim buffers so the
            # next checkpoint's env.reset() cannot write them.
            with torch.no_grad():
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

                in_contact_np = _projectile_in_contact(contact_sensors, num_envs, contact_threshold)

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
                    buf.append("probe_in_contact", np.array([in_contact_np[ei]]))
                    for sname, (lp, lq) in link_tf.items():
                        buf.append(f"{sname}_link_pos_w", lp[ei])
                        buf.append(f"{sname}_link_quat_w", lq[ei])
                    buf.append("actions", act_np[ei])
                    buf.step_done()

                # Step environment
                obs, _, dones, _ = env.step(actions)
                dones_np = dones.cpu().numpy() if isinstance(dones, torch.Tensor) else np.asarray(dones)
                tm = getattr(unwrapped, "termination_manager", None)
                if tm is not None and contact_term_names:
                    contact_done = torch.zeros(num_envs, dtype=torch.bool, device=unwrapped.device)
                    for name in contact_term_names:
                        contact_done |= tm.get_term(name)
                    for ei in np.flatnonzero(contact_done.cpu().numpy()):
                        frames = buffers[ei].data.get("probe_in_contact")
                        if frames:
                            frames[-1] = np.array([True])

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
                            pbar.update(1)
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

        tqdm.write(f"[INFO] Complete: {traj_counter} trajectories saved to {output_subdir}")
        return traj_counter

    pbar = tqdm(
        total=len(jobs) * args_cli.num_trajectories,
        desc="collect",
        unit="traj",
        dynamic_ncols=True,
    )
    for index, (run_name, checkpoint) in enumerate(jobs):
        if not simulation_app.is_running():
            break
        tqdm.write(f"[INFO] Collecting {index + 1}/{len(jobs)}: {checkpoint}")
        collected = 0
        try:
            resume_path = handle_deprecated_rsl_rl_checkpoint(checkpoint, _rsl_rl_version)
            tqdm.write(f"[INFO] Loading model checkpoint from: {resume_path}")
            runner.load(resume_path, load_cfg=infer_load)
            policy = runner.get_inference_policy(device=unwrapped.device)
            collected = _collect_one(policy, run_name, checkpoint, pbar)
        except Exception as exc:
            tqdm.write(f"[ERROR] Failed {checkpoint}: {exc}")
            if len(jobs) == 1:
                pbar.close()
                raise
        leftover = args_cli.num_trajectories - collected
        if leftover > 0:
            pbar.update(leftover)
    pbar.close()
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
