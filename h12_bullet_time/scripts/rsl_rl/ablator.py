"""Ablation study runner for H12 Bullet Time environment."""

import os
import subprocess
import sys
import json
import smtplib
import time
from email.mime.text import MIMEText
from pathlib import Path
from datetime import datetime, timedelta
from itertools import product
from dataclasses import dataclass, field, asdict
from typing import Any

_ENV_PATH = Path(__file__).parent / ".env"


def _load_env_file(path: Path = _ENV_PATH) -> dict[str, str]:
    if not path.exists():
        return {}
    vals = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" in line:
                k, _, v = line.partition("=")
                vals[k.strip()] = v.strip().strip("\"'")
    return vals


def send_notification(subject: str, body: str) -> bool:
    cfg = _load_env_file()
    host, port = cfg.get("SMTP_HOST", ""), int(cfg.get("SMTP_PORT", "587"))
    user, pw = cfg.get("SMTP_USER", ""), cfg.get("SMTP_PASS", "")
    to = cfg.get("NOTIFY_TO", "")
    if not all([host, user, pw, to]):
        return False
    try:
        msg = MIMEText(body)
        msg["Subject"], msg["From"], msg["To"] = subject, user, to
        with smtplib.SMTP(host, port) as s:
            s.starttls()
            s.login(user, pw)
            s.send_message(msg)
        print(f"[ABLATION] Notification sent to {to}")
        return True
    except Exception as e:
        print(f"[ABLATION] Notification failed: {e}")
        return False


def run_cmd(cmd: list[str], env: dict, verbose: bool = True) -> tuple[int, str]:
    """Run command, optionally streaming output. Returns (returncode, captured_output)."""
    captured = []
    proc = subprocess.Popen(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    for line in proc.stdout:
        if verbose:
            sys.stdout.write(line)
            sys.stdout.flush()
        captured.append(line)
    proc.wait()
    return proc.returncode, "".join(captured)

def _as_bool(v) -> bool:
    if isinstance(v, bool):
        return v
    return str(v).lower() in ("1", "true", "yes")


def _fmt_duration(seconds: float) -> str:
    seconds = max(0, int(round(seconds)))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}h {m:02d}m"
    if m:
        return f"{m}m {s:02d}s"
    return f"{s}s"


def _eta_text(done: int, total: int, elapsed: float) -> str:
    if done <= 0:
        return "ETA unknown until the first config finishes"
    avg = elapsed / done
    left = total - done
    if left <= 0:
        return f"all {total} configs done in {_fmt_duration(elapsed)}"
    remaining = avg * left
    finish = datetime.now() + timedelta(seconds=remaining)
    return (
        f"ETA {finish.strftime('%Y-%m-%d %H:%M:%S')} "
        f"({_fmt_duration(remaining)} left, avg {_fmt_duration(avg)}/config)"
    )


def _wm_run_suffix(params: dict) -> str:
    cur = str(params.get("CURRENT_OBS_TYPE", "LATENT")).upper().replace("_", "-")
    fut = str(params.get("FUTURE_OBS_TYPE", "LATENT")).upper().replace("_", "-")
    contact = "CONTACT" if _as_bool(params.get("CONTACT_PRED", True)) else "NOCONTACT"
    return f"CUR-{cur}-FUT-{fut}-{contact}"


def _parse_wm_stats(output: str) -> dict:
    for line in reversed(output.splitlines()):
        if "[WM_STATS]" in line:
            payload = line.split("[WM_STATS]", 1)[1].strip()
            try:
                return json.loads(payload)
            except json.JSONDecodeError:
                return {}
    return {}


_WM_VALUE_FLAGS = {
    "WM_CHECKPOINT": "--wm_checkpoint",
    "WM_CONFIG": "--wm_config",
    "WM_OUTPUT_CHECKPOINT": "--wm_output_checkpoint",
    "WM_MODE": "--wm_mode",
    "WM_CONTACT_THRESHOLD": "--wm_contact_threshold",
    "INFERENCE_FRAMES": "--inference_frames",
    "CONTEXT_STRIDE": "--context_stride",
    "WM_BATCH_SIZE": "--wm_batch_size",
    "WM_REDUCTION": "--wm_reduction",
    "WM_ODE_STEPS": "--wm_ode_steps",
    "WM_PRECISION": "--wm_precision",
    "WM_TRAJECTORIES_PER_CYCLE": "--wm_trajectories_per_cycle",
    "WM_EPOCHS_PER_CYCLE": "--wm_epochs_per_cycle",
    "WM_TOTAL_TRAJECTORIES": "--wm_total_trajectories",
    "WM_DATA_DIR": "--wm_data_dir",
    "WM_REPLAY_CYCLES": "--wm_replay_cycles",
    "WM_DIAG_INTERVAL": "--wm_diag_interval",
    "CURRENT_OBS_TYPE": "--wm_current_obs_type",
    "FUTURE_OBS_TYPE": "--wm_future_obs_type",
    "CONTACT_PRED": "--wm_contact_pred",
}
_WM_BOOL_FLAGS = {
    "WM_STOCHASTIC": "--wm_stochastic",
    "WM_NO_AMP": "--wm_no_amp",
    "WM_TRAIN_ENCODER": "--wm_train_encoder",
    "WM_NO_TRAIN_DYNAMICS": "--wm_no_train_dynamics",
}


def _append_wm_args(cmd: list[str], params: dict) -> None:
    if not params.get("WM_CHECKPOINT"):
        raise ValueError("WM_CHECKPOINT is required when USE_WORLD_MODEL is True")
    for key, flag in _WM_VALUE_FLAGS.items():
        val = params.get(key)
        if val is None or val == "":
            continue
        cmd.extend([flag, str(val)])
    for key, flag in _WM_BOOL_FLAGS.items():
        if _as_bool(params.get(key, False)):
            cmd.append(flag)


def _parse_train_log_dir(output: str) -> str:
    log_dir = ""
    for line in output.split("\n"):
        if "Run log directory:" in line:
            return line.split(":", 1)[-1].strip()
        if "Logging experiment in directory:" in line:
            log_dir = line.split(":", 1)[-1].strip()
    return log_dir


def _wm_play_fields(params: dict) -> dict:
    inf, stride, thresh = params.get("INFERENCE_FRAMES"), params.get("CONTEXT_STRIDE"), params.get("WM_CONTACT_THRESHOLD")
    return {
        "inference_frames": None if inf in (None, "") else int(inf),
        "context_stride": None if stride in (None, "") else int(stride),
        "wm_contact_threshold": None if thresh in (None, "") else float(thresh),
        "current_obs_type": None if params.get("CURRENT_OBS_TYPE") in (None, "") else str(params["CURRENT_OBS_TYPE"]),
        "future_obs_type": None if params.get("FUTURE_OBS_TYPE") in (None, "") else str(params["FUTURE_OBS_TYPE"]),
        "contact_pred": _as_bool(params.get("CONTACT_PRED", True)),
    }


@dataclass
class AblationResult:
    params: dict
    train_log_dir: str = ""
    test_metrics: dict = field(default_factory=dict)
    success: float = 0.0
    avg_inference_ms: float | None = None
    avg_inference_hz: float | None = None
    envs_per_s: float | None = None
    wm_size: int | None = None
    wm_name: str | None = None
    inference_frames: int | None = None
    context_stride: int | None = None
    wm_contact_threshold: float | None = None
    current_obs_type: str | None = None
    future_obs_type: str | None = None
    contact_pred: bool | None = None


def train_and_test(
    params: dict,
    num_envs: int = 4096,
    max_train_iters: int = 1000,
    ep_per_env: int = 1,
    task: str = "Isaac-H12-Bullet-Time-Hybrid-v0",
    headless: bool = True,
    verbose: bool = True,
    seed: int | None = None,
) -> AblationResult:
    """Train and test with given ablation parameters. Returns AblationResult."""

    env = os.environ.copy()
    for k, v in params.items():
        env[k] = str(v)

    script_dir = Path(__file__).parent
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"ablation_{run_id}"
    use_wm = _as_bool(params.get("USE_WORLD_MODEL", False))
    if "num_envs" in params:
        num_envs = int(params["num_envs"])
    else:
        num_envs = int(params.get("ABLATION_NUM_ENVS", num_envs))
    
    sensor_tag = str(params.get("ABLATION_SENSORS", "")).replace(":", "-").replace(";", "_")
    if use_wm:
        suffix = f"WM-{_wm_run_suffix(params)}"
        sensor_tag = f"{sensor_tag}-{suffix}" if sensor_tag else suffix
    train_script = "train_wm.py" if use_wm else "train.py"
    if use_wm and task == "Template-H12-Survive-Time-HYBRID":
        task = "Template-H12-Survive-Time-WM"
    train_cmd = [
        "python", str(script_dir / train_script),
        "--task", task,
        "--num_envs", str(num_envs),
        "--max_iterations", str(max_train_iters),
        "--run_name", sensor_tag,
    ]
    if headless:
        train_cmd.append("--headless")
    if seed is not None:
        train_cmd.extend(["--seed", str(seed)])
    if use_wm:
        _append_wm_args(train_cmd, params)
    
    print(f"\n{'='*60}\n[ABLATION] Training with params: {params}\n{'='*60}")
    train_returncode, train_output = run_cmd(train_cmd, env, verbose=verbose)

    log_dir = _parse_train_log_dir(train_output)
    wm_stats = _parse_wm_stats(train_output) if use_wm else {}
    result = AblationResult(
        params=params,
        train_log_dir=log_dir,
        avg_inference_ms=wm_stats.get("avg_inference_ms"),
        avg_inference_hz=wm_stats.get("avg_inference_hz"),
        envs_per_s=wm_stats.get("envs_per_s"),
        wm_size=wm_stats.get("wm_size"),
        wm_name=wm_stats.get("wm_name"),
        **(_wm_play_fields(params) if use_wm else {}),
    )

    if train_returncode != 0:
        print(f"[ABLATION] Training failed!")
        return result

    if use_wm:
        print(
            f"[ABLATION] WM stats: name={result.wm_name}, size={result.wm_size}, "
            f"inference_frames={result.inference_frames}, context_stride={result.context_stride}, "
            f"contact_threshold={result.wm_contact_threshold}, "
            f"avg_inference_ms={result.avg_inference_ms}, "
            f"avg_inference_hz={result.avg_inference_hz}, envs_per_s={result.envs_per_s}"
        )
        return result

    contact_threshold = params.get("ABLATION_CONTACT_THRESHOLD", 0.01)
    eval_output = script_dir / f"eval_results_{run_id}.json"
    throw_log_path = script_dir.parent.parent / "ablation_results" / "throw_log.json"
    throw_log_path.parent.mkdir(parents=True, exist_ok=True)
    eval_cmd = [
        "python", str(script_dir / "eval.py"),
        "--task", task,
        "--num_envs", str(num_envs),
        "--ep_per_env", str(ep_per_env),
        "--output_file", str(eval_output),
        "--contact_threshold", str(contact_threshold),
        "--throw_log_file", str(throw_log_path),
    ]
    if headless:
        eval_cmd.append("--headless")
    if seed is not None:
        eval_cmd.extend(["--seed", str(seed)])
    
    print(f"[ABLATION] Evaluating...")
    eval_returncode, _ = run_cmd(eval_cmd, env, verbose=verbose)
    
    metrics = {}
    if eval_output.exists():
        with open(eval_output) as f:
            metrics = json.load(f)
        eval_output.unlink()
    
    result.test_metrics = metrics
    result.success = metrics.get("success", 0.0)
    
    print(f"[ABLATION] Results: success={result.success}, metrics={metrics}")
    return result

def _wm_task(task: str) -> str:
    if task == "Template-H12-Survive-Time-HYBRID":
        return "Template-H12-Survive-Time-WM"
    return task


def record_wm_video(
    task: str,
    run_id: str,
    video_folder: str,
    log_dir: str,
    params: dict | None = None,
    num_envs: int = 1,
    video_length: int = 300,
    seed: int | None = None,
    timeout: int = 3600,
) -> bool:
    """Record a world-model policy video with play_wm_record.py."""
    script_dir = Path(__file__).parent
    env = os.environ.copy()
    if params:
        for k, v in params.items():
            env[k] = str(v)
    video_cmd = [
        "python", str(script_dir / "play_wm_record.py"),
        "--task", _wm_task(task),
        "--checkpoint", log_dir,
        "--num_envs", str(num_envs),
        "--video_length", str(video_length),
        "--video_folder", str(video_folder),
        "--video_name_prefix", run_id,
    ]
    wm_checkpoint = (params or {}).get("WM_CHECKPOINT")
    if wm_checkpoint:
        video_cmd.extend(["--wm_checkpoint", str(wm_checkpoint)])
    if seed is not None:
        video_cmd.extend(["--seed", str(seed)])
    print(f"[ABLATION] Recording WM video with {num_envs} env(s), {video_length} steps...")
    try:
        proc = subprocess.Popen(
            video_cmd, env=env,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1,
        )
        stdout, _ = proc.communicate(timeout=timeout)
        print(stdout)
        if proc.returncode != 0:
            print(f"[ABLATION] WM video recording failed with code {proc.returncode}")
            return False
        print(f"[ABLATION] WM video saved to {video_folder}")
        return True
    except subprocess.TimeoutExpired:
        print(f"[ABLATION] WM video recording timed out after {timeout}s, killing process...")
        proc.kill()
        proc.wait()
        return False
    except Exception as e:
        print(f"[ABLATION] WM video recording error: {e}")
        return False


def record_video(
    task: str,
    run_id: str,
    video_folder: str,
    params: dict | None = None,
    num_envs: int = 1,
    video_length: int = 300,
    timeout: int = 300,
) -> bool:
    """Record a video of the task with the given ablation parameters."""
    script_dir = Path(__file__).parent
    env = os.environ.copy()
    if params:
        for k, v in params.items():
            env[k] = str(v)
    
    video_cmd = [
        "python", str(script_dir / "play.py"),
        "--task", task,
        "--video",
        "--video_length", str(video_length),
        "--video_folder", str(video_folder),
        "--video_name_prefix", run_id,
        "--num_envs", str(num_envs),
        "--headless",
    ]
    
    print(f"[ABLATION] Recording video with {num_envs} env(s), {video_length} steps...")
    try:
        proc = subprocess.Popen(
            video_cmd, env=env,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1
        )
        stdout, _ = proc.communicate(timeout=timeout)
        print(stdout)
        if proc.returncode != 0:
            print(f"[ABLATION] Video recording failed with code {proc.returncode}")
            return False
        print(f"[ABLATION] Video saved to {video_folder}")
        return True
    except subprocess.TimeoutExpired:
        print(f"[ABLATION] Video recording timed out after {timeout}s, killing process...")
        proc.kill()
        proc.wait()
        return False
    except Exception as e:
        print(f"[ABLATION] Video recording error: {e}")
        return False


def load_cached_results(
    output_folder: str,
    combinations: list[dict],
    defaults: dict,
) -> tuple[list[dict], list[AblationResult]]:
    """Load cached results from previous ablation runs."""
    output_path = Path(output_folder)
    cached_results = []
    remaining_combinations = []
    
    existing_results = []
    for json_file in sorted(output_path.glob("ablation_results_*.json")):
        if json_file.is_dir():
            continue
        try:
            with open(json_file) as f:
                data = json.load(f)
                for item in data:
                    existing_results.append(item)
        except (json.JSONDecodeError, IOError) as e:
            print(f"[ABLATION] Warning: Could not load {json_file}: {e}")
            continue
    
    print(f"[ABLATION] Found {len(existing_results)} existing results in {output_folder}")
    
    def normalize_value(v):
        if isinstance(v, bool):
            return v
        if isinstance(v, str):
            if v.lower() == "true":
                return True
            if v.lower() == "false":
                return False
            try:
                if "." in v:
                    return float(v)
                return int(v)
            except ValueError:
                return v
        return v
    
    for params in combinations:
        full_params = {**defaults, **params}
        normalized_full = {k: normalize_value(v) for k, v in full_params.items()}
        
        matched = False
        for existing in existing_results:
            existing_params = existing.get("params", {})
            normalized_existing = {k: normalize_value(v) for k, v in {**defaults, **existing_params}.items()}
            
            if normalized_full == normalized_existing:
                has_eval = bool(existing.get("test_metrics"))
                has_wm = existing.get("wm_name") is not None
                if has_eval or has_wm:
                    play = _wm_play_fields(full_params)
                    result = AblationResult(
                        params=full_params,
                        train_log_dir=existing.get("train_log_dir", ""),
                        test_metrics=existing.get("test_metrics", {}),
                        success=existing.get("success", 0.0),
                        avg_inference_ms=existing.get("avg_inference_ms"),
                        avg_inference_hz=existing.get("avg_inference_hz"),
                        envs_per_s=existing.get("envs_per_s"),
                        wm_size=existing.get("wm_size"),
                        wm_name=existing.get("wm_name"),
                        inference_frames=existing.get("inference_frames", play["inference_frames"]),
                        context_stride=existing.get("context_stride", play["context_stride"]),
                        wm_contact_threshold=existing.get("wm_contact_threshold", play["wm_contact_threshold"]),
                        current_obs_type=existing.get("current_obs_type", play["current_obs_type"]),
                        future_obs_type=existing.get("future_obs_type", play["future_obs_type"]),
                        contact_pred=existing.get("contact_pred", play["contact_pred"]),
                    )
                    cached_results.append(result)
                    print(f"[ABLATION] Cache hit: {params} -> success={result.success}")
                    matched = True
                    break
        
        if not matched:
            remaining_combinations.append(params)
    
    print(f"[ABLATION] {len(cached_results)} cached, {len(remaining_combinations)} remaining to run")
    return remaining_combinations, cached_results


def run_ablation_study(
    param_grid: dict[str, list[Any]],
    output_folder: str = "ablation_results",
    training_iters: int = 3000,
    save_video: bool = False,
    video_num_envs: int = 1,
    video_length: int = 300,
    task: str = "Isaac-H12-Bullet-Time-Hybrid-v0",
    **kwargs,
) -> list[AblationResult]:
    """Run ablation study over parameter grid. Returns list of results."""
    _seed_val = param_grid.get("ABLATION_SEED", 42)
    seed = _seed_val[0] if isinstance(_seed_val, list) else _seed_val

    keys = list(param_grid.keys())
    values = list(param_grid.values())
    combinations = [dict(zip(keys, combo)) for combo in product(*values)]
    
    output_path = Path(output_folder)
    output_file = output_path / f"ablation_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    video_path = output_path / f"ablation_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}_videos"
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'#'*60}")
    print(f"# ABLATION STUDY: {len(combinations)} configurations")
    print(f"# Parameters: {keys}")
    print(f"# Base seed: {seed}")
    print(f"{'#'*60}\n")
    
    combinations, cached_results = load_cached_results(output_folder, combinations, DEFAULTS)
    results = list(cached_results)
    
    total_to_run = len(combinations)
    print(f"[ABLATION] Running {total_to_run} new configurations ({len(cached_results)} cached)\n")
    
    def _notify_seed_done(s):
        seed_results = [r for r in results if r.params.get("ABLATION_SEED") == s]
        avg = sum(r.success for r in seed_results) / max(len(seed_results), 1)
        send_notification(
            f"[Ablation] Seed {s} complete",
            f"Seed {s}: {len(seed_results)} configs, avg success={avg:.4f}\n"
            f"Results saved to: {output_file}",
        )

    prev_seed = None
    study_t0 = time.perf_counter()
    for i, params in enumerate(combinations):
        full_params = {**DEFAULTS, **params}
        config_seed = full_params.get("ABLATION_SEED", seed)

        if prev_seed is not None and config_seed != prev_seed:
            _notify_seed_done(prev_seed)
        prev_seed = config_seed

        elapsed = time.perf_counter() - study_t0
        print(f"\n[{i+1}/{total_to_run}] Running configuration... {_eta_text(i, total_to_run, elapsed)}")
        cfg_t0 = time.perf_counter()
        result = train_and_test(full_params, max_train_iters=training_iters, task=task, seed=config_seed, **kwargs)
        results.append(result)

        if save_video:
            sensor_tag = str(full_params.get("ABLATION_SENSORS", "")).replace(":", "-").replace(";", "_")
            if _as_bool(full_params.get("USE_WORLD_MODEL", False)) and result.wm_name and result.train_log_dir:
                record_wm_video(
                    task, f"ablation_{i}_{sensor_tag}",
                    str(video_path),
                    result.train_log_dir,
                    params=full_params,
                    num_envs=video_num_envs,
                    video_length=video_length,
                    seed=config_seed,
                )
            elif not _as_bool(full_params.get("USE_WORLD_MODEL", False)):
                record_video(
                    task, f"ablation_{i}_{sensor_tag}",
                    str(video_path),
                    params=full_params,
                    num_envs=video_num_envs,
                    video_length=video_length,
                )
        
        with open(output_file, "w") as f:
            json.dump([asdict(r) for r in results], f, indent=2)
        print(
            f"[ABLATION] Config {i+1}/{total_to_run} took {_fmt_duration(time.perf_counter() - cfg_t0)}; "
            f"{_eta_text(i + 1, total_to_run, time.perf_counter() - study_t0)}"
        )

    if prev_seed is not None:
        _notify_seed_done(prev_seed)

    send_notification(
        "[Ablation] Study complete",
        f"All {len(results)} configurations finished.\nResults saved to: {output_file}",
    )
    
    print(f"\n{'='*60}")
    print("ABLATION STUDY COMPLETE")
    print(f"{'='*60}")
    for r in results:
        extra = ""
        if r.wm_name is not None:
            extra = (
                f", wm={r.wm_name}, size={r.wm_size}, "
                f"avg_inference_ms={r.avg_inference_ms}, "
                f"hz={r.avg_inference_hz}, envs_per_s={r.envs_per_s}"
            )
        print(f"  {r.params} -> success={r.success}{extra}")
    print(f"\nResults saved to: {output_file}")
    
    return results

# Default ablation parameters
DEFAULTS = {
    "ABLATION_PROJECTILE_RADIUS": 0.15,
    "ABLATION_SENSORS": "RAY:DIST:X",
    "ABLATION_MAX_RANGE": 2.0,
    "ABLATION_DEBUG_VIS": False,
    "ABLATION_PROXIMITY_SCALE": -0.01,
    "ABLATION_CONTACT_SCALE": -0.5,
    "ABLATION_CONTACT_THRESHOLD": 0.03,
    "ABLATION_PROJECTILE_MASS": 0.1,
    "ABLATION_CONTACT_TERMINATION": True,
    "ABLATION_TERMINATION_ANGLE_THRESHOLD_DEG": 80,
    "ABLATION_TERMINATION_HEIGHT_THRESHOLD": 0.4,
    "ABLATION_PROJECTILE_MIN_SPEED": 4.0,
    "ABLATION_PROJECTILE_MAX_SPEED": 6.0,
    "ABLATION_PROJECTILE_MIN_SPAWN_DIST": 2.0,
    "ABLATION_PROJECTILE_MAX_SPAWN_DIST": 3.0,
    "ABLATION_PROJECTILE_MIN_HEIGHT": 1.0,
    "ABLATION_PROJECTILE_MAX_HEIGHT": 3.0,
    "ABLATION_SEED": 42,
    "ABLATION_NUM_ENVS": 4096,
    "USE_WORLD_MODEL": True,  # False: train.py + eval.py; True: train_wm.py
    "WM_CHECKPOINT": "/home/carson/GenTact/trybrid_skin_project/checkpoints/ToFWM-S/tof_wm_2000_long.pt",  # TOFWM .pt path; required when USE_WORLD_MODEL is True
    "WM_CONFIG": None,  # YAML for checkpoints that lack an embedded config
    "WM_OUTPUT_CHECKPOINT": None,  # adapted-weight save path; default <checkpoint_stem>_robot.pt
    "WM_MODE": "frozen",  # frozen = infer only; alternating = collect trajectories and retrain
    "WM_CONTACT_THRESHOLD": 0.7,
    "INFERENCE_FRAMES": 3, # Only run inference every N frames
    "CONTEXT_STRIDE": 3, # 3 For 20 hz, 6 For 10 hz trained model 
    "WM_BATCH_SIZE": 1024,  # WM inference micro-batch size
    "WM_REDUCTION": "flatten",  # mean-pool latent tokens (flatten keeps all tokens)
    "WM_ODE_STEPS": 2,  # flow integration steps; None uses checkpoint; fewer steps = lower latency
    "WM_TRAJECTORIES_PER_CYCLE": 100,  # X: completed trajectories per alternating retrain cycle
    "WM_EPOCHS_PER_CYCLE": 1,  # Y: WM train epochs per cycle
    "WM_TOTAL_TRAJECTORIES": 1000,  # Z: stop alternating collection after this many trajectories
    "WM_DATA_DIR": "wm_robot_data",  # H5 cycle directory under the PPO run log
    "WM_REPLAY_CYCLES": 1,  # train on the latest N cycle files (1 = new data only)
    "WM_DIAG_INTERVAL": 50,  # print WM inference latency every N steps; 0 disables
    "WM_STOCHASTIC": False,  # sample fresh flow noise at every prediction
    "WM_NO_AMP": False,  # disable autocast (forces fp32, overrides WM_PRECISION)
    "WM_PRECISION": "fp16",  # fp32 | fp16 | bf16 autocast dtype for WM encoder/dynamics/decode
    "WM_TRAIN_ENCODER": False,  # also train the encoder during alternating cycles
    "WM_NO_TRAIN_DYNAMICS": False,  # skip dynamics updates during alternating cycles
    "CURRENT_OBS_TYPE": "LATENT",  # RAW follows ABLATION_SENSORS; WM still encodes the full image. LATENT | NONE
    "FUTURE_OBS_TYPE": "LATENT",  # LATENT | DECODED (full image) | MIN-DECODED (per-sensor min) | CLOSEST-POINT | NONE
    "CONTACT_PRED": True,  # append WM contact-prediction flag to policy observations
}

if __name__ == "__main__":
    # ── Sensor shape × signal type study ─────────────────────────────────
    #
    # ABLATION_SENSORS format: "SHAPE:SIGNAL:MAX_RANGE"
    #   Shapes:  FIELD, RAY, CONE
    #   Signals: DIST, MINDIST, BIN, MINBIN, TRUE_POS, EVENT
    #   Combine with semicolons: "FIELD:DIST:4.0;RAY:MINDIST:4.0"
    #
    PARAM_GRID = {
        "ABLATION_SEED": [48, 49, 50, 51, 52],
        # "ABLATION_SEED": [44, 45, 46, 47],
        # "ABLATION_SEED": [43],
        "ABLATION_SENSORS": [
            # ── Single sensor shapes ──────────────────────────────────
            # Field sensor (spherical detection)
            # "FIELD:DIST:X",
            # "FIELD:BIN:X",
            # "FIELD:EVENT:X",
            # "FIELD:TRUE_POS:X",
            # Ray sensor (8x8 grid)
            # "RAY:DIST:X",
            "RAY:MINDIST:X",
            # "RAY:BIN:X",
            # "RAY:MINBIN:X",
            # "RAY:EVENT:X",
            # "RAY:TRUE_POS:X",
            # Cone sensor (conical receptive field, 30° default)
            # "CONE:DIST:X",
            # "CONE:BIN:X",
            # "CONE:EVENT:X",
            # "CONE:TRUE_POS:X",
            # ── Sensor combinations ───────────────────────────────────
            # "FIELD:DIST:X;RAY:DIST:X",

            #   # --- Masked Double Passing ---> Do this after training the single sensor models to find the best single sensor model.
            # "FIELD:DIST:X;FIELD:BIN:X",
            # "RAY:DIST:X;RAY:BIN:X",
            # "CONE:DIST:X;CONE:BIN:X",
        ],
        # --- Multimodel testing ---
        # "ABLATION_SENSORS": [
        #   # --- Masked Double Passing ---> Do this after training the single sensor models to find the best single sensor model.
        #   "FIELD:DIST:X;FIELD:BIN:X",
        #   "RAY:MINDIST:X;RAY:BIN:X",
        #   "CONE:DIST:X;CONE:BIN:X",
        # ],

        # "ABLATION_MAX_RANGE": [2.0, 1.0, 0.5, 0.2],

        # Test case
        # "ABLATION_SENSORS": ["FIELD:EVENT:X;FIELD:MINDIST:X"],
        # "ABLATION_MAX_RANGE": [2.0],

        # ── World-model inference-speed study ─────────────────────────
        # "USE_WORLD_MODEL": [True],
        # "ABLATION_NUM_ENVS": [1, 2, 4, 8, 16],
        "WM_CHECKPOINT": [
            # "/home/carson/GenTact/trybrid_skin_project/checkpoints/tof_wm_2000.pt",
            # "/home/carson/GenTact/trybrid_skin_project/checkpoints/ToFWM-S-e8/tof_wm.pt"
            # "/home/carson/GenTact/trybrid_skin_project/checkpoints/ToFWM-S-e16/tof_wm.pt"
            # "/home/carson/GenTact/trybrid_skin_project/checkpoints/c3r9/ToFWM-S-e8/tof_wm.pt"
            # "/home/carson/GenTact/trybrid_skin_project/checkpoints/c5r12/ToFWM-S-e8/tof_wm.pt"
            # "/home/carson/GenTact/trybrid_skin_project/checkpoints/c3r9/d5000enc-p6000dyn/ToFWM-S-e32/tof_wm.pt"
            "/home/carson/GenTact/trybrid_skin_project/checkpoints/c3r9/p6000/ToFWM-S-e8/tof_wm.pt"
        ],
        # "WM_CONFIG": [None],
        # "WM_OUTPUT_CHECKPOINT": [None],
        # "WM_MODE": ["frozen"],
        # "WM_CONTACT_THRESHOLD": [0.5],
        # "INFERENCE_FRAMES": [1, 5, 10],
        # "CONTEXT_STRIDE": [1, 3],
        # "WM_BATCH_SIZE": [256],
        # "WM_REDUCTION": ["mean"],
        # "WM_ODE_STEPS": [None],
        # "WM_TRAJECTORIES_PER_CYCLE": [4096],
        # "WM_EPOCHS_PER_CYCLE": [1],
        # "WM_TOTAL_TRAJECTORIES": [40960],
        # "WM_DATA_DIR": ["wm_robot_data"],
        # "WM_REPLAY_CYCLES": [1],
        # "WM_DIAG_INTERVAL": [50],
        # "WM_STOCHASTIC": [False],
        # "WM_NO_AMP": [False],
        # "WM_TRAIN_ENCODER": [False],
        # "WM_NO_TRAIN_DYNAMICS": [False],
        # RAW uses the sensor vector (MINDIST -> one distance per sensor). The WM still gets the full image.
        # DECODED is the full decoded image. MIN-DECODED mins that image to one distance per sensor.
        # "CURRENT_OBS_TYPE": ["RAW", "LATENT", "NONE"],
        "CURRENT_OBS_TYPE": ["RAW"],
        # "FUTURE_OBS_TYPE": ["LATENT", "DECODED", "MIN-DECODED", "CLOSEST-POINT", "NONE"],
        "FUTURE_OBS_TYPE": ["MIN-DECODED"],
        "CONTACT_PRED": [True],
        # "WM_PRECISION": ["fp16", "bf16"],
        
    }
    
    run_ablation_study(
        param_grid=PARAM_GRID,
        num_envs=4096,
        training_iters=3000,
        headless=True,
        task="Template-H12-Survive-Time-HYBRID",
        verbose=False,
        save_video=True,
        video_length=1000,
    )
