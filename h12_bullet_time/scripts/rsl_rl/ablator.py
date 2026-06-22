"""Ablation study runner for H12 Bullet Time environment."""

import os
import subprocess
import sys
import json
import smtplib
from email.mime.text import MIMEText
from pathlib import Path
from datetime import datetime
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

@dataclass
class AblationResult:
    params: dict
    train_log_dir: str = ""
    test_metrics: dict = field(default_factory=dict)
    success: float = 0.0


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
    
    sensor_tag = str(params.get("ABLATION_SENSORS", "")).replace(":", "-").replace(";", "_")
    train_cmd = [
        "python", str(script_dir / "train.py"),
        "--task", task,
        "--num_envs", str(num_envs),
        "--max_iterations", str(max_train_iters),
        "--run_name", sensor_tag,
    ]
    if headless:
        train_cmd.append("--headless")
    if seed is not None:
        train_cmd.extend(["--seed", str(seed)])
    
    print(f"\n{'='*60}\n[ABLATION] Training with params: {params}\n{'='*60}")
    train_returncode, train_output = run_cmd(train_cmd, env, verbose=verbose)
    
    log_dir = ""
    for line in train_output.split("\n"):
        if "Logging experiment in directory:" in line:
            log_dir = line.split(":")[-1].strip()
            break
    
    result = AblationResult(params=params, train_log_dir=log_dir)
    
    if train_returncode != 0:
        print(f"[ABLATION] Training failed!")
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
            normalized_existing = {k: normalize_value(v) for k, v in existing_params.items()}
            
            if normalized_full == normalized_existing:
                if existing.get("test_metrics"):
                    result = AblationResult(
                        params=existing_params,
                        train_log_dir=existing.get("train_log_dir", ""),
                        test_metrics=existing.get("test_metrics", {}),
                        success=existing.get("success", 0.0),
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
    for i, params in enumerate(combinations):
        full_params = {**DEFAULTS, **params}
        config_seed = full_params.get("ABLATION_SEED", seed)

        if prev_seed is not None and config_seed != prev_seed:
            _notify_seed_done(prev_seed)
        prev_seed = config_seed

        print(f"\n[{i+1}/{total_to_run}] Running configuration...")
        result = train_and_test(full_params, max_train_iters=training_iters, task=task, seed=config_seed, **kwargs)
        results.append(result)

        if save_video:
            sensor_tag = full_params.get("ABLATION_SENSORS", "").replace(":", "-").replace(";", "_")
            record_video(
                task, f"ablation_{i}_{sensor_tag}",
                video_path,
                params=full_params,
                num_envs=video_num_envs,
                video_length=video_length,
            )
        
        with open(output_file, "w") as f:
            json.dump([asdict(r) for r in results], f, indent=2)

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
        print(f"  {r.params} -> success={r.success}")
    print(f"\nResults saved to: {output_file}")
    
    return results

# Default ablation parameters
DEFAULTS = {
    "ABLATION_PROJECTILE_RADIUS": 0.15,
    "ABLATION_SENSORS": "FIELD:MINDIST:X",
    "ABLATION_MAX_RANGE": 4.0,
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
        # "ABLATION_SEED": [43, 44, 45, 46, 47, 48, 49, 50, 51, 52],
        "ABLATION_SEED": [53],
        "ABLATION_SENSORS": [
            # ── Single sensor shapes ──────────────────────────────────
            # Field sensor (spherical detection)
            # "FIELD:DIST:X",
            # "FIELD:BIN:X",
            # "FIELD:EVENT:X",
            "FIELD:TRUE_POS:X",
            # Ray sensor (8x8 grid)
            # "RAY:DIST:X",
            # "RAY:MINDIST:X",
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
        "ABLATION_MAX_RANGE": [2.0],
        
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
