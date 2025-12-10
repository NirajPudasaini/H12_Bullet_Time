"""Ablation study runner for H12 Bullet Time environment."""

import os
import subprocess
import sys
import json
from pathlib import Path
from datetime import datetime
from itertools import product
from dataclasses import dataclass, field, asdict
from typing import Any


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
    success: float = 0.0  # Populated from eval.py's compute_success()


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
    
    # Prepare environment with ablation overrides
    env = os.environ.copy()
    for k, v in params.items():
        env[k] = str(v)
    
    # Paths
    script_dir = Path(__file__).parent
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"ablation_{run_id}"
    
    # Build train command
    train_cmd = [
        "python", str(script_dir / "train.py"),
        "--task", task,
        "--num_envs", str(num_envs),
        "--max_iterations", str(max_train_iters),
        # "--run_name", run_name,
    ]
    if headless:
        train_cmd.append("--headless")
    if seed is not None:
        train_cmd.extend(["--seed", str(seed)])
    
    print(f"\n{'='*60}\n[ABLATION] Training with params: {params}\n{'='*60}")
    train_returncode, train_output = run_cmd(train_cmd, env, verbose=verbose)
    
    # Extract log directory from train output
    log_dir = ""
    for line in train_output.split("\n"):
        if "Logging experiment in directory:" in line:
            log_dir = line.split(":")[-1].strip()
            break
    
    result = AblationResult(params=params, train_log_dir=log_dir)
    
    if train_returncode != 0:
        print(f"[ABLATION] Training failed!")
        return result
    
    # Build eval command
    contact_threshold = params.get("ABLATION_CONTACT_THRESHOLD", 0.01)
    eval_output = script_dir / f"eval_results_{run_id}.json"
    eval_cmd = [
        "python", str(script_dir / "eval.py"),
        "--task", task,
        "--num_envs", str(num_envs),
        "--ep_per_env", str(ep_per_env),
        "--output_file", str(eval_output),
        "--contact_threshold", str(contact_threshold),
    ]
    if headless:
        eval_cmd.append("--headless")
    if seed is not None:
        eval_cmd.extend(["--seed", str(seed)])
    
    print(f"[ABLATION] Evaluating...")
    eval_returncode, _ = run_cmd(eval_cmd, env, verbose=verbose)
    
    # Read metrics from JSON
    metrics = {}
    if eval_output.exists():
        with open(eval_output) as f:
            metrics = json.load(f)
        eval_output.unlink()  # Clean up temp file
    
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
    """Record a video of the task with the given ablation parameters.
    
    Args:
        task: Task name
        run_id: Prefix for video filename
        video_folder: Where to save the video
        params: Ablation parameters to set as env vars
        num_envs: Number of environments (keep small for video! default=1)
        video_length: Length of video in steps
        timeout: Max seconds to wait for video recording
        
    Returns:
        True if video was recorded successfully, False otherwise
    """
    script_dir = Path(__file__).parent
    # Build environment with ablation overrides (same as train_and_test)
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
        "--num_envs", str(num_envs),  # Critical: keep small for rendering!
        "--headless",
    ]
    
    print(f"[ABLATION] Recording video with {num_envs} env(s), {video_length} steps...")
    try:
        proc = subprocess.Popen(
            video_cmd, env=env,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1
        )
        # Wait with timeout to prevent freezing
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
    """Load cached results from previous ablation runs.
    
    Scans the output folder for existing ablation_results_*.json files and
    matches them against the requested parameter combinations. Only exact
    parameter matches (same keys and values) with valid results are cached.
    
    Args:
        output_folder: Path to the output folder to scan
        combinations: List of parameter combinations to run
        defaults: Default parameter values to merge with combinations
        
    Returns:
        Tuple of (remaining_combinations, cached_results) where:
        - remaining_combinations: Combinations that still need to be run
        - cached_results: Results loaded from cache that match exactly
    """
    output_path = Path(output_folder)
    cached_results = []
    remaining_combinations = []
    
    # Load all existing results from JSON files
    existing_results = []
    for json_file in sorted(output_path.glob("ablation_results_*.json")):
        # Skip directories (like _videos folders)
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
        """Normalize values for comparison (env vars are strings)."""
        if isinstance(v, bool):
            return v
        if isinstance(v, str):
            if v.lower() == "true":
                return True
            if v.lower() == "false":
                return False
            try:
                # Try int first, then float
                if "." in v:
                    return float(v)
                return int(v)
            except ValueError:
                return v
        return v
    
    # Check each combination against existing results
    for params in combinations:
        full_params = {**defaults, **params}
        normalized_full = {k: normalize_value(v) for k, v in full_params.items()}
        
        # Look for exact match in existing results
        matched = False
        for existing in existing_results:
            existing_params = existing.get("params", {})
            normalized_existing = {k: normalize_value(v) for k, v in existing_params.items()}
            
            # Check for exact match (same keys and values)
            if normalized_full == normalized_existing:
                # Check that we have actual results (not a failed run)
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
    training_times: dict[str, int] = {},
    save_video: bool = False,
    video_num_envs: int = 1,
    video_length: int = 300,
    task: str = "Isaac-H12-Bullet-Time-Hybrid-v0",
    seed: int = 42,
    **kwargs,
) -> list[AblationResult]:
    """Run ablation study over parameter grid. Returns list of results.
    
    Args:
        seed: Base seed for reproducibility. Each configuration gets seed + config_index.
    """
    
    # Generate all parameter combinations
    keys = list(param_grid.keys())
    values = list(param_grid.values())
    combinations = [dict(zip(keys, combo)) for combo in product(*values)]
    
    # Ensure output directory exists
    output_path = Path(output_folder)
    output_file = output_path / f"ablation_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    video_path = output_path / f"ablation_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}_videos"
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'#'*60}")
    print(f"# ABLATION STUDY: {len(combinations)} configurations")
    print(f"# Parameters: {keys}")
    print(f"# Base seed: {seed}")
    print(f"{'#'*60}\n")
    
    # Load cached results and filter out already-completed combinations
    combinations, cached_results = load_cached_results(output_folder, combinations, DEFAULTS)
    results = list(cached_results)  # Start with cached results
    
    total_to_run = len(combinations)
    print(f"[ABLATION] Running {total_to_run} new configurations ({len(cached_results)} cached)\n")
    
    for i, params in enumerate(combinations):
        print(f"\n[{i+1}/{total_to_run}] Running configuration...")
        
        # Merge with defaults
        full_params = {**DEFAULTS, **params}
        # Each configuration gets a unique seed for reproducibility
        config_seed = seed + i + len(cached_results)
        max_train_iters = training_times.get(full_params.get("ABLATION_SENSOR_TYPE", "CAP"), 500)
        result = train_and_test(full_params, max_train_iters=max_train_iters, task=task, seed=config_seed, **kwargs)
        results.append(result)

        if save_video:
            record_video(
                task, f"ablation_{i}",
                video_path,
                params=full_params,
                num_envs=video_num_envs,
                video_length=video_length,
            )
        
        # Save intermediate results
        with open(output_file, "w") as f:
            json.dump([asdict(r) for r in results], f, indent=2)
    
    # Summary
    print(f"\n{'='*60}")
    print("ABLATION STUDY COMPLETE")
    print(f"{'='*60}")
    for r in results:
        print(f"  {r.params} -> success={r.success}")
    print(f"\nResults saved to: {output_file}")
    
    return results

# Default ablation parameters (match h12_bullet_time_env_cfg_hybrid.py)
DEFAULTS = {
    "ABLATION_PROJECTILE_RADIUS": 0.15,
    "ABLATION_MAX_RANGE": 1.0,
    "ABLATION_DEBUG_VIS": False,  # Disable vis for ablation runs
    "ABLATION_SENSOR_TYPE": "CAP",
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
}

if __name__ == "__main__":
    # Example ablation study configuration
    PARAM_GRID = {
        "ABLATION_SENSOR_TYPE": ["CAP", "TOF", "CAP_TOF"],
        # "ABLATION_MAX_RANGE": [0.001, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 1.0, 2.0, 3.0, 4.0],
        "ABLATION_MAX_RANGE": [4.0, 2.0, 1.0, 0.5, 0.2, 0.1],
        # "ABLATION_MAX_RANGE": [4.0, 2.0, 1.0, 0.5, 0.2, 0.15, 0.1],
        # "ABLATION_CONTACT_TERMINATION": [True, False],
        # "ABLATION_PROXIMITY_SCALE": [-0.001, -0.01, -0.1],
        # "ABLATION_CONTACT_SCALE": [-0.01, -0.1, -0.5, -1.0],
        # "ABLATION_PROJECTILE_MASS": [0.1, 1.0, 10.0],
        # Add more parameters to sweep here
    }

    TRAINING_TIMES = {
        "TOF": 800,
        "CAP": 800,
        "CAP_TOF": 1000,
    }
    
    run_ablation_study(
        param_grid=PARAM_GRID,
        num_envs=4096,
        training_times=TRAINING_TIMES,
        headless=True,
        task="Template-H12-Bullet-Time-HYBRID",
        verbose=False,
        save_video=True,
        video_length=1000,
        seed=42,  # Base seed for reproducibility
    )

