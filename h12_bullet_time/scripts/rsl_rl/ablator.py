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
    eval_output = script_dir / f"eval_results_{run_id}.json"
    eval_cmd = [
        "python", str(script_dir / "eval.py"),
        "--task", task,
        "--num_envs", str(num_envs),
        "--ep_per_env", str(ep_per_env),
        "--output_file", str(eval_output),
    ]
    if headless:
        eval_cmd.append("--headless")
    
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


def run_ablation_study(
    param_grid: dict[str, list[Any]],
    output_file: str = f"ablation_results/ablation_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
    max_train_iters: int = 1000,
    **kwargs,
) -> list[AblationResult]:
    """Run ablation study over parameter grid. Returns list of results."""
    
    # Generate all parameter combinations
    keys = list(param_grid.keys())
    values = list(param_grid.values())
    combinations = [dict(zip(keys, combo)) for combo in product(*values)]
    
    # Ensure output directory exists
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'#'*60}")
    print(f"# ABLATION STUDY: {len(combinations)} configurations")
    print(f"# Parameters: {keys}")
    print(f"{'#'*60}\n")
    
    results = []
    for i, params in enumerate(combinations):
        print(f"\n[{i+1}/{len(combinations)}] Running configuration...")
        
        # Merge with defaults
        full_params = {**DEFAULTS, **params}
        result = train_and_test(full_params, max_train_iters=max_train_iters, **kwargs)
        results.append(result)
        
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
    "ABLATION_MAX_RANGE": 4.0,
    "ABLATION_DEBUG_VIS": False,  # Disable vis for ablation runs
    "ABLATION_SENSOR_TYPE": "TOF",
    "ABLATION_PROXIMITY_SCALE": -0.001,
    "ABLATION_CONTACT_SCALE": -0.1,
    "ABLATION_CONTACT_THRESHOLD": 0.05,
}

if __name__ == "__main__":
    # Example ablation study configuration
    PARAM_GRID = {
        "ABLATION_SENSOR_TYPE": ["TOF", "CAP"],
        "ABLATION_MAX_RANGE": [0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 1.0, 2.0, 3.0, 4.0],
        # Add more parameters to sweep here
    }
    
    run_ablation_study(
        param_grid=PARAM_GRID,
        num_envs=4096,
        max_train_iters=500,
        headless=True,
        task="Template-H12-Bullet-Time-HYBRID",
        verbose=False,
    )

