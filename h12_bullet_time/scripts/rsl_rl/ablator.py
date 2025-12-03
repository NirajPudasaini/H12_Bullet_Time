# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Ablation study script for H12 Bullet Time environment.

This script runs multiple training experiments with varied hyperparameters
to study their effects on training performance.

Example usage:
    # Single parameter ablation on max_range
    python ablator.py --task Template-H12-Bullet-Time-CAP --max_range 0.1 0.15 0.2 0.25 0.3

    # With custom training settings
    python ablator.py --task Template-H12-Bullet-Time-CAP --max_range 0.1 0.2 --max_iterations 1000 --num_envs 2048

    # Sequential mode (one at a time, useful for debugging)
    python ablator.py --task Template-H12-Bullet-Time-CAP --max_range 0.1 0.2 --sequential

    # Dry run (show what would be run without executing)
    python ablator.py --task Template-H12-Bullet-Time-CAP --max_range 0.1 0.2 --dry_run
"""

import argparse
import subprocess
import sys
import os
import json
import itertools
from datetime import datetime
from pathlib import Path
from typing import Any


def parse_args():
    """Parse command line arguments for ablation study."""
    parser = argparse.ArgumentParser(
        description="Run ablation studies on H12 Bullet Time environment.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    # Required arguments
    parser.add_argument(
        "--task", 
        type=str, 
        default="Template-H12-Bullet-Time-CAP",
        help="Name of the task/environment to use."
    )
    
    # Ablation parameters - each accepts multiple values
    parser.add_argument(
        "--max_range",
        type=float,
        nargs="+",
        default=None,
        help="List of max_range values to test (capacitive sensor range in meters)."
    )
    parser.add_argument(
        "--projectile_radius",
        type=float,
        nargs="+",
        default=None,
        help="List of projectile_radius values to test (in meters)."
    )
    parser.add_argument(
        "--cap_proximity_scale",
        type=float,
        nargs="+",
        default=None,
        help="List of CAP proximity penalty scale values to test."
    )
    parser.add_argument(
        "--cap_contact_scale",
        type=float,
        nargs="+",
        default=None,
        help="List of CAP contact penalty scale values to test."
    )
    
    # Training parameters (passed through to train.py)
    parser.add_argument(
        "--num_envs",
        type=int,
        default=None,
        help="Number of environments to simulate."
    )
    parser.add_argument(
        "--max_iterations",
        type=int,
        default=None,
        help="Maximum number of training iterations per run."
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility."
    )
    
    # Ablation control
    parser.add_argument(
        "--sequential",
        action="store_true",
        help="Run experiments sequentially instead of in parallel subprocesses."
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Print the configurations that would be run without executing."
    )
    parser.add_argument(
        "--experiment_prefix",
        type=str,
        default="ablation",
        help="Prefix for experiment names in logs."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Directory to save ablation study results summary."
    )
    
    # Pass-through arguments for train.py
    parser.add_argument(
        "--video",
        action="store_true",
        help="Record videos during training."
    )
    parser.add_argument(
        "--headless",
        action="store_true", 
        help="Run in headless mode (no GUI)."
    )
    
    return parser.parse_args()


def generate_ablation_configs(args) -> list[dict[str, Any]]:
    """
    Generate all configurations for the ablation study.
    
    Returns a list of dictionaries, each containing the parameters for one run.
    """
    # Collect all ablation parameters that were specified
    ablation_params = {}
    
    if args.max_range is not None:
        ablation_params["max_range"] = args.max_range
    if args.projectile_radius is not None:
        ablation_params["projectile_radius"] = args.projectile_radius
    if args.cap_proximity_scale is not None:
        ablation_params["cap_proximity_scale"] = args.cap_proximity_scale
    if args.cap_contact_scale is not None:
        ablation_params["cap_contact_scale"] = args.cap_contact_scale
    
    if not ablation_params:
        print("[ERROR] No ablation parameters specified. Use --max_range, --projectile_radius, etc.")
        sys.exit(1)
    
    # Generate all combinations (Cartesian product)
    param_names = list(ablation_params.keys())
    param_values = list(ablation_params.values())
    
    configs = []
    for combo in itertools.product(*param_values):
        config = dict(zip(param_names, combo))
        configs.append(config)
    
    return configs


def build_run_name(config: dict[str, Any], prefix: str) -> str:
    """Build a descriptive run name from the configuration."""
    parts = [prefix]
    for key, value in config.items():
        # Format value nicely (remove decimals for whole numbers)
        if isinstance(value, float) and value == int(value):
            val_str = str(int(value))
        else:
            val_str = f"{value:.3f}".rstrip('0').rstrip('.')
        parts.append(f"{key}={val_str}")
    return "_".join(parts)


def run_training_subprocess(
    task: str,
    config: dict[str, Any],
    run_name: str,
    args,
) -> subprocess.Popen | None:
    """
    Launch a training subprocess with the given configuration.
    
    Uses environment variables to pass ablation parameters to the training script,
    which will be picked up by the environment configuration.
    """
    # Build the command
    script_dir = Path(__file__).parent
    train_script = script_dir / "train.py"
    
    cmd = [
        sys.executable,
        str(train_script),
        "--task", task,
        "--run_name", run_name,
    ]
    
    # Add optional training parameters
    if args.num_envs is not None:
        cmd.extend(["--num_envs", str(args.num_envs)])
    if args.max_iterations is not None:
        cmd.extend(["--max_iterations", str(args.max_iterations)])
    if args.seed is not None:
        cmd.extend(["--seed", str(args.seed)])
    if args.video:
        cmd.append("--video")
    if args.headless:
        cmd.append("--headless")
    
    # Set up environment variables for ablation parameters
    env = os.environ.copy()
    for key, value in config.items():
        env_key = f"ABLATION_{key.upper()}"
        env[env_key] = str(value)
    
    # Also pass config as JSON for easier parsing
    env["ABLATION_CONFIG"] = json.dumps(config)
    
    print(f"\n{'='*60}")
    print(f"[ABLATION] Starting run: {run_name}")
    print(f"[ABLATION] Config: {config}")
    print(f"[ABLATION] Command: {' '.join(cmd)}")
    print(f"{'='*60}\n")
    
    if args.dry_run:
        return None
    
    # Launch subprocess
    process = subprocess.Popen(
        cmd,
        env=env,
        stdout=subprocess.PIPE if not args.sequential else None,
        stderr=subprocess.STDOUT if not args.sequential else None,
    )
    
    return process


def run_ablation_study(args):
    """Main function to run the ablation study."""
    # Generate all configurations
    configs = generate_ablation_configs(args)
    
    print(f"\n{'#'*60}")
    print(f"# ABLATION STUDY")
    print(f"# Task: {args.task}")
    print(f"# Total configurations: {len(configs)}")
    print(f"# Mode: {'Sequential' if args.sequential else 'Parallel'}")
    print(f"{'#'*60}\n")
    
    # Print all configurations
    print("Configurations to run:")
    for i, config in enumerate(configs, 1):
        run_name = build_run_name(config, args.experiment_prefix)
        print(f"  {i}. {run_name}: {config}")
    print()
    
    if args.dry_run:
        print("[DRY RUN] No experiments will be executed.")
        for config in configs:
            run_name = build_run_name(config, args.experiment_prefix)
            run_training_subprocess(args.task, config, run_name, args)
        return
    
    # Run experiments
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    results = []
    
    if args.sequential:
        # Run one at a time, waiting for each to complete
        for i, config in enumerate(configs, 1):
            run_name = build_run_name(config, args.experiment_prefix)
            print(f"\n[ABLATION] Running experiment {i}/{len(configs)}")
            
            process = run_training_subprocess(args.task, config, run_name, args)
            if process is not None:
                return_code = process.wait()
                results.append({
                    "config": config,
                    "run_name": run_name,
                    "return_code": return_code,
                    "status": "completed" if return_code == 0 else "failed"
                })
                print(f"[ABLATION] Experiment {i} finished with return code: {return_code}")
    else:
        # Launch all processes (note: Isaac Sim typically needs sequential runs)
        print("[WARNING] Parallel mode may not work well with Isaac Sim due to GPU constraints.")
        print("[WARNING] Consider using --sequential for reliable results.")
        
        processes = []
        for config in configs:
            run_name = build_run_name(config, args.experiment_prefix)
            process = run_training_subprocess(args.task, config, run_name, args)
            if process is not None:
                processes.append((process, config, run_name))
        
        # Wait for all processes
        for process, config, run_name in processes:
            return_code = process.wait()
            results.append({
                "config": config,
                "run_name": run_name,
                "return_code": return_code,
                "status": "completed" if return_code == 0 else "failed"
            })
    
    # Save results summary
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = Path("logs/ablation_studies")
    
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_file = output_dir / f"ablation_summary_{timestamp}.json"
    
    summary = {
        "timestamp": timestamp,
        "task": args.task,
        "total_runs": len(configs),
        "results": results,
    }
    
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n{'#'*60}")
    print(f"# ABLATION STUDY COMPLETE")
    print(f"# Summary saved to: {summary_file}")
    print(f"{'#'*60}\n")
    
    # Print summary
    completed = sum(1 for r in results if r["status"] == "completed")
    failed = sum(1 for r in results if r["status"] == "failed")
    print(f"Results: {completed} completed, {failed} failed out of {len(configs)} total")


if __name__ == "__main__":
    args = parse_args()
    run_ablation_study(args)

