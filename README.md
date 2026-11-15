# H12 Bullet Time

This is an [**external Isaac Lab project**]((https://isaac-sim.github.io/IsaacLab/main/source/overview/own-project/template.html).) for testing obstacle-aware locomotion tasks with humanoid or quadruped robots. The project runs independently of the Isaac Lab core repository.  

---

## 🔹 Prerequisites

- Isaac Lab installed and running in a conda environment, [Follow](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html)

---

## 🔹 Installation

1. Clone the repository:

```bash
git clone https://github.com/NirajPudasaini/H12_Bullet_Time.git
cd H12_Bullet_Time/h12_bullet_time
```

2. Install the project in editable mode:

```bash
python -m pip install -e source/h12_bullet_time/
```

3. Check the tasks available in the project:
```bash
python scripts/list_envs.py
```

### Basic Training

To train :

```bash
cd H12_Bullet_Time/h12_bullet_time
python scripts/rsl_rl/train.py --task=Template-H12-Bullet-Time-HYBRID --num_envs=4 #(for visualization)
# python scripts/rsl_rl/train.py --task=Template-H12-Bullet-Time-v0 --num_envs=4096 --headless #(actual training)
python scripts/rsl_rl/train.py --task=Template-H12-Bullet-Time-HYBRID --num_envs=4096 --headless #(actual training)
```

Visualize logs
```bash
 tensorboard --logdir logs/rsl_rl/h12-bullet-time-ppo/
 ```

**Parameters:**
- `--task`: Task identifier (use `Template-H12-Bullet-Time-v0` for H12 standing task)
- `--num_envs`: Number of parallel environments
- `--headless`: Run without GUI for faster training
- `--max_iterations`: Maximum training iterations (default: 5000)



## Playing Trained Models

### Play with Latest Checkpoint

To visualize the trained policy:

```bash
cd H12_Bullet_Time/h12_bullet_time
python scripts/rsl_rl/play.py --task=Template-H12-Bullet-Time-v0 --num_envs=4 --use_last_checkpoint
```

### View Training Logs

Training logs are saved to: `logs/rsl_rl/h12-bullet-time-ppo/<timestamp>/`

Each run contains:
- `model_*.pt`: Model checkpoints at different iterations
- `events.out.tfevents.*`: TensorFlow event logs for TensorBoard
- `params/`: Configuration YAML files

### Run Ablation Study
```bash
cd H12_Bullet_Time
python3 scripts/rsl_rl/ablator.py # Ablation parameters defined in this script
```

### Plot Ablation Results
```bash
python3 h12_bullet_time/source/h12_bullet_time/h12_bullet_time/utils/plotting.py --data <data_file>
```

### Policy Configuration

PPO config: `source/h12_bullet_time/h12_bullet_time/tasks/manager_based/h12_bullet_time/agents/rsl_rl_ppo_cfg.py`

Key hyperparameters:
- `max_iterations`: Total training iterations (default: 5000)
- `num_steps_per_env`: Steps per environment before update (default: 16)
- `learning_rate`: Policy learning rate (default: 1.0e-3)
- `actor_hidden_dims`: Actor network hidden dimensions (default: [32, 32])
- `critic_hidden_dims`: Critic network hidden dimensions (default: [32, 32])

### Data collection
```bash
python scripts/rsl_rl/collect_data.py --task Template-H12-Bullet-Time-HYBRID --num_envs 64 --num_trajectories 100 --headless
# or
python scripts/rsl_rl/collect_data.py --task Template-H12-Survive-Time-HYBRID --num_envs 64 --num_trajectories 100 --headless --trajs_per_file 1000 --sensor_type CAP_TOF --static

python scripts/rsl_rl/collect_data_2.py --trajs_per_file 1 --num_trajectories 1 --num_envs 1 --task Template-H12-Survive-Time-HYBRID --output_dir collected_data --headless
```
--num_envs	64	Parallel environments
--num_trajectories	100	Total trajectories to collect
--max_traj_length	500	Max steps per trajectory
--min_traj_length	10	Discard shorter trajectories
--output_dir	collected_data	Output root directory
--trajs_per_file	50	Trajectories per H5 part file
--sensor_type: CAP | TOF | CAP_TOF
--static -> Restricts the robot from moving

## Video Playback of World Model Trajectories

WM inference settings (`inference_frames`, `context_stride`, `wm_contact_threshold`, checkpoint)
are read from the PPO run's `params/world_model.yaml`. Omit `--checkpoint` to use the latest run.

```bash
cd H12_Bullet_Time/h12_bullet_time
python scripts/rsl_rl/play_wm_record.py --task Template-H12-Survive-Time-WM --video_length 300
```

The script launches headless with cameras enabled so `RecordVideo` can capture `rgb_array` frames.
World-model observations are saved as `wm_observations.png` (current / selected future / contact)
and `wm_rollout.png` (all R predicted steps stacked above current). Both are timestamp-aligned.

## Perfect World Model (simulated future projectile)

The SIM-* future observations swap the PWM's learned dynamics for the true ball dynamics in sim, so the
policy sees the sensor reading the ball *would* produce right before it hits the robot.

**Environment** — `Template-H12-Survive-Time-Perfect-WM`
(`h12_survive_time_env_cfg_hybrid_perfect_wm.py`) is the WM environment plus a `Future_Projectile`: a
kinematic, collision-free, semi-transparent red sphere. Robot, projectile, rewards and terminations are
unchanged. Every step, the `future_projectile` policy term in `mdp/observations.py` returns an empty
observation and does the following:

1. Propagates the projectile ballistically for up to R = `ORACLE_ROLLOUT_FRAMES` frames. Each frame is
   `CONTEXT_STRIDE` control steps, matching the PWM. The path follows PhysX's semi-implicit Euler
   integration, so a free-flying ball's predicted position matches the real one exactly. z is clamped at
   the ground, and the robot is held at its current pose.
2. At every control step, tests whether the ball occupies the robot's URDF collision geometry. This
   uses the same surface test as ORACLE-CP, generalized to a batch of positions, and checking every
   step stops the ball from passing through thin links between frames.
3. Places `Future_Projectile` at the last frame **before** the first intersecting step. If nothing
   intersects within R, it goes at frame R. The position, the contact flag and the frame index are
   stored on `env._future_projectile_{pos_w,contact,frame}`.
4. Calls `update_future(pos_w)` on every sensor.

**Sensors** — `TofSensor` (RAY) and `CapacitiveSensor` (FIELD) have new `update_future(future_pos_w)`
methods that fill `data.future_dist_est` and `data.future_dist_est_normalized`. These have the same
meaning as `dist_est`/`dist_est_normalized`, but with the ball at the future pose; their shape is
`(N, S, 1, P)` for RAY and `(N, S, 1)` for FIELD. The current and future readings come from the same
helpers (`_ray_dist_est` / `_field_dist_est`), so they always use identical math. Outside the
Perfect-WM environment these buffers stay at max range.

**Future observation types** (`FUTURE_OBS_TYPE` in `ablator.py`, `--wm_future_obs_type` in `train_wm.py`):

| Type | Future features | WM calls for the future |
|------|-----------------|-------------------------|
| `SIM-RAW` | future ToF image (like `DECODED`) | none |
| `SIM-MINDIST` | per-sensor min of that image (like `MIN-DECODED`) | none |
| `SIM-LATENT` | PWM encoder latent of that image (like `LATENT`) | encoder only, no dynamics |

With `CONTACT_PRED=True` the sim contact flag is appended in place of the PWM's. The ablator automatically
switches to the Perfect-WM task when `FUTURE_OBS_TYPE` starts with `SIM-`. The shared feature code is in
`scripts/rsl_rl/sim_future.py`. `TOFWorldModel` is not modified: it runs with `future_obs_type="none"`, so
no rollouts happen.

**Verifying** — set `save_video=True` in `ablator.py` (or call `play_wm_record.py` with
`--task Template-H12-Survive-Time-Perfect-WM`). The video shows the real ball and the red, semi-transparent
`Future_Projectile`. In `wm_observations.png`, the *Future* row shows the simulated future ToF image and
the *Contact* row shows the sim contact flag.