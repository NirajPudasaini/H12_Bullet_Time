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