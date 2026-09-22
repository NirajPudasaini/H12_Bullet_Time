MID_NUM=20
FINAL_NUM=100

python scripts/rsl_rl/collect_data_2.py   --task Template-H12-Survive-Time-HYBRID   --ablation_results ablation_results/ablation_results_20260401_144930.json   --sensors RAY:TRUE_POS:X --max_range 2 --ckpt 500 --ckpt 1000 --ckpt 1500 --ckpt 2000 --ckpt 2500  --num_envs 1024 --num_trajectories $MID_NUM --trajs_per_file $MID_NUM

python scripts/rsl_rl/collect_data_2.py   --task Template-H12-Survive-Time-HYBRID   --ablation_results ablation_results/ablation_results_20260401_144930.json   --sensors RAY:DIST:X --max_range 2 --ckpt 500 --ckpt 1000 --ckpt 1500 --ckpt 2000 --ckpt 2500  --num_envs 1024 --num_trajectories $MID_NUM --trajs_per_file $MID_NUM

python scripts/rsl_rl/collect_data_2.py   --task Template-H12-Survive-Time-HYBRID   --ablation_results ablation_results/ablation_results_20260401_144930.json   --sensors RAY:MINDIST:X --max_range 2 --ckpt 500 --ckpt 1000 --ckpt 1500 --ckpt 2000 --ckpt 2500  --num_envs 1024 --num_trajectories $MID_NUM --trajs_per_file $MID_NUM

python scripts/rsl_rl/collect_data_2.py   --task Template-H12-Survive-Time-HYBRID   --ablation_results ablation_results/ablation_results_20260401_144930.json   --sensors RAY:TRUE_POS:X --max_range 2 --num_envs 1024 --num_trajectories $FINAL_NUM --trajs_per_file $FINAL_NUM

python scripts/rsl_rl/collect_data_2.py   --task Template-H12-Survive-Time-HYBRID   --ablation_results ablation_results/ablation_results_20260401_144930.json   --sensors RAY:DIST:X --max_range 2 --num_envs 1024 --num_trajectories $FINAL_NUM --trajs_per_file $FINAL_NUM

python scripts/rsl_rl/collect_data_2.py   --task Template-H12-Survive-Time-HYBRID   --ablation_results ablation_results/ablation_results_20260401_144930.json   --sensors RAY:MINDIST:X --max_range 2 --num_envs 1024 --num_trajectories $FINAL_NUM --trajs_per_file $FINAL_NUM