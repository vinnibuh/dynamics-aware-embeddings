#!/usr/bin/env bash
python3 scripts/visualise_clusters.py --env dmc_walker_walk --episodes_dir ./datasets/dreamer/dmc_walker_walk/1 --n_samples 100 --encoder_name DynE-4 --traj_len 4 --wandb True
python3 scripts/visualise_clusters.py --env dmc_walker_walk --episodes_dir ./datasets/dreamer/dmc_walker_walk/1 --n_samples 100 --encoder_name DynE-6 --traj_len 6 --wandb True
python3 scripts/visualise_clusters.py --env dmc_walker_walk --episodes_dir ./datasets/dreamer/dmc_walker_walk/1 --n_samples 100 --encoder_name DynE-8 --traj_len 8 --wandb True
