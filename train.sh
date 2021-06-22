#!/usr/bin/env bash
python3 scripts/visualise_clusters.py --env dmc_walker_stand --episodes_dir ./datasets/dreamer/dmc_walker_stand/1 --n_samples 100 --encoder_name DynE-4 --traj_len 4 --wandb True
python3 scripts/visualise_clusters.py --env dmc_walker_stand --episodes_dir ./datasets/dreamer/dmc_walker_stand/1 --n_samples 100 --encoder_name DynE-6 --traj_len 6 --wandb True
python3 scripts/visualise_clusters.py --env dmc_walker_stand --episodes_dir ./datasets/dreamer/dmc_walker_stand/1 --n_samples 100 --encoder_name DynE-8 --traj_len 8 --wandb True
python3 scripts/visualise_clusters.py --env dmc_walker_walk --encoder_env dmc_walker_stand --episodes_dir ./datasets/dreamer/dmc_walker_walk/1 --n_samples 100 --encoder_name DynE-4 --traj_len 4 --wandb True
python3 scripts/visualise_clusters.py --env dmc_walker_walk --encoder_env dmc_walker_stand --episodes_dir ./datasets/dreamer/dmc_walker_walk/1 --n_samples 100 --encoder_name DynE-6 --traj_len 6 --wandb True
python3 scripts/visualise_clusters.py --env dmc_walker_walk --encoder_env dmc_walker_stand --episodes_dir ./datasets/dreamer/dmc_walker_walk/1 --n_samples 100 --encoder_name DynE-8 --traj_len 8 --wandb True
python3 scripts/visualise_clusters.py --env dmc_hopper_hop --episodes_dir ./datasets/dreamer/dmc_hopper_hop/dreamer_finetuned --n_samples 100 --encoder_name DynE-4 --traj_len 4 --wandb True
python3 scripts/visualise_clusters.py --env dmc_hopper_hop --episodes_dir ./datasets/dreamer/dmc_hopper_hop/dreamer_finetuned --n_samples 100 --encoder_name DynE-6 --traj_len 6 --wandb True
python3 scripts/visualise_clusters.py --env dmc_hopper_hop --episodes_dir ./datasets/dreamer/dmc_hopper_hop/dreamer_finetuned --n_samples 100 --encoder_name DynE-8 --traj_len 8 --wandb True
python3 scripts/visualise_clusters.py --env dmc_hopper_hop --episodes_dir ./datasets/dreamer/dmc_hopper_hop/dreamer_finetuned --n_samples 100 --encoder_name DynE-4-kl-1e-3 --traj_len 4 --wandb True
python3 scripts/visualise_clusters.py --env dmc_hopper_hop --episodes_dir ./datasets/dreamer/dmc_hopper_hop/dreamer_finetuned --n_samples 100 --encoder_name DynE-4-kl-1e-2 --traj_len 4 --wandb True
python3 scripts/visualise_clusters.py --env dmc_hopper_hop --episodes_dir ./datasets/dreamer/dmc_hopper_hop/dreamer_finetuned --n_samples 100 --encoder_name DynE-4-kl-1e-1 --traj_len 4 --wandb True
python3 scripts/visualise_clusters.py --env dmc_hopper_hop --episodes_dir ./datasets/dreamer/dmc_hopper_hop/dreamer_finetuned --n_samples 100 --encoder_name DynE-4-kl-1 --traj_len 4 --wandb True

dvc add logdir
dvc push
git add .
git commit 'new viz for embeddings'
git push vinnibuh loss_modification