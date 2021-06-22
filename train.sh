#!/usr/bin/env bash
python3 scripts/train_decoder.py --name DynE-4-kl-1e-3 --env dmc_hopper_hop --wandb True --traj_len 4 
python3 scripts/train_decoder.py --name DynE-4-kl-1e-2 --env dmc_hopper_hop --wandb True --traj_len 4 
python3 scripts/train_decoder.py --name DynE-4-kl-1e-1 --env dmc_hopper_hop --wandb True --traj_len 4 
python3 scripts/train_decoder.py --name DynE-4-kl-1 --env dmc_hopper_hop --wandb True --traj_len 4 