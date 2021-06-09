#!/usr/bin/env bash
python3 ./scripts/train_decoder.py --traj_len 4 --wandb True --name DynE-4 --env gym_Thrower-v2
python3 ./scripts/train_decoder.py --traj_len 6 --wandb True --name DynE-6 --env gym_Thrower-v2
python3 ./scripts/train_decoder.py --traj_len 8 --wandb True --name DynE-8 --env gym_Thrower-v2