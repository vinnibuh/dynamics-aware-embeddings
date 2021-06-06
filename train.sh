#!/usr/bin/env bash
python3 scripts/train_td3.py --env gym_Thrower-v2 --wandb True --max_timesteps 200000 --start_timesteps 1000
python3 scripts/train_td3.py --env gym_Striker-v2 --wandb True --max_timesteps 200000 --start_timesteps 1000
