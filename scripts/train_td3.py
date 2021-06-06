"""
Main file for regular TD3.
Adapted from Scott Fujimoto's excellent implementation: https://github.com/sfujim/TD3/
"""

import numpy as np
import torch
import gym
import argparse
import pathlib
import sys

import wandb

sys.path.append('.')
import src.utils.util as util
import src.utils.buffers as buffers
import src.utils.wrappers as wrappers
from src.rl.TD3 import TD3


def define_config():
    config = util.AttrDict()
    # General parameters
    config.name = 'Vanilla-TD3'
    config.no_cuda = False
    config.seed = 0
    config.logdir = pathlib.Path('./logdir')
    config.models_dir = pathlib.Path('./models')

    # Dataset parameters
    config.env = 'dmc_hopper_hop'
    config.replay_size = int(1e5)

    # Training parameters
    config.max_timesteps = 1e7
    config.eval_freq = 1e3
    config.start_timesteps = 0
    config.expl_noise = 0.1
    config.batch_size = 128
    config.discount = 0.99
    config.tau = 0.005
    config.policy_noise = 0.2
    config.noise_clip = 0.5
    config.policy_freq = 2
    config.log_interval = int(1e4)
    config.render_freq = 5e3
    config.no_save_models = False

    # Wandb
    config.wandb = False
    config.wandb_proj = 'td3-training'
    return config


# Runs policy for X episodes and returns average reward
def evaluate_policy(policy, env, eval_episodes=10):
    avg_reward = 0.
    for episode in range(eval_episodes):
        obs = env.reset()
        policy.reset()
        done = False
        while not done:
            action = policy.select_action(np.array(obs))
            obs, reward, done, _ = env.step(action)
            avg_reward += reward

    avg_reward /= eval_episodes

    print("---------------------------------------")
    print("Evaluation over %d episodes: %f" % (eval_episodes, avg_reward))
    print("---------------------------------------")
    return avg_reward


def render_policy(policy, env, directory, filename, eval_episodes=5):
    frames = []
    for episode in range(eval_episodes):
        obs = env.reset()
        policy.reset()
        frames.append(env.render(mode='rgb_array'))
        done = False
        while not done:
            action = policy.select_action(np.array(obs))
            obs, reward, done, _ = env.step(action)
            frame = env.render(mode='rgb_array')

            frames.append(frame)

    util.save_gif(directory / '{}.mp4'.format(filename),
                  [torch.tensor(frame.copy()).float() / 255 for frame in frames],
                  color_last=True)


def main(config):
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    cuda = not config.no_cuda and torch.cuda.is_available()
    save_models = not config.no_save_models
    device = "cuda" if cuda else "cpu"

    model_path = config.models_dir / 'td3' / config.env / config.name
    model_path.mkdir(parents=True, exist_ok=True)

    log_path = config.logdir / 'rl' / 'TD3' / config.env / config.name
    log_path.mkdir(parents=True, exist_ok=True)
    util.write_options(config, log_path)

    render_path = log_path / 'render'
    render_path.mkdir(parents=True, exist_ok=True)

    eval_path = log_path / 'eval'
    eval_path.mkdir(parents=True, exist_ok=True)

    suite, task = config.env.split('_', 1)
    if suite == 'dmc':
        task = wrappers.DeepMindControl(task, seed=config.seed)
        env = wrappers.TimeLimit(task, 1000)
    elif suite == 'gym':
        env = gym.make(task)
        env.seed(config.seed)
    else:
        raise ValueError('Unsupported type of environment')

    env_max_steps = env._max_episode_steps
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])

    policy = TD3(state_dim, action_dim, max_action, device=device)
    replay_buffer = buffers.ReplayBuffer(max_size=config.replay_size)

    # Evaluate untrained policy
    evaluations = [(0, 0, evaluate_policy(policy, env))]

    if config.wandb:
        wandb.init(project=config.wandb_proj, entity='vinnibuh')
        wandb.config.update(util.clean_config(config))

    total_timesteps = 0
    timesteps_since_eval = 0
    timesteps_since_render = 0
    obs = env.reset()
    policy.reset()
    episode_reward = 0
    episode_timesteps = 0
    episode_num = 1
    done = False

    while total_timesteps < config.max_timesteps:

        if done:

            print("Total T: %d Episode Num: %d Episode T: %d Reward: %f" % (total_timesteps, episode_num, episode_timesteps, episode_reward))
            policy.train(replay_buffer, episode_timesteps, config=config)

            # Evaluate episode
            if timesteps_since_eval >= config.eval_freq:
                timesteps_since_eval %= config.eval_freq
                avg_reward = evaluate_policy(policy, env)
                wandb.log({'average reward over 10 episodes': avg_reward})
                wandb.log({'time step': total_timesteps})
                evaluations.append((episode_num, total_timesteps, avg_reward))

                if save_models:
                    policy.save("policy_{}".format(total_timesteps), directory=model_path)
                np.save(eval_path / "{}.npy".format(total_timesteps), np.stack(evaluations))

            if timesteps_since_render >= config.render_freq:
                timesteps_since_render %= config.render_freq
                render_policy(policy, env, render_path, total_timesteps)

            # Reset environment
            obs = env.reset()
            policy.reset()
            episode_reward = 0
            episode_timesteps = 0
            episode_num += 1

        # Select action randomly or according to policy
        if total_timesteps < config.start_timesteps:
            action = env.action_space.sample()
        else:
            action = policy.select_action(np.array(obs))
            util.noise_action(action, env.action_space, config.expl_noise)

        # Perform action
        new_obs, reward, done, _ = env.step(action)
        done_bool = 0 if episode_timesteps + 1 == env_max_steps else float(done)
        episode_reward += reward

        # Store data in replay buffer
        replay_buffer.add((obs, new_obs, action, reward, done_bool))

        obs = new_obs

        episode_timesteps += 1
        total_timesteps += 1
        timesteps_since_eval += 1
        timesteps_since_render += 1

    # Final evaluation
    evaluations.append((episode_num, total_timesteps, evaluate_policy(policy, env)))
    np.save(eval_path / "final.npy", np.stack(evaluations))
    render_policy(policy, env, render_path, 'final')
    if save_models:
        policy.save("final", directory=model_path)


if __name__ == "__main__":
    try:
        import colored_traceback

        colored_traceback.add_hook()
    except ImportError:
        pass
    parser = argparse.ArgumentParser()
    for key, value in define_config().items():
        parser.add_argument(f'--{key}', type=util.args_type(value), default=value)
    main(parser.parse_args())
