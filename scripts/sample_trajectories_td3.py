import numpy as np
import torch
import gym
import argparse
import pathlib
import sys

import wandb

sys.path.append('.')
import src.utils.util as util
import src.utils.wrappers as wrappers
import src.rl.TD3 as TD3


def define_config():
    config = util.AttrDict()
    # General parameters
    config.name = 'Vanilla-TD3'
    config.no_cuda = False
    config.seed = 0
    config.logdir = pathlib.Path('./logdir')
    config.model_prefix = 'final'
    config.model_dir = None

    # Dataset parameters
    config.env = 'dmc_hopper_hop'

    # Training parameters
    config.precision = 32
    config.episodes = 1e3
    config.eval_every = 10

    # Wandb
    config.wandb = False
    config.wandb_proj = 'td3-sampling'
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


def summarize_episode(episode, datadir, prefix):
    episodes, steps = util.count_episodes(datadir)
    length = len(episode['reward']) - 1
    ret = episode['reward'].sum()
    print(f'{prefix.title()} episode of length {length} with return {ret:.1f}.')
    metrics = [
        (f'{prefix}/return', float(episode['reward'].sum())),
        (f'{prefix}/length', len(episode['reward']) - 1),
        (f'episodes', episodes)]
    for k, v in metrics:
        wandb.log({k: v})


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

    log_path = config.logdir / 'rl' / 'TD3' / config.env / config.name
    log_path.mkdir(parents=True, exist_ok=True)

    if config.model_dir is None:
        config.model_dir = pathlib.Path('.') / 'models' / 'td3' / config.env / config.name

    episodes_path = log_path / 'episodes'
    episodes_path.mkdir(parents=True, exist_ok=True)
    util.write_options(config, log_path)

    suite, task = config.env.split('_', 1)
    if suite == 'dmc':
        task = wrappers.DeepMindControl(task, seed=config.seed)
        env = wrappers.TimeLimit(task, 1000)
    elif suite == 'gym':
        env = gym.make(task)
        env.seed(config.seed)
    else:
        raise ValueError('Unsupported type of environment')
    callbacks = []
    callbacks.append(lambda ep: util.save_episodes(episodes_path, [ep]))
    callbacks.append(
        lambda ep: summarize_episode(ep, episodes_path, 'sample'))
    env = wrappers.Collect(env, callbacks, config.precision)

    policy = TD3.load(config.model_prefix, config.model_dir)

    if config.wandb:
        wandb.init(project=config.wandb_proj, entity='vinnibuh')
        wandb.config.update(util.clean_config(config))

    total_episodes = 0
    env.reset()
    policy.reset()

    while total_episodes < config.episodes:
        # Evaluate episode
        avg_reward = evaluate_policy(policy, env, config.eval_every)
        wandb.log({'average reward over 10 episodes': avg_reward})
        wandb.log({'total episodes': total_episodes})

        # Reset environment
        env.reset()
        policy.reset()

        total_episodes += config.eval_every


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
