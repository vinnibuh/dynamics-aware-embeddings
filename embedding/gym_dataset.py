import random
import numpy as np
import os
import pathlib

import gym

import torch
from torch.utils.data import Dataset

from pixel_wrapper import PixelObservationWrapper
import wrappers


class GymData(Dataset):
    def __init__(self, env_name, traj_len, cache_size=100000, qpos_only=False, qpos_qvel=False, delta=True, whiten=True,
                 pixels=False, source_img_width=64, seed=None):
        self.suite, self.task = env_name.split('_', 1)
        self.traj_len = traj_len
        self.qpos_only = qpos_only
        self.qpos_qvel = qpos_qvel
        self.delta = delta
        self.whiten = whiten
        self.pixels = pixels
        self.source_img_width = source_img_width
        self.cache_size = cache_size
        self.cache = {}
        self.seed = seed

        self.done = False
        self.env_has_been_reset = False
        self.steps_since_reset = 0
        self.make_env()
        self.mean = 0
        self.std = 1
        if self.whiten:
            self.make_stats()

    def make_stats(self):
        states = []
        stat_samples = min(self.env_max_steps * 1000, 100000)
        print("Generating samples for computing statistics.")
        for i in range(stat_samples):
            if i % 10000 == 0:
                print("{}/{}".format(i, stat_samples))
            state, _ = self[i]
            states.append(state)
        states = torch.cat(states, 0)
        self.mean, self.std = states.mean(dim=0), states.std(dim=0)
        self.std[self.std == 0] = 1

    # a large number so we never re-instantiate a worker
    def __len__(self):
        return int(1e8)

    def get_obs(self):
        if self.suite == 'dmc':
            return self.env.observation()
        if self.qpos_only:
            return np.copy(self.env.sim.data.qpos)
        elif self.qpos_qvel:
            qpos = np.copy(self.env.sim.data.qpos)
            qvel = np.copy(self.env.sim.data.qvel)
            return np.concatenate([qpos.flat, qvel.flat])
        elif self.pixels:
            return self.env.observation()
        else:
            return np.copy(self.env._get_obs())

    def make_env(self):
        if self.suite == 'dmc':
            task = wrappers.DeepMindControl(self.task, seed=self.seed)
            self.env = wrappers.TimeLimit(task, 1000)
        elif self.suite == 'gym':
            self.env = gym.make(self.task)
        self.env_max_steps = self.env._max_episode_steps

        if hasattr(self.env, 'unwrapped'):
            self.env = self.env.unwrapped

        if self.pixels:
            self.env = PixelObservationWrapper(self.env, source_img_width=self.source_img_width)

        self.env.reset()
        # burn some steps to prevent workers from being correlated
        for _ in range(random.randrange(0, 10)):
            obs, act = self.generate_trajectory()

    def generate_trajectory(self):
        if self.done or self.steps_since_reset >= self.env_max_steps:
            self.env.reset()
            self.steps_since_reset = 0
            self.done = False
        obs = [self.get_obs()]
        actions = []
        for _ in range(self.traj_len):
            action = self.env.action_space.sample()
            actions.append(action)
            _, _, step_done, _ = self.env.step(action)
            self.done = self.done or step_done
            obs.append(self.get_obs())

        self.steps_since_reset += self.traj_len
        actions = torch.stack([torch.from_numpy(a).float() for a in actions])
        obs = torch.stack([torch.from_numpy(o).float() for o in obs])

        # only predict change in state, not absolute state
        if self.delta:
            delta_obs = torch.zeros_like(obs)
            for i in range(1, obs.size(0)):
                delta_obs[i] = obs[i] - obs[0]
            obs = delta_obs

        return obs, actions

    def __getitem__(self, i):
        if not self.env_has_been_reset:
            self.make_env()
            self.env_has_been_reset = True

        i = i % self.cache_size
        if i not in self.cache:
            self.cache[i] = self.generate_trajectory()

        obs, action = self.cache[i][0].float(), self.cache[i][1].float()

        # whitening observations helps account for the difference in scale between
        # positions and velocities
        if self.whiten:
            obs = (obs - self.mean) / self.std

        return obs, action


class AbstractActionsData(Dataset):
    def __init__(self, env_name, traj_len, encoder, qpos_only=False, qpos_qvel=False, delta=True, whiten=True,
                 pixels=False, source_img_width=64, seed=None):
        self.suite, self.task = env_name.split('_', 1)
        self.traj_len = traj_len
        self.encoder = encoder
        self.qpos_only = qpos_only
        self.qpos_qvel = qpos_qvel
        self.delta = delta
        self.whiten = whiten
        self.pixels = pixels
        self.source_img_width = source_img_width
        self.episodes = {}
        self.mapping = {}
        self.seed = seed

        self.mean = 0
        self.std = 1

    # a large number so we never re-instantiate a worker
    def __len__(self):
        return len(self.episodes.keys())

    def load_stats(self, directory):
        mean_path = pathlib.Path(directory) / 'hopper_hop_mean.pt'
        std_path = pathlib.Path(directory) / 'hopper_hop_std.pt'
        with mean_path.open('rb') as f:
            self.mean = torch.load(f)
        with std_path.open('rb') as f:
            self.std = torch.load(f)

    def load_from_directory(self, directory):
        directory = pathlib.Path(directory).expanduser()
        for idx, filename in enumerate(directory.glob('*.npz')):
            if filename not in self.mapping.keys():
                try:
                    with filename.open('rb') as f:
                        episode = np.load(f)
                        episode = {k: torch.from_numpy(episode[k]).float() for k in episode.keys()}
                        episode['obs'] = torch.cat([episode['position'],
                                                      episode['velocity'],
                                                      episode['touch']], 1)
                        episode.pop('position')
                        episode.pop('velocity')
                        episode.pop('touch')

                        self.episodes[idx] = episode
                        self.mapping[filename] = idx
                except Exception as e:
                    print(f'Could not load episode: {e}')
                    continue

    def transform_episode(self, i):
        obs, action, rewards = self[i]

        episode_size = obs.size(0)
        obs_dim = obs.size(1)
        action_dim = action.size(1)

        obs_reshaped = obs[:-1].reshape([episode_size // self.traj_len,
                                         self.traj_len,
                                         obs_dim])
        last_states = torch.unsqueeze(obs[4::4], 1)
        obs = torch.repeat_interleave(torch.cat([obs_reshaped, last_states], 1), 2, dim=0)

        action = action[1:].reshape([episode_size // self.traj_len,
                                     self.traj_len,
                                     action_dim])
        action = torch.repeat_interleave(action, 2, dim=0)
        with torch.no_grad():
            pred_states, mu, logvar = self.encoder(obs[:, 0], action)

        return mu, logvar

    def __getitem__(self, i):
        episode = self.episodes[i]
        obs = episode['obs']
        if self.delta:
            delta_obs = torch.zeros_like(obs)
            for i in range(1, obs.size(0)):
                delta_obs[i] = obs[i] - obs[0]
            obs = delta_obs

        if self.whiten:
            obs = (obs - self.mean) / self.std
        return obs, episode['action'], episode['reward']

def load_episodes(directory, rescan, length=None, balance=False, seed=0):
    directory = pathlib.Path(directory).expanduser()
    randomizer = np.random.RandomState(seed)
    cache = {}
    while True:
        for filename in directory.glob('*.npz'):
            if filename not in cache:
                try:
                    with filename.open('rb') as f:
                        episode = np.load(f)
                        episode = {k: episode[k] for k in episode.keys()}
                except Exception as e:
                    print(f'Could not load episode: {e}')
                    continue
                cache[filename] = episode
        keys = list(cache.keys())
        for index in randomizer.choice(len(keys), rescan):
            episode = cache[keys[index]]
            if length:
                total = len(next(iter(episode.values())))
                available = total - length
                if available < 1:
                    print(f'Skipped short episode of length {available}.')
                    continue
                if balance:
                    index = min(randomizer.randint(0, total), available)
                else:
                    index = int(randomizer.randint(0, available))
                episode = {k: v[index: index + length] for k, v in episode.items()}
            yield episode

def data_path(env_name, traj_len, cache_size, qpos_only, qpos_qvel, delta, whiten, pixels, source_img_width):
    name = "{}_len{}_n{}_qpos{}_qvel{}_delta{}_whiten{}_pixels{}".format(
        env_name, traj_len, cache_size, qpos_only, qpos_qvel, delta, whiten, pixels)
    if pixels:
        name += "_imgwidth{}".format(source_img_width)
    return 'data/{}.pt'.format(name)


def generate_and_save(env_name, traj_len, cache_size=100000, qpos_only=False, qpos_qvel=False,
                      delta=True, whiten=True, pixels=False, source_img_width=64):
    dataset = GymData(env_name, traj_len, cache_size, qpos_only, qpos_qvel, delta, whiten, pixels, source_img_width)
    print("Generating dataset to save.")
    for i in range(dataset.cache_size):
        _ = dataset[i]
        if i % 10000 == 0: print("{}/{}".format(i, dataset.cache_size))
    path = data_path(env_name, traj_len, cache_size, qpos_only, qpos_qvel, delta, whiten, pixels, source_img_width)
    data_folder = os.path.dirname(path)
    os.makedirs(data_folder, exist_ok=True)
    torch.save(dataset, path)
    return dataset


def load_or_generate(env_name, traj_len, cache_size=100000, qpos_only=False, qpos_qvel=False,
                     delta=True, whiten=True, pixels=False, source_img_width=64):
    path = data_path(env_name, traj_len, cache_size, qpos_only, qpos_qvel, delta, whiten, pixels, source_img_width)
    try:
        print("Attempting to load data from {}".format(path))
        dataset = torch.load(path)
    except FileNotFoundError:
        dataset = generate_and_save(env_name, traj_len, cache_size, qpos_only, qpos_qvel, delta, whiten, pixels,
                                    source_img_width)
    return dataset
