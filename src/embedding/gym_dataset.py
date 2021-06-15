import random
import numpy as np
import pathlib
import sys

import gym

import torch
from torch.utils.data import Dataset

sys.path.append('.')
import src.utils.wrappers as wrappers


class GymData(Dataset):
    def __init__(self, config):
        self._c = config
        self.suite, self.task = self._c.env.split('_', 1)
        self.cache = {}

        self.done = False
        self.env_has_been_reset = False
        self.steps_since_reset = 0
        self.make_env()
        self.mean = 0
        self.std = 1
        if self._c.whiten:
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
        return self._c.dataset_size

    def get_obs(self):
        if self.suite == 'dmc':
            return self.env.observation()
        if self._c.qpos_only:
            return np.copy(self.env.sim.data.qpos)
        elif self._c.qpos_qvel:
            qpos = np.copy(self.env.sim.data.qpos)
            qvel = np.copy(self.env.sim.data.qvel)
            return np.concatenate([qpos.flat, qvel.flat])
        elif self._c.pixels:
            return self.env.observation()
        else:
            return np.copy(self.env._get_obs())

    def make_env(self):
        if self.suite == 'dmc':
            task = wrappers.DeepMindControl(self.task, seed=self._c.seed)
            self.env = wrappers.TimeLimit(task, 1000)
        elif self.suite == 'gym':
            self.env = gym.make(self.task)
        self.env_max_steps = self.env._max_episode_steps

        if hasattr(self.env, 'unwrapped'):
            self.env = self.env.unwrapped

        if self._c.pixels:
            self.env = wrappers.PixelObservationWrapper(self.env, source_img_width=self._c.source_img_width)

        self.env.reset()
        # burn some steps to prevent workers from being correlated
        for _ in range(random.randrange(0, 10)):
            obs, act = self.generate_trajectory()

    def generate_trajectory(self):
        if self.done or self.steps_since_reset >= self.env_max_steps - self._c.traj_len:
            self.env.reset()
            self.steps_since_reset = 0
            self.done = False
        obs = [self.get_obs()]
        actions = []
        for _ in range(self._c.traj_len):
            action = self.env.action_space.sample()
            actions.append(action)
            _, _, step_done, _ = self.env.step(action)
            self.done = self.done or step_done
            obs.append(self.get_obs())

        self.steps_since_reset += self._c.traj_len
        actions = torch.stack([torch.from_numpy(a).float() for a in actions])
        obs = torch.stack([torch.from_numpy(o).float() for o in obs])

        # only predict change in state, not absolute state
        if self._c.delta:
            delta_obs = torch.zeros_like(obs)
            for i in range(1, obs.size(0)):
                delta_obs[i] = obs[i] - obs[0]
            obs = delta_obs

        return obs, actions

    def __getitem__(self, i):
        if not self.env_has_been_reset:
            self.make_env()
            self.env_has_been_reset = True

        i = i % self._c.dataset_size
        if i not in self.cache:
            self.cache[i] = self.generate_trajectory()

        obs, action = self.cache[i][0].float(), self.cache[i][1].float()

        # whitening observations helps account for the difference in scale between
        # positions and velocities
        if self._c.whiten:
            obs = (obs - self.mean) / self.std

        return obs, action


class AbstractActionsData(Dataset):
    def __init__(self, env, traj_len, encoder, action_repeat=2, qpos_only=False, qpos_qvel=False, delta=True, whiten=True,
                 pixels=False, source_img_width=64, seed=None):
        self.suite, self.task = env.split('_', 1)
        self.traj_len = traj_len
        self.encoder = encoder
        self.action_repeat = action_repeat
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

    def load_from_directory(self, directory, images=True):
        directory = pathlib.Path(directory).expanduser()
        for idx, filename in enumerate(directory.glob('*.npz')):
            if filename not in self.mapping.keys():
                try:
                    with filename.open('rb') as f:
                        episode = np.load(f)
                        episode = {k: torch.from_numpy(episode[k]).float() for k in episode.keys()}
                        if self.suite == 'dmc':
                            if 'hopper' in self.task:
                                episode['obs'] = torch.cat([episode['position'],
                                                            episode['velocity'],
                                                            episode['touch']], 1)
                                episode.pop('position')
                                episode.pop('velocity')
                                episode.pop('touch')
                            if 'walker' in self.task:
                                episode['obs'] = torch.cat([episode['orientations'],
                                                            torch.unsqueeze(episode['height'], 1),
                                                            episode['velocity']], 1)
                                episode.pop('orientations')
                                episode.pop('height')
                                episode.pop('velocity')
                            if not images:
                                episode.pop('image')
                        elif self.suite == 'gym':
                            pass
                        else:
                            raise NotImplementedError("This type of env is not supported")
                        self.episodes[idx] = episode
                        self.mapping[filename] = idx
                except Exception as e:
                    print(f'Could not load episode: {e}')
                    continue

    def transform_episode(self, i):
        obs, action, rewards = self[i]

        # account for action repeat and first zero action
        obs = torch.repeat_interleave(obs, self.action_repeat, dim=0)
        action = torch.repeat_interleave(action[1:], self.action_repeat, dim=0)

        action_dim = action.size(1)

        # account for not completely divisible sizes of episodes
        initial_episode_size = action.size(0)
        actual_episode_size = initial_episode_size - (initial_episode_size % self.traj_len)
        obs = obs[:actual_episode_size]
        action = action[:actual_episode_size]

        first_states = obs[:actual_episode_size][::self.traj_len]
        action = action.reshape([actual_episode_size // self.traj_len,
                                 self.traj_len,
                                 action_dim])
        with torch.no_grad():
            pred_states, mu, logvar = self.encoder(first_states, action)

        return mu, logvar, pred_states

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


def data_path(config):
    name = "{}_len{}_n{}_qpos{}_qvel{}_delta{}_whiten{}_pixels{}".format(
        config.env, config.traj_len, config.dataset_size, config.qpos_only,
        config.qpos_qvel, config.delta, config.whiten, config.pixels)
    if config.pixels:
        name += "_imgwidth{}".format(config.source_img_width)
    return config.dataset_dir / 'torch' / '{}.pt'.format(name)


def generate_and_save(config):
    dataset = GymData(config)
    print("Generating dataset to save.")
    for i in range(config.dataset_size):
        _ = dataset[i]
        if i % 10000 == 0:
            print("{}/{}".format(i, len(dataset)))
    path = data_path(config)
    data_folder = path.parent
    data_folder.mkdir(parents=True, exist_ok=True)
    torch.save(dataset, path)
    return dataset


def load_or_generate(config):
    path = data_path(config)
    try:
        print("Attempting to load data from {}".format(path))
        dataset = torch.load(path)
    except (FileNotFoundError, ModuleNotFoundError):
        dataset = generate_and_save(config)
    return dataset
