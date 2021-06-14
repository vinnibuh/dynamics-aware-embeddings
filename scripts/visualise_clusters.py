import torch
import sys
import argparse
import pathlib
import random
import numpy as np
import matplotlib.pyplot as plt
from MulticoreTSNE import MulticoreTSNE as TSNE
import matplotlib.cm as cm
import wandb

sys.path.append('.')
import src.utils.util as util
from src.embedding.gym_dataset import AbstractActionsData
from src.embedding.gym_dataset import load_or_generate


def define_config():
    config = util.AttrDict()
    # General parameters
    config.seed = 0
    config.models_dir = pathlib.Path('./models')
    config.dataset_dir = pathlib.Path('./datasets')
    config.episodes_dir = pathlib.Path('./datasets/dreamer/dmc_hopper_hop/dreamer_finetuned')
    config.log_dir = pathlib.Path('./logdir')

    # Dataset parameters
    config.env = 'dmc_hopper_hop'
    config.encoder_name = 'DynE-4'
    config.traj_len = 4
    config.dataset_size = int(1e5)
    config.qpos_only = False
    config.qpos_qvel = False
    config.delta = True
    config.whiten = True
    config.pixels = False
    config.source_img_width = 64

    # Visualisation parameters
    config.a_text = 0
    config.a_points = 1
    config.n_samples = 100

    # Wandb
    config.wandb = False
    config.wandb_proj = 'dyne-visualise'
    return config


def vis_embeddings(label, embeddings, words=[], a_points=1, a_text=0.6, ax=None):
    colors = cm.coolwarm(np.linspace(0, 1, len(embeddings)))
    x = embeddings[:, 0]
    y = embeddings[:, 1]
    ax.scatter(x, y, c=colors, alpha=a_points, label=label)
    for i, word in enumerate(words):
        ax.annotate(word, alpha=a_text, xy=(x[i], y[i]), xytext=(5, 2),
                    textcoords='offset points', ha='right', va='bottom', size=10)
    ax.legend(loc=4)
    ax.grid(True)


def tsne_plot_2d(label, embeddings, color_style='time', rewards=[], a_points=1, size=15,
                 log=False, wandb_label=None, ax=None, episodes_num=100):
    ax = plt if ax is None else ax
    if color_style == 'time':
        colors = cm.coolwarm(np.linspace(0, 1, int(len(embeddings)/episodes_num)))
        colors = np.tile(colors, (episodes_num, 1))
    elif color_style == 'rewards':
        rewards = np.array(rewards)
        scaled_rewards = rewards / rewards.max()
        colors = cm.coolwarm(scaled_rewards)
    elif color_style == 'episodes':
        colors = cm.coolwarm(np.linspace(0, 1, episodes_num))
        colors = np.repeat(colors, int(len(embeddings)/episodes_num), 0)
    x = embeddings[:,0]
    y = embeddings[:,1]
    ax.scatter(x, y, s=size, c=colors, alpha=a_points, label=label)
    ax.legend(loc=2)
    ax.grid(True)
    if log:
        wandb_label = label if wandb_label is None else wandb_label
        wandb.log({wandb_label: wandb.Image(plt)})


def compare_embeddings_1_by_1(config, label, data, n_samples, a_points=1, a_text=0.6, log=True, images_path=None):
    sample_idx = random.sample(range(len(data)), n_samples)
    fig, axes = plt.subplots(n_samples, 2, figsize=(30, 60))

    for idx, s in enumerate(sample_idx):

        raw_obs, raw_action, raw_reward = data[s]
        raw_action = torch.repeat_interleave(raw_action[1:], 2, dim=0)
        action_dim = raw_action.size(1)
        initial_episode_size = raw_action.size(0)
        actual_episode_size = initial_episode_size - (initial_episode_size % config.traj_len)
        raw_action = raw_action[:actual_episode_size]
        raw_embeddings = raw_action.reshape([actual_episode_size // config.traj_len,
                                             config.traj_len * action_dim])

        mu, logvar = data.transform_episode(s)

        numbers_ak = []
        embeddings_ak = []
        for k, vector in enumerate(raw_embeddings):
            embeddings_ak.append(vector.numpy())
            numbers_ak.append(k)

        tsne_ak_2d = TSNE(perplexity=30, n_components=2, init='pca', n_iter=3500, random_state=32)
        embeddings_ak_2d = tsne_ak_2d.fit_transform(embeddings_ak)

        vis_embeddings("episode {}, raw actions".format(s), embeddings_ak_2d,
                       numbers_ak, a_points=a_points, a_text=a_text, ax=axes[idx][0])

        dyne_numbers_ak = []
        dyne_emb_ak = []

        for k, vector in enumerate(mu):
            dyne_emb_ak.append(vector.numpy())
            dyne_numbers_ak.append(k)

        tsne_dyne_ak_2d = TSNE(perplexity=30, n_components=2, init='pca', n_iter=3500, random_state=32)
        embeddings_dyne_ak_2d = tsne_dyne_ak_2d.fit_transform(dyne_emb_ak)

        vis_embeddings("episode {}, dyne actions".format(s), embeddings_dyne_ak_2d,
                       dyne_numbers_ak, a_points=a_points, a_text=a_text, ax=axes[idx][1])

    fig.savefig(images_path / 'comparison.png', format='png', dpi=150, bbox_inches='tight')
    if log:
        wandb.log({label: wandb.Image(fig)})
    else:
        plt.show()


def compare_embeddings_in_group(config, data, n_samples, point_size=10, log=True, images_path=None):
    sample_idx = random.sample(range(len(data)), n_samples)

    raw_obs, raw_action, raw_reward = data[0]
    raw_action = torch.repeat_interleave(raw_action[1:], 2, dim=0)
    action_dim = raw_action.size(1)
    initial_episode_size = raw_action.size(0)
    actual_episode_size = initial_episode_size - (initial_episode_size % config.traj_len)

    rewards_ak = []
    embeddings_ak = []
    for k in sample_idx:
        raw_obs, raw_action, raw_reward = data[k]
        raw_embeddings = torch.repeat_interleave(raw_action[1:], 2, dim=0)[:actual_episode_size] \
            .reshape([actual_episode_size // config.traj_len,
                      config.traj_len * action_dim])

        rewards = torch.repeat_interleave(raw_reward[1:]/2, 2, dim=0)[:actual_episode_size] \
            .reshape([actual_episode_size // config.traj_len,
                      config.traj_len]).sum(axis=1)
        for idx, vector in enumerate(raw_embeddings):
            embeddings_ak.append(vector.numpy())
            rewards_ak.append(rewards[idx])

    tsne_ak_2d = TSNE(perplexity=30, n_components=2, init='random', n_iter=3500, random_state=32, n_jobs=8)
    embeddings_ak_2d = tsne_ak_2d.fit_transform(np.array(embeddings_ak))

    dyne_emb_ak = []
    for k in sample_idx:
        mu, logvar, _ = data.transform_episode(k)
        for idx, vector in enumerate(mu):
            dyne_emb_ak.append(vector.numpy())

    tsne_dyne_ak_2d = TSNE(perplexity=30, n_components=2, init='random', n_iter=3500, random_state=32, n_jobs=8)
    embeddings_dyne_ak_2d = tsne_dyne_ak_2d.fit_transform(np.array(dyne_emb_ak))

    fig, axes = plt.subplots(2, 2, figsize=(20, 20), constrained_layout=True)
    tsne_plot_2d('raw actions by time'.format(config.env), embeddings_ak_2d,
                 color_style='time', rewards=rewards_ak, size=point_size,
                 log=False, ax=axes[0][0], episodes_num=n_samples)

    tsne_plot_2d('dyne actions by time', embeddings_dyne_ak_2d,
                 color_style='time', rewards=rewards_ak, size=point_size,
                 log=False, ax=axes[0][1], episodes_num=n_samples)

    tsne_plot_2d('raw actions by rewards', embeddings_ak_2d,
                 color_style='rewards', rewards=rewards_ak, size=point_size,
                 log=False, ax=axes[1][0], episodes_num=n_samples)

    tsne_plot_2d('dyne actions by rewards', embeddings_dyne_ak_2d,
                 color_style='rewards', rewards=rewards_ak, size=point_size,
                 log=False, ax=axes[1][1], episodes_num=n_samples)

    fig.suptitle("{}_DynE-{}".format(config.env, config.traj_len), fontsize=16)
    fig.savefig(images_path / "{}_emb_comparison_{}_samples.png".format(config.env, config.n_samples),
                format='png', dpi=150, bbox_inches='tight')
    if log:
        wandb.log({'Embeddings Comparison': wandb.Image(fig)})


def main(config):
    torch.manual_seed(config.seed)
    random.seed(config.seed)
    np.random.seed(config.seed)

    encoder_path = config.models_dir / 'encoder' / config.env / config.encoder_name / 'encoder.pt'
    episodes_path = config.episodes_dir / 'episodes'
    images_path = config.log_dir / 'clusters' / config.env / config.encoder_name
    images_path.mkdir(parents=True, exist_ok=True)
    assert encoder_path.exists()
    assert episodes_path.exists()

    encoder = torch.load(encoder_path)
    data = AbstractActionsData(config.env, config.traj_len, encoder)

    gym_data = load_or_generate(config)
    data.mean, data.std = gym_data.mean, gym_data.std
    del gym_data

    data.load_from_directory(episodes_path)

    if config.wandb:
        wandb.init(project=config.wandb_proj, entity='vinnibuh')
        wandb.config.update(util.clean_config(config))

    compare_embeddings_in_group(config, data, config.n_samples, point_size=10,
                                log=config.wandb, images_path=images_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    for key, value in define_config().items():
        parser.add_argument(f'--{key}', type=util.args_type(value), default=value)
    main(parser.parse_args())
