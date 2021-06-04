import argparse
import random
import sys
import pathlib

import torch
import torch.utils.data
from torch import optim
from torch.nn import functional as F
import numpy as np
import wandb

sys.path.append('.')
import src.utils.util as util
import src.embedding.gym_dataset as gym_dataset
from src.embedding.vae_dyne_action import ActionDynEVAE


def define_config():
    config = util.AttrDict()
    # General parameters
    config.no_cuda = False
    config.seed = 0
    config.logdir = pathlib.Path('./logdir')
    config.dataset_dir = pathlib.Path('./datasets')
    config.models_dir = pathlib.Path('./models')

    # Dataset parameters
    config.env = 'dmc_hopper_hop'
    config.name = 'DynE'
    config.traj_len = 4
    config.dataset_size = int(1e5)
    config.batch_size = 128
    config.qpos_only = False
    config.qpos_qvel = False
    config.delta = True
    config.whiten = True
    config.pixels = False
    config.source_img_width = 64

    # Training parameters
    config.epochs = 10
    config.log_interval = int(1e4)
    config.lr = 1e-4
    config.kl = 1e-4

    # Encoder parameters
    config.n_layers = 1
    config.embed_size = None

    # Wandb
    config.wandb = False
    config.wandb_proj = 'dyne-training'
    return config


# Reconstruction + KL divergence losses summed over all elements and batch
def loss_function(recon_x, x, mu, logvar, obs_size):
    likelihood = F.mse_loss(recon_x, x.view(-1, obs_size),
                            size_average=False)

    # see Appendix B from VAE paper: https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    # scale the likelihood term by number of state dimensions to make this loss
    # invariant to the environment's observation space
    return likelihood / obs_size, KLD


def main(config):
    torch.manual_seed(config.seed)
    random.seed(config.seed)
    np.random.seed(config.seed)

    cuda = not config.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if cuda else "cpu")

    model_path = config.models_dir / 'encoder' / config.env / config.name
    model_path.mkdir(parents=True, exist_ok=True)
    log_path = config.logdir / 'encoder' / config.env / config.name
    log_path.mkdir(parents=True, exist_ok=True)
    util.write_options(config, log_path)

    # builds a dataset by stepping a gym env with random actions
    dataset = gym_dataset.load_or_generate(config)

    train_loader = torch.utils.data.DataLoader(dataset,
                                               batch_size=config.batch_size,
                                               shuffle=True,
                                               num_workers=0)

    # calculate the sizes of everything
    epoch_size = len(dataset)
    action_space = dataset.env.action_space
    obs_size = util.prod(dataset.get_obs().shape)
    action_size = util.prod(action_space.shape)
    traj_size = action_size * config.traj_len
    if config.embed_size is None:
        embed_size = action_size
    else:
        embed_size = config.embed_size

    model = ActionDynEVAE(config.n_layers, traj_size, embed_size, obs_size).to(device)
    optimizer = optim.Adam(model.parameters(), lr=config.lr, amsgrad=True)

    if config.wandb:
        wandb.init(project=config.wandb_proj, entity='vinnibuh')
        wandb.config.update(util.clean_config(config))

    for epoch in range(1, config.epochs + 1):

        model.train()

        train_loss = 0
        train_likelihood = 0
        train_kld = 0

        temp_batch = 0
        temp_loss = 0
        temp_likelihood = 0
        temp_kld = 0

        qpos_loss, qvel_loss = 0, 0
        for batch_idx, (states, actions) in enumerate(train_loader):
            states = states.to(device).float()
            actions = actions.to(device).float()

            # feed first states to the model
            pred_states, mu, logvar = model(states[:, 0], actions)
            # compute Reconstruction + KLD loss
            likelihood, kld = loss_function(pred_states, states[:, -1], mu, logvar, obs_size)
            loss = likelihood + config.kl * kld

            # compute qpos and qvel losses (For what purpose??)
            qpos_loss += F.mse_loss(pred_states[:, :obs_size // 2], states[:, -1, :obs_size // 2])
            qvel_loss += F.mse_loss(pred_states[:, obs_size // 2:], states[:, -1, obs_size // 2:])

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_likelihood += likelihood.item()
            train_kld += kld.item()

            temp_loss += loss.item()
            temp_likelihood += likelihood.item()
            temp_kld += kld.item()

            if batch_idx > 0 and (batch_idx * config.batch_size) % config.log_interval < config.batch_size:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * config.batch_size, epoch_size,
                    100. * batch_idx / len(train_loader),
                    loss.item() / config.batch_size))
                temp_size = (batch_idx - temp_batch) * config.batch_size
                temp_batch = batch_idx
                wandb.log({'epoch_progress': 100. * batch_idx / len(train_loader)})
                wandb.log({'mean batch loss': temp_loss / temp_size})
                wandb.log({'mean batch likelihood': temp_likelihood / temp_size})
                wandb.log({'mean batch kld': temp_kld / temp_size})
                temp_loss = 0
                temp_likelihood = 0
                temp_kld = 0

        print(('====> Epoch: {} Average loss: {:.4f}'
               '\tLL: {:.6f}\tKLD: {:.6f}').format(
            epoch, train_loss / epoch_size,
                   train_likelihood / epoch_size,
                   train_kld / epoch_size))
        wandb.log({'epoch': epoch})
        wandb.log({'mean epoch loss': train_loss / epoch_size})
        wandb.log({'mean epoch likelihood': train_likelihood / epoch_size})
        wandb.log({'mean epoch kld': train_kld / epoch_size})

        print("qpos loss: {:.6f}, qvel_loss: {:.6f}".format(
            qpos_loss / epoch_size, qvel_loss / epoch_size))
    torch.save(model, model_path / 'encoder.pt')


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
