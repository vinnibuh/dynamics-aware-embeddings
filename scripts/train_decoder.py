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
import embedding.util as util
import embedding.gym_dataset as gym_dataset
from embedding.action_decoder import ActionDecoder


def define_config():
    config = util.AttrDict()
    # General parameters
    config.no_cuda = False
    config.seed = 0
    config.encoder_path = None
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
    config.norm_loss = 1e-4

    # Decoder parameters
    config.n_layers = 1
    config.embed_size = None

    # Wandb
    config.wandb = False
    config.wandb_proj = 'dyne-decoders'
    return config


# sample from the marginal distribution of the encoder instead of the prior
def turn_to_z(actions, model, device):
    actions = actions.to(device).float()
    z = model.encode(actions)[0]
    return z


# used to whiten the latent space before training the decoder
def marginal_stats(train_loader, model, device):
    zs = []
    for i, (states, actions) in enumerate(train_loader):
        actions = actions.to(device).float()
        zs.append(model.encode(actions)[0])
    zs = torch.cat(zs, dim=0)
    mean, std = zs.mean(dim=0), zs.std(dim=0)
    white_zs = (zs - mean) / std
    white_max = white_zs.abs().max()
    return mean, std, white_max


def main(config):
    torch.manual_seed(config.seed)
    random.seed(config.seed)
    np.random.seed(config.seed)

    cuda = not config.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if cuda else "cpu")

    # define data paths
    decoder_path = config.models_dir / 'decoder' / config.env / config.name
    decoder_path.mkdir(parents=True, exist_ok=True)

    encoder_path = config.encoder_path
    if encoder_path is None:
        encoder_path = config.models_dir / 'encoder' / config.env / config.name / 'encoder.pt'

    log_path = config.logdir / 'decoder' / config.env / config.name
    log_path.mkdir(parents=True, exist_ok=True)
    util.write_options(config, log_path)

    # builds a dataset by stepping a gym env with random actions
    dataset = gym_dataset.load_or_generate(config)
    train_loader = torch.utils.data.DataLoader(dataset,
                                               batch_size=config.batch_size,
                                               shuffle=True,
                                               num_workers=0)

    # load encoder
    encoder = torch.load(encoder_path)

    # calculate the sizes of everything
    epoch_size = len(dataset)
    action_space = dataset.env.action_space
    action_size = util.prod(action_space.shape)
    if config.embed_size is None:
        embed_size = action_size
    else:
        embed_size = config.embed_size

    # define decoder
    decoder = ActionDecoder(
        config.n_layers,
        embed_size,
        config.traj_len,
        action_space).to(device)
    decoder_optimizer = optim.Adam(decoder.parameters(), lr=config.lr)

    # calculate necessary statistics
    z_stats = marginal_stats(train_loader, encoder, device)

    if config.wandb:
        wandb.init(project=config.wandb_proj, entity='vinnibuh')
        wandb.config.update(util.clean_config(config))

    for epoch in range(config.epochs):
        decoder_loss = 0
        decoder_recon_loss = 0
        decoder_norm_loss = 0

        temp_batch = 0
        temp_loss = 0
        temp_recon_loss = 0
        temp_norm_loss = 0

        for batch_idx, (states, actions) in enumerate(train_loader):
            z = turn_to_z(actions, encoder, device)
            z = (z - z_stats[0].detach()) / z_stats[1].detach()
            decoded_action = decoder(z)
            z_hat = encoder.encode(decoded_action)[0]
            z_hat_white = (z_hat - z_stats[0].detach()) / z_stats[1].detach()

            recon_loss = F.mse_loss(z_hat_white, z)
            norm_loss = decoded_action.norm(dim=2).sum()
            loss = recon_loss + config.norm_loss * norm_loss

            decoder_optimizer.zero_grad()
            loss.backward()
            decoder_optimizer.step()

            decoder_loss += loss.item()
            decoder_recon_loss += recon_loss.item()
            decoder_norm_loss += norm_loss.item()

            temp_loss += loss.item()
            temp_recon_loss += recon_loss.item()
            temp_norm_loss += norm_loss.item()

            if batch_idx > 0 and (batch_idx * config.batch_size) % config.log_interval < config.batch_size:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * config.batch_size, epoch_size,
                    100. * batch_idx / len(train_loader),
                    loss.item() / config.batch_size))
                temp_size = (batch_idx - temp_batch) * config.batch_size
                temp_batch = batch_idx
                wandb.log({'epoch_progress': 100. * batch_idx / len(train_loader)})
                wandb.log({'mean batch loss': temp_loss / temp_size})
                wandb.log({'mean batch recon loss': temp_recon_loss / temp_size})
                wandb.log({'mean batch norm loss': temp_norm_loss / temp_size})
                temp_loss = 0
                temp_recon_loss = 0
                temp_norm_loss = 0

        print((
            'ActionDecoder epoch: {}\tAverage loss: {:.4f}'
            '\tRecon loss: {:.6f}\tNorm loss: {:.6f}'
        ).format(
            epoch, decoder_loss / epoch_size,
            decoder_recon_loss / epoch_size,
            decoder_norm_loss / epoch_size))
        wandb.log({'epoch': epoch})
        wandb.log({'mean epoch loss': decoder_loss / epoch_size})
        wandb.log({'mean epoch recon loss': decoder_recon_loss / epoch_size})
        wandb.log({'mean epoch norm loss': decoder_norm_loss / epoch_size})

    # z_stats[2] is the max
    decoder.max_embedding = z_stats[2]
    torch.save(decoder, decoder_path / 'decoder.pt')


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
