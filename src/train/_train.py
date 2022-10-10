import os
import torch
import random
import datetime
import numpy as np
from typing import List
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter

from src.train._layers import *
from src.train._losses import *
from src.train._models import *
from src.train._utils import *
from src.data import CircuitFamily

import pdb

def train(cf: CircuitFamily, exp_name: str, model_save_dir: str, batch_size: int, pde:bool,
          num_epochs: int, latent_dim: int, optimizer_name: str, model_type: str, device: str,
          means_train: List[str], means_test: List[str], stds_train: List[str], stds_test: List[str], # TODO: is this necessary???
          learning_rate: float, log_dir: str, save_model_every: int, seed: int, max_grad: float,
          last_pad: int, **kwargs):
    """

    """
    #
    # first_pad: int, beta: float, gamma: float, p: float,
    # num_sparsified: int, recon_loss: str, pde: bool, scale_output: bool,
    # whiten: bool,
    # fig_save_dir: str,


    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    weights = torch.tensor(
        [1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1])
    num_DE_params = len(cf.param_ranges)

    print('Training model with data: {}'.format(cf.data_name))

    model_save_path = os.path.join(model_save_dir, exp_name + '.pt')

    # train_dataset = datasets.DatasetFolder(os.path.join(cf.data_dir, 'train'), custom_loader, # TODO: need to be careful here with dictionary lookup
    #                                        extensions=('pkl',))  # , transform=transforms.ToTensor())
    # test_dataset = datasets.DatasetFolder(os.path.join(cf.data_dir, 'test'), custom_loader,
    #                                       extensions=('pkl',))  # , transform=transforms.ToTensor())
    train_loader = load_data(cf.data_dir, tt='train', batch_size=batch_size, shuffle=True, drop_last=True)
    test_loader = load_data(cf.data_dir, tt='test', batch_size=batch_size, shuffle=False, drop_last=True)
    # train_dataset = CircuitFamilyDataset(os.path.join(cf.data_dir, 'train', '0'))  # , transform=transforms.ToTensor()) #TODO:what is the meaning of 0??
    # test_dataset = CircuitFamilyDataset(os.path.join(cf.data_dir, 'test', '0'))  # , transform=transforms.ToTensor())
    #
    # # Data Loader (Input Pipeline)
    # train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True,
    #                                            drop_last=True)
    # test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False,
    #                                           drop_last=True)

    # data, target = next(iter(test_loader))

    # # Initialize model
    # DE_ex = cf.DE_ex
    # L = None
    # # Build vector field mesh
    # if not pde:
    #     L = DE_ex.generate_mesh() # DE.min_dims, DE.max_dims, DE.num_lattice

    vae = load_model(model_type, cf.dim, cf.num_lattice, latent_dim, num_DE_params=num_DE_params, last_pad=last_pad)
    vae.to(device)

    optimizer = get_optimizer(optimizer_name, vae.parameters(), learning_rate)
    writer = SummaryWriter(log_dir=os.path.join(log_dir, datetime.datetime.now().strftime("%Y%m%d-%H%M%S")))

    for e in np.arange(num_epochs):
        epoch = e + 1


        # Train
        vae.train()

        train_observables = run_epoch('train', vae, cf, cf.L, cf.param_ranges, train_loader, epoch,
                                      optimizer=optimizer, max_grad=max_grad,
                                      means=means_train, stds=stds_train,
                                      device=device, writer=writer, pde=pde, weights=weights, **kwargs)

        # print('Epoch %d' % e)
        # Test
        vae.eval()
        test_observables = run_epoch('test', vae, cf, cf.L, cf.param_ranges, test_loader, epoch,
                                     means=means_test, stds=stds_test,
                                     device=device, writer=writer, pde=pde, weights=weights, **kwargs)

        # train_observables = run_epoch('train', vae, DE, L, cf.param_ranges, train_loader, optimizer, epoch,
        #                                 recon_loss=recon_loss, pad=first_pad, device=device, writer=writer, beta=beta,
        #                                 gamma=gamma, p=p, num_sparsified=num_sparsified, pde=pde, max_grad=max_grad,
        #                                 weights=weights, scale_output=scale_output, whiten=whiten, means=means_train,
        #                                 stds=stds_train)
        #
        # # Test
        # test_observables = run_epoch('test', vae, DE, L, cf.param_ranges, test_loader, epoch, recon_loss=recon_loss,
        #                               pad=first_pad, device=device, writer=writer, beta=beta, gamma=gamma, p=p,
        #                               num_sparsified=num_sparsified, pde=pde, weights=weights,
        #                               scale_output=scale_output, whiten=whiten, means=means_test, stds=stds_test)

        # Save and log
        if (e % save_model_every) == 0:
            torch.save(vae.state_dict(), model_save_path)
    torch.save(vae.state_dict(), model_save_path)
    return train_observables, test_observables


def run_epoch(tt, model, cf, L, param_ranges, dataloader, epoch, optimizer=None, recon_loss='BCE', pad=0, device='cpu',
              writer=None, show_every=25, beta=1.0, gamma=1.0, p=1.0, num_sparsified=0, pde=False, whiten=False,
              means=None, stds=None, max_grad=10.0, weights=None, scale_output=False, val_samples=1000, report_dimension=False, **kwargs):
    '''Run one epoch

       Positional arguments:
           tt: test or train
           model (): vae network
           cf (CircuitFamily): true family underlying the data
           L:
           param_ranges:
           dataloader (torch DataLoader object): torch dataloader iterating over training data
           epoch (int): number of current epoch

       Keyword arguments:
            optimizer (torch optim object): an optimizer
           loss_function (func): loss function
           pad (int) : how much to clip off each spatial dimension when comparing reconstruction to original
           device (str): which device to run on, one of 'cpu' or 'cuda'
           writer (Sumarizer or None): if not None, then a TB summarizer
           show_every (int): period at which to print losses in terminal
           beta (float): scalar weight of KL term in loss
           gamma (float): scalar weight of the sparsity term
           p (float): p in sparsity norm
           num_sparsified (int): how many parameters, from "left to right", to be included in the sparsity term
           pde (bool): if True, then data are assumed to be images; else, assumed to be flow fields
           TEMP REMOVED: whiten (bool): whether or not to whiten data
           means (array): channel-wise means for whitening
           stds (array): channel-wise stds for whitening
           max_grad (float): maximum norm for network gradient, used in clipping
           weights (array): layer-wise weights for use in the style loss
           scale_output (bool): if True, then output of network is automatically called to be within `param_ranges` '''

    to_train = (tt == 'train')
    ttt = tt.title()
    # Choose loss function
    recon_loss_function, return_activations = get_reconstruction_loss(recon_loss)

    # Train val batch
    if report_dimension:
        Y = to_batches(dataloader, val_samples, cf.dim, cf.num_lattice).to(device)

    if to_train and optimizer is None:
        raise ValueError('Optimizer missing in train iteration!')

    epoch_observables = {ttt + '/Loss/loss': 0,
                         ttt + '/Loss/RL': 0,
                         ttt + '/Loss/KLD': 0,
                         ttt + '/Loss/SP': 0,
                         ttt + '/ID/local': 0,
                         ttt + '/ID/global': 0
                         }

    display_loss = 0
    means = torch.tensor(means).to(device)
    stds = torch.tensor(stds).to(device)
    # TODO: whiten all data before loading to device?

    with grad_status(to_train):
        for batch_idx, (data, parameter) in enumerate(dataloader):
            # data, parameter = X[0], X[1]
            data = data.to(device)
            parameter = parameter.to(device)

            if whiten:
                data = whiten_data(data, means, stds)
                data = data.to(device)

            if to_train:
                optimizer.zero_grad()

            # Pass data through model
            model_out, mu, log_var = model(data, return_activations=return_activations)
            DE_params = model_out[0]

            # KL loss
            KLD = KL_loss(mu, log_var)
            if scale_output:
                DE_params = clamp_params(DE_params, param_ranges)

            # Instantiate and run DE
            #TODO: is it ok that we are giving the true minmaxdims
            flows = get_flows(cf, ndim=(data.ndim - 2), DE_params=DE_params, pde=pde)[0]

            if whiten: #TODO: why whitenning flows?
                flows = whiten_data(flows, means, stds)

            # Reconstruction loss
            RL = get_recon_loss(flows, data, model, model_out, recon_loss, recon_loss_function, return_activations,
                                weights, pad)

            # Sparsity loss
            par_norm = sparsity_loss(DE_params[:num_sparsified], p)

            # Total loss
            loss = RL + beta * KLD + gamma * par_norm #TODO: add back
            display_loss += loss.detach().cpu().numpy()

            # Backprop and clip
            if to_train:
                loss.backward()
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad)
                optimizer.step()

            # Logging and plotting
            if batch_idx % show_every == 0:

                # Log loss observables
                if report_dimension:
                    with torch.no_grad():
                        danco, lpca = model.id(Y) # TODO: wanted to skip batching the dataset (Y) so tried removing this
                        danco = danco.dimension_.astype(np.float32).item()
                        lpca = lpca.dimension_pw_.mean().astype(np.float32).item()
                        epoch_observables[ttt + '/ID/local'] = lpca #TODO: why are these refered as local and global? both are intrinsic dim estimation
                        epoch_observables[ttt + '/ID/global'] = danco
                        # for (key, obs) in zip(epoch_observables.keys(), [loss, RL, KLD, par_norm, lpca, danco]):
                #     if (key != ttt + '/ID/local') and (key != ttt + '/ID/global'):
                #         obs = obs.detach().cpu().numpy().item()
                #     # epoch_observables[key] += obs / int(len(dataloader) / float(show_every))
                #     epoch_observables[key] += obs / (len(dataloader) / float(show_every))  # TODO: temporary fix

                for (key, obs) in zip(epoch_observables.keys(), [loss, RL, KLD, par_norm]):
                    obs = obs.detach().cpu().numpy().item()
                    # epoch_observables[key] += obs / int(len(dataloader) / float(show_every))
                    epoch_observables[key] += obs / (len(dataloader) / float(show_every))  # TODO: temporary fix

                write_loss(writer, loss_vals=epoch_observables, idx=epoch * len(dataloader) + batch_idx)

                fig = plot_flows(flows, data, DE_params, parameter, L, pad, pde)
                writer.add_figure('flows_%s' % tt, fig)

                display_norml = show_every if batch_idx > 0 else 1
                print('{} Epoch: {} [{}/{} ({:.4f}%)]\tLoss: {:.6f}'.format(tt.title(),
                                                                            epoch,
                                                                            batch_idx * len(data),
                                                                            len(dataloader.dataset),
                                                                            100 * (batch_idx / float(len(dataloader))),
                                                                            display_loss / display_norml))
                display_loss = 0
    print('====> {} Epoch: {} Average loss: {:.4f}'.format(tt.title(), epoch, epoch_observables[ttt + '/Loss/loss']))
    return epoch_observables


def predict(cf: CircuitFamily, tt: str, num_samples: int,
          latent_dim: int, model_type: str, device: str, scale_output: bool,
          last_pad: int, **kwargs):
    """
    Predicts latent represetnation and circuit parameters
    """
    data_dim = cf.dim

    num_DE_params = len(cf.param_ranges)
    #model_save_path = os.path.join(model_save_dir, exp_name + '.pt')
    model = load_model(model_type,
                            data_dim,
                            cf.num_lattice,
                            latent_dim,
                            num_DE_params=num_DE_params,
                            last_pad=last_pad,
                            pretrained_path=kwargs['model_save_path'],
                            device=device)

    # Data-sets and loaders
    dl = load_data(cf.data_dir, tt, batch_size=num_samples, shuffle=False, drop_last=True) #TODO: why was shuffle=true?

    data, params = next(iter(dl))
    #TODO: replaced with what was here previously but should it be on the complete data always? do predict to get params
    data = data.to(cf.device)

    latent, _ = model.encode(data)
    latent = latent.detach().cpu().numpy()
    recon_params = model.forward(data)[0]
    recon_params = recon_params[0].detach().cpu().numpy()

    params = params.detach().cpu().numpy()

    if scale_output:
        recon_params = clamp_params(recon_params, cf.param_ranges)

    return latent, recon_params, params

