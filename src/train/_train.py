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

def train_model(X_train, X_test,
                y_train, y_test,
                p_train, p_test,
                net, exp_name,
                num_epochs = 5, 
                learning_rate=1e-4, momentum=0.0,
                optimizer='SGD',
                batch_size=10,
                beta=1e-3,
                fp_normalize=False,
                device='cuda',
                log_dir='/home/mgricci/runs',
                log_period = 10):

    '''Train net.'''
    if optimizer == 'Adam':
        opt = torch.optim.Adam(net.parameters(), lr=learning_rate)
    elif optimizer == 'SGD':
        opt = torch.optim.SGD(net.parameters(), lr=lr,momentum=momentum)
    else:
        raise ValueError('Unknown optimizer!')
        
    writer = SummaryWriter(log_dir=log_dir)

    loss_names = ['T', 'R', 'S', 'P']

    for e in range(num_epochs):
        print(f'Running Epoch {e}')
        
        for i, (data, labels, pars) in enumerate(zip([X_train, X_test], [y_train, y_test], [p_train, p_test])):
            train  = (i == 0) 
            mode = 'Training' if train else 'Validating'
            losses = run_epoch(data, labels, pars,
                               net, e, opt,
                               train=train,
                               batch_size=batch_size,
                               beta=beta,
                               fp_normalize=fp_normalize,
                               device=device)
            
            avg_losses = [np.mean(loss) for loss in losses]

            print(f'{mode}: Total loss: %.2f, Recon loss: %.2f,  Sparsity loss: %.2f, Parameter loss: %.2f' % tuple(avg_losses))
            writer.add_scalars(f"/Loss/Total",{mode + exp_name:avg_losses[0]} , e)
            writer.add_scalars(f"/Loss/Recon",{mode + exp_name:avg_losses[1]}, e)
            writer.add_scalars(f"/Loss/Sparsity",{mode + exp_name:avg_losses[2]}, e)
            writer.add_scalars(f"/Loss/Parameter",{mode + exp_name:avg_losses[3]}, e)

    return net

def run_epoch(data, labels, gt_pars, net, epoch, opt,
              train=False,
              batch_size=10,
              beta=1e-3,
              fp_normalize=False,
              device='cuda', 
              return_embeddings=False):

    '''One pass through either the training, validating or testing data.'''
    tloss_history  = []
    rloss_history  = []
    sloss_history  = []
    ploss_history  = []
    embeddings     = []

    library = net.library

    if train:
        net.train()
    else:
        net.eval()
        
    euclidean = normalized_euclidean if fp_normalize else euclidean_loss
        
    num_iter = int(np.ceil(data.shape[0] / float(batch_size)))
    num_lattice = data.shape[2]
    for i in range(num_iter):

        effective_batch_size = min(batch_size, len(data) - i*batch_size)
        batch        = torch.tensor(data[i * batch_size:i*batch_size + effective_batch_size]).to(device).float()
        batch_labels = torch.tensor(labels[i * batch_size:i*batch_size + effective_batch_size])
        batch_pars   = torch.tensor(gt_pars[i * batch_size:i*batch_size + effective_batch_size]).to(device).float()
        dim = batch.shape[1]
        
        if train:
            opt.zero_grad()
        
        # Forward pass
        z    = net.emb(net.conv(batch).reshape(effective_batch_size, -1))
        pars  = net.fc(z)
        pars = pars.reshape(-1,library.shape[-1], dim)

        # Save embeddings? 
        if return_embeddings:
            embeddings.append(z)

        # Reconstruction using fn dictionary
        recon = torch.einsum('sl,bld->bsd',library.to(device),pars).reshape(effective_batch_size, num_lattice,num_lattice,dim).permute(0,3,1,2)
        
        recon_loss      = euclidean(recon, batch)

        # Sparsity
        sparsity_loss   = pars.reshape(effective_batch_size, -1).abs().mean(1)
        gt_sparsity     = batch_pars.reshape(effective_batch_size, -1).abs().mean(1)
        vis_sparsity    = (sparsity_loss / gt_sparsity).mean()
        total_loss      = recon_loss + beta * sparsity_loss.mean()

        par_loss = euclidean_loss(pars,batch_pars)
        if train:
            total_loss.backward()
            opt.step()
        
        tloss_history.append(total_loss.detach().cpu().numpy())
        rloss_history.append(recon_loss.detach().cpu().numpy())
        sloss_history.append(vis_sparsity.detach().cpu().numpy())
        ploss_history.append(par_loss.detach().cpu().numpy())

    if return_embeddings:
        embeddings = torch.cat(tuple([emb for emb in embeddings])).reshape(-1, net.latent_dim)
        return [tloss_history, rloss_history, sloss_history, ploss_history], embeddings
    else:
        return [tloss_history, rloss_history, sloss_history, ploss_history]
