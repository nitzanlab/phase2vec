import os
import torch
import numpy as np
import matplotlib.pyplot as plt

def get_optimizer(optimizer_name, model_params, lr):
    """
    Select optimizer
    optimizer_name - name of optimizer (Adam/SGD)
    model_params - model weights/parameters
    lr - learning rate
    """
    if optimizer_name == 'Adam':
        optimizer = torch.optim.Adam(model_params, lr=lr)
    elif optimizer_name == 'SGD':
        optimizer = torch.optim.SGD(model_params, lr=lr)
    else:
        raise ValueError('Optimizer "{}" not recognized.'.format(optimizer_name))
    return optimizer


#def load_data(data_dir, tt, **kwargs):
#    """
#    Get dataloader
#    """
#    tt_dataset = SystemFamilyDataset(os.path.join(data_dir, tt, '0'))
#    tt_loader = torch.utils.data.DataLoader(dataset=tt_dataset, **kwargs)
#    return tt_loader


def to_batches(loader, val_samples, dim, dx):
    """

    """
    Y = []
    for batch_idx, (y, _) in enumerate(loader):
        Y.append(y[0])
        if batch_idx * loader.batch_size > val_samples: break
    Y = torch.stack(Y).reshape(-1, dim, dx, dx)[:val_samples]
    return Y

def whiten_data(data, means, stds): #TODO: use defaults
    """

    """
    data = ((data - means.reshape(1, data.shape[1], *[1] * (data.ndim - 2))) / (
            stds.reshape(1, data.shape[1], *[1] * (data.ndim - 2)))).float()
    # data = ((data - means.reshape(1,data.shape[1],1,1))/(stds.reshape(1,data.shape[1],1,1))).float()
    return data


def clamp_params(params, ranges):
    """
    Clamp parameters within acceptable ranges #TODO: (maybe remove)
    """
    for d, rg in enumerate(ranges):
        params[:, d] = torch.sigmoid(params[:, d]) * (rg[1] - rg[0]) + rg[0]
    return params

def get_flows(sf, ndim, DE_params, pde):
    """
    Flow generator
    """
    permuted_dims = [ndim] + list(range(ndim))
    if not pde:
        # Get ODE vector field on mesh
        # parDEs = [torch.tensor(sf.generate_flow(params=dep, train=False)) for dep in DE_params]
        # flows = torch.stack([parDE.permute(*permuted_dims) for parDE in parDEs])
        kwargs = sf.data_info
        kwargs['train'] = False
        parDEs = [sf.DE(params=dep, **kwargs) for dep in DE_params]
        flows = torch.stack([parDE.forward(0, torch.tensor(sf.L)).permute(*permuted_dims) for parDE in parDEs])
    else:
        # Get PDE steady state
        # TODO: handle!
        parDEs = [sf.generate_model(sf.num_lattice, params=DE_params, train=False) for dep in DE_params]
        flows = torch.stack(
            [parDE.run(2000, alpha=.1, noise_magnitude=0.2).permute(*permuted_dims) for parDE in parDEs])
        flows = flows[-1, ...].reshape(-1, sf.num_lattice, sf.num_lattice, 2)
    return flows, parDEs

def plot_flows(recon_data, data, recon_pars, pars, L, pad, pde): #TODO: move from here
    c = data.shape[1]
    if c < 4:
        if c == 2:
            fig, axes = plt.subplots(2, 4, figsize=(15, 7.5))
        else:
            fig, axes = plt.subplots(nrows=2, ncols=3, subplot_kw={'projection': '3d'}, figsize=(15, 7.5))
        spatial = data.shape[2:]
        slices = [slice(0, c)] + [slice(pad, sd - pad) for sd in spatial]
        coords = [np.squeeze(L[..., i]).detach().cpu().numpy() for i in range(L.shape[-1])]
        for a, axs in enumerate(axes.T):
            img = data.detach().cpu().numpy()[a].reshape(*data.shape[1:])[slices]
            recon_img = recon_data.detach().cpu().numpy()[a].reshape(*data.shape[1:])[slices]
            par = pars[a]
            recon_par = recon_pars[a]
            if not pde:
                axs[0].quiver(*coords, *img)
                axs[1].quiver(*coords, *recon_img)
            else:
                axs[0].imshow(img[0, ...])
                axs[1].imshow(recon_img[0, ...])
            axs[0].set_title(r'$\alpha=$' + '(' + ', '.join(
                [str(np.around(p.cpu().numpy(), decimals=2)) for p in par]) + ')', fontsize=8)
            axs[1].set_title(r'$\bar{\alpha}=$' + '(' + ', '.join(
                [str(np.around(p.detach().cpu().numpy(), decimals=2)) for p in recon_par]) + ')', fontsize=8)
        return fig

def write_loss(writer, loss_vals, idx):
    """

    """
    for loss_name, loss_val in loss_vals.items():
        writer.add_scalar(loss_name, loss_val, idx)

def get_recon_loss(recon_data, data, model, model_out, recon_loss, recon_loss_function, return_activations, weights, pad):
    """

    """
    if recon_loss == 'style':
        model.activations = []
        recon_data = recon_data.detach().data
        with torch.no_grad():
            model_out_target, _, _ = model(recon_data, return_activations=return_activations)
        RL = recon_loss_function(model_out, model_out_target, weights=weights)
    else:
        RL = recon_loss_function(recon_data, data, pad=pad)
    return RL


def grad_status(to_train):
    """
    Sets whether to propagate gradient or not
    """
    if to_train:
        return torch.enable_grad()
    else:
        return torch.no_grad()


def laplacian(A):
    '''Calculate laplacian of array'''
    A = F.pad(A,(0,0,1,1,1,1))
    return A[:-2,1:-1] + A[1:-1,:-2] - 4*A[1:-1,1:-1] + A[1:-1,2:] + A[2:,1:-1]
