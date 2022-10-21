# prerequisites
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torchdiffeq import odeint, odeint_adjoint
from ._layers import convNd, MLP, CNN, dCNN
from src.data import sindy_library
import pdb

from pl_bolts.models.autoencoders.components import resnet18_decoder, resnet18_encoder


def load_model(model_type, pretrained_path=None, device='cpu', **kwargs):
    """load_model: loads a neural network which maps from flows to either flows or parameters

       Positional arguments:

           model_type (str): name of model (see below)
           data_dim (int): number of dimensionin the phase space (i.e. number of array channels in data)
           num_lattice (int): number of points in the side of the cubic mesh on which velocities are measured in the data
           latent_dim (int): dimension of latent space

        Keyword arguments:
            num_DE_params (int): number of parameters in underlying system. Relevant for "Par" models (see below)
            last_pad (int): amount to pad output"""

    if model_type == 'CNNwFC_exp_emb':
        model = CNNwFC_exp_emb(**kwargs)
    elif model_type == 'Conv2dAE':
        model = Conv2dAE(**kwargs)
    else:
        raise ValueError("Haven't added other models yet!")

    if pretrained_path is not None:
        model.load_state_dict(torch.load(pretrained_path, map_location=torch.device(device)))
    return model.to(device)

class CNNwFC(nn.Module):
    def __init__(self, in_shape, poly_order=3, latent_dim=64,
            num_conv_layers=1, kernel_sizes=[5], kernel_features=[16],
            pooling_sizes=[],
            strides = [1],
            num_fc_hid_layers=0, fc_hid_dims=[],
            min_dims=[-1.,-1.],
            max_dims=[1.,1.],
            finetune=False, batch_norm=False, dropout=False, dropout_rate=.5, activation_type='relu',last_pad=None):

        super(CNNwFC, self).__init__()
        self.finetune = finetune
        self.enc = CNN(in_shape, num_conv_layers=num_conv_layers,
                        kernel_sizes=kernel_sizes, kernel_features=kernel_features,
                        pooling_sizes = pooling_sizes,
                        strides = strides,
                        batch_norm=batch_norm, dropout=dropout, dropout_rate=dropout_rate,
                        activation_type=activation_type)

        with torch.no_grad():
            x = torch.rand(1, *in_shape)
            y = self.enc(x)
            self.out_shape = y.shape[1:]
            self.out_size = torch.prod(torch.tensor(y.shape))

        self.poly_order = poly_order

        spatial_coords = [np.linspace(mn, mx, self.num_lattice) for (mn, mx) in zip(min_dims, max_dims)]
        mesh = np.meshgrid(*spatial_coords)
        self.L = torch.tensor(np.concatenate([ms[..., None] for ms in mesh], axis=-1))
        
        library, library_terms = sindy_library(self.L.reshape(self.num_lattice**self.dim,self.dim), poly_order)
        self.library = library.float()

        self.dec = MLP(self.out_size, num_DE_params,
                         num_hid_layers=num_fc_hid_layers, hid_dims=fc_hid_dims,
                         batch_norm=batch_norm, dropout=dropout, dropout_rate=dropout_rate,
                         activation_type=activation_type)

    def forward(self,x):
        if self.finetune:
            with torch.no_grad():
                x = self.enc(x)
        else:
            x = self.enc(x)
        x = x.reshape(-1, self.out_size)
        return self.fc(x)

class CNNwFC_exp_emb(nn.Module):
    def __init__(self, in_shape, poly_order=3, latent_dim=64,
            num_conv_layers=1, kernel_sizes=[5], kernel_features=[16],
            pooling_sizes=[],
            strides = [1],
            num_fc_hid_layers=0, fc_hid_dims=[],
            min_dims=[-1.,-1.],
            max_dims=[1.,1.],
            batch_norm=False, dropout=False, dropout_rate=.5, activation_type='relu', last_pad=None):

        super(CNNwFC_exp_emb, self).__init__()

        self.dim         = in_shape[0]
        self.num_lattice = in_shape[1]
        self.latent_dim = latent_dim
        self.enc = CNN(in_shape, num_conv_layers=num_conv_layers,
                        kernel_sizes=kernel_sizes, kernel_features=kernel_features,
                        pooling_sizes = pooling_sizes,
                        strides = strides,
                        batch_norm=batch_norm, dropout=dropout, dropout_rate=dropout_rate,
                        activation_type=activation_type)

        with torch.no_grad():
            x = torch.rand(1, *in_shape)
            y = self.env(x)
            self.out_shape = y.shape[1:]
            self.out_size = torch.prod(torch.tensor(y.shape))

        self.emb = MLP(self.out_size, latent_dim,
                         num_hid_layers=0, hid_dims=[],
                         batch_norm=batch_norm, dropout=dropout, dropout_rate=dropout_rate,
                         activation_type=activation_type)

        self.poly_order = poly_order

        spatial_coords = [np.linspace(mn, mx, self.num_lattice) for (mn, mx) in zip(min_dims, max_dims)]
        mesh = np.meshgrid(*spatial_coords)
        self.L = torch.tensor(np.concatenate([ms[..., None] for ms in mesh], axis=-1))
        
        library, library_terms = sindy_library(self.L.reshape(self.num_lattice**self.dim,self.dim), poly_order)
        self.library = library.float()

        self.dec = MLP(latent_dim, self.dim * len(library_terms),
                       num_hid_layers=num_fc_hid_layers, hid_dims=fc_hid_dims,
                       batch_norm=batch_norm, dropout=dropout, dropout_rate=dropout_rate,
                       activation_type=activation_type)

    def forward(self,x):
        if self.finetune:
            with torch.no_grad():
                x = self.enc(x)
        else:
            x = self.enc(x)
        x = x.reshape(-1, self.out_size)
        return self.fc(self.emb(x))

class Conv2dAE(nn.Module):
    def __init__(self, in_shape, latent_dim=64,
            num_conv_layers=1, kernel_sizes=[5], kernel_features=[16],
            pooling_sizes=[],
            strides = [1],
            num_fc_hid_layers=0, fc_hid_dims=[],
            batch_norm=False, dropout=False, dropout_rate=.5, activation_type='relu',
            finetune=False,last_pad=False, min_dims=None, max_dims=None, poly_order=None):

        super(Conv2dAE, self).__init__()
        self.finetune = finetune
        self.latent_dim = latent_dim
        self.enc = CNN(in_shape, num_conv_layers=num_conv_layers,
                        kernel_sizes=kernel_sizes, kernel_features=kernel_features,
                        pooling_sizes = pooling_sizes,
                        strides = strides,
                        batch_norm=batch_norm, dropout=dropout, dropout_rate=dropout_rate,
                        activation_type=activation_type)

        with torch.no_grad():
            x = torch.rand(1, *in_shape)
            y = self.enc(x)
            self.out_shape = y.shape[1:]
            self.out_size = torch.prod(torch.tensor(y.shape))

        self.emb   = MLP(self.out_size, latent_dim,
                         num_hid_layers=0, hid_dims=[],
                         batch_norm=batch_norm, dropout=dropout, dropout_rate=dropout_rate,
                         activation_type=activation_type)

        self.deemb = MLP(latent_dim, self.out_size,
                         num_hid_layers=0, hid_dims=[],
                         batch_norm=batch_norm, dropout=dropout, dropout_rate=dropout_rate,
                         activation_type=activation_type)

        deconv_kernel_features = kernel_features[::-1][:-1] + [in_shape[0]]
        self.deconv = dCNN(self.out_shape, num_conv_layers=num_conv_layers,
                           kernel_sizes=kernel_sizes[::-1], kernel_features=deconv_kernel_features,
                           pooling_sizes = pooling_sizes[::-1],
                           strides = strides,
                           batch_norm=batch_norm, dropout=dropout, dropout_rate=dropout_rate,
                           activation_type=activation_type,
                           last_pad=last_pad)

        self.dec = lambda x : self.deconv(self.deemb(x).reshape(-1,*self.out_shape))

    def forward(self,x):
        if self.finetune:
            with torch.no_grad():
                x = self.enc(x)
        else:
            x = self.enc(x)
        return self.dec(x)
