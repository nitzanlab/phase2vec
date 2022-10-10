# prerequisites
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torchdiffeq import odeint, odeint_adjoint
from ._layers import convNd
from skdim import id as ID
import pdb

from pl_bolts.models.autoencoders.components import resnet18_decoder, resnet18_encoder


def load_model(model_type, data_dim=None, num_lattice=None, latent_dim=None, pretrained_path=None, device='cpu' , conditional=None, **kwargs):
    """load_model: loads a neural network which maps from flows to either flows or parameters

       Positional arguments:

           model_type (str): name of model (see below)
           data_dim (int): number of dimensionin the phase space (i.e. number of array channels in data)
           num_lattice (int): number of points in the side of the cubic mesh on which velocities are measured in the data
           latent_dim (int): dimension of latent space

        Keyword arguments:
            num_DE_params (int): number of parameters in underlying system. Relevant for "Par" models (see below)
            last_pad (int): amount to pad output"""

    if model_type == 'VAE':
        model = VAE([data_dim] + data_dim * [num_lattice], 512, 256, latent_dim)
    elif model_type == 'ResVAE':
        model = ResVAE([data_dim] + data_dim * [num_lattice], 512, latent_dim, last_pad=last_pad)
    elif model_type == 'Conv2dVAE':
        model = Conv2dVAE([data_dim] + data_dim * [num_lattice], 512, 256, latent_dim, conditional=conditional)
    elif model_type == 'Conv2dAE':
        model = Conv2dAE(**kwargs)
    elif model_type == 'Conv3dVAE':
        model = Conv3dVAE([data_dim] + data_dim * [num_lattice], 64, 32, latent_dim)
    elif model_type == 'VGGVAE':
        model = VGGVAE([data_dim] + data_dim * [num_lattice], latent_dim)
    elif model_type == 'ResParAE':
        model = ResParAE(kwargs['in_shape'], kwargs['z_dim'], kwargs['num_DE_params'],**kwargs)
    elif model_type == 'Conv2dParAE':
        model = Conv2dParAE(**kwargs)
    elif model_type == 'Conv3dParAE':
        model = Conv3dParAE([data_dim] + data_dim * [num_lattice], 64, 32, latent_dim, num_DE_params)
    elif model_type == 'ConvNdParAE':
        model = ConvNdParAE([data_dim] + data_dim * [num_lattice], 64, 32, latent_dim, num_DE_params)
    elif model_type == 'CNNwFC':
        model = CNNwFC(**kwargs)
    elif model_type == 'CNNwFC_exp_emb':
        model = CNNwFC_exp_emb(**kwargs)
    else:
        raise ValueError('Model type not recognized!')

    if pretrained_path is not None:
        model.load_state_dict(torch.load(pretrained_path, map_location=torch.device(device)))
    return model.to(device)


class VAE(nn.Module):
    """Fully-connected two-layer VAE"""

    def __init__(self, in_shape, h_dim1, h_dim2, z_dim, return_activations=False, conditional=False):
        super(VAE, self).__init__()
        """VAE: fully-connected VAE

            Positional Arguments:
            in_shape (iterable): An iterable indicating array size (c x h x w)
            h_dim1 (int): features in first hidden dimension
            h_dim2 (int): features in second hidden dimension


            Keyword Arguments:
                return_activations (bool): if True, returns activation of encoder"""
        self.return_activations = return_activations
        self.activations = []
        self.conditional = conditional

        # encoder part
        self.in_shape = in_shape
        self.fc1 = nn.Linear(in_shape[0] * in_shape[1] * in_shape[2] + 1*(conditional), h_dim1)
        self.fc2 = nn.Linear(h_dim1, h_dim2)
        self.fc31 = nn.Linear(h_dim2, z_dim)
        self.fc32 = nn.Linear(h_dim2, z_dim)
        # decoder part
        self.fc4 = nn.Linear(z_dim + 1*(conditional), h_dim2)
        self.fc5 = nn.Linear(h_dim2, h_dim1)
        self.fc6 = nn.Linear(h_dim1, in_shape[0] * in_shape[1] * in_shape[2])

    def encode(self, x):
        if len(x.shape) > 2:
            x = x.reshape(x.shape[0], -1)
        h1 = F.relu(self.fc1(x))
        h2 = F.relu(self.fc2(h1))
        h31 = self.fc31(h2)
        h32 = self.fc32(h2)

        if self.return_activations:
            self.activations += [h1, h2]
        return h31, h32

    def sampling(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)  # return z sample

    def decode(self, z):
        h = F.relu(self.fc4(z))
        h = F.relu(self.fc5(h))
        return self.fc6(h)

    def id(self, x, n_neighbors=100):
        latent, _ = self.encode(x)
        latent = latent.detach().cpu().numpy()
        return ID.DANCo().fit(latent), ID.lPCA().fit_pw(latent, n_neighbors=n_neighbors, n_jobs=1)

    def forward(self, x, y=None, return_activations=False):
        in_shape = x.shape
        if len(x.shape) > 2:
            x = x.reshape(x.shape[0], -1)
        self.return_activations = return_activations
        if self.conditional:
            x = torch.cat((x,y),dim=-1)
        mu, log_var = self.encode(x)
        z = self.sampling(mu, log_var)
        if self.conditional:
            z = torch.cat((z,y),dim=-1)
        dec = self.decode(z).reshape(in_shape)
        return self.activations + [dec], mu, log_var

class Conv2dVAE(nn.Module):
    def __init__(self, in_shape, h_dim1, h_dim2, z_dim, conditional=False, return_activations=False):
        super(Conv2dVAE, self).__init__()
        """Conv2dVAE: convolutional VAE

            Positional Arguments:
            in_shape (iterable): An iterable indicating array size (c x h x w)
            h_dim1 (int): features in first hidden dimension
            h_dim2 (int): features in second hidden dimension
            z_dim (int):  features in latent space

            Keyword Arguments:
                return_activations (bool): if True, returns activation of encoder"""

        self.return_activations = return_activations
        self.activations = []
        self.conditional = conditional

        self.in_shape = in_shape if not conditional else [in_shape[0] + 1] + in_shape[1:]
        self.out_shape = [h_dim2] + [((ins - 5 + 1) - 5 + 1) for ins in in_shape[1:]]
        self.out_size = np.prod(self.out_shape)

        # Initializing the 2 convolutional layers and 2 full-connected layers for the encoder
        self.encConv1 = nn.Conv2d(self.in_shape[0], h_dim1, 5)
        self.encConv2 = nn.Conv2d(h_dim1, h_dim2, 5)
        self.encFC1 = nn.Linear(self.out_size, z_dim)
        self.encFC2 = nn.Linear(self.out_size, z_dim)

        # Initializing the fully-connected layer and 2 convolutional layers for decoder
        self.decFC1 = nn.Linear(z_dim + 1*(conditional), self.out_size)
        self.decConv1 = nn.ConvTranspose2d(h_dim2, h_dim1, 5)
        self.decConv2 = nn.ConvTranspose2d(h_dim1, self.in_shape[0] - 1*conditional, 5)

    def encode(self, x):
        # Input is fed into 2 convolutional layers sequentially
        # The output feature map are fed into 2 fully-connected layers to predict mean (mu) and variance (logVar)
        # Mu and logVar are used for generating middle representation z and KL divergence loss
        h1 = F.relu(self.encConv1(x))
        h2 = F.relu(self.encConv2(h1))
        if self.return_activations:
            self.activations += [h1, h2]
        h2 = h2.reshape(-1, self.out_size)
        mu = self.encFC1(h2)
        logVar = self.encFC2(h2)

        return mu, logVar

    def reparameterize(self, mu, logVar):
        # Reparameterization takes in the input mu and logVar and sample the mu + std * eps
        std = torch.exp(logVar / 2)
        eps = torch.randn_like(std)
        return mu + std * eps

    def decoder(self, z):
        # z is fed back into a fully-connected layers and then into two transpose convolutional layers
        # The generated output is the same size of the original input
        x = F.relu(self.decFC1(z))
        x = x.reshape(-1, *self.out_shape)
        x = F.relu(self.decConv1(x))
        x = self.decConv2(x)
        return x

    def id(self, x, n_neighbors=100):
        latent, _ = self.encode(x)
        latent = latent.detach().cpu().numpy()
        return ID.DANCo().fit(latent), ID.lPCA().fit_pw(latent, n_neighbors=n_neighbors, n_jobs=1)

    def forward(self, x, y=None, return_activations=False):
        self.return_activations = return_activations
        # The entire pipeline of the VAE: encoder -> reparameterization -> decoder
        # output, mu, and logVar are returned for loss computation
        if self.conditional:
            label_channel = torch.ones(x.shape[0],1,x.shape[2],x.shape[3]).to(x.device)
            label_channel = label_channel * y.reshape(-1, 1, 1, 1)
            x = torch.cat((x,label_channel),dim=1)
        mu, logVar = self.encode(x)
        z = self.reparameterize(mu, logVar)
        if self.conditional:
            z = torch.cat((z,y.unsqueeze(1)),dim=1)
        return self.activations + [self.decoder(z)], mu, logVar

class Conv2dAE(nn.Module):
    def __init__(self, in_shape, num_DE_params, z_dim=64,
            num_conv_layers=1, kernel_sizes=[5], kernel_features=[16],
            pooling_sizes=[],
            strides = [1],
            num_fc_hid_layers=0, fc_hid_dims=[],
            batch_norm=False, dropout=False, dropout_rate=.5, activation_type='relu',
            conditional=False, return_activations=False, finetune=False, last_pad=False):

        super(Conv2dAE, self).__init__()
        self.finetune = finetune
        self.z_dim = z_dim
        self.conv = CNN(in_shape, num_conv_layers=num_conv_layers,
                        kernel_sizes=kernel_sizes, kernel_features=kernel_features,
                        pooling_sizes = pooling_sizes,
                        strides = strides,
                        batch_norm=batch_norm, dropout=dropout, dropout_rate=dropout_rate,
                        activation_type=activation_type)

        with torch.no_grad():
            x = torch.rand(1, *in_shape)
            y = self.conv(x)
            self.out_shape = y.shape[1:]
            self.out_size = torch.prod(torch.tensor(y.shape))

        self.emb   = MLP(self.out_size, z_dim,
                         num_hid_layers=0, hid_dims=[],
                         batch_norm=batch_norm, dropout=dropout, dropout_rate=dropout_rate,
                         activation_type=activation_type)

        self.deemb = MLP(z_dim, self.out_size,
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

    def forward(self,x):
        x = self.conv(x).reshape(-1,self.out_size)
        x = self.deemb(self.emb(x)).reshape(-1,*self.out_shape)
        return self.deconv(x)

class MLP(nn.Module):
    def __init__(self, in_dim, out_dim, num_hid_layers=1, hid_dims=[64], batch_norm=False, dropout=False, dropout_rate=.5, activation_type='relu'):
        super(MLP, self).__init__()

        if activation_type == 'relu':
            activation = torch.nn.ReLU()
        elif activation_type == 'softplus':
            activation = torch.nn.Softplus()
        elif activation_type == 'tanh':
            activation = torch.nn.Tanh()
        elif activation_type == None:
            activation = None
        else:
            raise ValueError('Activation type not recognized!')

        layers = []

        for l in range(num_hid_layers + 1):
            if l == 0:
                if num_hid_layers == 0:
                    layers.append(torch.nn.Linear(in_dim, out_dim))
                else:
                    layers.append(torch.nn.Linear(in_dim, hid_dims[l]))
                    if batch_norm:
                        layers.append(torch.nn.BatchNorm1d(hid_dims[l]))
                    if activation !=  None:
                        layers.append(activation)
                    if dropout:
                        layers.append(torch.nn.Dropout(p=dropout_rate))
            if l > 0:
                if l == num_hid_layers:
                    layers.append(torch.nn.Linear(hid_dims[l-1], out_dim))
                else:
                    layers.append(torch.nn.Linear(hid_dims[l-1], hid_dims[l]))
                    if batch_norm:
                        layers.append(torch.nn.BatchNorm1d(hid_dims[l]))
                    if activation !=  None:
                        layers.append(activation)
                    if dropout:
                        layers.append(torch.nn.Dropout(p=dropout_rate))
        self.layers = torch.nn.Sequential(*layers)
    def forward(self,x):
        return self.layers(x)

class CNNwFC(nn.Module):
    def __init__(self, in_shape, num_DE_params,
            num_conv_layers=1, kernel_sizes=[5], kernel_features=[16],
            pooling_sizes=[],
            strides = [1],
            num_fc_hid_layers=0, fc_hid_dims=[],
            batch_norm=False, dropout=False, dropout_rate=.5, activation_type='relu',
            conditional=False, return_activations=False, finetune=False):

        super(CNNwFC, self).__init__()
        self.finetune = finetune
        self.conv = CNN(in_shape, num_conv_layers=num_conv_layers,
                        kernel_sizes=kernel_sizes, kernel_features=kernel_features,
                        pooling_sizes = pooling_sizes,
                        strides = strides,
                        batch_norm=batch_norm, dropout=dropout, dropout_rate=dropout_rate,
                        activation_type=activation_type)

        with torch.no_grad():
            x = torch.rand(1, *in_shape)
            y = self.conv(x)
            self.out_shape = y.shape[1:]
            self.out_size = torch.prod(torch.tensor(y.shape))

        self.fc = MLP(self.out_size, num_DE_params,
                         num_hid_layers=num_fc_hid_layers, hid_dims=fc_hid_dims,
                         batch_norm=batch_norm, dropout=dropout, dropout_rate=dropout_rate,
                         activation_type=activation_type)

    def forward(self,x):
        if self.finetune:
            with torch.no_grad():
                x = self.conv(x)
        else:
            x = self.conv(x)
        x = x.reshape(-1, self.out_size)
        return self.fc(x)

class CNNwFC_exp_emb(nn.Module):
    def __init__(self, in_shape, num_DE_params, z_dim=64,
            num_conv_layers=1, kernel_sizes=[5], kernel_features=[16],
            pooling_sizes=[],
            strides = [1],
            num_fc_hid_layers=0, fc_hid_dims=[],
            batch_norm=False, dropout=False, dropout_rate=.5, activation_type='relu',
            conditional=False, return_activations=False, finetune=False):

        super(CNNwFC_exp_emb, self).__init__()
        self.finetune = finetune
        self.z_dim = z_dim
        self.conv = CNN(in_shape, num_conv_layers=num_conv_layers,
                        kernel_sizes=kernel_sizes, kernel_features=kernel_features,
                        pooling_sizes = pooling_sizes,
                        strides = strides,
                        batch_norm=batch_norm, dropout=dropout, dropout_rate=dropout_rate,
                        activation_type=activation_type)

        with torch.no_grad():
            x = torch.rand(1, *in_shape)
            y = self.conv(x)
            self.out_shape = y.shape[1:]
            self.out_size = torch.prod(torch.tensor(y.shape))

        self.emb = MLP(self.out_size, z_dim,
                         num_hid_layers=0, hid_dims=[],
                         batch_norm=batch_norm, dropout=dropout, dropout_rate=dropout_rate,
                         activation_type=activation_type)

        self.fc = MLP(z_dim, num_DE_params,
                         num_hid_layers=num_fc_hid_layers, hid_dims=fc_hid_dims,
                         batch_norm=batch_norm, dropout=dropout, dropout_rate=dropout_rate,
                         activation_type=activation_type)

    def forward(self,x):
        if self.finetune:
            with torch.no_grad():
                x = self.conv(x)
        else:
            x = self.conv(x)
        x = x.reshape(-1, self.out_size)
        return self.fc(self.emb(x))

class CNN(nn.Module):
    def __init__(self, in_shape, num_conv_layers=1, 
            kernel_sizes=[], kernel_features=[],
            pooling_sizes = [],
            strides = [1],
            batch_norm=False, dropout=False, dropout_rate=.5,
            activation_type='relu'):

        super(CNN, self).__init__()

        if activation_type == 'relu':
            activation = torch.nn.ReLU()
        elif activation_type == 'softplus':
            activation = torch.nn.Softplus()
        elif activation_type == 'tanh':
            activation = torch.nn.Tanh()
        elif activation_type == None:
            activation = None
        else:
            raise ValueError('Activation type not recognized!')

        conv_layers = []

        for l in range(num_conv_layers):
            in_channels = in_shape[0] if l==0 else kernel_features[l-1]
            conv_layers.append(torch.nn.Conv2d(in_channels, kernel_features[l], kernel_sizes[l], stride=strides[l]))
            if batch_norm:
                conv_layers.append(torch.nn.BatchNorm2d(kernel_features[l]))
            if activation != None:
               conv_layers.append(activation)
            if len(pooling_sizes) >= l +1:
                conv_layers.append(torch.nn.MaxPool2d(pooling_sizes[l]))

        self.conv_layers = torch.nn.Sequential(*conv_layers)

        with torch.no_grad():
            x = torch.rand(1, *in_shape)
            y = self.conv_layers(x)
            self.out_shape = y.shape[1:]
            self.out_size = torch.prod(torch.tensor(y.shape))

    def forward(self, x):
        return self.conv_layers(x)

class dCNN(nn.Module):
    def __init__(self, in_shape, num_conv_layers=1, 
            kernel_sizes=[], kernel_features=[],
            pooling_sizes = [],
            strides = [1],
            batch_norm=False, dropout=False, dropout_rate=.5,
            activation_type='relu', last_pad=False):

        super(dCNN, self).__init__()

        self.last_pad = last_pad

        if activation_type == 'relu':
            activation = torch.nn.ReLU()
        elif activation_type == 'softplus':
            activation = torch.nn.Softplus()
        elif activation_type == 'tanh':
            activation = torch.nn.Tanh()
        elif activation_type == None:
            activation = None
        else:
            raise ValueError('Activation type not recognized!')

        deconv_layers = []
        for l in range(num_conv_layers):
            in_channels = in_shape[0] if l==0 else kernel_features[l-1]
            output_padding = 1 if l == (num_conv_layers - 1) and last_pad else 0
            deconv_layers.append(torch.nn.ConvTranspose2d(in_channels, kernel_features[l], kernel_sizes[l], stride=strides[l],output_padding=output_padding))
            if batch_norm:
                deconv_layers.append(torch.nn.BatchNorm2d(kernel_features[l]))
            if activation != None:
               deconv_layers.append(activation)
            if len(pooling_sizes) >= l +1:
                deconv_layers.append(torch.nn.MaxPool2d(pooling_sizes[l]))

        self.num_conv_layers = num_conv_layers
        self.deconv_layers = torch.nn.Sequential(*deconv_layers)

        with torch.no_grad():
            x = torch.rand(1, *in_shape)
            y = self.deconv_layers(x)
            self.out_shape = y.shape[1:]
            self.out_size = torch.prod(torch.tensor(y.shape))

    def forward(self, x):
        return self.deconv_layers(x)

class Conv2dParAE(nn.Module):
    def __init__(self, in_shape, z_dim, num_DE_params,
            num_conv_layers=1, kernel_sizes=[5], kernel_features=[16],
            num_enc_fc_hid_layers=0, enc_fc_hid_dims=[],
            num_dec_fc_hid_layers=1, dec_fc_hid_dims=[64],
            batch_norm=False, dropout=False, dropout_rate=.5, activation_type='relu',
            conditional=False, return_activations=False):
        super(Conv2dParAE, self).__init__()
        """Conv2dParAE: convolutional encoer and FC decoder with num_DE_params-sized output

            Positional Arguments:
            in_shape (iterable): An iterable indicating array size (c x h x w)
            h_dim1 (int): features in first hidden dimension
            h_dim2 (int): features in second hidden dimension
            z_dim (int):  features in latent space
            num_DE_params (int): output size, number of parameters in the underlying dynamical system

            Keyword Arguments:
                conditional (bool): if True, uses a condtional VAE with label arguments
                return_activations (bool): if True, returns activation of encoder"""

        self.return_activations = return_activations
        self.conditional = conditional
        self.activations = []

        self.in_shape = in_shape if not conditional else [in_shape[0] + 1] + in_shape[1:]

        self.enc = CNN(self.in_shape, num_conv_layers=num_conv_layers,
                       kernel_sizes=kernel_sizes, kernel_features=kernel_features,
                       batch_norm=batch_norm, dropout=dropout, dropout_rate=dropout_rate,
                       activation_type=activation_type)

        self.fc_mu = MLP(self.enc.out_size, z_dim,
                         num_hid_layers=num_enc_fc_hid_layers, hid_dims=enc_fc_hid_dims,
                         batch_norm=batch_norm, dropout=dropout, dropout_rate=dropout_rate,
                         activation_type=activation_type)

        self.fc_var = MLP(self.enc.out_size, z_dim,
                         num_hid_layers=num_enc_fc_hid_layers, hid_dims=enc_fc_hid_dims,
                         batch_norm=batch_norm, dropout=dropout, dropout_rate=dropout_rate,
                         activation_type=activation_type)

        self.dec = MLP(z_dim, num_DE_params, 
                       num_hid_layers=num_dec_fc_hid_layers, hid_dims=dec_fc_hid_dims,
                       batch_norm=batch_norm, dropout=dropout, dropout_rate=dropout_rate,
                       activation_type=activation_type)

    def encode(self, x):
        # Input is fed into 2 convolutional layers sequentially
        # The output feature map are fed into 2 fully-connected layers to predict mean (mu) and variance (logVar)
        # Mu and logVar are used for generating middle representation z and KL divergence loss
        x  = self.enc(x).reshape(-1, self.enc.out_size)
        mu = self.fc_mu(x)
        logVar = self.fc_var(x)

        return mu, logVar

    def reparameterize(self, mu, logVar):
        # Reparameterization takes in the input mu and logVar and sample the mu + std * eps
        std = torch.exp(logVar / 2)
        eps = torch.randn_like(std)
        return mu + std * eps

    def decode(self, z):
        # z is fed back into a fully-connected layers and then into two transpose convolutional layers
        # The generated output is the same size of the original input
        return self.dec(z)

    def id(self, x, n_neighbors=100):
        latent, _ = self.encode(x)
        latent = latent.detach().cpu().numpy()
        return ID.DANCo().fit(latent), ID.lPCA().fit_pw(latent, n_neighbors=n_neighbors, n_jobs=1)

    def forward(self, x, y=None, return_activations=False):
        self.return_activations = return_activations
        # The entire pipeline of the VAE: encoder -> reparameterization -> decoder
        # output, mu, and logVar are returned for loss computation

        if self.conditional:
            label_channel = torch.ones(x.shape[0], 1, x.shape[2], x.shape[3]).to(x.device)
            label_channel = label_channel * y.reshape(-1, 1, 1, 1)
            x = torch.cat((x,label_channel),dim=1)
        mu, logVar = self.encode(x)
        z = self.reparameterize(mu, logVar)
        if self.conditional:
            z = torch.cat((z,y.unsqueeze(1)),dim=1)
        return self.activations + [self.decode(z)], mu, logVar


class Conv3dVAE(nn.Module):
    def __init__(self, in_shape, h_dim1, h_dim2, z_dim, return_activations=False):
        super(Conv3dVAE, self).__init__()
        """Conv3dVAE: 3d convolutional VAE

            Positional Arguments:
                in_shape (iterable): An iterable indicating array size (c x h x w)
                h_dim1 (int): features in first hidden dimension
                h_dim2 (int): features in second hidden dimension
                z_dim (int):  features in latent space

            Keyword Arguments:
                return_activations (bool): if True, returns activation of encoder"""

        self.activations = []
        self.return_activations = return_activations

        self.in_shape = in_shape

        current_shape = in_shape[1:]
        for _ in range(4):
            current_shape = [(cs - 8 + 1) for cs in current_shape]

        self.out_shape = [h_dim2] + current_shape
        self.out_size = np.prod(self.out_shape)

        # Initializing the 2 convolutional layers and 2 full-connected layers for the encoder
        self.encConv1 = nn.Conv3d(in_shape[0], h_dim1, 8)
        self.encConv2 = nn.Conv3d(h_dim1, h_dim2, 8)
        self.encConv3 = nn.Conv3d(h_dim2, h_dim2, 8)
        self.encConv4 = nn.Conv3d(h_dim2, h_dim2, 8)
        self.encFC1 = nn.Linear(self.out_size, z_dim)
        self.encFC2 = nn.Linear(self.out_size, z_dim)

        # Initializing the fully-connected layer and 2 convolutional layers for decoder
        self.decFC1 = nn.Linear(z_dim, self.out_size)
        self.decConv1 = nn.ConvTranspose3d(h_dim2, h_dim2, 8)
        self.decConv2 = nn.ConvTranspose3d(h_dim2, h_dim2, 8)
        self.decConv3 = nn.ConvTranspose3d(h_dim2, h_dim1, 8)
        self.decConv4 = nn.ConvTranspose3d(h_dim1, self.in_shape[0], 8)

    def encode(self, x):

        # Input is fed into 2 convolutional layers sequentially
        # The output feature map are fed into 2 fully-connected layers to predict mean (mu) and variance (logVar)
        # Mu and logVar are used for generating middle representation z and KL divergence loss
        h1 = F.relu(self.encConv1(x))
        h2 = F.relu(self.encConv2(h1))
        h3 = F.relu(self.encConv3(h2))
        h4 = F.relu(self.encConv4(h3))

        if self.return_activations:
            self.activations += [h1, h2, h3, h4]
        h4 = h4.view(-1, self.out_size)
        mu = self.encFC1(h4)
        logVar = self.encFC2(h4)

        return mu, logVar

    def reparameterize(self, mu, logVar):

        # Reparameterization takes in the input mu and logVar and sample the mu + std * eps
        std = torch.exp(logVar / 2)
        eps = torch.randn_like(std)
        return mu + std * eps

    def decode(self, z):

        # z is fed back into a fully-connected layers and then into two transpose convolutional layers
        # The generated output is the same size of the original input
        x = F.relu(self.decFC1(z))
        x = x.view(-1, *self.out_shape)
        x = F.relu(self.decConv1(x))
        x = F.relu(self.decConv2(x))
        x = F.relu(self.decConv3(x))
        x = self.decConv4(x)
        return x

    def id(self, x, n_neighbors=100):
        latent, _ = self.encode(x)
        latent = latent.detach().cpu().numpy()
        return ID.DANCo().fit(latent), ID.lPCA().fit_pw(latent, n_neighbors=n_neighbors, n_jobs=1)

    def forward(self, x, return_activations=False):
        self.return_activations = return_activations
        # The entire pipeline of the VAE: encoder -> reparameterization -> decoder
        # output, mu, and logVar are returned for loss computation
        mu, logVar = self.encode(x)
        z = self.reparameterize(mu, logVar)
        return self.activations + [self.decode(z)], mu, logVar


class VGGVAE(nn.Module):
    """ResNet VAE"""

    def __init__(self, in_shape, z_dim, pretrained=True, train_encoder=False, return_activations=False):
        super(VGGVAE, self).__init__()
        """VGGVAE: VAE with VGG19 encoder

            Positional Arguments:
                in_shape (iterable): An iterable indicating array size (c x h x w)
                z_dim (int): dimension of latent space

            Keyword Arguments:
                pretrained (bool): if True, uses pretrained encoder. Necessary for style loss.
                train_encoder (bool): if True, trains encoder
                return_Activation (bool): if True, returns encoder activation. 
            """
        self.return_activations = return_activations
        self.activations = []
        self.train_encoder = train_encoder
        self.encoder = torch.hub.load('pytorch/vision:v0.10.0', 'vgg19', pretrained=pretrained).features
        encoder0_weights = self.encoder[0].weight.data[:, :in_shape[0], ...]
        self.encoder[0] = torch.nn.Conv2d(in_shape[0], 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.encoder[0].weight.data = encoder0_weights

        # Turn off gradients
        if not train_encoder:
            for layer in self.encoder:
                for pr in layer.parameters():
                    pr.requires_grad = False

        # Calculate out_shape
        current_shape = in_shape[1:]
        for layer in self.encoder:
            try:
                padding = 2 * [layer.padding] if type(layer.padding) == int else layer.padding

                kernel_size = 2 * [layer.kernel_size] if type(layer.kernel_size) == int else layer.kernel_size
                stride = 2 * [layer.stride] if type(layer.stride) == int else layer.stride

                current_shape = [int(np.floor((cs + 2 * p - (k - 1) - 1) / float(s) + 1)) for (cs, p, k, s) in
                                 zip(current_shape, padding, kernel_size, stride)]
            except:
                continue

        self.out_shape = [512] + current_shape
        if self.out_shape[1] < 1 or self.out_shape[2] < 1:
            raise ValueError('Image is too small for VGG19!')
        self.out_size = np.prod(np.array(self.out_shape))

        #  distribution parameters
        self.fc_mu = nn.Linear(self.out_size, z_dim)
        self.fc_var = nn.Linear(self.out_size, z_dim)

        self.fc_dec = nn.Linear(z_dim, self.out_size)

        # Decoder FC
        self.decoder = torch.nn.Sequential(nn.ConvTranspose2d(self.out_shape[0], 128, 3),
                                           nn.Upsample(scale_factor=2),
                                           nn.ReLU(),
                                           nn.ConvTranspose2d(128, 64, 3),
                                           nn.Upsample(scale_factor=2),
                                           nn.ReLU(),
                                           nn.ConvTranspose2d(64, 32, 3),
                                           nn.Upsample(scale_factor=2),
                                           nn.ReLU(),
                                           nn.ConvTranspose2d(32, 16, 3),
                                           nn.Upsample(size=in_shape[1:]),
                                           nn.Conv2d(16, in_shape[0], 3, padding='same'),
                                           nn.Sigmoid())

    def encode(self, x):
        for layer in self.encoder:
            x = layer(x)
            if self.return_activations:
                self.activations.append(x.detach().clone())
        x = x.reshape(x.shape[0], -1)
        mu = self.fc_mu(x)
        var = self.fc_var(x)
        return self.fc_mu(x), self.fc_var(x)  # mu, log_var

    def sampling(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)  # return z sample

    def decode(self, z):
        dec = self.fc_dec(z).reshape(-1, *self.out_shape)
        dec = self.decoder(dec)
        # return F.sigmoid(dec)
        return dec

    def id(self, x, n_neighbors=100):
        latent, _ = self.encode(x)
        latent = latent.detach().cpu().numpy()
        return ID.DANCo().fit(latent), ID.lPCA().fit_pw(latent, n_neighbors=n_neighbors, n_jobs=1)

    def forward(self, x, return_activations=False):
        self.return_activations = return_activations
        mu, log_var = self.encode(x)
        z = self.sampling(mu, log_var)
        return self.activations + [self.decode(z)], mu, log_var


class ResVAE(nn.Module):
    """ResNet VAE"""

    def __init__(self, in_shape, enc_out_dim, z_dim, last_pad=0, return_activations=False):
        super(ResVAE, self).__init__()
        """ResVAE: VAE with ResNet encoder and decoder

            Positional Arguments:
                in_shape (iterable): An iterable indicating array size (c x h x w)
                enc_out_dim (int): output channels of encoder
                z_dim (int): dimension of latent space

            Keyword Arguments:
                last_pad (int): how much to pad each spatial dimension of the output
                return_activation (bool): if True, returns encoder acctivations"""

        # TODO: Add pretraining option
        self.return_activations = return_activations
        self.activations = []

        self.encoder = resnet18_encoder(False, False)
        self.decoder = resnet18_decoder(
            latent_dim=z_dim,
            input_height=in_shape[1],
            first_conv=False,
            maxpool1=False
        )
        self.encoder.conv1 = torch.nn.Conv2d(
            in_shape[0], 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.decoder.conv1 = torch.nn.Conv2d(
            64, in_shape[0], kernel_size=1, stride=1, padding=last_pad, bias=False)

        # Encoder layers
        self.layers = [self.encoder.conv1,
                       self.encoder.bn1,
                       self.encoder.relu,
                       self.encoder.maxpool,
                       self.encoder.layer1,
                       self.encoder.layer2,
                       self.encoder.layer3,
                       self.encoder.layer4]

        # Calculate out_shape
        x = torch.rand(1, *in_shape)
        with torch.no_grad():
            for layer in self.layers:
                x = layer(x)
        self.out_shape = x.shape[1:]
        self.out_size = self.out_shape[0] * self.out_shape[1] * self.out_shape[2]

        if self.out_shape[1] < 1 or self.out_shape[2] < 1:
            raise ValueError('Image is too small for ResNet!')
        # self.out_size = np.prod(np.array(self.out_shape))

        #  distribution parameters
        self.fc_mu = nn.Linear(self.out_size, z_dim)
        self.fc_var = nn.Linear(self.out_size, z_dim)

        # Encoder layers
        self.layers = [self.encoder.conv1,
                       self.encoder.bn1,
                       self.encoder.relu,
                       self.encoder.maxpool,
                       self.encoder.layer1,
                       self.encoder.layer2,
                       self.encoder.layer3,
                       self.encoder.layer4]

    def encode(self, x):
        for layer in self.layers:
            x = layer(x)
            if self.return_activations:
                self.activations.append(x.detach().clone())
        x = x.reshape(x.shape[0], -1)
        return self.fc_mu(x), self.fc_var(x)  # mu, log_var

    def sampling(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)  # return z sample

    def decode(self, z):
        dec = self.decoder(z)
        # return F.sigmoid(dec)
        return dec

    def id(self, x, n_neighbors=100):
        latent, _ = self.encode(x)
        latent = latent.detach().cpu().numpy()
        return ID.DANCo().fit(latent), ID.lPCA().fit_pw(latent, n_neighbors=n_neighbors, n_jobs=1)

    def forward(self, x, return_activations=False):
        self.return_activations = return_activations
        mu, log_var = self.encode(x)
        z = self.sampling(mu, log_var)
        return self.activations + [self.decode(z)], mu, log_var


class Conv3dParAE(nn.Module):
    def __init__(self, in_shape, h_dim1, h_dim2, z_dim, num_DE_params, return_activations=False):
        super(Conv3dParAE, self).__init__()
        """Conv3dParAE: VAE with 3d convolutional encoder and FC decoder mapping to the parameter space

            Positional Arguments:
                in_shape (iterable): An iterable indicating array size (c x h x w)
                h_dim1 (int): features in first hidden dimension
                h_dim2 (int): features in second hidden dimension
                z_dim (int):  features in latent space
                num_DE_params (int): number of parameters in underlying system, and thus size of decoder output

            Keyword Arguments:
                return_activations (bool): if True, returns activation of encoder"""

        self.activations = []
        self.return_activations = return_activations

        self.in_shape = in_shape

        current_shape = in_shape[1:]
        for _ in range(4):
            current_shape = [(cs - 5 + 1) for cs in current_shape]

        self.out_shape = [h_dim2] + current_shape
        self.out_size = np.prod(self.out_shape)

        # Initializing the 2 convolutional layers and 2 full-connected layers for the encoder
        self.encConv1 = nn.Conv3d(in_shape[0], h_dim1, 5)
        self.encConv2 = nn.Conv3d(h_dim1, h_dim2, 5)
        self.encConv3 = nn.Conv3d(h_dim2, h_dim2, 5)
        self.encConv4 = nn.Conv3d(h_dim2, h_dim2, 5)
        self.encFC1 = nn.Linear(self.out_size, z_dim)
        self.encFC2 = nn.Linear(self.out_size, z_dim)

        # Initializing the fully-connected layer and 2 convolutional layers for decoder

        self.decoder = nn.Sequential(nn.Linear(z_dim, 64),
                                     nn.ReLU(),
                                     nn.Linear(64, 64),
                                     nn.ReLU(),
                                     nn.Linear(64, num_DE_params))

    def encode(self, x):

        # Input is fed into 2 convolutional layers sequentially
        # The output feature map are fed into 2 fully-connected layers to predict mean (mu) and variance (logVar)
        # Mu and logVar are used for generating middle representation z and KL divergence loss
        h1 = F.relu(self.encConv1(x))
        h2 = F.relu(self.encConv2(h1))
        h3 = F.relu(self.encConv3(h2))
        h4 = F.relu(self.encConv4(h3))

        if self.return_activations:
            self.activations += [h1, h2, h3, h4]
        h4 = h4.view(-1, self.out_size)
        mu = self.encFC1(h4)
        logVar = self.encFC2(h4)

        return mu, logVar

    def reparameterize(self, mu, logVar):

        # Reparameterization takes in the input mu and logVar and sample the mu + std * eps
        std = torch.exp(logVar / 2)
        eps = torch.randn_like(std)
        return mu + std * eps

    def decode(self, z):

        # z is fed back into a fully-connected layers and then into two transpose convolutional layers
        # The generated output is the same size of the original input
        return self.decoder(z)

    def id(self, x, n_neighbors=100):
        latent, _ = self.encode(x)
        latent = latent.detach().cpu().numpy()
        return ID.DANCo().fit(latent), ID.lPCA().fit_pw(latent, n_neighbors=n_neighbors, n_jobs=1)

    def forward(self, x, return_activations=False):
        self.return_activations = return_activations
        # The entire pipeline of the VAE: encoder -> reparameterization -> decoder
        # output, mu, and logVar are returned for loss computation
        mu, logVar = self.encode(x)
        z = self.reparameterize(mu, logVar)
        return self.activations + [self.decoder(z)], mu, logVar


class ConvNdParAE(nn.Module):
    def __init__(self, in_shape, h_dim1, h_dim2, z_dim, num_DE_params, return_activations=False):
        super(ConvNdParAE, self).__init__()
        """ConvNdParAE: VAE with Nd convolutional encoder and FC decoder mapping to the parameter space

            Positional Arguments:
                in_shape (iterable): An iterable indicating array size (c x h x w)
                h_dim1 (int): features in first hidden dimension
                h_dim2 (int): features in second hidden dimension
                z_dim (int):  features in latent space
                num_DE_params (int): number of parameters in underlying system, and thus size of decoder output

            Keyword Arguments:
                return_activations (bool): if True, returns activation of encoder"""

        self.activations = []
        self.return_activations = return_activations

        self.in_shape = in_shape
        num_dims = len(in_shape) - 1

        # current_shape = in_shape[1:]
        # for _ in range(2):
        #    current_shape = [(cs - 3 + 1) for cs in current_shape]

        # self.out_shape = [h_dim2] + current_shape
        # self.out_size = np.prod(self.out_shape)

        # Initializing the 2 convolutional layers and 2 full-connected layers for the encoder

        self.encConv1 = convNd(in_channels=in_shape[0],
                               out_channels=h_dim1,
                               num_dims=num_dims,
                               kernel_size=3,
                               stride=tuple([1 for _ in range(num_dims)]),
                               padding=0,
                               kernel_initializer=torch.nn.init.xavier_uniform_,
                               bias_initializer=torch.nn.init.zeros_)
        self.encConv2 = convNd(in_channels=h_dim1,
                               out_channels=h_dim2,
                               kernel_size=3,
                               num_dims=num_dims,
                               stride=tuple([1 for _ in range(num_dims)]),
                               padding=0,
                               kernel_initializer=torch.nn.init.xavier_uniform_,
                               bias_initializer=torch.nn.init.zeros_)

        with torch.no_grad():
            x = torch.rand(1, *in_shape)
            x = self.encConv1(x)
            x = self.encConv2(x)
        self.out_shape = x.shape
        self.out_size = np.prod(self.out_shape)

        self.encFC1 = nn.Linear(self.out_size, z_dim)
        self.encFC2 = nn.Linear(self.out_size, z_dim)

        # Initializing the fully-connected layer and 2 convolutional layers for decoder

        self.decoder = nn.Sequential(nn.Linear(z_dim, 64),
                                     nn.ReLU(),
                                     nn.Linear(64, 64),
                                     nn.ReLU(),
                                     nn.Linear(64, num_DE_params))

    def encode(self, x):
        # Input is fed into 2 convolutional layers sequentially
        # The output feature map are fed into 2 fully-connected layers to predict mean (mu) and variance (logVar)
        # Mu and logVar are used for generating middle representation z and KL divergence loss
        h1 = torch.tanh(self.encConv1(x))
        h2 = torch.tanh(self.encConv2(h1))

        if self.return_activations:
            self.activations += [h1, h2]
        h2 = h2.view(-1, self.out_size)
        mu = self.encFC1(h2)
        logVar = self.encFC2(h2)

        return mu, logVar

    def reparameterize(self, mu, logVar):
        # Reparameterization takes in the input mu and logVar and sample the mu + std * eps
        std = torch.exp(logVar / 2)
        eps = torch.randn_like(std)
        return mu + std * eps

    def decode(self, z):
        # z is fed back into a fully-connected layers and then into two transpose convolutional layers
        # The generated output is the same size of the original input
        return self.decoder(z)

    def id(self, x, n_neighbors=100):
        latent, _ = self.encode(x)
        latent = latent.detach().cpu().numpy()
        return ID.DANCo().fit(latent), ID.lPCA().fit_pw(latent, n_neighbors=n_neighbors, n_jobs=1)

    def forward(self, x, return_activations=False):
        self.return_activations = return_activations
        # The entire pipeline of the VAE: encoder -> reparameterization -> decoder
        # output, mu, and logVar are returned for loss computation
        mu, logVar = self.encode(x)
        z = self.reparameterize(mu, logVar)
        return self.activations + [self.decoder(z)], mu, logVar


class ResParAE(nn.Module):
    """ResNet VAE"""

    def __init__(self, in_shape, z_dim, num_DE_params,
            last_pad=0, return_activations=False, conditional=False,
            num_enc_fc_hid_layers=0, enc_fc_hid_dims=[0], 
            num_dec_fc_hid_layers=1, dec_fc_hid_dims=[64], dropout=False,
            dropout_rate=.5, batch_norm=False, activation_type='relu'):
        super(ResParAE, self).__init__()
        """ResVAE: VAE with ResNet encoder and FC decoder mapping to the underlying parameter space

            Positional Arguments:
                in_shape (iterable): An iterable indicating array size (c x h x w)
                enc_out_dim (int): output channels of encoder
                z_dim (int): dimension of latent space
                num_DE_params (int): number of parameters in underlying system, and thus size of decoder output

            Keyword Arguments:
                last_pad (int): how much to pad each spatial dimension of the output
                return_activation (bool): if True, returns encoder acctivations"""
        self.return_activations = return_activations
        self.activations = []
        self.conditional = conditional

        self.encoder = resnet18_encoder(False, False)

        self.in_shape = in_shape if not conditional else [in_shape[0] + 1] + in_shape[1:]
        self.encoder.conv1 = torch.nn.Conv2d(
            self.in_shape[0], 64, kernel_size=5, stride=1, padding=1, bias=False)

        # Encoder layers
        self.layers = [self.encoder.conv1,
                       self.encoder.bn1,
                       self.encoder.relu,
                       self.encoder.maxpool,
                       self.encoder.layer1,
                       self.encoder.layer2,
                       self.encoder.layer3,
                       self.encoder.layer4]

        # Calculate out_shape
        x = torch.rand(1, *in_shape)
        with torch.no_grad():
            for layer in self.layers:
                x = layer(x)
        self.out_shape = x.shape[1:]
        self.out_size = self.out_shape[0] * self.out_shape[1] * self.out_shape[2]

        if self.out_shape[1] < 1 or self.out_shape[2] < 1:
            raise ValueError('Image is too small for ResNet!')
        # self.out_size = np.prod(np.array(self.out_shape))

        #  distribution parameters
        self.fc_mu = MLP(self.out_size, z_dim,
                         num_hid_layers=num_enc_fc_hid_layers, hid_dims=enc_fc_hid_dims,
                         batch_norm=batch_norm, dropout=dropout, dropout_rate=dropout_rate,
                         activation_type=activation_type)

        self.fc_var = MLP(self.out_size, z_dim,
                         num_hid_layers=num_enc_fc_hid_layers, hid_dims=enc_fc_hid_dims,
                         batch_norm=batch_norm, dropout=dropout, dropout_rate=dropout_rate,
                         activation_type=activation_type)



        self.dec = MLP(z_dim, num_DE_params, 
                       num_hid_layers=num_dec_fc_hid_layers, hid_dims=dec_fc_hid_dims,
                       batch_norm=batch_norm, dropout=dropout, dropout_rate=dropout_rate,
                       activation_type=activation_type)

        self.global_id = ID.DANCo()
        self.local_id = ID.lPCA()

    def encode(self, x):
        for layer in self.layers:
            x = layer(x)
            if self.return_activations:
                self.activations.append(x.detach().clone())
        pdb.set_trace()
        x = x.reshape(x.shape[0], -1)
        return self.fc_mu(x), self.fc_var(x)  # mu, log_var

    def sampling(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)  # return z sample

    def decode(self, z):
        dec = self.dec(z)
        # return F.sigmoid(dec)
        return dec

    def id(self, x, n_neighbors=100):
        latent, _ = self.encode(x)
        latent = latent.detach().cpu().numpy()
        return self.global_id.fit(latent), self.local_id.fit_pw(latent, n_neighbors=n_neighbors, n_jobs=1)

    def forward(self, x, y=None,return_activations=False):
        self.return_activations = return_activations
        if self.conditional:
            x = torch.cat((x,y),dim=-1)
        mu, log_var = self.encode(x)
        z = self.sampling(mu, log_var)
        if self.conditional:
            z = torch.cat((z,y),dim=-1)
        return self.activations + [self.decode(z)], mu, log_var

class hyperCNN(nn.Module):
    def __init__(self, hyper_in_shape, in_dim, out_dim, num_hid_layers, num_hid_units, batch_size, hyper_num_conv_features=[32], hyper_num_conv_layers=1, hyper_kernel_sizes=[5,5], hyper_num_fc_layers=1,hyper_num_fc_features=[],return_inv=False):    
        super(hyperCNN,self).__init__()
        '''hyperCNN: convolutional network taking in stacked vector fields and outputing a dictionary of parameters for a target MLP                                                                                                                Positional arguments:
        * hyper_in_shape (list): shape of input vector fields (2*dim x height x width)                                        * in_dim (int): number of input units in target network
        * out_dim (int): number of output units in target network (equal to in_dim for a neural ODE target net)
        * num_hid_layers (int): number of hidden layers in target network
        * num_hid_units (int): number of hidden features in target network hidden layer
        * batch_size (int): batch_size of TARGET network
                                                                                                                              Keyword arguments:
        * hyper_num_hid_features (int): number of conv features in hypernetwork layers
        * hyper_num_hid_layers (int): number of conv layesr in hypernetwork
        
        Returns:
        * param_dict (dict): a dictionary with keys 'Wi' and 'bi' whose values are the ith weight matrix or bias vector, respectively, of the target network.'''
        self.num_hid_layers        = num_hid_layers
        self.num_hid_units         = num_hid_units 
        self.hyper_num_conv_layers = hyper_num_conv_layers
        self.out_dim               = out_dim
        self.return_inv            = return_inv
        num_out                    = 0
                                                                                                                              # Make dictionary of empty parameters for the target network
        self.param_dict = {}
        self.inv_param_dict = {}
                                                                                                                              # For first layer, hidden layers, last layer
        for l in range(num_hid_layers + 1):
            # If first layer
            if l == 0 and num_hid_layers > 0:
                # First linear: in_dim x num_hid_units
                num_out += in_dim * num_hid_units
                # First bias
                num_out += num_hid_units
                self.param_dict[f'W{l}'] =  torch.zeros((batch_size, in_dim, num_hid_units))
                self.param_dict[f'b{l}'] =  torch.zeros(batch_size, num_hid_units)
                if return_inv:
                    self.inv_param_dict[f'W{l}'] =  torch.zeros((batch_size, out_dim, num_hid_units))
                    self.inv_param_dict[f'b{l}'] =  torch.zeros(batch_size, num_hid_units)
                    num_out += out_dim * num_hid_units
                    num_out += num_hid_units
 
            elif l == 0 and num_hid_layers == 0:
                # First linear: in_dim x num_hid_units
                num_out += in_dim * out_dim
                # First bias
                num_out += out_dim
                self.param_dict[f'W{l}'] =  torch.zeros((batch_size, in_dim, out_dim))
                self.param_dict[f'b{l}'] =  torch.zeros(batch_size, out_dim)
                if return_inv:
                    self.inv_param_dict[f'W{l}'] =  torch.zeros((batch_size, out_dim, in_dim))
                    self.inv_param_dict[f'b{l}'] =  torch.zeros(batch_size, in_dim)
                    num_out += in_dim * out_dim
                    num_out += in_dim

            # If interior layer
            elif l > 0 and l < num_hid_layers:
                # Hid linear: num_hid_units x num_hid_units
                num_out += num_hid_units**2
                num_out += num_hid_units
                self.param_dict[f'W{l}'] =  torch.zeros((batch_size, num_hid_units, num_hid_units))
                self.param_dict[f'b{l}'] =  torch.zeros(batch_size, num_hid_units)
                if return_inv:
                    self.inv_param_dict[f'W{l}'] =  torch.zeros((batch_size, num_hid_units, num_hid_units))
                    self.inv_param_dict[f'b{l}'] =  torch.zeros(batch_size, num_hid_units)
                    num_out += num_hid_units**2
                    num_out += num_hid_units
            # If last layer
            else:
                # Last: num_hid_units x out_dim
                num_out += num_hid_units * out_dim
                num_out += out_dim
                self.param_dict[f'W{l}'] =  torch.zeros((batch_size, num_hid_units, out_dim))
                self.param_dict[f'b{l}'] =  torch.zeros(batch_size, out_dim)
                if return_inv:
                    self.inv_param_dict[f'W{l}'] =  torch.zeros((batch_size, num_hid_units, in_dim))
                    self.inv_param_dict[f'b{l}'] =  torch.zeros(batch_size, in_dim)
                    num_out += num_hid_units * in_dim
                    num_out += in_dim
                
        # Build hyperCNN
        layers = [nn.Conv2d(hyper_in_shape[0],hyper_num_conv_features[0], hyper_kernel_sizes[0]), nn.ReLU()]
        for i in range(hyper_num_conv_layers - 1):
            layers += [nn.Conv2d(hyper_num_conv_features[i], hyper_num_conv_features[i+1], hyper_kernel_sizes[i+1])]
            layers += [nn.ReLU()]
        
        # Add linear layer after calculating output shape of conv part
        with torch.no_grad():
            x = torch.rand(1,*hyper_in_shape)
            for layer in layers:
                x = layer(x)
            self.out_shape = x.shape
            self.out_size = np.prod(self.out_shape)

        # FC component
        in_features = self.out_size
        for j in range(hyper_num_fc_layers):
            if j < hyper_num_fc_layers - 1:
                layers += [nn.Linear(in_features, hyper_num_fc_features[j]), nn.ReLU()]
                in_features = hyper_num_fc_features[j]
            else:
                layers += [nn.Linear(in_features,num_out)]
        self.layers = nn.ModuleList(layers) 
    
    def forward(self,x,y):
        batch_size = x.shape[0]
        z = x - y
        # Get output
        for l, layer in enumerate(self.layers):
            if l == (2*self.hyper_num_conv_layers - 1):
                z = z.reshape(batch_size, -1)
            z = layer(z)
        # Reshape for target network and update parameter dictionary
        p = 0
        for l in range(self.num_hid_layers + 1):
            # Weight
            weight_shape = self.param_dict[f'W{l}'].shape
            num_params = np.prod(weight_shape[1:])

            self.param_dict[f'W{l}'] = z[:, p:p+num_params].reshape(*weight_shape)
            p += num_params
            
            #Bias
            bias_shape = self.param_dict[f'b{l}'].shape
            num_params = np.prod(bias_shape[1:])
            
            self.param_dict[f'b{l}'] = z[:, p:p+num_params].reshape(*bias_shape)
            p += num_params
        if self.return_inv:
            for l in range(self.num_hid_layers + 1):
                # Weight
                weight_shape = self.inv_param_dict[f'W{l}'].shape
                num_params = np.prod(weight_shape[1:])
                self.inv_param_dict[f'W{l}'] = z[:, p:p+num_params].reshape(*weight_shape)
                p += num_params
                
                #Bias
                bias_shape = self.inv_param_dict[f'b{l}'].shape
                num_params = np.prod(bias_shape[1:])
                
                self.inv_param_dict[f'b{l}'] = z[:, p:p+num_params].reshape(*bias_shape)
                p += num_params

        if self.return_inv:
            return self.param_dict, self.inv_param_dict
        else:
            return self.param_dict

class TargetAE(nn.Module):
    def __init__(self,param_dict):
        super(TargetAE,self).__init__()
        '''flow_ode_target: neural ODE whose parameters are inherited from a hypernetwork output
        
        Positional arguments:
        * param_dict (dict): see hyperCNN help string
        
        Keyword arguments: 
        *adjoint (bool): if true, calculate gradients with adjoint sensitivity method
        * solver_method (str): method for solving ode'''

        self.param_dict = param_dict

    def forward(self, x, return_summed=False):
        par_batch_size = self.param_dict['W0'].shape[0]
        dim = self.param_dict['W0'].shape[1]
        x = x.reshape(par_batch_size, -1, dim)
        # Calculate time derivative using param_dict. NOTE: this relies on proper ordering of dictionary so not too stable
        for l, (key, params) in enumerate(self.param_dict.items()):
            # If linear operation
            if key[0] == 'W':
                # We calculate num_pairs different flows simultaneously
                # (num_pairs x in_features x out_features) X (num_pairs x num_inits x in_features) --> num_pairs x num_inits x out_features
                x = torch.einsum('pfg,pif->pig',params,x)
        # Elif bias
            elif key[0] == 'b':
                x = x + params.unsqueeze(1)
                if l < len(self.param_dict) - 1:
                    x = F.relu(x)
        if return_summed:
            return x.reshape(-1,dim).sum(0)
        else:
            return x
