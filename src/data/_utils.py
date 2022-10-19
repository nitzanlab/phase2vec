import torch
import numpy as np
import os

def load_dataset(data_path):
    # Load data
    X_train = np.load(os.path.join(data_path, 'X_train.npy'))
    X_val = np.load(os.path.join(data_path, 'X_val.npy'))

    # Load labels
    y_train = np.load(os.path.join(data_path, 'y_train.npy'))
    y_val = np.load(os.path.join(data_path, 'y_val.npy'))

    # Load pars
    p_train = np.load(os.path.join(data_path, 'p_train.npy'))
    p_val = np.load(os.path.join(data_path, 'p_val.npy'))
    return X_train, X_val, y_train, y_val, p_train, p_val

def jacobian(f,spacings=1):
    '''Returns the Jacobian of a batch of planar vector fields shaped batch x dim x spatial x spatial'''
    num_dims = f.shape[1]
    return torch.stack([torch.stack(torch.gradient(f[:,i],dim=list(range(1,num_dims+1)), spacing=spacings)) for i in range(num_dims)]).movedim(2,0)

def curl(f, spacings=1):
    '''Returns the curl of a batch of 2d (3d) vector fields shaped batch x dim x spatial x spatial (x spatial)'''
    num_dims = f.shape[1]
    if num_dims > 4:
        raise ValueError('Curl is only defined for dim <=3.')
    elif num_dims < 3:
        b = f.shape[0]
        s = f.shape[-1]
        f = torch.tile(f.unsqueeze(-1),(s,))
        f = torch.cat((f, torch.zeros(b,1,s,s,s)), dim=1)
        spacings = [sp for sp in spacings]
        spacings.append(spacings[-1])
        spacings = tuple(spacings)

    J = jacobian(f,spacings=spacings)

    dFxdy = J[:,0,1]
    dFxdz = J[:,0,2]
    dFydx = J[:,1,0]
    dFydz = J[:,1,2]
    dFzdx = J[:,2,0]
    dFzdy = J[:,2,1]
    return torch.stack([dFzdy - dFydz, dFxdz - dFzdx, dFydx - dFxdy]).movedim(1,0)

def divergence(f,spacings=1):
    '''Returns the divergence of a batch of planar vector fields shaped batch x dim x spatial x spatial'''

    # J.shape = batch x dim x dim x [spatial]^n
    J = jacobian(f,spacings=spacings)
    return torch.diagonal(J,dim1=1,dim2=2).sum(-1)

def laplacian(f):
    '''Calculate laplacian of vector field'''
    num_dims = f.shape[1]
    if num_dims>3:
        raise ValueError('Laplacian not yet implemented for dim>2.')
    return torch.stack([divergence(torch.stack(torch.gradient(f[:,i], dim=[1,2])).movedim(1,0)) for i in range(num_dims)]).movedim(1,0)
