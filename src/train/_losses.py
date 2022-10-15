import torch
import torch.nn.functional as F
import numpy as np
from numba import jit
from torch.autograd import Function


def slice_to_loss(recon_x, x, pad):
    """
    Slices prediction and truth for loss computation
    """
    batch_size = x.shape[0]
    c = x.shape[1]
    spatial = x.shape[2:]
    recon_x = recon_x.reshape(batch_size, c, *spatial)
    slices = [slice(0, sd) for sd in x.shape[:2]] + [slice(pad, sd - pad) for sd in spatial]
    recon_x = recon_x[slices]
    x = x[slices]
    return recon_x, x

################################################## Reconstruction losses ###############################################

def euclidean_loss(recon_x, x, pad=0):
    """
    Euclidean loss
    """
    recon_x, x = slice_to_loss(recon_x, x, pad)
    return ((recon_x - x) ** 2).mean()


def normalized_euclidean(recon,gt, eps=1e-5, norm_type='single', measurement='speed'):

    '''Euclidean distance between two arrays with spatial dimensions. Normalizes pointwise across the spatial dimension by the norm of the second argument'''
    batch_size =recon.shape[0]
    if measurement=='speed':
        m_r = torch.sqrt(recon[:,0]**2 + recon[:,1]**2).unsqueeze(1)
        m_gt = torch.sqrt(gt[:,0]**2 + gt[:,1]**2).unsqueeze(1)
    elif measurement == 'jacobian':
        m_r = torch.real(torch.linalg.det(jacobian(recon).permute(0,3,4,1,2))).unsqueeze(1)
        m_gt = torch.real(torch.linalg.det(jacobian(gt).permute(0,3,4,1,2))).unsqueeze(1)
    elif measurement == 'divergence':
        m_r = divergence(recon).unsqueeze(1).abs()
        m_gt = divergence(gt).unsqueeze(1).abs()
    elif measurement == 'curl':
        m_r = (curl(recon)**2).sum(1).unsqueeze(1)
        m_gt = (curl(gt)**2).sum(1).unsqueeze(1)
    elif measurement == 'laplacian':
        # Should you use the inverse of this?
        m_r = (laplacian(recon)**2).sum(1).unsqueeze(1)
        m_gt = (laplacian(gt)**2).sum(1).unsqueeze(1)
    else:
        raise ValueError('Measurement not recognized!')

    if norm_type == 'single':
        den = m_gt + eps
    elif norm_type == 'product':
        den = m_gt * m_recon + eps
    elif norm_type == 'sum':
        den = m_gt + m_recon + eps
    else:
        raise ValueError('Denominator type not recognized!')


    # normalize pointwise by gt norm
    d = torch.sqrt((((recon- gt)**2) / den).reshape(batch_size,-1).mean(1))

    return d.mean()
