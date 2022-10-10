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

def BCE_loss(recon_x, x, pad=0):
    """
    Binary cross-entropy loss
    """
    batch_size = x.shape[0]
    recon_x, x = slice_to_loss(recon_x, x, pad)
    return F.binary_cross_entropy(recon_x.reshape(batch_size, -1), x.reshape(batch_size, -1), reduction='mean')


def euclidean_loss(recon_x, x, pad=0):
    """
    Euclidean loss
    """
    recon_x, x = slice_to_loss(recon_x, x, pad)
    return ((recon_x - x) ** 2).mean()

def style_loss(model_out, model_out_target, weights=None):
    """
    Computes l2 difference of Gram matrices of weights of two models
    """
    est_activations = model_out[:-1]
    target_activations = model_out_target[:-1]

    L = len(est_activations)

    if weights == None:
        weights = torch.ones(L, )

    loss = 0
    for l, (est, target) in enumerate(zip(est_activations, target_activations)):
        B = est.shape[0]
        C = est.shape[1]
        M = est.shape[2] * est.shape[3]
        G_est = torch.einsum('bchw,bchw->bhw', est, est).reshape(B, -1)
        G_target = torch.einsum('bchw,bchw->bhw', target, target).reshape(B, -1)

        loss += weights[l] * ((G_est - G_target) ** 2).sum(-1) / (4 * C ** 2 * M ** 2)
    return loss.mean() / (B * L)


def weighted_euclidean_loss(recon,gt, eps=1e-5):
    '''Euclidean distance between two arrays with spatial dimensions. 
    Normalizes pointwise across the spatial dimension by the norm of the second argument'''
    batch_size =recon.shape[0]
    r_recon = torch.sqrt(recon[:,0]**2 + recon[:,1]**2).unsqueeze(1)
    r_gt = torch.sqrt(gt[:,0]**2 + gt[:,1]**2).unsqueeze(1)
    
    # normalize pointwise by gt norm
    d = torch.sqrt((((recon- gt)**2) / (r_gt + eps)).reshape(batch_size,-1).mean(1))
    
    return d.mean()

def get_reconstruction_loss(recon_loss):
    """
    Select loss function
    """
    recon_loss_function = None
    return_activations = False

    if recon_loss == 'BCE':
        recon_loss_function = BCE_loss
    elif recon_loss == 'euclidean':
        recon_loss_function = euclidean_loss
    elif recon_loss == 'style':
        recon_loss_function = style_loss
        return_activations = True
    else:
        raise ValueError('Loss function not recognized!')

    return recon_loss_function, return_activations

################################################## ? losses ###############################################

def KL_loss(mu, log_var): # TODO: actually, not sure what we are evaluating here, why defined this way??
    """

    """
    KLD = -.5 * torch.mean(1 + log_var - mu.pow(2) - log_var.exp())
    return KLD

##################################################### Sparsity losses ##################################################

def sparsity_loss(params, p):
    """
    Lp norm of inferred parameters
    """
    return torch.norm(params, dim=-1, p=p).mean()

# https://github.com/Sleepwalking/pytorch-softdtw.git

@jit(nopython = True)
def compute_softdtw(D, gamma):
  B = D.shape[0]
  N = D.shape[1]
  M = D.shape[2]
  R = np.ones((B, N + 2, M + 2)) * np.inf
  R[:, 0, 0] = 0
  for k in range(B):
    for j in range(1, M + 1):
      for i in range(1, N + 1):
        r0 = -R[k, i - 1, j - 1] / gamma
        r1 = -R[k, i - 1, j] / gamma
        r2 = -R[k, i, j - 1] / gamma
        rmax = max(max(r0, r1), r2)
        rsum = np.exp(r0 - rmax) + np.exp(r1 - rmax) + np.exp(r2 - rmax)
        softmin = - gamma * (np.log(rsum) + rmax)
        R[k, i, j] = D[k, i - 1, j - 1] + softmin
  return R

@jit(nopython = True)
def compute_softdtw_backward(D_, R, gamma):
  B = D_.shape[0]
  N = D_.shape[1]
  M = D_.shape[2]
  D = np.zeros((B, N + 2, M + 2))
  E = np.zeros((B, N + 2, M + 2))
  D[:, 1:N + 1, 1:M + 1] = D_
  E[:, -1, -1] = 1
  R[:, : , -1] = -np.inf
  R[:, -1, :] = -np.inf
  R[:, -1, -1] = R[:, -2, -2]
  for k in range(B):
    for j in range(M, 0, -1):
      for i in range(N, 0, -1):
        a0 = (R[k, i + 1, j] - R[k, i, j] - D[k, i + 1, j]) / gamma
        b0 = (R[k, i, j + 1] - R[k, i, j] - D[k, i, j + 1]) / gamma
        c0 = (R[k, i + 1, j + 1] - R[k, i, j] - D[k, i + 1, j + 1]) / gamma
        a = np.exp(a0)
        b = np.exp(b0)
        c = np.exp(c0)
        E[k, i, j] = E[k, i + 1, j] * a + E[k, i, j + 1] * b + E[k, i + 1, j + 1] * c
  return E[:, 1:N + 1, 1:M + 1]

class _SoftDTW(Function):
  @staticmethod
  def forward(ctx, D, gamma):
    dev = D.device
    dtype = D.dtype
    gamma = torch.Tensor([gamma]).to(dev).type(dtype) # dtype fixed
    D_ = D.detach().cpu().numpy()
    g_ = gamma.item()
    R = torch.Tensor(compute_softdtw(D_, g_)).to(dev).type(dtype)
    ctx.save_for_backward(D, R, gamma)
    return R[:, -2, -2]

  @staticmethod
  def backward(ctx, grad_output):
    dev = grad_output.device
    dtype = grad_output.dtype
    D, R, gamma = ctx.saved_tensors
    D_ = D.detach().cpu().numpy()
    R_ = R.detach().cpu().numpy()
    g_ = gamma.item()
    E = torch.Tensor(compute_softdtw_backward(D_, R_, g_)).to(dev).type(dtype)
    return grad_output.view(-1, 1, 1).expand_as(E) * E, None

class SoftDTW(torch.nn.Module):
  def __init__(self, gamma=1.0, normalize=False):
    super(SoftDTW, self).__init__()
    self.normalize = normalize
    self.gamma=gamma
    self.func_dtw = _SoftDTW.apply

  def calc_distance_matrix(self, x, y):
    n = x.size(1)
    m = y.size(1)
    d = x.size(2)
    x = x.unsqueeze(2).expand(-1, n, m, d)
    y = y.unsqueeze(1).expand(-1, n, m, d)
    dist = torch.pow(x - y, 2).sum(3)
    return dist

  def forward(self, x, y):
    assert len(x.shape) == len(y.shape)
    squeeze = False
    if len(x.shape) < 3:
      x = x.unsqueeze(0)
      y = y.unsqueeze(0)
      squeeze = True
    if self.normalize:
      D_xy = self.calc_distance_matrix(x, y)
      out_xy = self.func_dtw(D_xy, self.gamma)
      D_xx = self.calc_distance_matrix(x, x)
      out_xx = self.func_dtw(D_xx, self.gamma)
      D_yy = self.calc_distance_matrix(y, y)
      out_yy = self.func_dtw(D_yy, self.gamma)
      result = out_xy - 1/2 * (out_xx + out_yy) # distance
    else:
      D_xy = self.calc_distance_matrix(x, y)
      out_xy = self.func_dtw(D_xy, self.gamma)
      result = out_xy # discrepancy
    return result.squeeze(0) if squeeze else result
