#import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
#plt.ioff()
from mpl_toolkits import mplot3d
import torch
from torch.distributions.normal import Normal
import torch.nn.functional as F
import numpy as np
from torchdiffeq import odeint, odeint_adjoint
#matplotlib.rcParams['text.usetex'] = True
from tqdm import tqdm
import os, sys
import pickle
import pdb


class FlowCircuitPDE(torch.nn.Module):
    """
    Super class for circuits of Partial Differential Equations
    """

    alpha = .1
    min_dims = [-1.0, -1.0]
    max_dims = [1.0, 1.0]
    num_lattice = 28

    def __init__(self, params, labels, adjoint=False, solver_method='euler', train=False):
        super().__init__()
        self.solver = odeint_adjoint if adjoint else odeint
        self.solver_method = solver_method
        self.params = torch.nn.Parameter(params, requires_grad=True) if train else params
        self.labels = labels

    def run(self, T, alpha=.1, init=torch.zeros((5))):
        grid = torch.linspace(0, T, int(T / alpha))
        return self.solver(self, init, grid, rtol=1e-3, atol=1e-5, method=self.solver_method)

    def _plot_trajectory_2d(self, T, alpha, L, min_dims, max_dims, ax=None):
        """
        Plots trajectory over a 2d grid
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(6, 6))
        for ics in [L[0, ...], L[-1, ...], L[:, 0, :], L[:, -1, :]]:
            for ic in ics:
                trajectory = self.run(T, alpha=alpha, init=ic).detach().cpu().numpy()
                p = ax.plot(trajectory[..., 0], trajectory[..., 1], linewidth=.1, alpha=1., color='blue')
                ax.scatter(trajectory[-1, 0], trajectory[-1, 1], c='red', marker='x', s=64)

        ax.set_xlim([min_dims[0], max_dims[0]])
        ax.set_ylim([min_dims[1], max_dims[1]])
        ax.set_xlabel(self.labels[0])
        ax.set_ylabel(self.labels[1])

        if ax is None:
            plt.show()
            plt.close()


    def _plot_trajectory_3d(self, T, alpha, L, min_dims, max_dims, ax=None):
        """
        Plots trajectory over a 3d grid
        """
        lx, ly, lz = L.shape
        if ax is None:
            fig = plt.figure(figsize=(6, 6, 6))
            ax = plt.axes(projection='3d')
        for x in range(lx):
            for y in range(ly):
                for z in range(lz):
                    ic = L[x, y, z]
                    trajectory = self.run(T, alpha=alpha, init=ic).detach().cpu().numpy()
                    p = ax.plot(trajectory[..., 0], trajectory[..., 1], trajectory[..., 2], linewidth=1, alpha=.5,
                                color='blue')
                    # ax.scatter(trajectory[-1,0],trajectory[-1,1], trajectory[-1,2], c='red', marker='x', s=64)

        ax.set_xlim([min_dims[0], max_dims[0]])
        ax.set_ylim([min_dims[1], max_dims[1]])
        ax.set_zlim([min_dims[2], max_dims[2]])
        ax.set_xlabel(self.labels[0])
        ax.set_ylabel(self.labels[1])
        ax.set_zlabel(self.labels[2])

        if ax is None:
            plt.show()
            plt.close()


    def plot_trajectory(self, T, alpha=.1, min_dims=[-1.0, -1.0], max_dims=[1.0, 1.0], num_lattice=28, ax=None):
        """
        Plot runs of multiple trajectories
        """
        spatial_coords = [torch.linspace(mn, mx, num_lattice) for (mn, mx) in zip(min_dims, max_dims)]
        mesh = torch.meshgrid(*spatial_coords)
        L = torch.cat([ms[..., None] for ms in mesh], dim=-1)

        ndims = len(max_dims)
        if ndims == 2:
            self._plot_trajectory_2d(T, alpha, L, min_dims, max_dims, ax=ax)
        elif ndims == 3:
            self.plot_trajectory_3d(T, alpha, L, min_dims, max_dims, ax=ax)


    # def forward(self, t, z, **kwargs): # TODO: We can have the super class Gray-Scott specific and then this can go there
    #     # Unpack parameters
    #
    #     view_dims = (z.ndim - 1) * [1]
    #     V1 = self.other_params[0].view(*view_dims, -1)
    #     V2 = self.other_params[1].view(*view_dims, -1)
    #     k11 = self.other_params[2].view(*view_dims, -1)
    #     k12 = self.other_params[3].view(*view_dims, -1)
    #     k21 = self.other_params[4].view(*view_dims, -1)
    #     k22 = self.other_params[5].view(*view_dims, -1)
    #     n11 = self.other_params[6].view(*view_dims, -1)
    #     n12 = self.other_params[7].view(*view_dims, -1)
    #     n21 = self.other_params[8].view(*view_dims, -1)
    #     n22 = self.other_params[9].view(*view_dims, -1)
    #     b1 = self.other_params[10].view(*view_dims, -1)
    #     b2 = self.other_params[11].view(*view_dims, -1)
    #     mu1 = self.other_params[12].view(*view_dims, -1)
    #     mu2 = self.other_params[13].view(*view_dims, -1)
    #
    #     K = torch.cat((torch.cat([k11.unsqueeze(-1), k12.unsqueeze(-1)], dim=-1),
    #                    torch.cat([k21.unsqueeze(-1), k22.unsqueeze(-1)], dim=-1)), dim=-2)
    #     N = torch.cat((torch.cat([n11.unsqueeze(-1), n12.unsqueeze(-1)], dim=-1),
    #                    torch.cat([n21.unsqueeze(-1), n22.unsqueeze(-1)], dim=-1)), dim=-2)
    #
    #     V = torch.cat((V1, V2), dim=-1)
    #     b = torch.cat((b1, b2), dim=-1)
    #     mu = torch.cat((mu1, mu2), dim=-1)
    #     couplings = self.params.reshape(*K.shape)
    #     N = N.reshape(*K.shape)
    #
    #     zdot = V * torch.prod(1. / (1. + (K / z.unsqueeze(-2)) ** (couplings * N)), dim=-1) + b  # - mu*z
    #     # zdot = torch.clamp(zdot, -10,10)
    #     return zdot


class GrayScott(FlowCircuitPDE):
    """
    Gray Scott diffusion-reaction model:

        udot = -uv^2 + F(1-u) + Du Lu
        vdot = uv^2 - (F+k)v + Dv Lv

    Where:
        - Du, Dv - diffusion coefficients
        - Lu, Lv - Laplacians of u and v
        - F - rate inwhich u is replenished
        - k - rate of v's removal
    """
    labels = ['u', 'v']
    params = torch.ones(16)
    num_lattice = 64
    min_dims = []
    max_dims = []

    def __init__(self, n, params=params, labels=labels, train=False, device='cpu', adjoint=False, solver_method='euler'):
        super().__init__()

        self.n = n
        self.device = device

        self.Du = params[-2]
        self.Dv = params[-1]
        tmp = torch.tensor([1., 1., -1., 0., 100., 3.1623, 3.1623, .1, 3.1623, 2., 2., 2., 2., 100., .1, .1, 1., 1.])
        self.network_model = GrayScottNetwork(params=tmp) #TODO: remove this

        self.solver_method = solver_method
        if adjoint:
            self.solver = odeint_adjoint
        else:
            self.solver = odeint

    def initial_condition(self, noise_magnitude=0.0):
        u = (torch.ones((self.n, self.n)) + (noise_magnitude*np.random.random((self.n, self.n)))).float().to(self.device)
        v = (torch.zeros((self.n, self.n)) + (noise_magnitude*np.random.random((self.n, self.n)))).float().to(self.device)

        x, y = torch.meshgrid(torch.linspace(0, 1, self.n).to(self.device),
                              torch.linspace(0, 1, self.n).to(self.device))

        mask = (0.4 < x) & (x < 0.6) & (0.4 < y) & (y < 0.6)

        u[mask] = 0.50
        v[mask] = 0.25

        return u, v

    def periodic_bc(self, u):
        u = u.reshape(self.n, self.n)
        u[0, :] = u[-2, :]
        u[-1, :] = u[1, :]
        u[:, 0] = u[:, -2]
        u[:, -1] = u[:, 1]

        return u.reshape(-1)

    def laplacian(self, u):
        """
        second order finite differences
        """
        u = F.pad(u, (1, 1, 1, 1))
        return u[:-2, 1:-1] + u[1:-1, :-2] - 4*u[1:-1, 1:-1] + u[1:-1, 2:] + u[2:, 1:-1]

    def forward(self, t, X):
        U, V = X[:, 0], X[:, 1]

        U = torch.clamp(U, min=0.0)
        V = torch.clamp(V, min=0.0)

        u = U.reshape(self.n, self.n)
        v = V.reshape(self.n, self.n)

        Lu = self.laplacian(u).reshape(-1)
        Lv = self.laplacian(v).reshape(-1)

        Z = torch.cat([U.unsqueeze(-1), V.unsqueeze(-1)], dim=-1)
        reaction = self.network_model.forward(0, Z) # TODO: move this to joint super class

        delta_U = (self.Du*Lu + reaction[..., 0])
        delta_V = (self.Dv*Lv + reaction[..., 1])
        #delta_U = (.1*Lu -U*V**2 + .0545*(1 - U))
        #delta_V = (.05*Lv + U*V**2 - (.062 + .0545)*V)

        delta_U = self.periodic_bc(delta_U)
        delta_V = self.periodic_bc(delta_V)

        delta = torch.cat([delta_U.unsqueeze(-1), delta_V.unsqueeze(-1)], dim=-1)

        if delta.mean() != delta.mean():
            pdb.set_trace()
        return delta

    def run(self, T, alpha=.1, noise_magnitude=0.0):
        U, V = self.initial_condition(noise_magnitude=noise_magnitude)
        y = torch.cat([U.reshape(-1).unsqueeze(-1), V.reshape(-1).unsqueeze(-1)], dim=-1)
        num_steps = int(T/float(alpha))
        grid = torch.linspace(0, T, num_steps)
        trajectory = self.solver(self, y, grid, rtol=1e-3, atol=1e-5, method=self.solver_method)

        return trajectory

    def sample_parameters(self):
        samples = []
        for p, pr in enumerate(self.params):
            dist = Normal(pr, self.sigmas[p])
            samples.append(dist.rsample())
        return samples

    def set_U(self,U_trajectory, which='final'):
        which_U = U_trajectory[-1].reshape(self.n,self.n)

        if which == 'final':
            self.final_U = which_U.clone().detach().cpu().numpy()
        else:
            self.init_U = which_U.clone().detach().cpu().numpy()

class GrayScottNetwork(FlowCircuitPDE):
    """

    """
    params = torch.ones(18)
    dim = 2
    labels = [r'$\theta_{}$'.format(i) for i in range(dim)] # TODO: check that len(mesh) == dim
    num_lattice = 10
    min_dims = [0, 0]
    max_dims = [10, 10]

    def __init__(self, params=params, train=False, **kwargs):
        super().__init__(params=params, train=train, **kwargs)

        num_couplings = 4 if params.shape[0] == 18 else 9
        self.num_dim = int(np.sqrt(num_couplings))
        num_other     = params.shape[0] - num_couplings
        self.params = torch.nn.Parameter(params[:num_couplings], requires_grad=True) if train else params[:num_couplings]
        self.params = self.params.float()

        #reshape_dims = self.num_dim * [1] 
        #self.other_params = params[-num_other:].reshape(*reshape_dims, -1).float()
        #self.other_params = params[-num_other:].float()
        self.other_params = torch.tensor([100.,3.1623,3.1623,.1,3.1623,2.,2.,2.,2.,100.,.1,.1,1.,1.]).float()


    def plot_trajectory(self, T, alpha=.1, min_dims=min_dims, max_dims=max_dims, num_lattice=num_lattice, ax=None):
        num_dims = len(min_dims)
        spatial_coords = [torch.linspace(mn, mx, num_lattice) for (mn,mx) in zip(min_dims, max_dims)]
        mesh = torch.meshgrid(*spatial_coords)
        L = torch.cat([ms[...,None] for ms in mesh], dim=-1)

        if ax is None:
            if num_dims == 2:
                fig, ax = plt.subplots(figsize=(6,6))
            else:
                fig = plt.figure(figsize=(6,6,6))
                ax = plt.axes(projection='3d')

        if num_dims == 2:
            for ics in [L[0,...], L[-1,...],L[:,0,:],L[:,-1,:]]:
                for ic in ics:
                    trajectory = self.run(T, alpha=alpha, init=ic).detach().cpu().numpy()
                    p = ax.plot(trajectory[...,0], trajectory[...,1], linewidth=1, alpha=1., color='blue')
                    ax.scatter(trajectory[-1,0],trajectory[-1,1], c='red', marker='x', s=64)

            ax.set_xlim([min_dims[0], max_dims[0]])
            ax.set_ylim([min_dims[1], max_dims[1]])
            ax.set_xlabel(self.labels[0])
            ax.set_ylabel(self.labels[1])
        else:
            for x in range(num_lattice):
                        for y in range(num_lattice):
                            for z in range(num_lattice):
                                ic = L[x, y, z]
                                trajectory = self.run(T, alpha=alpha, init=ic).detach().cpu().numpy()
                                p = ax.plot(trajectory[..., 0], trajectory[..., 1], trajectory[..., 2], linewidth=1, alpha=.5, color='blue')

        #if ax is None:
        #    plt.show()
        #    plt.close()

