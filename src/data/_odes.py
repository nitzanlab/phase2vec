#import matplotlib
#matplotlib.use('Agg')
import pandas as pd
import matplotlib.pyplot as plt
#plt.ioff()
from mpl_toolkits import mplot3d
import math
import torch
import numpy as np
from torchdiffeq import odeint, odeint_adjoint
from functorch import vmap, jacrev
import torch.nn.functional as F
from mpl_toolkits.axes_grid1 import make_axes_locatable
from src.data._polynomials import sindy_library, library_size
# from IPython.display import display, Math, Latex
#from flow_encoding.utils import stable_sigmoid
#matplotlib.rcParams['text.usetex'] = True
import pdb
from time import time
from sklearn import linear_model
from src.utils._utils import curl


class FlowCircuitODE(torch.nn.Module):
    """
    Super class for all circuits of Ordinary Differential Equations.
    Default is a system with one parameter over two dimensions
    """
    
    # ODE params
    n_params = 1
    params = torch.zeros(n_params)

    # ODE variables
    labels = ['x', 'y']
    dim = 2

    # recommended lattice ranges
    min_dims = [-2, -2]
    max_dims = [2, 2]

    # recommended param ranges
    recommended_param_ranges = [[-1, 1]] * n_params

    # configs excluded from info
    exclude = ['params', 'solver', 'library']

    eq_string = None

    def __init__(self, params=params, labels=labels, adjoint=False, solver_method='euler', train=False, device='cuda',
                 num_lattice=28, min_dims=min_dims, max_dims=max_dims, boundary_type=None, boundary_radius=1e1, boundary_gain=1.0,time_direction=1, **kwargs):
        """
        Initialize ODE. Defaults to a 2-dimensional system with a single parameter

        params - torch array allocated for circuit params (default: 0)
        labels - labels of the params (default: ['x', 'y'])
        adjoint - torch solver for differential equations (default: using odeint)
        solver_method - solver method (default: 'euler')
        train - boolean setting whether the parameters are trainable (default: False)
        device - (default: 'cuda')
        num_lattice - resolution of lattice of initial conditions
        min_dims - dimension's minimums of lattice of initial conditions
        max_dims - dimension's maximums of lattice of initial conditions
        """
        super().__init__()
        self.solver = odeint_adjoint if adjoint else odeint
        self.solver_method = solver_method
        self.params = torch.nn.Parameter(params, requires_grad=True) if train else params
        self.params = torch.tensor(self.params) if not isinstance(self.params, torch.Tensor) else self.params
        self.params = self.params.float()
        self.labels = labels
        self.dim = len(labels)
        self.num_lattice = num_lattice
        self.min_dims = min_dims #if min_dims is not None else [-2, ] * self.dim
        self.max_dims = max_dims #if max_dims is not None else [2, ] * self.dim
        self.device = device
        self.boundary_type = boundary_type
        self.boundary_radius = torch.tensor(boundary_radius).float()
        self.boundary_gain = torch.tensor(boundary_gain).float()
        self.boundary_box = [min_dims, max_dims]
        self.time_direction = time_direction
        self.polynomial_terms = sindy_library(torch.ones((1, self.dim)), poly_order=3, include_sine=False, include_exp=False)[1]


    def run(self, T, alpha, init=None,return_summed=False):
        """
        Run circuit

        T - length of run
        alpha - resolution
        init - initial startpoint

        returns:
        """

        init = torch.zeros(self.dim) if init is None else init
        T = T * self.time_direction
        grid = torch.linspace(0, T, abs(int(T / alpha)))
        boundary_dict = {'boundary_type':self.boundary_type, 
                         'boundary_radius':self.boundary_radius,
                         'boundary_gain':self.boundary_gain, 
                         'boundary_box':self.boundary_box}
        trajectory = self.solver(self, init, grid, rtol=1e-3, atol=1e-5, options=boundary_dict, method=self.solver_method)
        if return_summed:
            return trajectory[-1].reshape(-1,self.dim).sum(0)
        else:
            return trajectory


    def get_lattice_params(self, min_dims=None, max_dims=None, num_lattice=None):
        """
        Substitute lattice parameters if provided
        """
        min_dims = self.min_dims if min_dims is None else min_dims
        max_dims = self.max_dims if max_dims is None else max_dims
        num_lattice = self.num_lattice if num_lattice is None else num_lattice

        return min_dims, max_dims, num_lattice


    def _plot_trajectory_2d(self, L, fig=None, ax=None, density=1.0, which_dims=[0,1], title=''):
        """
        Plots trajectory over a 2d grid
        """
        # TODO: add handling of different dims

        #min_dims, max_dims, _ = self.get_lattice_params(min_dims, max_dims)
        xdim, ydim = which_dims
        if ax is None:
            fig, ax = plt.subplots(figsize=(6, 6))
        if isinstance(self, NeuralODE):
            UV = self.forward(0, L.clone().reshape(1,-1,self.dim))[0].reshape(L.shape).detach()
        else:
            UV = self.forward(0, L.clone())
        X = L[...,0].cpu().numpy()
        Y = L[...,1].cpu().numpy()
        U = UV[...,0].cpu().numpy()
        V = UV[...,1].cpu().numpy()
        R = np.sqrt(U**2 + V**2)

        stream = ax.streamplot(X,Y,U,V,density=density,color=R,cmap='autumn',integration_direction='forward')
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.1)
        cax.set_xticks([])
        cax.set_xticklabels([])
        cax.set_yticks([])
        cax.set_yticklabels([])
        
        if fig:
            fig.colorbar(stream.lines, ax=cax)
        
        ax.set_xlim([self.min_dims[xdim], self.max_dims[xdim]]) # TODO: replace with min of L
        ax.set_ylim([self.min_dims[ydim], self.max_dims[ydim]])
        ax.set_xlabel(self.labels[xdim])
        ax.set_ylabel(self.labels[ydim])
        ax.set_title(title,fontsize=8)


    def _plot_trajectory_3d(self, T, alpha, L, min_dims=None, max_dims=None, ax=None, which_dims=[0,1,2], title=''):
        """
        Plots trajectory over a 3d grid
        """
        min_dims, max_dims, _ = self.get_lattice_params(min_dims, max_dims)

        lx, ly, lz = L.shape[:3] # TODO: which dims
        # ax_n = ax is None
        # if ax is None:
        fig = plt.figure()
        ax = plt.axes(projection='3d')
        for x in range(lx):
            for y in range(ly):
                for z in range(lz):
                    ic = L[x, y, z]
                    trajectory = self.run(T, alpha=alpha, init=ic).detach().cpu().numpy()
                    p = ax.plot(trajectory[..., 0], trajectory[..., 1], trajectory[..., 2], linewidth=1, alpha=.5,
                                color='blue')
                    ax.scatter(trajectory[-1, 0], trajectory[-1, 1],  trajectory[-1, 2], c='red', marker='x', s=64)

        ax.set_xlim([min_dims[0], max_dims[0]])
        ax.set_ylim([min_dims[1], max_dims[1]])
        ax.set_zlim([min_dims[2], max_dims[2]])
        ax.set_xlabel(self.labels[0])
        ax.set_ylabel(self.labels[1])
        ax.set_zlabel(self.labels[2])
        ax.set_title(title)
        plt.show()
        # if ax_n:
        #     plt.show()
        #     plt.close()


    def generate_mesh(self, min_dims=None, max_dims=None, num_lattice=None):
        """
        Creates a lattice over coordinate range
        """
        min_dims, max_dims, num_lattice = self.get_lattice_params(min_dims, max_dims, num_lattice)

        spatial_coords = [np.linspace(mn, mx, num_lattice) for (mn, mx) in zip(min_dims, max_dims)]
        # TODO: This is used because pytorch 1.9.0 does not have an indexing keyword argument which can we used to switch to xy ordering for the mesh. We should update to pytorch 1.10.0. 
        mesh = np.meshgrid(*spatial_coords)
        L = np.concatenate([ms[..., None] for ms in mesh], axis=-1)
        return torch.from_numpy(L)


    def plot_trajectory(self, T=None, alpha=None, min_dims=None, max_dims=None, num_lattice=None, ax=None, which_dims=[0,1], title=''):
        """
        Plot multiple trajectories
        """
        L = self.generate_mesh(min_dims=self.min_dims, max_dims=self.max_dims, num_lattice=self.num_lattice)
        L = L.to(self.device)

        mesh = (L[...,0], L[...,1])

        if self.dim == 2:
            self._plot_trajectory_2d(L=L, ax=ax, which_dims=which_dims, title=title)
            # self._plot_trajectory_2d(T, alpha, mesh, L, ax=ax, which_dims=which_dims, title=title)
        elif self.dim == 3:
            self._plot_trajectory_3d(T, alpha, mesh, L, min_dims, max_dims, ax=ax, which_dims=which_dims, title=title)


    def params_str(self, s=''):
        """
        Returns a string representation of the circuit's parameters
        """
        if self.eq_string:
            s = s + self.eq_string % tuple(self.params)
        else:
            s = (','.join(np.round(np.array(self.params), 3).astype(str))) if s == '' else s
        return s


    def plot_vector_field(self, which_dims=[0, 1], ax=None, min_dims=None, max_dims=None, num_lattice=None, title=''):
        """
        Plot the vector field induced on a lattice
        """
        if ax is None:
            fig, ax = plt.subplots(1)
        L = self.generate_mesh(min_dims=min_dims, max_dims=max_dims, num_lattice=num_lattice).to(self.device)
        flow = self.forward(0, torch.tensor(L).detach().float()).detach().numpy()
        xdim, ydim = which_dims

        coords = [np.squeeze(L[..., a]) for a in range(self.dim)]
        img = [np.squeeze(flow[..., a]) for a in range(self.dim)]
        ax.quiver(*coords, *img)

        ax.set_xlabel(self.labels[xdim])
        ax.set_ylabel(self.labels[ydim])
        ax.set_title(self.params_str(title))


    def get_info(self, exclude=exclude):
        """
        Return dictionary with the configuration of the circuit
        """
        data_info = self.__dict__
        data_info = {k: v for k, v in data_info.items() if not k.startswith('_')}

        for p in exclude:
            if p in data_info.keys():
                _ = data_info.pop(p)

        for k,v in data_info.items():
            if isinstance(v, torch.Tensor):
                data_info[k] = v.tolist()

        return data_info

    def fit_polynomial_representation(self, poly_order=None, fit_with='lstsq', return_rt=False,**kwargs):
        """
        Return the polynomial representations of the circuit
        Assumes two dimensions, x and y
        :param poly_order: order of the polynomial
        :param fit_with: method to fit the polynomial, options are 'lstsq' or 'lasso'
        :param kwargs: additional arguments to pass to the fitting method
        """
        if self.dim != 2:
            raise ValueError('Polynomial representation only implemented for 2D systems')
        poly_order = poly_order if poly_order else self.poly_order
        # defaults to least square fitting
        L = self.generate_mesh()
        z = self.forward(0, torch.tensor(L).float().reshape(-1, self.dim)).reshape(self.num_lattice, self.num_lattice, self.dim)
        zx = z[:,:,0].numpy().flatten()
        zy = z[:,:,1].numpy().flatten()

        library, library_terms = sindy_library(L.reshape(self.num_lattice**self.dim, self.dim), poly_order=poly_order)
        if fit_with == 'lstsq':
            start = time()
            mx = np.linalg.lstsq(library, zx, **kwargs)[0]
            my = np.linalg.lstsq(library, zy, **kwargs)[0]
            stop = time()
        elif fit_with == 'lasso':
            clf = linear_model.Lasso(**kwargs)
            start = time()
            clf.fit(library, zx)
            mx = clf.coef_
            clf.fit(library, zy)
            my = clf.coef_
            stop = time()
        else:
            raise ValueError('fit_with must be either "lstsq" or "lasso"')
        dx = pd.DataFrame(data={lb:m.astype(np.float64) for lb,m in  zip(library_terms,mx)}, index=[self.__class__.__name__])
        dy = pd.DataFrame(data={lb:m.astype(np.float64) for lb,m in  zip(library_terms,my)}, index=[self.__class__.__name__])
        if return_rt:
            return dx, dy, stop - start
        else:
            return dx, dy


    def get_polynomial_representation(self):
        """
        Return the polynomial representations of the circuit
        Assumes two dimensions, x and y
        """
        # TODO: if not configured, tried computing with fitting method
        dx = pd.DataFrame(columns=self.polynomial_terms)
        dy = pd.DataFrame(columns=self.polynomial_terms)
        return dx, dy

    def curl(self, L):

        if self.dim > 3:
            raise ValueError('Curl is only defined for dim <=3.')
        J = vmap(jacrev(self.forward,argnums=1),in_dims=(None,0))
        Jx = J(torch.tensor(0.),L.reshape(-1,2))
        dFxdy = Jx[:,0,1]
        dFydx = Jx[:,1,0]

        if self.dim < 3:
            dFxdz = dFzdx = dFzdy = dFydz = torch.zeros_like(dFxdy)
        else:
            dFxdz = Jx[:,0,2]
            dFydz = Jx[:,1,2]
            dFzdx = Jx[:,2,0]
        return torch.stack([dFzdy - dFydz, dFxdz - dFzdx, dFydx - dFxdy]).movedim(1,0)


#    def forward(self, t, z):
#        
#        if self.bound_type == 'periodic':
#            periodic = []
#            for d in range(self.dim):
#
#                d_min = self.min_dims[d]
#                d_max = self.max_dims[d]
#                periodic.append(periodic_boundary(z[...,d],d_min,d_max))
#            z = torch.stack(periodic).permute(*range(len(z.shape))[1:],0)
#        zdot = self._forward(0,z)
#
#        if self.bound_type == 'soft_neumann':
#            return (zdot * torch.sigmoid(self.boundary_gain*(self.bound - torch.norm(z,2,dim=-1,keepdim=True)))).float()
#        else:
#            return zdot#.float().float()
#
# def periodic_boundary(z,z_min,z_max):
#     return ((z - z_min) % (z_max - z_min)) + z_min





class SaddleNode(FlowCircuitODE):
    """
    Saddle node bifurcation
        xdot = r + x^2
        ydot = -y
    """

    min_dims = [-1.,-1.]
    max_dims = [1.,1.]

    recommended_param_ranges=[[-1.,1.]]
    recommended_param_groups=[[[-1.,0.]], [[0.,1.]]]

    eq_string = r'$\dot{x}_0 = %.02f - x_0^2; \dot{x}_1 = -1$'
    short_name = 'sn'

    def forward(self, t, z, **kwargs):
        x = z[..., 0]
        y = z[..., 1]

        xdot = self.params[0] - x**2 
        ydot = -y 
        zdot = torch.cat([xdot.unsqueeze(-1), ydot.unsqueeze(-1)], dim=-1)
        return zdot#.float()
        #if self.bounded:
            #return (zdot * torch.sigmoid(self.boundary_gain*(self.bound - torch.norm(z,2,dim=-1,keepdim=True)))).float()
        #    return zdot#.float().float()
        #else:
        #    return zdot#.float().float()

    def get_polynomial_representation(self):
        dx = pd.DataFrame(0.0, index=['saddle_node'], columns=self.polynomial_terms)
        dy = pd.DataFrame(0.0, index=['saddle_node'], columns=self.polynomial_terms)
        params = [p.numpy() for p in self.params]
        dx.loc['saddle_node']['1'] = params[0]
        dx.loc['saddle_node']['$x_0^2$'] = -1
        dy.loc['saddle_node']['$x_1$'] = -1
        return dx, dy

class Pitchfork(FlowCircuitODE):
    """
    (Supercritical) pitchfork bifurcation:
        xdot = rx - x^3
        ydot = -y
    """
    min_dims=[-1.,-1.]
    max_dims=[1.,1.]
    recommended_param_ranges=[[-1.,1.]]
    recommended_param_groups=[[[-1.,0.]],[[0.,1.]]]
    eq_string = r'$\dot{x}_0 = %.02f x_0 - x_0^3; \dot{x}_1 = -1$'
    short_name = 'pf'

    def forward(self, t, z, **kwargs):
        x = z[..., 0]
        y = z[..., 1]

        xdot = self.params[0]*x - x**3
        ydot = -y
        zdot = torch.cat([xdot.unsqueeze(-1), ydot.unsqueeze(-1)], dim=-1)
        return zdot#.float()

    def get_polynomial_representation(self):
        dx = pd.DataFrame(0.0, index=['pitchfork'], columns=self.polynomial_terms)
        dy = pd.DataFrame(0.0, index=['pitchfork'], columns=self.polynomial_terms)
        params = [p.numpy() for p in self.params]
        dx.loc['pitchfork']['$x_0$'] = params[0]
        dx.loc['pitchfork']['$x_0^3$'] = -1
        dy.loc['pitchfork']['$x_1$'] = -1
        return dx, dy

class Homoclinic(FlowCircuitODE):
    """
    Homoclinic (saddle-loop) bifurcation:
        xdot = y
        ydot = mu * y + x - x^2 + xy
    """

    #min_dims = [-2,-2]
    #max_dims = [2,2]
    min_dims = [-1,-1]
    max_dims = [1,1]
    recommended_param_ranges=[[-1.2,-.5]]
    recommended_param_groups = [
                                   [[-1.2, -.8645]],[[-.8645,-0.5]]   
                               ]

    short_name='hc'

    eq_string = r'$\dot{x}_0 = x_1; \dot{x}_1=%.02f x_1 + x_0 - x_0^2 + x_0x_1$'

    def forward(self, t, z, **kwargs):
        x = z[..., 0]
        y = z[..., 1]
        a = self.params[0]

        #xdot = y
        #ydot = self.params[0]*y + x - x**2 + x*y
        v = 2 * x
        w = 2 * y
        xdot = w
        ydot = self.params[0]*w + v -v**2 + v*w
       
        zdot = torch.cat([xdot.unsqueeze(-1), ydot.unsqueeze(-1)], dim=-1)
        return zdot#.float()

    def get_polynomial_representation(self):
        dx = pd.DataFrame(0.0, index=['homoclinic'], columns=self.polynomial_terms)
        dy = pd.DataFrame(0.0, index=['homoclinic'], columns=self.polynomial_terms)
        params = [p.numpy() for p in self.params]
        dx.loc['homoclinic']['$x_1$'] = 2
        dy.loc['homoclinic']['$x_1$'] = 2*params[0]
        dy.loc['homoclinic']['$x_0$'] = 2.
        dy.loc['homoclinic']['$x_0^2$'] = -4.
        dy.loc['homoclinic']['$x_0x_1$'] = 4.
        return dx, dy



class Transcritical(FlowCircuitODE):
    """
    (Supercritical) pitchfork bifurcation:
        xdot = rx - x^3
        ydot = -y
    """

    min_dims=[-1.,-1.]
    max_dims=[1.,1.]
    recommended_param_ranges=[[-1.,1.]]
    recommended_param_groups=[[[-1.,0.]],[[0.,1.]]]
    

    short_name='tc'

    eq_string = r'$\dot{x}_0 = %.02f x_0 - x_0^2; \dot{x}_1 = -1$'

    def forward(self, t, z, **kwargs):
        x = z[..., 0]
        y = z[..., 1]

        xdot = self.params[0]*x - x**2
        ydot = -y
        zdot = torch.cat([xdot.unsqueeze(-1), ydot.unsqueeze(-1)], dim=-1)
        return zdot#.float()

    def get_polynomial_representation(self):
        dx = pd.DataFrame(0.0, index=['transcritical'], columns=self.polynomial_terms)
        dy = pd.DataFrame(0.0, index=['transcritical'], columns=self.polynomial_terms)
        params = [p.numpy() for p in self.params]
        dx.loc['transcritical']['$x_0$'] = params[0]
        dx.loc['transcritical']['$x_0^2$'] = -1
        dy.loc['transcritical']['$x_1$'] = -1
        return dx, dy

class SimpleOscillator(FlowCircuitODE):
    """
    Simple oscillator:

        rdot = r * (a - r^2)
        xdot = rdot * cos(theta) - r * sin(theta)
        ydot = rdot * sin(theta) + r * cos(theta)

    Where:
        - a - angular velocity
        - r - the radius of (x,y)
        - theta - the angle of (x,y)
    """

    min_dims = [-1.,-1.]
    max_dims = [1.,1.]
    recommended_param_ranges=[[-1.,1.]]
    recommended_param_groups=[[[-1.,0.]],[[0.,1.]]]
    eq_string = r'$\dot{x} = r(%.02f - r^2); \dot{\theta} = -1$'
    short_name='so'

    def forward(self, t, z, **kwargs):
        x = z[..., 0]
        y = z[..., 1]

        r = torch.sqrt(x**2 + y**2)
        theta = torch.atan2(y,  x)
        rdot = r * (self.params[0] - r ** 2)
        # thetadot = torch.tensor(-1.)

        xdot = torch.cos(theta) * rdot - r * torch.sin(theta)
        ydot = torch.sin(theta) * rdot + r * torch.cos(theta)
        zdot = torch.cat([xdot.unsqueeze(-1), ydot.unsqueeze(-1)], dim=-1)
        return zdot#.float()

class NoisySimpleOscillator(FlowCircuitODE):
    """
    Noisy Simple oscillator:

        rdot = r * (a - r^2)
        xdot = rdot * cos(theta) - r * sin(theta)
        ydot = rdot * sin(theta) + r * cos(theta)

    Where:
        - a - angular velocity
        - r - the radius of (x,y)
        - theta - the angle of (x,y)
    """

    def forward(self, t, z, **kwargs):
        x = z[..., 0]
        y = z[..., 1]
        w = z[...,2:]

        r = torch.sqrt(x**2 + y**2)
        theta = torch.atan2(y,  x)
        rdot = r * (self.params[0] - r ** 2)
        # thetadot = torch.tensor(-1.)

        xdot = torch.cos(theta) * rdot - r * torch.sin(theta)
        ydot = torch.sin(theta) * rdot + r * torch.cos(theta)
        wdot = torch.normal
        zdot = torch.cat([xdot.unsqueeze(-1), ydot.unsqueeze(-1)], dim=-1)
        return zdot#.float()

class SelkovOscillator(FlowCircuitODE):
    """
    Selkov oscillator:
        xdot = -x + ay + x^2y
        ydot = b - ay - x^2y

    """
    #min_dims = [0,0]
    #max_dims = [3.,3.]
    min_dims = [-1.,-1.]
    max_dims = [1.,1.]

    recommended_param_ranges=[[.1,.11],[.2,1.0]]
    
    recommended_param_groups=[
                                [[.1,.11],[.2,.4,]],[[.1,.11],[.4,1.0]]
                             ]
    eq_string=r'$\dot{x}_0 = -x_0 + {0:%.02f} x_1 + x_0^2 x_1; \dot{x}_1= {1:%.02f} - {0:%.02f} x_1 - x_0x_1$'
    short_name='sl'

    def forward(self, t, z, **kwargs):
        x = z[...,0]
        y = z[...,1]

        a, b = self.params
        v = (x + 1) * 1.5
        w = (y + 1) * 1.5
        xdot = -v + a*w + v**2*w
        ydot = b - a*w - v**2*w
        zdot = torch.cat([xdot.unsqueeze(-1), ydot.unsqueeze(-1)], dim=-1)
        return zdot#.float()

    def get_polynomial_representation(self):
        dx = pd.DataFrame(0.0, index=['selkov'], columns=self.polynomial_terms)
        dy = pd.DataFrame(0.0, index=['selkov'], columns=self.polynomial_terms)
        params = [p.numpy() for p in self.params]
        dx.loc['selkov']['1'] = 1.5 * (params[0]) + 1.875
        dx.loc['selkov']['$x_0$'] = 5.25
        dx.loc['selkov']['$x_1$'] = 1.5 * params[0] + 3.375
        dx.loc['selkov']['$x_0^2$'] = 3.375
        dx.loc['selkov']['$x_0x_1$'] = 6.75
        dx.loc['selkov']['$x_0^2x_1$'] = 3.375
        dy.loc['selkov']['1'] = -1.5 * params[0] + params[1] - 3.375
        dy.loc['selkov']['$x_0$'] = -6.75
        dy.loc['selkov']['$x_1$'] = -1.5 * params[0] - 3.375
        dy.loc['selkov']['$x_0^2$'] = -3.375
        dy.loc['selkov']['$x_0x_1$'] = -6.75 
        dy.loc['selkov']['$x_0^2x_1$'] = -3.375

        return dx, dy

class VanDerPolOscillator(FlowCircuitODE):

    #min_dims = [-6.5, -6.5]
    #max_dims = [6.5, 6.5]
    min_dims = [-1., -1.]
    max_dims = [1., 1.]

    recommended_param_ranges=[[.005,.05]]
    recommended_param_groups=[[[.005,.05]]]
    eq_string=r'$\dot{x}_0=x_1; \dot{x}_1 = %.02f (1-x_0^2)x_1 - x_0$'
    short_name='vp'

    def forward(self, t, z, **kwargs):

        x = z[...,0]
        y = z[...,1]

        mu = self.params[0]
        #xdot = y
        #ydot = mu * (1-x**2) * y - x
        v = 6.5*x
        w = 6.5*y
        xdot = w
        ydot = mu * (1-v**2) * w - v
        
        zdot = torch.cat([xdot.unsqueeze(-1), ydot.unsqueeze(-1)], dim=-1)
        return zdot#.float()

    def get_polynomial_representation(self):
        dx = pd.DataFrame(0.0, index=['vanderpol'], columns=self.polynomial_terms)
        dy = pd.DataFrame(0.0, index=['vanderpol'], columns=self.polynomial_terms)
        params = [p.numpy() for p in self.params]
        dx.loc['vanderpol']['$x_1$'] = 6.5
        dy.loc['vanderpol']['$x_1$'] = 6.5*params[0]
        dy.loc['vanderpol']['$x_0^2x_1$'] = -274.625 * params[0]
        dy.loc['vanderpol']['$x_0$'] = -6.5 
        return dx, dy

class AlonCircuit(FlowCircuitODE):
    """
    One cell type, one ligand circuit from Hart et al. PNAS 2012 (Eq. 7-8):

        Xdot = (beta(c) - alpha(c)) * X
        cdot = beta3 + beta2 * X - alpha0 * f(c) * X - gamma * c

    Where:
        - X - cells
        - c - ligand quantity
        - beta - X proliferation rate as a function of c
        - alpha - X removal rate as a function of c
        - f - c uptake rate as a function of X
        - gamma - c degradation rate as a function of its own abundance
    """
    
    labels = ['X', 'c']
    n_params = 6
    params = torch.ones(n_params)

#    min_dims = [0.0, 0.0]
#    max_dims = [20.0, 10.0]
    min_dims = [-1.,-1.]
    max_dims = [1.,1.]

    recommended_param_ranges = [[0, 1]] * n_params
    recommended_param_groups = [recommended_param_ranges]
    eq_string = r'$\dot{x}_0=(%.02f \frac{3 * x_1^2}{x^2 + 1} - %.02f x_1) * x_0; \dot{x}_1=%.02f + %.02f x_0 - %.02f \frac{x_1^2}{x^2 + 1} * x_0 - %.02f x_1'
    short_name='al'

    def __init__(self, params=params, alpha0=1.0, beta2=1.0, beta3=1.0,
                 gamma=5.0, labels=labels, min_dims=min_dims, max_dims=max_dims, **kwargs):
            
        super().__init__(params=params, labels=labels, min_dims=min_dims, max_dims=max_dims, **kwargs)

        if len(self.params) != AlonCircuit.n_params:
            ValueError('Alon Circuit requires %d params' % AlonCircuit.n_params)
        if np.any(np.array(min_dims) < 0):
            ValueError('Ligand and cell quantities are non-negative (min_dims is negative)')
        self.alpha0 = alpha0
        self.beta2 = beta2
        self.beta3 = beta3
        self.gamma = gamma

    def proliferation(self, c):
        return 3.0 * c ** 2 / (c ** 2 + 1)

    def removal(self, c):
        return c

    def uptake(self, c):
        return c ** 2 / (c ** 2 + 1)

    def forward(self, t, z, **kwargs):
        v = z[..., 0]
        w = z[..., 1]

        X = (v + 1) * 10
        c = (v + 1) * 5

        Xdot = (self.params[0] * self.proliferation(c) - self.params[1] * self.removal(c)) * X
        cdot = self.params[2] * self.beta3 + \
               self.params[3] * self.beta2 * X - \
               self.params[4] * self.alpha0 * self.uptake(c) * X - \
               self.params[5] * self.gamma * c
        zdot = torch.cat([Xdot.unsqueeze(-1), cdot.unsqueeze(-1)], dim=-1)
        return zdot#.float()

class LotkaVolterra(FlowCircuitODE):
    labels = ['rabbit', 'lynx']
    n_params = 1
    min_dims = [-1.,-1.]
    max_dims = [1.,1.]

    recommended_param_ranges = [[0.,2.]] 
    recommended_param_groups = [[[0.,1.]], [[1.,2.]]]

    def forward(self, t, z, **kwargs):
        x = z[...,0]
        y = z[...,1]

        mu = self.params[0]

        v = 12*(x + 1)
        w = 6*(y + 1)
        xdot = v * (1-w)
        ydot = mu * w * (v  -1)

        zdot = torch.cat([xdot.unsqueeze(-1), ydot.unsqueeze(-1)], dim=-1)
        return zdot
    
    def get_polynomial_representation(self):
        dx = pd.DataFrame(0.0, index=['lotkavolterra'], columns=self.polynomial_terms)
        dy = pd.DataFrame(0.0, index=['lotkavolterra'], columns=self.polynomial_terms)
        params = self.params.numpy()
        dx.loc['lotkavolterra']['$x_0$'] = 1.
        dx.loc['lotkavolterra']['$x_0x_1$'] = -1.
        dy.loc['lotkavolterra']['$x_1$'] = -1*params[0]
        dy.loc['lotkavolterra']['$x_0x_1$'] = params[0]
        return dx, dy
        

class FitzHughNagumo(FlowCircuitODE):
    """
    Fitzhugh Nagumo model (a prototype of an excitable system, e.g. neuron)

        vdot = v - v^3 / 3. - w + I
        wdot = v + a - b * w

    Where:
        - I - external stimulus
    """

    labels = ['v', 'w'] # TODO: can the voltage be negative???
    n_params = 4
    params = torch.zeros(n_params) #tensor([0, 0, 0, 0])
    #min_dims = [-3.0, -3.0]
    #max_dims = [3.0, 3.0]
    min_dims = [-1.,-1.]
    max_dims = [1.,1.]

    recommended_param_ranges = [[0,.7],[12.5,12.6],[.7,.8],[.8,.9]] 
    recommended_param_groups = [
                                   [[0,.35],[12.5,12.6],[.7,.8],[.8,.9]] ,[[.35,.7],[12.5,12.6],[.7,.8],[.8,.9]] 
                               ]
    eq_string=r'$\dot{x}_0 = %.02f + x_0  - x_1 - x_0^3/3; \dot{x}_1 = \frac{1}{%.02f}(x_0 + %.02f - %.02f x_1)$'
    short_name='fn'

    def __init__(self, params=params, labels=labels, min_dims=min_dims, max_dims=max_dims, **kwargs):
        super().__init__(params, labels, min_dims=min_dims, max_dims=max_dims, **kwargs)

    def forward(self, t, z, **kwargs):
        v = z[..., 0]
        w = z[..., 1]

        I = self.params[0]
        tau = self.params[1]
        a = self.params[2]
        b = self.params[3]

        x = v * 3
        y = w * 3
        vdot = x - x**3 / 3. - y + I
        wdot = (1. / tau) * (x + a - b*y)
        zdot = torch.cat([vdot.unsqueeze(-1), wdot.unsqueeze(-1)], dim=-1)
        return zdot#.float()

    def get_polynomial_representation(self):
        """
        Considering v as x and w as y, return the polynomial representation of the system.
        """
        dx = pd.DataFrame(0.0, index=['fitzhughnagumo'], columns=self.polynomial_terms)
        dy = pd.DataFrame(0.0, index=['fitzhughnagumo'], columns=self.polynomial_terms)
        params = self.params.numpy()
        dx.loc['fitzhughnagumo']['1'] = params[0]
        dx.loc['fitzhughnagumo']['$x_0$'] = 3.
        dx.loc['fitzhughnagumo']['$x_1$'] = -3.
        dx.loc['fitzhughnagumo']['$x_0^3$'] = -9.
        dy.loc['fitzhughnagumo']['$x_0$'] = (1/params[1]) * 3
        dy.loc['fitzhughnagumo']['1'] = (1 / params[1]) * params[2]
        dy.loc['fitzhughnagumo']['$x_1$'] = (1/params[1]) * (-3*params[2])
        return dx, dy

class HodgkinHuxley(FlowCircuitODE): # TODO: not working
    """
    Hodgkin Huxley modelling initiation and propagation of action potential across neurons:

        Vmdot = (-1. / Cm) * (gK * n ** 4 * (Vm - Vk) + gNa * m ** 3 * h * (Vm - VNa) + gl * (Vm - Vl) - I)
        ndot  = alpha_n * (1 - n) - beta_n * n
        mdot  = alpha_m * (1 - m) - beta_m * m
        hdot  = alpha_h * (1 - h) - beta_n * h

    Where:
        - Cm - membrane capacity
        - gK,gNa,gl - conductances
        - I - current per unit area
        - Vk, VNa, Vl - resting potentials
        - Vm - membrane potential
    """
    labels = ['Vm', 'n', 'm', 'h']  # TODO: not sure about order
    n_params = 11
    params = torch.ones(n_params, )

    num_lattice = 10
    dim = 4
    min_dims = [-80., 0., 0., 0.]
    max_dims = [20., 1., 1., 1.]

    recommended_param_ranges = [[0, 1]] * n_params

    def __init__(self, params=params, labels=labels, num_lattice=num_lattice, min_dims=min_dims, max_dims=max_dims, **kwargs):
        super().__init__(params, labels, num_lattice=num_lattice, min_dims=min_dims, max_dims=max_dims, **kwargs)

    # def rate_constant(self, v, a, b, c, d, e):
    #     return (a*v + b) / (torch.exp((d - v) / e) + c)

    def alpha_m_rate(self, V):
        return 0.1 * (V + 40.0) / (1.0 - torch.exp(-(V + 40.0) / 10.0))

    def beta_m_rate(self, V):
        return 4.0 * torch.exp(-(V + 65.0) / 18.0)

    def alpha_h_rate(self, V):
        return 0.07 * torch.exp(-(V + 65.0) / 20.0)

    def beta_h_rate(self, V):
        return 1.0 / (1.0 + torch.exp(-(V + 35.0) / 10.0))

    def alpha_n_rate(self, V):
        return 0.01 * (V + 55.0) / (1.0 - torch.exp(-(V + 55.0) / 10.0))

    def beta_n_rate(self, V):
        return 0.125 * torch.exp(-(V + 65) / 80.0)


    def forward(self, t, z, **kwargs):

        # State variables
        Vm = z[..., 0]
        n  = z[..., 1]
        m  = z[..., 2]
        h  = z[..., 3]

        # Parameters
        I         = self.params[0]
        c_Cm        = self.params[1]
        c_gK        = self.params[2]
        c_gNa       = self.params[3]
        c_gl        = self.params[4]
        c_alpha_n = self.params[5]
        c_beta_n  = self.params[6]
        c_alpha_m = self.params[7]
        c_beta_m  = self.params[8]
        c_alpha_h = self.params[9]
        c_beta_h  = self.params[10]

        # Resting potentials
        Vk  = -77
        VNa = 50
        Vl  = -54

        # Membrane capacitance
        Cm = c_Cm * 1.0

        # Conductances
        gK = c_gK * 120
        gNa = c_gNa * 36
        gl = c_gl * .3


        # Rate constants
        alpha_n = c_alpha_n * self.alpha_n_rate(Vm) #self.rate_constant(Vm, -.01, .1, -1, 10., 10.)
        beta_n  = c_beta_n * self.beta_n_rate(Vm) #self.rate_constant(Vm, 0., .125, 0., 0., 80.)
        alpha_m = c_alpha_m * self.alpha_m_rate(Vm) #self.rate_constant(Vm, -.1, 2.5, -1., 25, 10.)
        beta_m  = c_beta_m * self.beta_m_rate(Vm) #self.rate_constant(Vm, 0., 4., 0., 0., 18.,)
        alpha_h = c_alpha_h * self.alpha_h_rate(Vm) #self.rate_constant(Vm, 0., .07, 0., 0., 20.)
        beta_h  = c_beta_h * self.beta_h_rate(Vm) #self.rate_constant(Vm, 0., 1., 1., 30., 10.)

        Vmdot = (-1. / Cm) * (gK * n ** 4 * (Vm - Vk) + gNa * m ** 3 * h * (Vm - VNa) + gl * (Vm - Vl) - I)
        ndot  = alpha_n * (1 - n) - beta_n * n
        mdot  = alpha_m * (1 - m) - beta_m * m
        hdot  = alpha_h * (1 - h) - beta_n * h

        zdot = torch.cat([Vmdot.unsqueeze(-1), ndot.unsqueeze(-1), mdot.unsqueeze(-1), hdot.unsqueeze(-1)], dim=-1)
        return zdot#.float()
#        if self.bounded:
#            return (zdot * torch.sigmoid(self.boundary_gain*(self.bound - torch.norm(z,2,dim=-1,keepdim=True)))).float()
#        else:
#            return zdot#.float()
#

class Kuramoto(FlowCircuitODE):
    """
    System of oscillators, obtaining an intrinsic intrinsic and coupled frequencies:
    Default is Kuramoto of two oscillators.
    """

    # params = torch.ones(( 30 )) #TODO: why 30?
    dim = n_oscillators = 2
    labels = [r'$\theta_%d$' % i for i in range(n_oscillators)]

    min_dims = n_oscillators * [0.0]
    max_dims = n_oscillators * [2 * math.pi]

    #TODO: learning only couplings
    n_params = (n_oscillators ** 2) + n_oscillators
    recommended_param_ranges = [[0, 1]] * n_params

    def set_n_oscillators(self, n_oscillators=None, labels=None, min_dims=None, max_dims=None, params=None):
        self.dim = self.n_oscillators = n_oscillators
        self.labels = [r'$\theta_%d$' % i for i in range(n_oscillators)] if labels is None else labels
        self.min_dims = n_oscillators * [0.0] if min_dims is None else min_dims
        self.max_dims = n_oscillators * [2 * math.pi] if max_dims is None else max_dims

        num_couplings = self.n_oscillators ** 2
        num_units = self.n_oscillators

        self.n_params = (n_oscillators ** 2) + n_oscillators

        if params is not None:
            if len(params) != (num_couplings + num_units):
                raise ValueError('Input params should be of length n_oscillators**2 + n_oscillators. '
                                 'Currently configured to {} oscillators (expecting {} params), but given {} parameters'.format(self.n_oscillators, self.n_params, len(params)))
            self.params = params[:num_couplings]
            self.omega = params[-num_units:]
        else:
            self.params = torch.ones(num_couplings)
            self.omega = torch.ones(num_units)


        # num_couplings = int(np.sqrt(params.shape[0]))**2
        # num_units = params.shape[0] - num_couplings
        # self.params = self.params[:num_couplings]
        # self.params = torch.nn.Parameter(params[:num_couplings], requires_grad=True) if train else params[:num_couplings]
        # self.params = self.params.float()

    def __init__(self, params=None, labels=None, min_dims=None, max_dims=None, n_oscillators=2, **kwargs):
        # self.set_n_oscillators(n_oscillators, labels, min_dims, max_dims, params)
        # TODO: constructor MUST be called first, need to fix!
        super().__init__(params=self.params, labels=self.labels, min_dims=self.min_dims, max_dims=self.max_dims, **kwargs)

        # num_couplings = int(np.sqrt(params.shape[0]))**2
        # num_units = params.shape[0] - num_couplings
        # self.params = self.params[:num_couplings]
        # self.params = torch.nn.Parameter(params[:num_couplings], requires_grad=True) if train else params[:num_couplings]
        # self.params = self.params.float()


    def run(self, T, **kwargs):
        self.omega = torch.randn(self.n_oscillators)*.5
        return super().run(T, **kwargs)
    
    # def plot_trajectory(self, T, min_dims=min_dims, max_dims=max_dims, **kwargs):
    #     super().plot_trajectory(T, min_dims=min_dims, max_dims=max_dims, **kwargs)

    def forward(self, t, z, **kwargs):
        z = z % (2 * math.pi)
        z = z.unsqueeze(-2)
        perm_dims = list(range(z.ndim - 2)) + [z.ndim-1, z.ndim-2]
        phase_diffs = torch.sin(z.permute(*perm_dims) - z)
        couplings = torch.tensor(self.params.reshape(self.n_oscillators, self.n_oscillators)) # TODO: added tensor (?)
        zdot = (couplings * phase_diffs).sum(-2) + self.omega.to(self.device)
        return zdot#.float()

    def get_info(self, exclude=[]):
        """

        """
        exclude = exclude + ['omega'] + super().exclude
        return super().get_info(exclude)


class Lorenz(FlowCircuitODE):
    """
    Lorenz system of equations, prone to chaotic behavior:

        xdot = sigma * (y - x)
        ydot = x * (rho - z) - y
        zdot = x * y - beta * z
    """
    labels = ['u', 'v', 'w']
    n_params = dim = 3
    params = torch.tensor([0] * dim)

    min_dims = [-10.0, -20.0, -5.0]
    max_dims = [20.0, 20.0, 40.0]

    recommended_param_ranges = [[0, 1]] * n_params

    def __init__(self, params=params, labels=labels, min_dims=min_dims, max_dims=max_dims, **kwargs):
        super().__init__(params, labels, min_dims=min_dims, max_dims=max_dims, **kwargs)

    # TODO: check where x_min=-10.0 was integrated here
    # def plot_trajectory(self, T, min_dims=min_dims, max_dims=max_dims, **kwargs):
    #     super().plot_trajectory(T, min_dims=min_dims, max_dims=max_dims, **kwargs)

    def forward(self, t, q, **kwargs):
        x = q[..., 0]
        y = q[..., 1]
        z = q[..., 2]
 
        sigma = self.params[0]
        rho   = self.params[1]
        beta  = self.params[2]

        xdot = sigma * (y - x)
        ydot = x * (rho - z) - y
        zdot = x * y - beta * z

        qdot = torch.cat([xdot.unsqueeze(-1), ydot.unsqueeze(-1), zdot.unsqueeze(-1)],dim=-1)
        if self.bounded:
            return (qdot * torch.sigmoid(self.boundary_gain*(self.bound - torch.norm(q,1,dim=-1,keepdim=True)))).float()
        else:
            return qdot.float()

class Linear(FlowCircuitODE):

    min_dims = [-1.,-1.]
    max_dims = [1., 1.]
    recommended_param_ranges = 4 * [[-1.,1.]]

    recommended_param_groups = [recommended_param_ranges]

    def forward(self, t, z, **kwargs):

        A = torch.tensor([[self.params[0], self.params[1]],[self.params[2], self.params[3]]])
        return torch.einsum('ij,...j', A,z)

    def get_polynomial_representation(self):
        dx = pd.DataFrame(0.0, index=['linear'], columns=self.polynomial_terms)
        dy = pd.DataFrame(0.0, index=['linear'], columns=self.polynomial_terms)
        params = [p.numpy() for p in self.params]
        dx.loc['linear']['$x_0$'] = params[0]
        dx.loc['linear']['$x_1$'] = params[1]
        dy.loc['linear']['$x_0$'] = params[2]
        dy.loc['linear']['$x_1$'] = params[3]
        return dx, dy

class Conservative(FlowCircuitODE):
 
    recommended_param_ranges = 10*[[-3., 3.]]
    recommended_param_groups = [recommended_param_ranges]
 
    def __init__(self, params=None, labels=['x', 'y'], min_dims=[-1.0,-1.0], max_dims=[1.0,1.0], poly_order=3, include_sine=False, include_exp=False, **kwargs):
        super().__init__(params, labels, min_dims=min_dims, max_dims=max_dims, **kwargs)

        self.poly_order = int(poly_order)
        L = self.generate_mesh(min_dims=self.min_dims, max_dims=self.max_dims, num_lattice=self.num_lattice)
        self.library, self.library_terms = sindy_library(L.reshape(self.num_lattice**self.dim, self.dim), self.poly_order,
                                      include_sine=include_sine, include_exp=include_exp)
        self.params = torch.tensor(params).float()
    def forward(self, t, z, **kwargs):
        phi = torch.einsum('sl,l->s', self.library, self.params).reshape(self.num_lattice, self.num_lattice) # generate scalar field
        zdot = torch.stack([dphi for dphi in torch.gradient(phi,dim=[0,1])]).permute(1,2,0)
        return zdot#.float()

class Incompressible(FlowCircuitODE):
    min_dims = [-1.,-1.]
    max_dims = [1., 1.]
    
    recommended_param_ranges = 8*[[-3., 3.]]
    recommended_param_groups = [recommended_param_ranges]
 
    def __init__(self, params=None, labels=['x', 'y'], min_dims=[-1.0,-1.0], max_dims=[1.0,1.0], poly_order=3, include_sine=False, include_exp=False, **kwargs):
        super().__init__(params, labels, min_dims=min_dims, max_dims=max_dims, **kwargs)
        self.poly_order = int(poly_order)
        L = self.generate_mesh(min_dims=self.min_dims, max_dims=self.max_dims, num_lattice=self.num_lattice)
        L = torch.flip(L,(-1,))
        self.im_unit = torch.sqrt(torch.tensor(-1).cfloat())

        cL = L[...,0].cfloat() + self.im_unit * L[...,1].cfloat()
        self.library, self.library_terms = sindy_library(cL.reshape(self.num_lattice**self.dim, 1), self.poly_order,
                                      include_sine=include_sine, include_exp=include_exp)
        self.library = self.library.cfloat()
        self.params = torch.tensor(params).float()
    def forward(self, t, z, **kwargs):
        
        # Convert params to complex
        params = self.params.reshape(-1,self.dim)
        params = (params[...,0].cfloat() + self.im_unit * params[...,1].cfloat()).unsqueeze(-1)
        f_z = torch.einsum('sl,ld->sd', self.library, params).reshape(self.num_lattice, self.num_lattice)
        phi = torch.real(f_z)
        psi = torch.imag(f_z)
        [dphi_dx,dphi_dy] = torch.gradient(phi,dim=[0,1])
        [dpsi_dx,dpsi_dy] = torch.gradient(psi,dim=[0,1])
        zdot = torch.cat([dphi_dx.unsqueeze(-1), dphi_dy.unsqueeze(-1)],dim=-1)
        return zdot

class Polynomial(FlowCircuitODE):
    """
    Polynomial system of equations up to poly_order. e.g., for poly_order=2, the system is:
        xdot = 
        ydot = -y
    """
    min_dims = [-1.,-1.]
    max_dims = [1., 1.]
    
    recommended_param_ranges = 20*[[-3., 3.]]
    recommended_param_groups = [recommended_param_ranges]
 
    # TODO: handle defaults?
    def __init__(self, params=None, labels=['x', 'y'], min_dims=[-1.0,-1.0], max_dims=[1.0,1.0], poly_order=3, include_sine=False, include_exp=False, **kwargs):
        """
        Initialize the polynomial system.
        :param params: correspond to library terms for dim1 concatenated with library terms for dim2 ()
        :param poly_order: the order of the polynomial system
        """
        super().__init__(params, labels, min_dims=min_dims, max_dims=max_dims, **kwargs)

        self.poly_order = int(poly_order)
        L = self.generate_mesh(min_dims=self.min_dims, max_dims=self.max_dims, num_lattice=self.num_lattice)
        self.library, self.library_terms = sindy_library(L.reshape(self.num_lattice**self.dim, self.dim), self.poly_order,
                                      include_sine=include_sine, include_exp=include_exp)
        self.library = self.library.float()
        self.params = params

    def forward(self, t, z, **kwargs):
        params = self.params.reshape(-1, self.dim)
        z_shape = z.shape
        z = z.reshape(-1,self.dim)
        library, _ = sindy_library(z, self.poly_order,
                                   include_sine=False, include_exp=False)
        
        zdot = torch.einsum('sl,ld->sd', library.to(params.device).float(), params.float())
        zdot = zdot.reshape(*z_shape)
    
        # xdot = data[:, 0]
        # ydot = data[:, 1]
        # zdot = torch.cat([xdot.unsqueeze(-1), ydot.unsqueeze(-1)], dim=-1)
        return zdot#.float()

    def params_str(self, s=''):
        """
        Sparse representation of the parameters.
        """
        nterms = len(self.library_terms)
        if (nterms * self.dim)!= len(self.params):
        # raise ValueError('Number of parameters does not match number of library terms.')
            return s
        eqs = []
        params = self.params.numpy()
        for a in range(self.dim):
            eq = r'$\dot{x_' + f'{a}' + '}' + '= $'
            first_term = True
            for lt, pr in zip(self.library_terms, params[a * nterms:(a + 1) * nterms]):
                if np.abs(pr) > 0:
                    sgn = np.sign(pr)
                    if first_term:
                        conj = '' if sgn > 0 else '-'
                    else:
                        conj = '+' if sgn > 0 else '-'
                    first_term = False
                    eq += f' {conj} {np.abs(pr):.3f}' + lt
            eqs.append(eq)
        
        return s + '\n'.join(eqs)

    def get_polynomial_representation(self):
        nterms = len(self.library_terms)
        dx = pd.DataFrame(0.0, index=['polynomial'], columns=self.library_terms) # should we keep these terms or set according to poly_order?
        dy = pd.DataFrame(0.0, index=['polynomial'], columns=self.library_terms)
        params = [p.numpy() for p in self.params]
        for ilb,lb in enumerate(self.library_terms):
            dx[lb] = params[ilb]
            dy[lb] = params[ilb + nterms]
        return dx, dy

class NeuralODE(FlowCircuitODE):
    params = {}
    labels = ['u','v']
    #min_dims = [-10.,-10.]
    #max_dims = [10.,10.]

    def __init__(self, params=params, labels=labels, min_dims=[-2.,-2.], max_dims=[2.,2.], squeeze=True, train=False, **kwargs):
        super().__init__(params, labels, min_dims=min_dims, max_dims=max_dims, **kwargs)

        self.squeeze = squeeze

    def forward(self,t,x):
        par_batch_size = self.params['W0'].shape[0]
        dim = self.params['W0'].shape[1]
        x = x.reshape(par_batch_size,-1,dim)
        for l, (key,par) in enumerate(self.params.items()):
            if key[0] == 'W':
                x = torch.einsum('pfg,pif->pig',par,x)
            elif key[0] == 'b':
                x = x + par.unsqueeze(1)
                if l < len(self.params) - 1:
                    x = F.relu(x)
        if self.squeeze:
            x = x / torch.norm(x,2,dim=-1,keepdim=True)
        return x.reshape(-1, dim)



# class GrayScottNetwork(FlowCircuitODE):
#     """
#     Reaction equations of morphogens (Scholes et al.):

#         F_i = V_i * P_i(x) + b_i - mu_i * x_i

#         Where:
#         - P_i = \prod_j 1 / (1 + k_ij/xj) ^ n_ij) - product over hill function
#         - n_ij - hill coefficient
#         - k_ij - connectivity (couplings)
#         - V_i - max value of morphogen
#         - b_i - const term
#         - mu_i -

#     """

#     labels = ['A', 'B']
#     params = np.array([ 100, # k11
#                         100, # k12
#                         100, # k21
#                         0, ])# k22
#     min_dims = [0.0, 0.0]
#     max_dims = [400.0, 400.0]


#     @staticmethod
#     def non_competitive(z, K, H):
#         H[K < 0] = -H[K < 0]
#         P = torch.prod(1. / (1. + (K / z.unsqueeze(-2)) ** H), dim=-1)
#         return P

#     # @staticmethod
#     # def competitive(z, K, H): # TODO: Not working :(
#     #     A = ( K / z.unsqueeze(-2)) ** (-H)
#     #     # A = (z.unsqueeze(-2) / K) ** H
#     #     denom = 1. + torch.sum(A, dim=-1)
#     #     K0 = K
#     #     K0[K < 0] = 0
#     #     B = (K0 / z.unsqueeze(-2)) ** (-H)
#     #     nom = torch.sum(B, dim=-1)
#     #     P = nom / denom
#     #     return P

#     activation = non_competitive


#     def set_n_morphogens(self, n_morphogens=2, labels=None, min_dims=None, max_dims=None, params=None):
#         self.dim = self.n_morphogens = n_morphogens
#         self.labels = list('ABC'[:n_morphogens]) if labels is None else labels
#         self.min_dims = n_morphogens * [0.0] if min_dims is None else min_dims
#         self.max_dims = n_morphogens * [400.0] if max_dims is None else max_dims
#         self.params = np.array([100, 0, 3.1623, 3.16230, 0, 0, 0.1, 0]) if params is None else params

#     # @staticmethod
#     # def set_competitive(to_competitive=True, **kwargs):
#     #     if to_competitive:
#     #         GrayScottNetwork.activation = GrayScottNetwork.competitive
#     #     else:
#     #         GrayScottNetwork.activation = GrayScottNetwork.non_competitive



#     def __init__(self, params=None, labels=None, min_dims=min_dims, max_dims=max_dims, n_morphogens=2, **kwargs):
#         self.set_n_morphogens(n_morphogens, labels, min_dims, max_dims, params)
#         super().__init__(self.params, self.labels, min_dims=self.min_dims, max_dims=self.max_dims, **kwargs)
#         # num_couplings = GrayScottNetwork.n_morphogens ** 2
#         # num_other = self.params.shape[0] - num_couplings
#         # self.params = self.params[:num_couplings]
#         # self.other_params = self.params[-num_other:]
#         self.N = torch.ones(self.n_morphogens, self.n_morphogens) * 2
#         self.V = torch.tensor([3.1623] * self.n_morphogens)
#         self.b = torch.tensor([0.1] * self.n_morphogens)
#         self.mu = torch.tensor([0.01] * self.n_morphogens)

#     def forward(self, t, z, **kwargs):
#         # TODO: send to function to handle 3 params
#         K = torch.tensor(self.params.reshape(self.n_morphogens, self.n_morphogens))

#         if K.shape != self.N.shape: # TODO: at the end, become unequal shape, not sure why
#             print('K and N of different shapes!')
#         P = GrayScottNetwork.activation(z, K, self.N)
#         zdot = self.V * P + self.b - self.mu * z

#         return zdot#.float()

# def stable_sigmoid(z):
#     return torch.where(z<0,torch.exp(z) / (1 + torch.exp(z)), 1. / (1 + torch.exp(-1*z)))


if __name__ == '__main__':
    
    # fig, ax = plt.subplots(1)
    dim = 2
    poly_order = 3
    params = np.random.uniform(low=-2, high=2, size=dim * library_size(dim, poly_order))
    # kwargs = {'device': 'cpu', 'poly_order': poly_order, 'params': params} # 'min_dims': [-2, -2], 'max_dims': [2, 2],
    # DE = Polynomial(**kwargs)

    params = [2]
    kwargs = {'device': 'cpu', 'params': params} # 'min_dims': [-2, -2], 'max_dims': [2, 2],
    DE = SaddleNode(**kwargs)
    
    dx,dy = DE.get_polynomial_representation()
    print(dx)
    print(dy)
    fig, ax = plt.subplots(1,1, figsize=(10,10))
    DE.plot_vector_field(ax=ax)#
    # # DE.run(T=50, alpha=0.1)
    # # DE.plot_trajectory(T=50, alpha=0.1)

    # # DE = FitzhughNagumo(**kwargs)
    # # DE.plot_trajectory(T=50, alpha=0.1, ax=ax[1])
    # plt.savefig('polynomial_vector_field.png')
    plt.show()
