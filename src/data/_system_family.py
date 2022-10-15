import os
import torch
import pickle
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from src.data._odes import *

class SystemFamily():
    """
    Family of ODE or PDE systems
    """

    @staticmethod
    def get_generator(data_name):
        """
        Selecting supported ODE or PDE generator
        """
        if data_name == 'alon':
            generator = AlonSystem

        elif data_name == 'lotka_volterra':
            generator = LotkaVolterra

        elif data_name == 'simple_oscillator':
            generator = SimpleOscillator

        elif data_name == 'vanderpol':
            generator = VanDerPolOscillator

        elif data_name == 'selkov':
            generator = SelkovOscillator

        elif data_name == 'saddle_node':
            generator = SaddleNode

        elif data_name == 'pitchfork':
            generator = Pitchfork

        elif data_name == 'transcritical':
            generator = Transcritical

        elif data_name == 'homoclinic':
            generator = Homoclinic

        elif (data_name == 'fitzhugh_nagumo') or (data_name == 'fn'):
            generator = FitzHughNagumo

        elif (data_name == 'hodgkin_huxley') or (data_name == 'hh'):
            generator = HodgkinHuxley

        elif data_name == 'lorenz':
            generator = Lorenz

        elif data_name == 'neural_ode':
            generator = NeuralODE

        elif data_name == 'polynomial':
            generator = Polynomial

        elif data_name == 'linear':
            generator = Linear

        elif data_name == 'conservative':
            generator = Conservative
        
        elif data_name == 'incompressible':
            generator = Incompressible

        # elif data_name == 'kuramoto': # TODO: fix constructor
        #     generator = Kuramoto
            # generator.set_n_oscillators(**kwargs)

        # elif data_name == 'grayscott':
        #     generator = GrayScott
        # elif data_name == 'grayscottnetwork': # TODO: commented out for now bc unclear (noa regularization?)
        #     generator = GrayScottNetwork
            # generator.set_n_morphogens(**kwargs)
            # generator.set_competitive(**kwargs)
        else:
            raise ValueError(f'Unknown data, `{data_name}`! Try `alon`.')

        return generator

    def __init__(self, data_name, data_dir=None, device=None, min_dims=None, max_dims=None, num_lattice=64, param_ranges=None, param_groups=None, default_sampler='uniform', **kwargs):
        """
        Generate a system family
        :param data_name: name of system
        :param param_ranges: range for each param in model, 
        :param data_dir: directory to save data
        :param device: device to use
        :param min_dims: minimum range of dimensions to use
        :param max_dims: maximum range of dimensions to use
        :param kwargs: any arguments of a general system
        """
        self.data_name = data_name
        self.pde = False
         
        DE = SystemFamily.get_generator(self.data_name)
        self.data_dir = os.path.abspath(data_dir) if data_dir is not None else '.'

        # if not provided, use ode suggested params
        self.param_ranges = DE.recommended_param_ranges if param_ranges is None else param_ranges
        self.param_groups = DE.recommended_param_groups if param_groups is None else param_groups
        self.min_dims = DE.min_dims if min_dims is None else min_dims
        self.max_dims = DE.max_dims if max_dims is None else max_dims

        # Num classes
        if self.data_name == 'linear':
            self.num_classes = 5
        elif self.data_name == 'polynomial':
            self.num_classes = 1
        else:
            self.num_classes = len(self.param_groups)

        # general DE params
        self.device = device
        params = self.params_random(1)
        DE_ex = DE(params=params[0], device=device, min_dims=self.min_dims, max_dims=self.max_dims, **kwargs)
        data_info = DE_ex.get_info()
        data_info = {**self.__dict__, **data_info} # merge dictionaries
        self.data_info = data_info
        self.num_lattice = num_lattice
        self.dim = DE_ex.dim
        self.DE = DE
        self.DE_ex = DE_ex
        if self.data_name != 'grayscott': # TODO: ask isinstance(generator, ODE/PDE)
            # TODO: can generator.param can be used as default?
            self.L = self.DE_ex.generate_mesh(num_lattice=num_lattice) # min_dims, max_dims, num_lattice
            self.L = self.L.to(device).float()

        self.kwargs = kwargs
        if default_sampler == 'uniform':
            self.param_sampler = self.params_random
        elif default_sampler == 'extreme':
            self.param_sampler = self.params_extreme
        elif default_sampler == 'sparse':
            self.param_sampler = self.params_sparse
        elif default_sampler == 'control':
            self.param_sampler = self.params_control
        else:
            raise ValueError('Param sampler not recognized.')
       
    def set_label(self, model, params, linear=False, polynomial=False):
        if linear:
            A = np.array([[params[0],params[1]],[params[2], params[3]]])
            tr = np.trace(A)
            det = np.linalg.det(A)
            delta = tr**2 - 4*det

            if det < 0:
                # Saddle
                label = 0
            elif det > 0:
                 if tr > 0:
                     if det < delta:
                         # Source
                         label = 1
                     elif det > delta:
                         # Spiral source
                         label = 2
                     else:
                         # Degenerate source
                         label = -1
                 elif tr < 0:
                     if det < delta:
                         # Sink
                         label = 3
                     elif det > delta:
                         # Spiral sink
                         label = 4
                     else:
                         # Degenerate sink
                         label = -1
                 else:
                     # Center
                     label=-1
            else:
                # Line
                label = -1
            model.label = label

        elif polynomial:
            model.label = 0
        else:
            for g, gp in enumerate(self.param_groups):
                if np.all([par >= gp[p][0] and par <= gp[p][1] for p, par in enumerate(params)]): 
                    model.label = g

    def generate_model(self, params, set_label=True, **kwargs):
        """
        Generate model with params
        """
        kwargs = {**self.data_info, **kwargs}  # merge dictionaries
        if self.data_name == 'grayscott':
            model = self.DE(self.num_lattice, params=torch.tensor(params), **kwargs)
        elif self.data_name == 'neural_ode':
            model = self.DE(params, **kwargs)
        else:
            model = self.DE(params=torch.tensor(params), **kwargs)
            if set_label:
                self.set_label(model, params, linear=(self.data_name=='linear'), polynomial=(self.data_name=='polynomial'))
        return model

    def generate_flow(self, params, **kwargs):
        """
        Generate flow over system with params
        """
        model = self.generate_model(params, **kwargs)
        if self.data_name != 'grayscott':
            flow = model.forward(0, torch.tensor(self.L).float())
        else:
            print('TODO')
            T = 10000  # 2000
            alpha = 100  # 1.
            flow = model.run(T, alpha=alpha, noise_magnitude=0.2)  # TODO: what is this noise magnitude?
            flow = flow[-1, ...].reshape(self.num_lattice, self.num_lattice, 2).detach().numpy()
        return flow


    def make_data(self, num_samples):
        """
        Generates data of system
        data_name - name of system
        param_ranges - range for each param in model
        """

        train_dir = os.path.abspath(os.path.join(self.data_dir, 'train', '0'))
        test_dir = os.path.abspath(os.path.join(self.data_dir, 'test', '0'))

        for dr in [train_dir, test_dir]:
            if not os.path.exists(dr):
                os.makedirs(dr)

        # all_params = []
        # for rg in self.param_ranges:
        #     all_params.append(np.random.rand(2*num_samples) * (rg[1] - rg[0]) + rg[0])
        # all_params = np.array(all_params)
        all_params = self.params_random(2*num_samples).T
        
        for dt, dr, nm in zip([all_params[:, :num_samples], all_params[:, num_samples:]], [train_dir, test_dir], ['Train', 'Test']):
            print('Generating {} Set'.format(nm))
            for d, params in tqdm(enumerate(dt.T)):
                #data = np.array([.1, .05, .0545, .062])

                flow = self.generate_flow(params)

                fn = os.path.join(dr,  'flow%05i.pkl' % d)
                with open(fn, 'wb') as f:
                    pickle.dump((flow, params), f)

        # # params set upon data generation
        # self.data_info['train_dir'] = train_dir
        # self.data_info['test_dir'] = test_dir
        # self.data_info['num_lattice'] = num_lattice
        # self.data_info['min_dims'] = min_dims
        # self.data_info['max_dims'] = max_dims

######################################## Param sampling methods ########################################################

    def params_random(self, num_samples):
        """
        Return random sampling of params in range
        """
        params = np.zeros((num_samples, len(self.param_ranges)))
        
        for i,p in enumerate(self.param_ranges):
            params[:, i] = np.random.uniform(low=p[0], high=p[1], size=num_samples)
        return params

    def params_extreme(self):
        """
        Return array of extreme (minimal, intermediate and maximal bounds) param combinations
        """
        spatial_coords = [torch.tensor([mn, np.mean([mn,mx]), mx]) for (mn, mx) in self.param_ranges]
        mesh = torch.meshgrid(*spatial_coords)
        params = torch.cat([ms[..., None] for ms in mesh], dim=-1)

        return params

    def params_sparse(self, num_samples, p=0.5):
        """
        Samples which parameters to set with Binomial distribution with probability p and then 
        randomly assigns them (for families of many parameters)
        """
        which_params = np.random.binomial(1, p, (num_samples, len(self.param_ranges)))
        params = self.params_random(num_samples)
        return (params * which_params)

    def params_control(self, num_samples, max_coeff=3., prop_zero=.6, prop_non_unit=.3):

        # proportion of unit parameters
        prop_unit = 1 - prop_zero - prop_non_unit

        # Initialize parameter vector
        num_terms = int(len(self.param_ranges) / self.dim)
        x =  (2*max_coeff) * torch.rand(num_terms, self.dim)

        # zero out `prop_zero` parameters
        coeffs = torch.where(x/(2*max_coeff) < prop_zero, torch.zeros_like(x), x)

        # Add 1 coeffs
        coeffs = torch.where((x/(2*max_coeff) >= prop_zero)*(x/(2*max_coeff)< (prop_zero + prop_unit/2)), torch.ones_like(coeffs), coeffs)

        # Add -1 coeffs
        coeffs = torch.where((x/(2*max_coeff) >=prop_zero + prop_unit/2)*(x/(2*max_coeff) < (prop_zero + prop_unit)), -1*torch.ones_like(coeffs), coeffs)

        # Add random coeffs
        coeffs = torch.where(x/(2*max_coeff)>prop_zero + prop_unit, (2*max_coeff) * torch.rand(num_terms, self.dim) - max_coeff, coeffs)

        # Are both equations identically 0?
        one_zero_eq = (coeffs.sum(0)[0] * coeffs.sum(0)[1] == 0)
        if one_zero_eq:
            # Make some parameters randomly +/- 1
            for i in range(self.dim):
                ind = np.random.randint(num_terms)
                sgn  = 2 * torch.rand(1) - 1
                sgn /= sgn.abs()
                coeffs[ind,i] = sgn * 1.
        return coeffs.reshape(-1).unsqueeze(0).numpy()

    def plot_vector_fields(self, params=None, param_selection='random', add_trajectories=False, **kwargs):
        """
        Plot vector fields of system
        :param params: array of params for system
        :param param_selection: plot extreme (minimal, intermediate and maximal bounds) param combinations
        :param kwargs: additional params for sampling method
        """
        if param_selection == 'random':
            params = self.params_random(**kwargs)
        elif param_selection == 'extreme':
            params = self.params_extreme()
        # elif param_selection == 'grid':
        #     pass
        elif param_selection == 'sparse':
            params = self.params_sparse(**kwargs)
        else:
            raise ValueError('param_selection must be random, extreme or sparse')
        
        num_samples = params.shape[0]
        skip = 1

        nrow = ncol = int(np.ceil(np.sqrt(num_samples)))
        if add_trajectories:
            ncol = 2
            nrow = num_samples
            skip = 2
        
        fig, axs = plt.subplots(nrow, ncol, figsize=(6*ncol, 6*nrow))
        axs = axs.flatten()
        for i in range(num_samples):
            ax = axs[skip*i]
            model = self.generate_model(params[i, :])
            model.plot_vector_field(ax=ax)
            if add_trajectories:
                ax = axs[skip*i+1]
                model.plot_trajectory(ax=ax)
        # plt.suptitle(self.data_name)
        plt.tight_layout()
        plt.show()

if __name__ == '__main__':
    # sf = SystemFamily(data_name='simple_oscillator', param_ranges=[[-1, 1]], data_dir=None, device='cpu', min_dims=[-0.5, -0.5], max_dims=[0.5, 0.5])
    poly_order = 3
    dim = 2
    nparams = library_size(dim, poly_order=poly_order) * 2
    # import pdb; pdb.set_trace()
    # sf = SystemFamily(data_name='simple_oscillator', param_ranges=[[-2, 2]], data_dir=None, device='cpu')
    sf = SystemFamily(data_name='polynomial', param_ranges=[[-2, 2]] * nparams, data_dir=None, device='cpu', poly_order=poly_order)
    # DE_inst = sf.DE(params=[1], **sf.data_info)
    # DE_inst.plot_vector_field()
    # plt.show()
    sf.plot_vector_fields(param_selection='random', num_samples=4, add_trajectories=True)
    plt.show()
    # sf.plot_vector_fields(param_selection='sparse', num_samples=4, p=0.2)
    # plt.show()



