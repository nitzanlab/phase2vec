import os
import torch
import pickle
import random
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from phase2vec.data._odes import *
from sklearn.model_selection import train_test_split
from phase2vec.utils import ensure_dir
 

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

        else:
            raise ValueError(f'Unknown data, `{data_name}`! Try `alon`.')

        return generator


    @staticmethod
    def get_sampler(sampler_type):
        """
        Selecting supported sampler
        """
        if (sampler_type == 'uniform') or (sampler_type == 'random'):
            sampler = SystemFamily.params_random
        elif sampler_type == 'extreme':
            sampler = SystemFamily.params_extreme
        elif sampler_type == 'sparse':
            sampler = SystemFamily.params_sparse
        elif sampler_type == 'control':
            sampler = SystemFamily.params_control
        else:
            raise ValueError('Param sampler not recognized.')

        return sampler



    def __init__(self, data_name, device=None, min_dims=None, max_dims=None, num_lattice=64, 
                param_ranges=None, param_groups=None, seed=0, **kwargs):
        """
        Generate a system family
        :param data_name: name of system
        :param param_ranges: range for each param in model, 
        :param device: device to use
        :param min_dims: minimum range of dimensions to use
        :param max_dims: maximum range of dimensions to use
        :param kwargs: any arguments of a general system
        """

        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        self.data_name = data_name
        self.pde = False
         
        DE = SystemFamily.get_generator(self.data_name)
        # self.data_dir = os.path.abspath(data_dir) if data_dir is not None else '.'

        # if not provided, use ode suggested params
        self.param_ranges = DE.recommended_param_ranges if param_ranges is None else param_ranges
        self.param_groups = DE.recommended_param_groups if param_groups is None else param_groups
        self.min_dims = DE.min_dims if min_dims is None else min_dims
        self.max_dims = DE.max_dims if max_dims is None else max_dims

        # general DE params
        self.device = device
        params = self.params_random(1)
        DE_ex = DE(params=params[0], device=device, min_dims=self.min_dims, max_dims=self.max_dims, num_lattice=num_lattice, **kwargs)
        data_info = DE_ex.get_info()
        # data_info = {**self.__dict__, **data_info} # merge dictionaries
        self.data_info = data_info
        self.num_lattice = num_lattice
        self.dim = DE_ex.dim
        self.DE = DE
        self.DE_ex = DE_ex
        if self.data_name != 'grayscott': # TODO: ask isinstance(generator, ODE/PDE)
            # TODO: can generator.param can be used as default?
            self.coords = self.DE_ex.generate_mesh() # min_dims, max_dims, num_lattice
            self.coords = self.coords.to(device).float()

        self.kwargs = kwargs

        # self.param_sampler = SystemFamily.get_sampler(sampler_type)

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

######################################## Flow noise functions ########################################################

    def noise_flow_gaussian(self, flow, noise_level):
        """
        Add gaussian noise to flow
        """
        noise = np.random.randn(flow.shape) * noise_level
        return flow + noise

    def noise_flow_mask(self, flow, noise_level):
        """
        Add mask noise to flow
        """
        mask = np.random.randn(flow.shape) < noise_level
        return flow * mask

    def noise_flow(self, flow, noise_type, noise_level):
        """
        Add noise to parameters
        """
        if noise_type == 'gaussian':
            return self.noise_flow_gaussian(flow, noise_level)
        if noise_type == 'mask':
            return self.noise_flow_mask(flow, noise_level)
        else:
            return flow

######################################## Params noise functions ########################################################

    def noise_params_gaussian(self, params, noise_level):
        """
        Add gaussian noise to parameters
        """
        noise = torch.randn_like(params) * noise_level
        return params + noise

    def noise_params(self, params, noise_type, noise_level):
        """
        Add noise to parameters
        """
        if noise_type == 'params_gaussian':
            return self.noise_params_gaussian(params, noise_level)
        else:
            return params

######################################## Data generation ########################################################

    def generate_flows(self, num_samples, noise_type=None, noise_level=None, sampler_type='random', params=None):
        """
        Generate original and perturbed params and flows
        """
        # kwargs = {**self.data_info, **kwargs}
        sampler = SystemFamily.get_sampler(sampler_type)
        params = params if params else sampler(self, num_samples)
        
        params_pert = self.noise_params(params, noise_type, noise_level=noise_level)
        
        vectors = []
        vectors_pert = []

        for p, p_pert in zip(params, params_pert):

            DE = self.DE(params=p, **self.data_info)
            _, v = DE.get_vector_field()
            
            DE_pert = self.DE(params=p_pert, **self.data_info)
            if noise_type == 'trajectory':
                _, v_pert = DE_pert.get_vector_field_from_trajectory()
            else:
                _, v_pert = DE_pert.get_vector_field()
            
            vectors.append(v)
            vectors_pert.append(v_pert)

        vectors = np.stack(vectors)
        vectors_pert  = np.stack(vectors_pert)

        _, vectors_pert = self.noise_flow(vectors_pert, noise_type, noise_level=noise_level)

        return params_pert, vectors_pert, params, vectors


    # def make_data(self, test_size, train_size, sampler_type='uniform', noise_type=None, **kwargs):
    #     """
    #     Generates data of system
    #     """
    #     ensure_dir(self.data_dir)
        # train_dir = os.path.abspath(os.path.join(self.data_dir, 'train', '0'))
        # test_dir = os.path.abspath(os.path.join(self.data_dir, 'test', '0'))

        # for dr in [train_dir, test_dir]:
        #     if not os.path.exists(dr):
        #         os.makedirs(dr)

        # num_samples = test_size + train_size
        # params_pert, flow_pert, params, flow = self.generate_flows(num_samples=num_samples, noise_type=noise_type, sampler_type=sampler_type, **kwargs)

        # all_data   = torch.stack(all_data).numpy().transpose(0,3,1,2)
        # all_pars   = torch.stack(all_pars).numpy()
        # all_labels = np.array(all_labels)

        # split = train_test_split(params_pert, flow_pert, params, flow, test_size=test_size, train_size=train_size, stratify=all_labels, random_state=seed)
        # filenames = ['p_pert', 'X_pert', 'p', 'X'] * ['_'] * ['train', 'test']
        # for dt, nm in zip(split, ['X_train', 'X_test', 'y_train', 'y_test', 'p_train', 'p_test']):
        #     np.save(os.path.join(self.data_dir, nm + '.npy'), dt)

        # for dt, dr, nm in zip([all_params[:, :num_samples], all_params[:, num_samples:]], [train_dir, test_dir], ['Train', 'Test']):
        #     print('Generating {} Set'.format(nm))
        #     for d, params in tqdm(enumerate(dt.T)):
        #         #data = np.array([.1, .05, .0545, .062])

        #         flow = self.generate_flow(params)

        #         fn = os.path.join(dr,  'flow%05i.pkl' % d)
        #         with open(fn, 'wb') as f:
        #             pickle.dump((flow, params), f)

######################################## Plotting ########################################################

    def plot_noised_vector_fields(self, num_samples, noise_type, noise_level, params=None, **kwargs):
        """
        Plot original and perturbed vector fields
        """
        params_pert, flow_pert, params, flow = self.generate_flows(num_samples=num_samples, params=params, noise_type=noise_type, noise_level=noise_level, **kwargs)
        # self.plot_vector_fields(params_pert, flow_pert, params, flow, **kwargs)
        fig, axs = plt.subplots(num_samples,2, figsize=(10,5))
        for i, (f, f_pert) in enumerate(zip(flow, flow_pert)):
            self.plot_vector_field(f, ax=axs[i, 0])
            self.plot_vector_field(f_pert, ax=axs[i, 1])

        plt.show()
    
    def plot_vector_fields(self, params=None, sampler_type='uniform', add_trajectories=False, **kwargs):
        """
        Plot vector fields of system
        :param params: array of params for system
        :param param_selection: plot extreme (minimal, intermediate and maximal bounds) param combinations
        :param kwargs: additional params for sampling method
        """
        sampler = SystemFamily.get_sampler(sampler_type)
        params = sampler(self, **kwargs)
        num_samples = params.shape[0]
        skip = 1

        nrow = ncol = int(np.ceil(np.sqrt(num_samples)))
        if add_trajectories:
            ncol = 2
            nrow = num_samples
            skip = 2
        
        fig, axs = plt.subplots(nrow, ncol, figsize=(6*ncol, 6*nrow), tight_layout=False, constrained_layout=True)
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
    # poly_order = 3
    # dim = 2
    # nparams = library_size(dim, poly_order=poly_order) * 2
    # sf = SystemFamily(data_name='polynomial', param_ranges=[[-2, 2]] * nparams, data_dir=None, device='cpu', poly_order=poly_order)
    # sf.plot_vector_fields(param_selection='random', num_samples=4, add_trajectories=True)
    # plt.show()

    # params = [10, 28, 8/3]
    # param_ranges = np.array([params]).T @ np.array([[0.5, 1.5]])
    # param_ranges = param_ranges.tolist()
    # min_dims = [-30, -30, 0]
    # max_dims = [30, 30, 60]
    # poly_order = 2
    # sf = SystemFamily(data_name='lorenz', param_ranges=param_ranges, data_dir=None, device='cpu', poly_order=poly_order)
    # sf.plot_vector_fields(sampler_type='random', num_samples=4, add_trajectories=True)
    # plt.show()

    poly_order = 3
    dim = 2
    nparams = library_size(dim, poly_order=poly_order) * 2
    sf = SystemFamily(data_name='polynomial', param_ranges=[[-2, 2]] * nparams, device='cpu', poly_order=poly_order)
    
    sf.plot_noised_vector_fields(num_samples=2, noise_type='gaussian', noise_level=1, )

