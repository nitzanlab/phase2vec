import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from src.plotting._visualizer import Visualizer
plt.ioff()
plt.style.use('ggplot')

from src.data import CircuitFamily
from src.utils import ensure_dir, read_yaml



class VisualizerGridsearch(object):
    """
    Visualizes results of multiple training experiments over the same data
    """

    def __init__(self, gs_config_path, fig_save_dir, tt, num_samples, **kwargs): #TODO: why care about seed so much here?
        self.kwargs = kwargs
        self.tt = tt
        self.num_samples = num_samples
        self.gs_config_path = gs_config_path
        self.fig_save_dir = fig_save_dir

        gs_info = read_yaml(self.gs_config_path)
        self.gs_name = gs_info['gs_name']
        self.gs_params_info = gs_info['parameters']
        self.gs_scans_info = gs_info['scans'][0]['scan']
        self.data_info = read_yaml(self.gs_params_info['data_config'])
        self.cf = CircuitFamily(**self.data_info)

        self.param_ranges = self.cf.param_ranges
        self.gs_fig_dir = os.path.join(self.fig_save_dir, self.gs_name)
        ensure_dir(self.gs_fig_dir)

        # gridsearch directory
        gs_dir = os.path.join(self.gs_params_info['model_save_dir'], self.gs_name)
        if not os.path.isdir(gs_dir):
            raise ValueError('Gridsearch models directory {} not found'.format(gs_dir))

        # read all training experiment files
        self.train_configs = glob.glob(os.path.join(gs_dir, '*_train.yaml'))
        # self.result_configs = glob.glob(os.path.join(gs_dir, '*_result.yaml'))
        if len(self.train_configs) == 0:
            raise ValueError('No training configs found in {}'.format(gs_dir))
        self.train_infos = {}
        for tc in self.train_configs:
            train_info = read_yaml(tc)
            result_info = read_yaml(tc.replace('_train.yaml', '_results.yaml')) #TODO: "try"
            self.train_infos[train_info['exp_name']] = {**train_info, **result_info}
        self.result_names = list([k for k in result_info.keys() if self.tt.lower() in k.lower()])
        # self.result_infos = []
        # for rc in self.result_configs:
        #     self.result_infos.append(read_yaml(rc))

        self._get_hyperparam_df()
        self.visualizers = {}
        self._ini_visualizer()

    def _ini_visualizer(self):
        """
        Initialize each training experiment visualizer
        """
        if len(self.visualizers) > 0:
            print('Visualizers are already initialized.')
            return
        for exp_name, train_info in self.train_infos.items():
            #TODO: edit save dir?
            self.visualizers[exp_name] = Visualizer(self.cf, train_info, fig_save_dir=self.fig_save_dir, tt=self.tt, num_samples=self.num_samples, **self.kwargs)


    def visualize_multi(self, **kwargs):
        """
        Visualize each training experiment result
        """
        for exp_name, V in self.visualizers.items():
            V.visualize_selected(**kwargs)


    def _get_hyperparam_df(self):
        """

        """
        def get_grid(row):
            if row['scale'] == 'linear':
                grid = np.linspace(row['range min'], row['range max'], row['range steps'])
            else:
                grid = np.logspace(row['range min'], row['range max'], row['range steps'])
            return grid

        def get_step_size(row):
            if row['scale'] == 'linear':
                step_size = (row['range max'] - row['range min']) / row['range steps']
            else:
                step_size = np.diff(row['grid'])
            return step_size

        hyperparam_df = pd.DataFrame(self.gs_scans_info)
        hyperparam_df.index = hyperparam_df['parameter']
        hyperparam_df['range min'] = hyperparam_df['range'].apply(lambda x: x[0])
        hyperparam_df['range max'] = hyperparam_df['range'].apply(lambda x: x[1])
        hyperparam_df['range steps'] = hyperparam_df['range'].apply(lambda x: x[2])
        hyperparam_df['grid'] = hyperparam_df.apply(get_grid, axis=1)
        hyperparam_df['step size'] = hyperparam_df.apply(get_step_size, axis=1)

        self.hyperparam_df = hyperparam_df

    def _check_hyperparams(self, hyperparams):
        hyperparams_present = list(self.hyperparam_df.index)
        missing_hyperparams = list(set(hyperparams) - set(hyperparams_present))
        if len(missing_hyperparams) > 0:
            raise ValueError('Hyperparams {} is/are missing from the gridsearch.'.format(missing_hyperparams))


    def visualize_across_hyperparams(self, x_hyperparam='beta', y_hyperparam = 'gamma'):
        """Visualize sparsity-accuracy tradeoff"""
        print('Visualizing training experiment results across two hyperparamers')

        self._check_hyperparams([x_hyperparam, y_hyperparam])
        train_df = pd.DataFrame(self.train_infos).T

        x = self.hyperparam_df.loc[x_hyperparam]['grid']
        y = self.hyperparam_df.loc[y_hyperparam]['grid']
        x_step_size = self.hyperparam_df.loc[x_hyperparam]['step size']
        y_step_size = self.hyperparam_df.loc[x_hyperparam]['step size']
        bb, gg = np.meshgrid(x, y, indexing='ij')
        #
        plotting_vars = {result_name: np.zeros_like(bb) for result_name in self.result_names}

        for key in self.result_names:
            for r, res in train_df.iterrows():
                res_x = res[x_hyperparam]
                res_y = res[y_hyperparam]
                plotting_vars[key][x == res_x, y == res_y] = res[key] #TODO: make sure we use all....

            fig = plt.figure(111, figsize=(8, 8))
            ax = fig.add_subplot(111, projection='3d')
            ax.bar3d(bb.ravel(), gg.ravel(), np.zeros_like(bb).ravel(), x_step_size, y_step_size,
                     plotting_vars[key].ravel(), shade=True)
            ax.set_xlabel(r'$\%s' % x_hyperparam) #TODO: how does this behave if not greek letter?
            ax.set_ylabel(r'$\%s' % y_hyperparam)
            ax.set_zlabel(key)
            key_str = key.replace('/', '_').lower()
            plt.savefig(os.path.join(self.gs_fig_dir, '{}_{}_{}.png'.format(key_str, x_hyperparam, y_hyperparam)))
            plt.close()

    def visualize_mean_per_hyperparam(self, hyperparam='gamma'):
        """

        """
        print('Visualizing parameter means vs gamma')
        self._check_hyperparams([hyperparam])
        hyperparam_df = self.hyperparam_df.copy()

        exp_param_diffs = []
        for exp_name, V in self.visualizers.items():
            param_diff = np.mean(np.abs(V.recon_params - V.params), axis=0) #TODO: add across dim
            exp_param_diffs.append({'exp_name': exp_name, 'param_diff': param_diff})

        train_df = pd.DataFrame(self.train_infos).T
        exp_param_diff_df = pd.DataFrame(exp_param_diffs)
        exp_param_diff_df = pd.merge(train_df, exp_param_diff_df, on=['exp_name'])

        hyperparam_diff_df = exp_param_diff_df.groupby(hyperparam).mean().reset_index()
        plt.plot(hyperparam_diff_df[hyperparam], hyperparam_diff_df['param_diff'])
        plt.xlabel(r'$\%s$' % hyperparam)
        plt.ylabel(r'$\langle \alpha \rangle - \langle \bar{\alpha} \rangle$')
        plt.legend() # param_names
        plt.savefig(os.path.join(self.gs_fig_dir, '{}_param_diffs.png'.format(self.tt)))
        plt.close()

        # num_DE_params = len(self.param_ranges)
        # param_mx_train = np.zeros((num_DE_params, len(y)))
        # param_mx_test = np.zeros((num_DE_params, len(y)))
        # param_mx_train_recon = np.zeros((num_DE_params, len(y)))
        # param_mx_test_recon = np.zeros((num_DE_params, len(y)))


        # for r, res in self.results.iterrows():
        #     model_save_path = os.path.join(res['model_save_dir'], res['exp_name'] + '.pt')
        #     model = load_model(res['model_type'],
        #                        res['data_dim'],
        #                        res['num_lattice'],
        #                        res['latent_dim'],
        #                        num_DE_params=num_DE_params,
        #                        last_pad=res['last_pad'],
        #                        pretrained_path=model_save_path,
        #                        device=self.device)

        #     _, _, train_params, test_params, _, _, train_recon_params, test_recon_params, _, _, _, _ = self.fetch_data(
        #         res['data_dir'], res['data_name'], self.num_samples, res['pde'], res['scale_output'], res['param_ranges'],
        #         model)
        #
        #     res_gamma = res['gamma']
        #     param_mx_train[:, gamma == res_gamma] = train_params.mean(0).unsqueeze(1).detach().cpu().numpy()
        #     param_mx_test[:, gamma == res_gamma] = test_params.mean(0).unsqueeze(1).detach().cpu().numpy()
        #     param_mx_train_recon[:, gamma == res_gamma] = train_recon_params.mean(0).unsqueeze(1).detach().cpu().numpy()
        #     param_mx_test_recon[:, gamma == res_gamma] = test_recon_params.mean(0).unsqueeze(1).detach().cpu().numpy()
        #
        # train_param_diffs = param_mx_train - param_mx_train_recon
        # test_param_diffs = param_mx_test - param_mx_test_recon
        #
        # for tt, mx in zip(['train', 'test'], [train_param_diffs, test_param_diffs]):
        #     plt.plot(gamma, mx.T)
        #     plt.xlabel(r'$\gamma$')
        #     plt.ylabel(r'$\langle \alpha \rangle - \langle \bar{\alpha} \rangle$')
        #     plt.legend()#param_names)
        #     plt.savefig(os.path.join(self.gs_fig_dir, '{}_param_diffs.png'.format(tt)))
        #     plt.close()
