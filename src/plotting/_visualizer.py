import os
import torch
import random
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

plt.ioff()
plt.style.use('ggplot')

from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from src.data import CircuitFamily

import pdb
from scipy.stats import linregress
from src.utils import nearly_square, ensure_dir, read_yaml

from src.train import predict
from itertools import combinations
from typing import List


class Visualizer(object):
    """
    Reads multiple samples of an experiment to visualize their individual and collective representations
    """
    def __init__(self, cf: CircuitFamily, train_info: dict,
                 fig_save_dir: str, tt: str, device: str, seed: int, num_samples: int):

        torch.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)

        # TODO: check if predict file doesnt exist
        # TODO: ensure path

        train_info_cp = train_info.copy()
        train_info_cp['device'] = device
        self.latent, self.recon_params, self.params = predict(cf=cf, tt=tt, num_samples=num_samples, **train_info_cp)
        self.num_samples = self.latent.shape[0]
        # self.device = device
        self.tt = tt
        self.cf = cf
        self.param_ranges = cf.param_ranges
        self.latent_dim = train_info['latent_dim']

        self.fig_save_dir = fig_save_dir
        if not os.path.exists(self.fig_save_dir):
            os.makedirs(self.fig_save_dir)

        self.num_DE_params = len(self.param_ranges)

    def visualize_latent(self, proj: str, **kwargs):
        print('Visualizing latent space...')
        # danco, lpca = self.model.id(self.flows)

        proj_dim = max(self.num_DE_params, 3)
        proj_dim = min(proj_dim, self.latent_dim)
        if self.latent.shape[1] > 2:
            if proj == 'TSNE':
                proj_latent = TSNE(n_components=proj_dim, perplexity=50.0).fit_transform(self.latent)[:, :3]
            elif proj == 'PCA':
                proj_model = PCA(n_components=proj_dim).fit(self.latent)
                print('Explained variance: {}'.format(proj_model.explained_variance_ratio_))
                proj_latent = proj_model.transform(self.latent)[:, :3]
            else:
                raise ValueError('Projection not recognized!')
        else:
            proj_latent = self.latent

        # Coloring
        all_param_ind_combs = list(combinations(range(self.num_DE_params), min(3, self.num_DE_params)))
        all_colorings = []
        for comb in all_param_ind_combs:
            # param_comb = self.params[:, comb].cpu() - torch.tensor(
            #     [rg[0] for rg in [self.param_ranges[c] for c in comb]])
            param_comb = self.params[:, comb] - np.array(
                [rg[0] for rg in [self.param_ranges[c] for c in comb]]) #TODO: not sure here..

            mx = param_comb.max(0)
            # mx, amx = param_comb.max(0)
            coloring = param_comb / mx#.unsqueeze(0)
            all_colorings.append(coloring)
            # all_colorings.append(coloring.cpu().numpy())
        all_colorings = np.array(all_colorings)
        if all_colorings.shape[1] < 3:
            remaining_dims = 3 - all_colorings.shape[1]
            zero_pad = np.zeros(self.num_samples, remaining_dims)
            all_colorings = np.concatenate((all_colorings, zero_pad), axis=-1)

        num_colors = min(len(all_colorings), 10)
        for c in range(num_colors):
            coloring = all_colorings[c]
            fig = plt.figure(figsize=(12, 12))
            ax = fig.add_subplot(111, projection='3d')
            ax.scatter(proj_latent[:, 0], proj_latent[:, 1], proj_latent[:, 2], c=coloring, s=6, alpha=1.0)
            ax.tick_params(axis='both', which='major', labelsize=16)
            ax.tick_params(axis='both', which='minor', labelsize=16)
            ax.set_xlabel('PC1', fontsize=36, labelpad=16.0)
            ax.set_ylabel('PC2', fontsize=36, labelpad=16.0)
            ax.set_zlabel('PC3', fontsize=36, labelpad=20.0)
            # ax.set_title('ID (DANCo): ' + str(np.around(danco.dimension_, decimals=4)) + ', ID (lPCA): ' + str(
            #     np.mean(lpca.dimension_pw_)))
            param_str = '_'.join([str(p) for p in all_param_ind_combs[c]])
            plt.savefig(os.path.join(self.fig_save_dir, '{}_latent{}.png'.format(self.tt, param_str)))
            plt.close()

    def visualize_fits(self, **kwargs):
        print('Visualizing parameter fits...')

        # Circuit recon
        colors = np.random.rand(self.num_DE_params, 3)
        fig_shape = nearly_square(self.num_DE_params)
        fig, axes = plt.subplots(fig_shape[0], fig_shape[1], figsize=(2 * fig_shape[0], 2 * fig_shape[1]))
        if self.params.shape[1] == 1:
            axes = np.array([axes])
        for c, (par, est, ax) in enumerate(zip(self.params.T[:len(colors)],
                                               self.recon_params.T[:len(colors)],
                                               axes.reshape(-1))):
            p1, p0, r_value, _, _ = linregress(par, est)
            x = np.linspace(self.param_ranges[c][0], self.param_ranges[c][1], num=10)
            y = p1 * x + p0
            ax.scatter(par, est, color=colors[c])
            ax.set_title(r'$r^2={:.4f}$'.format(r_value), fontsize=8)
            ax.plot(x, y, color='k')
            ax.set_xlabel('True parameter')
            ax.set_ylabel('Est. parameter')
            ax.set_ylim(*self.param_ranges[c])
        plt.tight_layout()
        plt.savefig(os.path.join(self.fig_save_dir, '{}_param_scatter.png'.format(self.tt)))
        plt.close()


    def visualize_flows(self, vis_res, vis_alpha, vis_time, which_dims=[0,1], num_samples=3, **kwargs):
        print('Visualizing flows...')
        fig, axes = plt.subplots(2, num_samples, figsize=(9, 6))

        def plot_trajectory(pr, title_pref, ax):
            title= title_pref + '(' + ', '.join(
                [str(np.around(p, decimals=2)) for p in pr]) + ')'
            DE_inst = self.cf.DE(params=pr, **self.cf.data_info)
            DE_inst.plot_trajectory(T=vis_time, alpha=vis_alpha, num_lattice=vis_res, ax=ax, title=title, which_dims=which_dims)

        title_pref = r'$\alpha=$'
        title_pref_recon = r'$\bar{\alpha}=$'

        for (pr, pr_recon, axs) in zip(self.params[:num_samples, :], self.recon_params[:num_samples], axes.T):
            plot_trajectory(pr, title_pref=title_pref, ax=axs[0])
            plot_trajectory(pr_recon, title_pref=title_pref_recon, ax=axs[1])

            # if (self.data_dim < 4 and not self.pde) or (which_dims is not None and not self.pde):
        #     # Compare dynamics
        #     if self.data_dim == 2 or which_dims is not None:
        #         fig, axes = plt.subplots(2, 3, figsize=(9, 6))
        #     else:
        #         fig, axes = plt.subplots(nrows=2, ncols=3, subplot_kw={'projection': '3d'}, figsize=(12, 9))

        plt.subplots_adjust(wspace=.4)
        plt.tight_layout()
        plt.savefig(os.path.join(self.fig_save_dir, '{}_recon_dynamics.png'.format(self.tt)))
        plt.close()


    def visualize_selected(self, visualize_latent=False, visualize_fits=False, visualize_flows=False, **kwargs):
        """
        Performs selected visualizing functions
        """
        if visualize_latent:
            self.visualize_latent(**kwargs)
        if visualize_fits:
            self.visualize_fits(**kwargs)
        if visualize_flows:
            self.visualize_flows(**kwargs)
