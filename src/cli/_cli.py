import os
import sys
import click
import uuid
import time
import numpy as np
from functools import partial

from src.train import train
from src.plotting import Visualizer, VisualizerGridsearch

from src.gridsearch import generate_gridsearch_worker_params, get_gridsearch_default_scans, results_to_df
from src.gridsearch import wrap_command_with_local, wrap_command_with_slurm, write_jobs
from src.utils import command_with_config, ensure_dir, get_command_defaults
from src.utils import write_yaml, read_yaml, timestamp, strtuple_to_list, str_to_list, get_last_config
from src.data import CircuitFamily
from sklearn.model_selection import train_test_split
import torch

# Here we will monkey-patch click Option __init__
# in order to force showing default values for all options
orig_init = click.core.Option.__init__


def new_init(self, *args, **kwargs):
    orig_init(self, *args, **kwargs)
    self.show_default = True


click.core.Option.__init__ = new_init  # type: ignore


@click.group()
@click.version_option()
def cli():
    pass

################################################## Single Experiment ###################################################

@cli.command(name="generate-dataset", cls=command_with_config('config_file'), context_settings=dict(ignore_unknown_options=True, allow_extra_args=True), 
help="Generates a data set of vector fields.")
@click.option('--data-dir', '-f', type=str)
@click.option('--data-set-name', '-d', type=str, default='dataset')
@click.option('--system-names', '-s', type=str, multiple=True, default=['simple_oscillator'])
@click.option('--num-samples', '-m', type=int, default=1000)
@click.option('--samplers', '-sp', type=str, multiple=True, default=['uniform'])
@click.option('--class-props', '-c', type=float, multiple=True, default=[1.0])
@click.option('--test-size', '-t', type=float, default=.25)
@click.option('--config-file', type=click.Path())
#@click.pass_context
def generate_dataset(data_dir, data_set_name, system_names, num_samples, samplers, class_props, test_size, config_file):
    """
    Generates train and test data for one data set

    Positional arguments:

    data_set_name (str): name of data set and folder to save all data in.
    system_names (list of str): names of data to generate
    num_samples (int): number of total samples to generate
    samplers (list of strings): for each system, a string denoting the type of sampler used. 
    class_props (list of floats): for each system, a float controlling proportion of total data this system will comprise.

    """

    # Living dangerously
    import warnings
    warnings.filterwarnings("ignore")
    # TODO: For now, no control here over param ranges, min or max dims. Just use cf defaults
    # TODO: Add noise params
    # TODO: Add control for poly params

    save_dir = os.path.join(data_dir, data_set_name)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    all_data   = []
    all_labels = []
    all_pars   = []

    cfs = [CircuitFamily(data_name=system_name, default_sampler=sampler) for (system_name, sampler) in zip(system_names, samplers)]
    num_labels_per_group = [len(cf.param_groups) for cf in cfs]
    cum_labels_per_group = [0] + list(np.cumsum(num_labels_per_group))[:-1]

    for d, system_name in enumerate(system_names):

        print(f'Generating {system_name} data.')

        sampler       = samplers[d]
        cf            = CircuitFamily(data_name=system_name, default_sampler=sampler)
        class_samples = int(class_props[d] * num_samples)

        linear       = (system_name == 'linear')
        polynomial    = (system_name == 'polynomial')

        gen_pars = cf.param_sampler(class_samples)
        systems  = [cf.generate_model(pr, linear=linear, polynomial=polynomial) for pr in gen_pars]
        data     = [system.forward(0,cf.L) for system in systems]
        labels   = [system.label + cum_labels_per_group[d] for system in systems]

        if system_name in ['simple_oscillator', 'alon']:
            save_pars = [system.fit_polynomial_representation(poly_order=3) for system in systems]
        else:
            save_pars = [system.get_polynomial_representation() for system in systems]

        save_pars = [torch.cat((torch.tensor(dx.to_numpy()), torch.tensor(dy.to_numpy()))).transpose(1,0).float() for (dx, dy) in save_pars]

        all_data   += data
        all_pars   += save_pars
        all_labels += labels

    all_data   = torch.stack(all_data).numpy().transpose(0,3,1,2)
    all_pars   = torch.stack(all_pars).numpy()
    all_labels = np.array(all_labels)

    split = train_test_split(all_data, all_labels, all_pars, test_size=test_size, stratify=all_labels)

    for dt, nm in zip(split, ['X_train', 'X_test', 'y_train', 'y_test', 'p_train', 'p_test']):
        np.save(os.path.join(save_dir, nm + '.npy'), dt)

@cli.command(name='train', cls=command_with_config('config_file'), help='train a VAE to learn reduced models')
@click.argument("data-config", type=click.Path())
@click.option('--exp-name', type=str)
@click.option('--num-epochs', type=int, default=10)
@click.option('--batch-size', type=int, default=16)
@click.option('--latent-dim', type=int, default=16)
@click.option('--first-pad', type=int, default=0)
@click.option('--last-pad', type=int, default=0)
@click.option('--beta', type=float, default=1.0)
@click.option('--gamma', type=float, default=0.0)
@click.option('--p', type=float, default=1.0)
@click.option('--num-sparsified', type=int, default=1)
@click.option('--model-type', type=str, default='ResParAE')
@click.option('--recon-loss', type=str, default='euclidean')
@click.option('--device', type=str, default='cuda')
@click.option('--pde', is_flag=True)
@click.option('--scale-output', is_flag=True)
@click.option('--whiten', is_flag=True)
@click.option('--means-train', type=str, multiple=True, default=[]) # TODO: are these different???
@click.option('--means-test', type=str, multiple=True, default=[])
@click.option('--stds-train', type=str, multiple=True, default=[])
@click.option('--stds-test', type=str, multiple=True, default=[])
@click.option('--optimizer_name', type=str, default='Adam')
@click.option('--learning-rate', type=float, default=.0001)
@click.option('--max-grad', type=float, default=10.0)
# @click.option('--data-dir', type=str)
@click.option('--model-save-dir', type=str)
# @click.option('--fig-save-dir', type=str)
@click.option('--log-dir', type=str)
@click.option('--save-model-every', type=int, default=1)
@click.option('--seed', type=int, default=0)
@click.option('--config-file', type=click.Path())
def call_train(data_config, exp_name, model_save_dir, log_dir, config_file, **kwargs):
    """
    Train an autoencoder on data of a circuit family
    """
    data_info = read_yaml(data_config)
    cf = CircuitFamily(**data_info)

    if exp_name is None:
        id = str(uuid.uuid4())
        exp_name = cf.data_name + '_' + id

    # TODO: removing
    # model_save_dir = os.path.join(model_save_dir, cf.data_name)
    # log_dir = os.path.join(log_dir, cf.data_name)
    ensure_dir(log_dir)
    ensure_dir(model_save_dir)

    start = time.time()

    train_observables, test_observables = train(cf, exp_name=exp_name, model_save_dir=model_save_dir, log_dir=log_dir, **kwargs)

    print('Successfully trained on data config: {}'.format(data_config))

    stop = time.time()
    duration = stop - start

    train_info = kwargs
    train_info['data_config'] = data_config
    train_info['exp_name'] = exp_name
    train_info['model_save_dir'] = model_save_dir
    train_info['log_dir'] = log_dir
    train_info['model_save_path'] = os.path.join(model_save_dir, exp_name + '.pt')

    train_config = os.path.join(model_save_dir, exp_name + '_train.yaml')
    write_yaml(train_config, train_info)
    print('Train config file: {} \n'.format(train_config))

    results = {**train_observables, **test_observables, 'train_time': duration}
    results_config = os.path.join(model_save_dir, exp_name + '_results.yaml')
    write_yaml(results_config, results)
    print('Train/test observables: {}'.format(results_config))

@cli.command(name="generate-data-config", help="Generates a configuration file that holds editable options for a dataset.")
@click.option('--output-file', '-o', type=click.Path(), default='data-config.yaml')
def generate_train_config(output_file):
    output_file = os.path.abspath(output_file)
    write_yaml(output_file, get_command_defaults(generate_dataset))
    print(f'Successfully generated train config file at "{output_file}".')

@cli.command(name="generate-train-config", help="Generates a configuration file that holds editable options for training parameters.")
@click.option('--output-file', '-o', type=click.Path(), default='train-config.yaml')
def generate_train_config(output_file):
    output_file = os.path.abspath(output_file)
    write_yaml(output_file, get_command_defaults(call_train))
    print(f'Successfully generated train config file at "{output_file}".')

@cli.command(name='visualize', help='visualize the results of training')
@click.argument("data-config", type=click.Path())
@click.argument('train-config', type=click.Path())
@click.argument('fig_save_dir', type=click.Path())
@click.option('--num-samples', type=int, default=2000)
@click.option('--tt', type=str, default='test')
@click.option('--vis-res', type=int, default=10)
@click.option('--proj', type=str, default='PCA')
@click.option('--vis-time', type=int, default=50)
@click.option('--vis-alpha', type=float, default=.01)
@click.option('--visualize-latent', is_flag=True)
@click.option('--visualize-fits', is_flag=True)
@click.option('--visualize-flows', is_flag=True)
@click.option('--which-dims', type=str, multiple=True, default=[])
@click.option('--device', type=str, default='cuda')
@click.option('--seed', type=int, default=0)
def visualize(data_config, train_config, fig_save_dir, num_samples, tt,
              visualize_latent, visualize_fits, visualize_flows, which_dims, device, seed, **kwargs):
    """
    Visualize results of a single training experiment
    """
    data_info = read_yaml(data_config)
    cf = CircuitFamily(**data_info)

    train_config_path = get_last_config(train_config, suffix='_train')
    if train_config_path is None:
        train_config_path = get_last_config(os.path.join(train_config, cf.data_name), suffix='_train')


    train_info = read_yaml(train_config_path)

    fig_save_dir = os.path.join(fig_save_dir, cf.data_name)

    V = Visualizer(cf, train_info, tt=tt, num_samples=num_samples, fig_save_dir=fig_save_dir, device=device, seed=seed)

    V.visualize_selected(visualize_latent, visualize_fits, visualize_flows, **kwargs)



############################################ Multiple Training Experiments #############################################

@cli.command(name='write-gridsearch-jobs', help='grid search hyperparameters')
@click.argument("scan_file", type=click.Path(exists=True))
@click.option('--save-dir', type=click.Path(), default=os.path.join(os.getcwd(), 'worker-conf'),
              help="Directory to save worker configurations")
@click.option('--cluster-type', default='local', type=click.Choice(['local', 'slurm']))
@click.option('--slurm-partition', type=str, default='main', help="Partition on which to run jobs. Only for SLURM")
@click.option('--slurm-ncpus', type=int, default=1, help="Number of CPUs per job. Only for SLURM")
@click.option('--slurm-memory', type=str, default="2GB", help="Amount of memory per job. Only for SLURM")
@click.option('--slurm-wall-time', type=str, default='6:00:00', help="Max wall time per job. Only for SLURM")
def write_gridsearch_jobs(scan_file, save_dir, cluster_type, slurm_partition, slurm_ncpus, slurm_memory, slurm_wall_time): #TODO: this was previously grid-search
    """
    Write command lines for gridsearch training experiments
    """
    worker_dicts = generate_gridsearch_worker_params(scan_file)

    if cluster_type == 'local':
        cluster_wrap = wrap_command_with_local
    elif cluster_type == 'slurm':
        cluster_wrap = partial(wrap_command_with_slurm, partition=slurm_partition, ncpus=slurm_ncpus,
                               memory=slurm_memory, wall_time=slurm_wall_time)
    else:
        raise ValueError(f'Unsupported cluster-type {cluster_type}')
    full_save_dir = os.path.join(save_dir, read_yaml(scan_file)['gs_name'])
    save_dir = ensure_dir(full_save_dir)
    write_jobs(worker_dicts, cluster_wrap, full_save_dir)
    sys.stderr.write(f'{len(worker_dicts)} jobs written to {save_dir}\n')


@cli.command(name="generate-gridsearch-config",
             help="Generates a configuration file that holds editable options for gridsearching hyperparameters.")
@click.option('--output-file', '-o', type=click.Path(), default='gridsearch-config.yaml')
def generate_gridsearch_config(output_file):
    """
    Generate gridsearch config
    """
    gs_name = 'gs_' + timestamp()
    params = {
        'scans': get_gridsearch_default_scans(),
        'parameters': get_command_defaults(call_train),
        'gs_name': gs_name
    }

    output_file = os.path.abspath(output_file)
    write_yaml(output_file, params)
    print(f'Successfully generated gridsearch config file at "{output_file}".')

#TODO: put logs  also in dir with gs name

@cli.command(name="aggregate-gridsearch-results", help="Aggregate Gridsearch results.") # TODO: is this in use?
@click.argument("results_directory", type=click.Path(exists=True))
def aggregate_gridsearch_results(results_directory):
    results = results_to_df(results_directory)
    results.to_csv(os.path.join(results_directory, 'gridsearch-aggregate-results.tsv'), sep='\t', index=False)



@cli.command(name='visualize-gridsearch', help='Visualize the results of gridsearch training experiments')
@click.argument("gs-config-path", type=click.Path())
@click.argument('fig-save-dir', type=click.Path())
@click.option('--num-samples', type=int, default=2000)
@click.option('--tt', type=str, default='test')
@click.option('--vis-res', type=int, default=10)
@click.option('--proj', type=str, default='PCA')
@click.option('--vis-time', type=int, default=50)
@click.option('--vis-alpha', type=float, default=.01)
@click.option('--visualize-latent', is_flag=True)
@click.option('--visualize-fits', is_flag=True)
@click.option('--visualize-flows', is_flag=True)
@click.option('--visualize-beta-gamma', is_flag=True)
@click.option('--device', type=str, default='cuda')
@click.option('--seed', type=int, default=0)
def visualize_gridsearch(gs_config_path, fig_save_dir, device, seed, num_samples, visualize_beta_gamma, tt, **kwargs):
    """
    Visualize gridsearch
    """
    ensure_dir(fig_save_dir) # TODO: move to visualizer? here can add a default option
    VG = VisualizerGridsearch(gs_config_path, fig_save_dir=fig_save_dir, device=device, seed=seed, num_samples=num_samples, tt=tt)
    # TODO: if visualize_beta_gamma:
    VG.visualize_across_hyperparams()
    VG.visualize_mean_per_hyperparam()
    VG.visualize_multi()


if __name__ == '__main__':
    cli()
