import os
import sys
import click
import uuid
import time
import numpy as np
from functools import partial

from phase2vec.train import train_model, run_epoch, load_model

from phase2vec.utils import command_with_config, ensure_dir, get_command_defaults
from phase2vec.utils import update_yaml, write_yaml, read_yaml, timestamp, strtuple_to_list, str_to_list, get_last_config
from phase2vec.data import SystemFamily, sindy_library, load_dataset
from sklearn.model_selection import train_test_split, KFold
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import classification_report

import torch
import random

# Living dangerously
import warnings
warnings.filterwarnings("ignore")
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
@click.option('--data-name', '-d', type=str, default='dataset')
@click.option('--train-size', '-m', type=int, default=500)
@click.option('--test-size', '-m', type=int, default=500)
@click.option('--sampler-type', '-sp', type=str, default='uniform')
@click.option('--num-lattice', '-n', type=int, default=64) # 
@click.option('--min-dims', '-mi', type=list) # , default=[-1.,-1.]
@click.option('--max-dims', '-mn', type=list) # , default=[1.,1.]
@click.option('--labels', '-mx', type=list) # , default=['x','y']
@click.option('--param-ranges', '-pr', type=list)
@click.option('--noise-type', '-nt', type=click.Choice([None, 'gaussian', 'masking', 'parameter', 'trajectory']), default=None)
@click.option('--noise-mag', '-n', type=float, default=0.0)
@click.option('--seed', '-se', type=int, default=0)
@click.option('--config-file', type=click.Path())
def generate_dataset(data_dir, data_name, train_size, test_size, sampler_type, num_lattice, min_dims, max_dims, labels, param_ranges, noise_type, noise_mag, seed, config_file):
    """
    Generates train and test data for one data set

    Positional arguments:

    data_name (str): name of data set and folder to save all data in.
    system_names (list of str): names of data to generate
    num_samples (int): number of total samples to generate
    sampler_type (list of strings): for each system, a string denoting the type of sampler_type used. 
    system_props (list of floats): for each system, a float controlling proportion of total data this system will comprise.
    val_size (float): proportion in (0,1) of data allocated to validation set
    num_lattice (int): number of points for all dimensions in the equally spaced grid on which velocity is measured
    min_dims (list of floats): the lower bounds for each dimension in phase space
    max_dims (list of floats): the upper bounds for each dimension in phase space
    moise_type (None, 'gaussian', 'masking', 'parameter'): type of noise to apply to the data, including None. Gaussian means white noise on the vector field; masking means randomly zeroing out vectors; parameter means gaussian noise added to the parameters
    noise_mag (float): amount of noise, interpreted differently according to each noise type. If gaussian, then the std of the applied noise relative to each vector field's natural std; if masking, proportio to be masked; if parameter, then just the std of applied noise. 
    seed (int): random seed
    """

    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    save_dir = os.path.join(data_dir, data_name)
    ensure_dir(save_dir)

    # read config file
    # data_info = {}
    # if config_file is not None:
    #     data_info = read_yaml(config_file)
    # param_ranges = strtuple_to_list(param_ranges)
    param_ranges = param_ranges if param_ranges is None else [str_to_list(x) for x in param_ranges] #TODO: better handle!
    sf = SystemFamily(data_name=data_name, num_lattice=num_lattice, min_dims=min_dims, max_dims=max_dims, labels=labels, param_ranges=param_ranges, seed=seed)
    
    num_samples = train_size + test_size
    res = sf.generate_flows(num_samples=num_samples, 
                                                  noise_type=noise_type, 
                                                  noise_level=noise_mag,
                                                  sampler_type=sampler_type)
    params_pert = res[0] 
    vectors_pert = res[1]
    savenames = ['X', 'p']

    if (test_size > 0) and (train_size > 0):
        split = train_test_split(vectors_pert, params_pert, test_size=test_size, train_size=train_size, random_state=seed)
    else:
        split = [vectors_pert, params_pert]
    
    tt = ['train'] * (train_size > 0) + ['test'] * (test_size > 0)
    filenames = [f'{s}_{t}' for s in savenames for t in tt]

    for dt, nm in zip(split, filenames):
        np.save(os.path.join(save_dir, nm + '.npy'), dt)

    # # Unique forms
    # redundant_forms = 1* (all_pars != 0)
    # unique_forms    = np.unique(redundant_forms, axis=0)

    # np.save(os.path.join(save_dir, 'forms.npy'), unique_forms)


    save_dir = os.path.join(data_dir, data_name)

    # param_ranges = strtuple_to_list(param_ranges)
    # param_ranges = param_ranges * times_param_ranges
    # min_dims = str_to_list(min_dims)
    # max_dims = str_to_list(max_dims)

    # kwargs = {ctx.args[i][2:].replace('-','_'):ctx.args[i+1] for i in range(0,len(ctx.args),2)}
    # cf = CircuitFamily(data_name=data_name, param_ranges=param_ranges, device=device, data_dir=save_dir, min_dims=min_dims, max_dims=max_dims, **kwargs)
    # cf.make_data(num_samples=num_samples)
    data_config = os.path.join(save_dir, 'data_config.yaml')
    write_yaml(data_config, sf.data_info)
    print('Successfully generated data for {}. Config file: {}'.format(data_name, data_config))


# @cli.command(name='train', cls=command_with_config('config_file'), help='train a VAE to learn reduced models')
# @click.argument("data-config", type=click.Path())
# @click.argument("net-config", type=click.Path())
# @click.option('--exp-name', type=str)
# @click.option('--num-epochs', type=int, default=10)
# @click.option('--batch-size', type=int, default=64)
# @click.option('--beta', type=float, default=1.0)
# @click.option('--fp_normalize', is_flag=True, default=True)
# @click.option('--device', type=str, default='cpu')
# @click.option('--optimizer', type=str, default='Adam')
# @click.option('--learning-rate', type=float, default=.0001)
# @click.option('--momentum', type=float, default=0.0)
# @click.option('--model-save-dir', type=str)
# @click.option('--log-dir', type=str)
# @click.option('--log-period', type=int, default=10)
# @click.option('--seed', type=int, default=0)
# @click.option('--config-file', type=click.Path())
# def call_train(data_config, net_config, exp_name, num_epochs, batch_size, beta, fp_normalize, device, optimizer, learning_rate, momentum, model_save_dir, log_dir, log_period, seed, config_file):
#     """
#     Train vector field embeddings
#     """
#     data_info = read_yaml(data_config)
#     net_info = read_yaml(net_config)

#     if exp_name is None:
#         id = str(uuid.uuid4())
#         exp_name = sf.data_name + '_' + id

#     ensure_dir(log_dir)
#     model_save_dir = os.path.join(model_save_dir, exp_name)
#     ensure_dir(model_save_dir)

#     start = time.time()

#     data_path = os.path.join(data_info['data_dir'], data_info['data_name'])

#     X_train, X_test, y_train, y_test, p_train, p_test = load_dataset(data_path)

#     model_type = net_info['net_class']
#     pretrained_path = net_info['pretrained_path']
#     AE = net_info['ae']
#     del net_info['net_class']
#     del net_info['pretrained_path']
#     del net_info['ae']
#     del net_info['output_file']

#     net = load_model(model_type, pretrained_path=pretrained_path, device=device, **net_info)

#     net = train_model(X_train, X_test,
#                       y_train, y_test,
#                       p_train, p_test,
#                       net, exp_name,
#                       num_epochs=num_epochs,
#                       learning_rate=learning_rate,
#                       momentum=momentum,
#                       optimizer=optimizer,
#                       batch_size=batch_size,
#                       beta=beta,
#                       fp_normalize=fp_normalize,
#                       device=device,
#                       log_dir=log_dir,
#                       log_period=log_period,
#                       AE=AE)

#     torch.save(net.state_dict(), os.path.join(model_save_dir, 'model.pt'))

# @cli.command(name="evaluate", help='Evaluates a trained model on a data set.')
# @click.argument('data-path', type=str)
# @click.argument('net-config', type=str)
# @click.argument('train-config', type=str)
# @click.option('--pretrained-path', type=str)
# @click.option('--results-dir', type=str)
# @click.option('--output-file', '-o', type=click.Path(), default='training_results.yaml')
# def evaluate(data_path, net_config, train_config, pretrained_path, results_dir, output_file):

#     net_info = read_yaml(net_config)
#     train_info = read_yaml(train_config)

#     if results_dir is None:
#         results_dir = '.'
#     ensure_dir(results_dir)
    
#     output_file = os.path.join(results_dir, output_file)

#     X_train, X_test, y_train, y_test, p_train, p_test = load_dataset(data_path)
#     model_type = net_info['net_class']
#     pretrained_path = net_info['pretrained_path'] if pretrained_path is None else os.path.join(pretrained_path,'model.pt')
#     AE = net_info['ae']
#     del net_info['net_class']
#     del net_info['pretrained_path']
#     del net_info['ae']
#     del net_info['output_file']

#     net = load_model(model_type, pretrained_path=pretrained_path, device=train_info['device'], **net_info)

#     for i, (name, data, labels, pars) in enumerate(zip(['train', 'test'], [X_train, X_test],[y_train, y_test],[p_train, p_test])):
#         losses, embeddings = run_epoch(data, labels, pars,
#                                    net, 0, None,
#                                    train=False,
#                                    batch_size=train_info['batch_size'],
#                                    beta=train_info['beta'],
#                                    fp_normalize=train_info['fp_normalize'],
#                                    device=train_info['device'],
#                                    return_embeddings=True,
#                                    AE=AE)

#         np.save(os.path.join(results_dir,f'embeddings_{name}.npy'), embeddings.detach().cpu().numpy())

#         loss_dict = {f'{name}_total_loss': str(np.mean(losses[0])), f'{name}_recon_loss': str(np.mean(losses[1])), f'{name}_sparsity_loss': str(np.mean(losses[2])), f'{name}_parameter_loss': str(np.mean(losses[3]))}

#         yaml_fn = write_yaml if i == 0 else update_yaml
#         for (key, value) in loss_dict.items():
#             print(key + f': {value}')
#         yaml_fn(output_file, loss_dict)

# @cli.command(name='classify', help="Trains a logistic regression classifier on labeled embeddings.")
# @click.argument('data-path', type=str)
# @click.option('--feature-name', type=str, default='embeddings')
# @click.option('--classifier', type=click.Choice(['logistic_regressor', 'k_means']), default='logistic_regressor')
# @click.option('--results-dir', type=str, default='.')
# @click.option('--penalty', type=str, default='l2')
# @click.option('--num-c', type=int, default=11)
# @click.option('--k', type=int, default=10)
# @click.option('--multi-class', type=str, default='ovr')
# @click.option('--verbose', type=int, default=0)
# @click.option('--seed', type=int, default=0)
# @click.option('--output-file', '-o', type=click.Path(), default='classifier_results.yaml')
# def classify(data_path, feature_name, classifier, results_dir, penalty, num_c, k, multi_class, verbose, seed, output_file):

#     X_train, X_test, y_train, y_test, p_train, p_test = load_dataset(data_path)

#     z_train = np.load(os.path.join(results_dir, f'{feature_name}_train.npy')).reshape(X_train.shape[0], -1)
#     z_test = np.load(os.path.join(results_dir, f'{feature_name}_test.npy')).reshape(X_test.shape[0], -1)

#     Cs = np.logspace(-5, 5, num_c)
#     kf = KFold(n_splits=int(len(y_train) / float(k)))

#     clf_params = {
#         'cv': kf,
#         'random_state': seed,
#         'dual': False,
#         'solver': 'lbfgs',
#         'class_weight': 'balanced',
#         'multi_class': multi_class,
#         'refit': True,
#         'scoring': 'accuracy',
#         'tol': 1e-2,
#         'max_iter': 5000,
#         'verbose': verbose}

#     # Load and train classifier
#     if penalty != 'none':
#         clf_params.update({
#             'Cs': Cs,
#             'penalty': penalty
#         })

#     if classifier == 'logistic_regressor':
#         clf = LogisticRegressionCV(**clf_params).fit(z_train, y_train)
#     elif clasisfier == 'k_means':
#         num_classes = len(np.unique(y_train))
#         clf = KMeans(n_clusters=num_classes, random_state=seed).fit(z_train)

#     for nm, lb_true, lb_pred in zip(['train', 'test'], [y_train, y_test], [clf.predict(z_train), clf.predict(z_test)]):
#         report = classification_report(lb_true, lb_pred, output_dict=True)

#         fn = os.path.join(results_dir, f'{nm}_' + output_file)
#         for (key, value) in report.items():
#             print(key + f': {value}')
#         write_yaml(fn, report)
#         print('\n')

# @cli.command(name="generate-net-config", help="Generates a configuration file that holds editable options for a deep net.")
# @click.option('--net-class', type=str, default='CNNwFC_exp_emb')
# @click.option('--ae', is_flag=True, default=False)
# @click.option('--latent-dim', type=int, default=100)
# @click.option('--in-shape', type=list, default=[2,64,64])
# @click.option('--num-conv-layers', type=int, default=3)
# @click.option('--kernel-sizes', type=list, default=3*[3])
# @click.option('--kernel-features', type=list, default=3*[128])
# @click.option('--strides', type=list, default=3*[2])
# @click.option('--pooling-sizes', type=list, default=[])
# @click.option('--min-dims', '-mi', type=list, default=[-1.,-1.])
# @click.option('--max-dims', '-ma', type=list, default=[1.,1.])
# @click.option('--num-fc-hid-layers', type=int, default=2)
# @click.option('--fc-hid-dims', type=list, default=2*[128])
# @click.option('--poly_order', type=int, default=3)
# @click.option('--batch-norm', is_flag=True, default=True)
# @click.option('--dropout', is_flag=True, default=True)
# @click.option('--dropout-rate', type=float, default=.1)
# @click.option('--activation-type', type=str, default='relu')
# @click.option('--last-pad', is_flag=True, default=False)
# @click.option('--pretrained-path', type=str)
# @click.option('--output-file', '-o', type=click.Path(), default='net-config.yaml')
# def generate_net_config(**args):
#     output_file = args['output_file']
#     output_file = os.path.abspath(output_file)
#     write_yaml(output_file, args)
#     print(f'Successfully generated net config file at "{output_file}".')

@cli.command(name="generate-data-config", help="Generates a configuration file that holds editable options for a dataset.")
@click.option('--output-file', '-o', type=click.Path(), default='data-config.yaml')
def generate_data_config(output_file):
    output_file = os.path.abspath(output_file)
    write_yaml(output_file, get_command_defaults(generate_dataset))
    print(f'Successfully generated data config file at "{output_file}".')

# @cli.command(name="generate-train-config", help="Generates a configuration file that holds editable options for training parameters.")
# @click.option('--output-file', '-o', type=click.Path(), default='train-config.yaml')
# def generate_train_config(output_file):
#     output_file = os.path.abspath(output_file)
#     write_yaml(output_file, get_command_defaults(call_train))
#     print(f'Successfully generated train config file at "{output_file}".')

if __name__ == '__main__':
    outdir = '/Users/nomo/PycharmProjects/phase2vec/output' # '../output'
    data_dir = os.path.join(outdir, 'data', 'nd')

    # testing
    train_size = test_size = 10
    data_name = 'polynomial'
    sampler_type = 'random'
    # generate_dataset(['--data-dir', data_dir, '--train-size', train_size, '--test-size', test_size, '--data-name', data_name, '--sampler-type', sampler_type])

    dim = 3
    library_size = 20
    train_size = 100
    test_size = 0
    data_name = 'polynomial'
    sampler_type = 'random'
    min_dims = [-30,-30,0]
    max_dims = [30,30,60]
    labels = ['x_%d' % i for i in range(dim)]
    num_lattice = 10
    # [10,28,8/3]
    # generate_dataset(['--data-dir', data_dir, 
    #                   '--train-size', train_size, 
    #                   '--test-size', test_size, 
    #                   '--data-name', data_name, 
    #                   '--sampler-type', sampler_type, 
    #                   '--min-dims', min_dims,
    #                   '--max-dims', max_dims,
    #                   '--labels', labels,
    #                   '--param-ranges', ['-30.0, 30.0'] * (dim * library_size) ,
    #                   ])
                      

    dim = 3
    train_size = 0
    test_size = 100
    data_name = 'lorenz'
    sampler_type = 'random'
    fold_mn = 0.5
    fold_mx = 1.5
    param_ranges = ['%.02f, %.02f' % (fold_mn * v, fold_mx * v) for v in [10,28,8/3]]
    generate_dataset(['--data-dir', data_dir, 
                      '--train-size', train_size, 
                      '--test-size', test_size, 
                      '--data-name', data_name, 
                      '--sampler-type', sampler_type, 
                      '--min-dims', min_dims,
                      '--max-dims', max_dims,
                      '--labels', labels,
                      '--param-ranges', param_ranges,
                      ])

                      
