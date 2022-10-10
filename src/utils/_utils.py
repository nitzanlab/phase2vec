import torch
import glob
import pickle
import numpy as np
import errno
import os
from typing import Type
from datetime import datetime
import click
import ruamel.yaml as yaml
import torch.nn.functional as F
import pdb
def str_to_list(s):
    """
    Converts str of list of floats to a list of floats
    """
    if isinstance(s, str):
        return [float(i) for i in s.split(',')]
    else:
        return s

def strtuple_to_list(param_ranges_tuple):
    """
    Converts tuple of strs of floats to a list of lists of floats
    """
    if isinstance(param_ranges_tuple, tuple):
        param_ranges = [] # TODO: handle input mistakes
        for ir, r in enumerate(param_ranges_tuple):
            vals = r.split(',')
            param_ranges.append([float(vals[0]), float(vals[1])])
        return param_ranges
    else:
        return param_ranges_tuple

def get_last_config(fname, suffix=''):
    """

    """
    if os.path.isfile(fname):
        return fname
    if os.path.isdir(fname):
        list_of_files = glob.glob(os.path.join(fname, '*%s.yaml' % suffix))
        latest_file = max(list_of_files, key=os.path.getctime)
        return latest_file



# from https://stackoverflow.com/questions/46358797/
# python-click-supply-arguments-and-options-from-a-configuration-file
def command_with_config(config_file_param_name: str) -> Type[click.Command]:
    """ Create and return a class inheriting `click.Command` which accepts a configuration file
        containing arguments/options accepted by the command.

        The returned class should be passed to the `@click.Commnad` parameter `cls`:

        ```
        @cli.command(name='command-name', cls=command_with_config('config_file'))
        ```

    Parameters:
        config_file_param_name (str): name of the parameter that accepts a configuration file

    Returns:
        class (Type[click.Command]): Class to use when constructing a new click.Command
    """

    class custom_command_class(click.Command):

        def invoke(self, ctx):
            # grab the config file
            config_file = ctx.params[config_file_param_name]
            param_defaults = {p.human_readable_name: p.default for p in self.params
                              if isinstance(p, click.core.Option)}
            param_defaults = {k: tuple(v) if type(v) is list else v for k, v in param_defaults.items()}
            param_cli = {k: tuple(v) if type(v) is list else v for k, v in ctx.params.items()}

            if config_file is not None:

                config_data = read_yaml(config_file)
                # modified to only use keys that are actually defined in options
                config_data = {k: tuple(v) if isinstance(v, yaml.comments.CommentedSeq) else v
                               for k, v in config_data.items() if k in param_defaults.keys()}

                # find differences btw config and param defaults
                diffs = set(param_defaults.items()) ^ set(param_cli.items())

                # combine defaults w/ config data
                combined = {**param_defaults, **config_data}

                # update cli params that are non-default
                keys = [d[0] for d in diffs]
                for k in set(keys):
                    combined[k] = ctx.params[k]

                ctx.params = combined

            return super().invoke(ctx)

    return custom_command_class


def get_command_defaults(command: click.Command):
    """ Get the default values for the options of `command`
    """
    return {tmp.name: tmp.default for tmp in command.params if not tmp.required}



def read_yaml(yaml_file: str) -> dict:
    """ Read a yaml file into dict object

    Parameters:
        yaml_file (str): path to yaml file

    Returns:
        return_dict (dict): dict of yaml contents
    """
    with open(yaml_file, 'r') as f:
        yml = yaml.YAML(typ='safe')
        return yml.load(f)


def write_yaml(yaml_file: str, data: dict) -> None:
    """ Write a dict object into a yaml file

    Parameters:
        yaml_file (str): path to yaml file
        data (dict): dict of data to write to `yaml_file`
    """
    yaml.SafeDumper.ignore_aliases = lambda *args: True

    with open(yaml_file, 'w') as f:
        yaml.safe_dump(data, f)


def update_yaml(yaml_file: str, data: dict) -> None:
    """ Update a yaml file with a dict object

    Parameters:
        yaml_file (str): path to yaml file
        data (dict): dict of data to write to `yaml_file`
    """

    with open(yaml_file, 'r') as f:
        yml = yaml.YAML(typ='safe')
        cur_yaml = yml.load(f)
        cur_yaml.update(data)

    write_yaml(yaml_file, cur_yaml)


def ensure_dir(path: str) -> str:
    """ Creates the directories specified by path if they do not already exist.

    Parameters:
        path (str): path to directory that should be created

    Returns:
        return_path (str): path to the directory that now exists
    """
    path = os.path.abspath(path)
    if not os.path.exists(path):
        try:
            os.makedirs(path)
        except OSError as exception:
            # if the exception is raised because the directory already exits,
            # than our work is done and everything is OK, otherwise re-raise the error
            # THIS CAN OCCUR FROM A POSSIBLE RACE CONDITION!!!
            if exception.errno != errno.EEXIST:
                raise
    return path


def timestamp():
    """
    Return string of current time
    """
    now = datetime.now()
    return now.strftime("%H:%M:%S_%m_%d_%Y")


def nearly_square(n):
    """

    """
    pairs = [(i, int(n / i)) for i in range(1, int(n ** 0.5) + 1) if n % i == 0]
    diffs = [np.abs(a - b) for (a, b) in pairs]
    min_pair = pairs[np.argmin(diffs)]
    if 1 in min_pair:
        q = int(np.ceil(np.sqrt(n)))
        return (q, q)
    else:
        return min_pair

def stable_sigmoid(x):
    return torch.where(x < 0, torch.exp(x) / (1 + torch.exp(x)), torch.exp(-1*x) / ((1+torch.exp(-1*x))))

def curl(f, spacings=1):
    '''Returns the curl of a batch of 2d (3d) vector fields shaped batch x dim x spatial x spatial (x spatial)'''
    num_dims = f.shape[1]
    if num_dims > 4:
        raise ValueError('Curl is only defined for dim <=3.')
    elif num_dims < 3:
        b = f.shape[0]
        s = f.shape[-1]
        f = torch.tile(f.unsqueeze(-1),(s,))
        f = torch.cat((f, torch.zeros(b,1,s,s,s)), dim=1)
        spacings = [sp for sp in spacings]
        spacings.append(spacings[-1])
        spacings = tuple(spacings)
    #J = torch.rot90(jacobian(f,spacings=spacings),-1, (1,2))
    J = jacobian(f,spacings=spacings)
    # J = [[dFxdx, dFxdy],[dFydx,dFydy]]
    # or
    # J = [[dFxdx, dFxdy, dFxdz],[dFydx,dFydy,dFydz],[dFzdx, dFzdy, dFzdz]]
    # curl = nabla X F = [[dFzdy - dFydz],[dFxdz - dFzdx],[dFydx - dFxdy]]

    dFxdy = J[:,0,1]
    dFxdz = J[:,0,2]
    dFydx = J[:,1,0]
    dFydz = J[:,1,2]
    dFzdx = J[:,2,0]
    dFzdy = J[:,2,1]
    return torch.stack([dFzdy - dFydz, dFxdz - dFzdx, dFydx - dFxdy]).movedim(1,0)

def divergence(f,spacings=1):
    '''Returns the divergence of a batch of planar vector fields shaped batch x dim x spatial x spatial'''

    # J.shape = batch x dim x dim x [spatial]^n
    J = jacobian(f,spacings=spacings)
    #return torch.diagonal(torch.rot90(J,-1, (1,2)),dim1=1,dim2=2).sum(-1)
    return torch.diagonal(J,dim1=1,dim2=2).sum(-1)

def jacobian(f,spacings=1):
    '''Returns the Jacobian of a batch of planar vector fields shaped batch x dim x spatial x spatial'''
    num_dims = f.shape[1]
    return torch.stack([torch.stack(torch.gradient(f[:,i],dim=list(range(1,num_dims+1)), spacing=spacings)) for i in range(num_dims)]).movedim(2,0)

def laplacian(f):
    '''Calculate laplacian of vector field'''
    num_dims = f.shape[1]
    if num_dims>3:
        raise ValueError('Laplacian not yet implemented for dim>2.')
    return torch.stack([divergence(torch.stack(torch.gradient(f[:,i], dim=[1,2])).movedim(1,0)) for i in range(num_dims)]).movedim(1,0)
