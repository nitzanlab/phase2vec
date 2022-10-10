import glob
import itertools
import os
import sys
import uuid
from typing import Any, List, Protocol, Literal

import numpy as np
import pandas as pd

from src.utils import ensure_dir, read_yaml, write_yaml


def get_scan_values(scale: Literal['log', 'linear', 'list'], range: List, type='float') -> List:
    """ Generate concrete scan values given a scan specification
    Parameters:
        scale (Literal['log', 'linear', 'list']): type of scale to scan.
        range (List): arguments for the scale
        type (dtype-like): dtype for the scale
    Returns:
        (List): List of values to scan over
    """
    if scale == 'list':
        return np.array(range).astype(type).tolist()
    elif scale == 'linear':
        return np.linspace(*range).astype(type).tolist()
    elif scale == 'log':
        ls_args = [np.log10(x) if i < 2 else x for i, x in enumerate(range)]
        return np.logspace(*ls_args).astype(type).tolist()
    else:
        raise ValueError(f'Unknown scale {scale}')


class CommandWrapper(Protocol):
    def __call__(self, cmd: str, output: str = None, **kwds: Any) -> str: ...


def wrap_command_with_slurm(cmd: str, partition: str, ncpus: int, memory: str, wall_time: str,
                            output: str = None) -> str:
    """ Wraps a command to be run as a SLURM sbatch job
    Parameters:
        cmd (str): Command to be wrapped
        partition (str): Partition on which to run this job
        ncpus (int): Number of CPU cores to allocate to this job
        memory (str): Amount of memory to allocate to this job. ex: "2GB"
        wall_time (str): Amount of wall time allocated to this job. ex: "1:00:00"
        output (str): Path of file to write output to
    Returns:
        (str): the slurm wrapped command
    """
    preamble = f'sbatch --partition {partition} --nodes 1 --ntasks-per-node 1 --cpus-per-task {ncpus} --mem {memory} --time {wall_time}'
    if output is not None:
        preamble += f' --output "{output}"'
    escaped_cmd = cmd.replace('"', r'\"')
    return f'{preamble} --wrap "{escaped_cmd}";'


def wrap_command_with_local(cmd: str, output: str = None) -> str:
    """ Wraps a command to be run locally. Admittedly, this does not do too much
    Parameters:
        cmd (str): Command to be wrapped
        output (str): Path of file to write output to
    Returns:
        (str): the wrapped command
    """
    if output is not None:
        return cmd
    else:
        return cmd + f' > "{output}"'


def generate_gridsearch_worker_params(scan_file: str) -> List[dict]:
    """ Given a path to YAML scan configuration file, read the contents
        and generate a dictionary for each implied job
    Parameters:
        scan_file (str): path to a yaml scan configuration file
    Returns:
        (List[dict]): a list of dicts, each dict containing parameters to a single job
    """
    scan_settings = read_yaml(scan_file)
    base_parameters = scan_settings['parameters']
    scans = scan_settings['scans']

    # Rename model_save_dir for this gs
    model_save_dir = os.path.join(base_parameters['model_save_dir'], scan_settings['gs_name'])
    ensure_dir(model_save_dir)
    base_parameters['model_save_dir'] = model_save_dir

    worker_dicts = []
    for scan in scans:

        scan_param_products = []
        if 'scan' in scan:
            scan_param_gens = {}
            for sp in scan['scan']:
                scan_param_gens[sp['parameter']] = get_scan_values(sp['scale'], sp['range'], sp['type'])

            for params in itertools.product(*scan_param_gens.values()):
                scan_param_products.append(dict(zip(scan_param_gens.keys(), params)))
        else:
            # single empty dict, since we want to run with the scan parameters once
            # even though we dont have any iterable parameters
            scan_param_products.append({})

        if 'parameters' in scan:
            scan_base_params = scan['parameters']
        else:
            scan_base_params = {}

        for spp in scan_param_products:
            # scan-specific params, combo of static and dynamic
            scan_params = {**scan_base_params, **spp}
            # final params incorporating scan-specific params
            final_params = {**base_parameters, **scan_params}

            id = str(uuid.uuid4())
            exp_name = base_parameters['data_name'] + '_' + id
            final_params['exp_name'] = exp_name

            # rename the job, based on the specific params
            # name_suffix = '_'.join([f'{k}-{v}' for k, v in scan_params.items()])
            # final_params['name'] = f"{final_params['data_name']}_{name_suffix}"

            worker_dicts.append(final_params)

    return worker_dicts


def write_jobs(worker_dicts: List[dict], cluster_format: CommandWrapper, dest_dir: str) -> None:
    """ Write job configurations to YAML files, and write job invocations to stdout
    Parameters:
        worker_dicts (List[dict]): Job configurations to write
        cluster_format (Callable[[str], str]): A callable to format the job invoation for a given environment
        dest_dir (str): directory to write job configurations to
    """
    for worker in worker_dicts:
        # worker['uuid'] = uuid.uuid4()
        worker_dest = os.path.join(dest_dir, f"{worker['exp_name']}.yaml")
        write_yaml(worker_dest, worker)
        ensure_dir(worker['model_save_dir'])
        output = os.path.join(worker['model_save_dir'], f"{worker['exp_name']}.log")
        work_cmd = f'flow-encoding train --config-file "{worker_dest}";'
        full_cmd = cluster_format(work_cmd, output=output)

        sys.stdout.write(full_cmd + '\n')


def get_gridsearch_default_scans() -> List:
    """ Generate default scan configuration
    """
    return [
        {
            'scan': [
                {
                    'parameter': 'beta',
                    'type': 'float',
                    'scale': 'linear',
                    'range': [.1, 1.0, 2]
                }, {
                    'parameter': 'gamma',
                    'type': 'float',
                    'scale': 'linear',
                    'range': [.1, 1.0, 2]
                }
            ]
        }]


def results_to_df(path: str) -> pd.DataFrame:
    """ Find and aggregate grid search results
    Parameters:
        path (str): path to search for experiments
    """
    experiments = glob.glob(os.path.join(path, '*.yaml'))

    exp_data = []
    for exp in experiments:
        id = uuid.uuid4()
        exp_data.append(read_yaml(exp))

    ## tag each dict with the model ID

    # time_data = {f'time_{k}': v for k, v in data['train_time']}
    # exp_data.append({
    #    'id': id,
    #    **data['parameters'],
    #    **time_data,
    #    **data['model_performance']
    # })
    return pd.DataFrame(exp_data)
