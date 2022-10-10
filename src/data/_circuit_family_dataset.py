import os
import pickle
import torch
import glob
from torch.utils.data import Dataset


def custom_loader(input):
    """
    Pickle loader
    """
    (data, parameter) = pickle.load(open(input, 'rb'))
    data_dim = len(data.shape) - 1
    permuted_dims = [data_dim] + list(range(data_dim))
    return torch.tensor(data).permute(*permuted_dims), torch.tensor(parameter)


class CircuitFamilyDataset(Dataset):
    def __init__(self, data_dir, transform_data=None, transform_label=None):
        self.data_dir = data_dir
        self.fname_list = glob.glob(os.path.join(self.data_dir, '*.pkl'))
        # self.transform_data = transform_data
        # self.transform_label = transform_label

    def __len__(self):
        return len(self.fname_list)

    def __getitem__(self, item):
        fname = self.fname_list[item]
        (data, parameter) = pickle.load(open(fname, 'rb'))
        data_dim = len(data.shape) - 1
        permuted_dims = [data_dim] + list(range(data_dim))
        return torch.tensor(data).permute(*permuted_dims), torch.tensor(parameter)
