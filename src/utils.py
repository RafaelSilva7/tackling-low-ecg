from copy import copy
import json
import torch
import torch.nn as nn
import random
import numpy as np
from datetime import datetime
from pathlib import Path



class JsonEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.ndarray, torch.Tensor)):
            return obj.tolist()
        return super().default(obj)


def interp_dataset(data, new_size: int, mode: str, inplace=False):
    '''Interpolate the samples of a `Dataset`.
    
    Args:
        data (Dataset): Dataset to be resampled.
        new_size (int): Output sampling size.
        mode (str): Algorithm used for upsampling: `'nearest'` | `'linear'` | `'bilinear'` | `'bicubic'` | `'trilinear'` | `'area'` | `'nearest-exact'`. Default: `'nearest'`.
        inplace (bool):  Whether to modify the Dataset rather than creating a new one. Default = `False`.
        
    Return:
        Resampled dataset (`Dataset`).
    '''
    resampled = nn.functional.interpolate(data.samples, new_size, mode=mode)

    if inplace:
        data.samples = resampled.float()
    else:
        new_data = copy(data)
        new_data.samples = resampled.float()

        return new_data


def load_json(path) -> dict:
    with open(path) as f:
        conf = json.load(f)
    return conf


def save_json(path, data):
    with open(path, 'w') as f:
        json.dump(data, f, cls=JsonEncoder)


def name_with_datetime(prefix='default', drop_min=False):
    _format = "%Y%m%d_%H" if drop_min else "%Y%m%d_%H%M%S"
    now = datetime.now()
    return prefix + '__' + now.strftime(_format)


def task_dir(task_conf: dict):
    '''Create dir name by combining the `model`, `train_mode` and `dataset` name.'''
    dir_name = Path(task_conf['dataset'])
    dir_name = dir_name / task_conf['model'] / task_conf['train_mode'] 

    return dir_name


def set_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)  # Numpy module.
    random.seed(seed)  # Python random module.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True