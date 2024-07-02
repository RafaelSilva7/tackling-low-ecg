import torch

from .mlp import MLPClassifier
from .linear import LinearClassifier
from .oates_mlp import OatesMLPClassifier
from .fcn import FCNClassifier
from .resnet import ResNetClassifier



def classifier(model: str,
               model_args: dict, 
               optim_args: dict):
    '''Instanciate a classifier Head Model.
    
    Args:
        model (str): Head Model, values `mlp` | `linear` | `oates-mlp` | `fcn`.
        model_args (dict): Feature extractor constructor args.
        optim_args (dict): Optimizer constructor args.
        device (str): Device used for training and inference. Default "cuda".

    Return:
        Head model and it optimizer.
    '''
    if model == 'mlp':
        head_model = MLPClassifier(**model_args)

    if model == 'linear':
        head_model = LinearClassifier(**model_args)

    if model == 'oates-mlp':
        head_model = OatesMLPClassifier(**model_args)

    if model == 'fcn':
        head_model = FCNClassifier(**model_args)

    if model == 'resnet':
        head_model = ResNetClassifier(**model_args)

    head_optim = torch.optim.Adam(head_model.parameters(), **optim_args)

    return head_model, head_optim
