import torch
from typing import Optional

from .ts2vec import TS2Vec, trainer as ts2vec_trainer
from .tstcc import Encoder, trainer as tstcc_trainer



def instance_model(model: str,
                   model_args: dict, 
                   optim_args: dict,
                   tuning_mode: str, 
                   model_params: Optional[dict] = None,
                   device: Optional[str] = "cuda"):
    '''Instanciate the SSL Model's feature extractor (encoder) and load its pretrained params.
    
    Args:
        model (str): SSL Model, values `cae` | `ts2vec` | `tstcc`.
        model_args (dict): Feature extractor constructor args.
        optim_args (dict): Optimizer constructor args.
        tuning_mode (str): Tuning mode, `linear-probing` | `fine-tuning`.
        model_params (dic, optional): Pretrained parameters of the feature extractor model.
        If `None` the model will be instanciated with random parameters. Default `None`.
        device (str, optional): Device used for training and inference. Default "cuda".

    Return:
        Feature extractor (encoder) and it optimizer.
    '''
    if model == "ts2vec":
        ssl_model = TS2Vec(device=device, **model_args)

        # load pretrained model and set the tuning mode.
        if model_params:
            ssl_model.load_state_dict(model_params) 
            
        ssl_model.tuning_mode(tuning_mode)
        optimizer = torch.optim.AdamW(ssl_model._net.parameters(), **optim_args)

    if model == "tstcc":
        ssl_model = Encoder(**model_args['encoder'])

        # load pretrained model and set the tuning mode.
        if model_params:
            ssl_model.load_state_dict(model_params['encoder'])
            
        ssl_model.tuning_mode(tuning_mode)
        optimizer = torch.optim.Adam(ssl_model.parameters(), **optim_args)
    
    return ssl_model, optimizer


def get_trainer(model: str):
    '''Get the trainer of the Feature extractor model.'''

    if model == "ts2vec":
        return ts2vec_trainer

    if model == "tstcc":
        return tstcc_trainer
