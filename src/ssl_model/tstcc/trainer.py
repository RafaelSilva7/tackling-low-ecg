from typing import Optional
from datetime import datetime

import torch
import torch.optim as optim

from src.dataloader import Dataset
from src.upsampler import ESPCN1d
from .tc import TC
from .model import BaseModel as Encoder
from .dataloader import Load_Dataset
from .augmentations import ConfigAug



def pretrain(data: tuple[Dataset, Dataset], 
             train_params: dict,
             model_args: dict,
             optim_args: dict,
             upsampler_args: dict = None,
             device: Optional[str] = "cuda") -> tuple[dict, dict]:
    '''Pretrain the TS-TCC feature extractor model in the pretext task.

    Args:
        data (tuple[Dataset, Dataset]): Train and validation dataset.
        train_params (dict): Training params, epochs and batch size.
        model_args (dict): TS-TCC args to instance the model.
        optim_args (dict): Adam optimizer's constructor args.
        device (str, optional): Device used to run the model training. Default = "cuda".

    Returns:
        pretrained TS-TCC params (dict), training logs (dict)
    '''
    train_data, val_data = data
    
    # create agumented versions of the datasets
    config_aug = ConfigAug(**model_args['augmentation'])
    
    train_dataset = Load_Dataset(train_data.to_dict(), config_aug, 
                                 training_mode="self_supervised")
    
    val_dataset = Load_Dataset(val_data.to_dict(), config_aug, 
                               training_mode="self_supervised")

    # TS-TCC model and its optimizers
    tc = TC(device=device, **model_args['tc_model']) # temporal contrasting module
    tc_optim = optim.Adam(tc.parameters(), **optim_args)

    encoder = Encoder(**model_args['encoder']) # feature extractor module
    encoder_optim = torch.optim.Adam(encoder.parameters(), **optim_args)

    if upsampler_args:
        upsampler = ESPCN1d(**upsampler_args)
        upsampler_optim = torch.optim.Adam(upsampler.parameters(), **optim_args)
    else:
        upsampler = None
        upsampler_optim = None

    since = datetime.now()
    print(f"\nSelf-supervised TS-TCC pretraining ...")
    tstcc_params, logs = tc.fit_ssl(encoder, encoder_optim, tc_optim, 
                                    train_dataset, val_dataset, 
                                    upsampler=upsampler, 
                                    upsampler_optim=upsampler_optim,
                                    n_epochs=train_params['epochs'], 
                                    batch_size=train_params['batch'], 
                                    **model_args['cc_loss'])
    time = datetime.now() - since
    logs['time'] = time.total_seconds()

    print(f"\nFinished! Training time {time}")
    print(f"Validation loss: {logs['val_loss_mean']:.6f} ({logs['val_loss_std']:.6f} std)")

    return tstcc_params, logs
