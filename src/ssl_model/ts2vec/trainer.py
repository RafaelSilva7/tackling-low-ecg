from typing import Optional
from datetime import datetime

import torch.optim as optim

from src.dataloader import Dataset
from .ts2vec import TS2Vec



def pretrain(data: tuple[Dataset, Dataset], 
             train_params: dict,
             model_args: dict,
             optim_args: dict,
             device: Optional[str] = "cuda") -> tuple[dict, dict]:
    '''Pretrain the TS2Vec feature extractor model in the pretext task.

    Args:
        data (tuple[Dataset, Dataset]): Train and validation dataset.
        train_params (dict): Training params, epochs and batch size.
        model_args (dict): TS2Vec args to instance the model.
        optim_args (dict): Adam optimizer's constructor args.
        device (str, optional): Device used to run the model training. Default = "cuda".

    Returns:
        pretrained TS2Vec params (dict), training logs (dict)
    '''
    print(f"\nSelf-supervised TS2Vec pretraining ...")
    train_data, val_data = data

    ts2vec = TS2Vec(device=device, **model_args)
    ts2vec_optim = optim.AdamW(ts2vec._net.parameters(), **optim_args)
    
    # re-arrange the dimensions to ts2vec pattern (n_instance, n_timestamps, n_features)
    train_data.permute(dims=(0, 2, 1))
    val_data.permute(dims=(0, 2, 1))

    since = datetime.now()
    ts2vec_params, logs = ts2vec.fit_ssl(train_data.samples.numpy(), 
                                         val_data.samples.numpy(),
                                         ts2vec_optim, 
                                         train_params['batch'], 
                                         train_params['epochs'])
    ssl_time = datetime.now() - since
    logs['time'] = ssl_time.total_seconds()

    print(f"\nFinished! Training time {ssl_time}")
    print(f"Validation loss: {logs['val_loss_mean']:.6f} ({logs['val_loss_std']:.6f} std)")

    return ts2vec_params, logs
