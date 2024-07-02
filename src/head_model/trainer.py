import numpy as np
from typing import Any, Optional

import torch
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from src.ssl_model.ts2vec import TS2Vec
from src.upsampler import ESPCN1d


def fit(train_dataloader: DataLoader,
        head_model,
        head_optim: Optimizer, 
        criterion: Any, 
        ssl_model: Optional[Any] = None,
        ssl_optim: Optional[Optimizer] = None, 
        upsampler_model: ESPCN1d = None,
        upsampler_optim: Optimizer = None,
        fit_mode: Optional[str] = 'fine-tuning',
        device: Optional[str] = "cuda"
        ) -> tuple[np.ndarray, list[torch.Tensor]]:
    '''Fit the model to a downstream task.

    Args:
        train_dataloader (DataLoader): Dataloader of the training data.
        head_model: Head model.
        head_optim (Optimizer): Optimizer of the Head model.
        criterion: Loss function used to train the model.
        ssl_model: Feature extractor model (ssl model).
        ssl_optim (Optimizer): Optimizer of the feature extractor (ssl model).
        fit_mode (str): Training mode ['fine-tuning', 'linear-probing']. Default 'fine-tuning'.

    Returns:
        train loss (np.ndarray), batch predictions (list[Tensor])
    '''
    head_model.train()

    train_losses = []
    predictions = []

    # self-supervised tuning --------------------------------------
    if ssl_model is not None: 
        if fit_mode == 'linear-probing':
            ssl_model.eval()

            if upsampler_model:
               upsampler_model.eval() 
               upsampler_model.requires_grad_(False)
        
        else: 
            ssl_model.train()

            if upsampler_model:
               upsampler_model.train() 

        for (inputs, labels) in train_dataloader:
                inputs = inputs.to(device) 
                labels = labels.to(device)
                
                head_optim.zero_grad()
                ssl_optim.zero_grad()

                # upsampler the input data
                if upsampler_model:
                    upsampler_optim.zero_grad()
                    inputs = upsampler_model(inputs)

                # obtain the predictions
                repr = ssl_model.encode(inputs)
                pred = head_model(repr)

                loss = criterion(pred, labels)
                loss.backward()

                head_optim.step()
                ssl_optim.step()
                if upsampler_optim:
                    upsampler_optim.step()

                if isinstance(ssl_model, TS2Vec):
                    ssl_model.swa_update_params()

                train_losses.append(loss.item())
                predictions.append(pred.detach().cpu())

    # supervised tuning -----------------------------
    else: 
        if upsampler_model:
            upsampler_model.train() 
        
        for (inputs, labels) in train_dataloader:
            inputs = inputs.to(device) 
            labels = labels.to(device)
            
            head_optim.zero_grad()

            # upsampler the input data
            if upsampler_model:
                upsampler_optim.zero_grad()
                inputs = upsampler_model(inputs)

            # obtain the predictions
            pred = head_model(inputs)

            loss = criterion(pred, labels)
            loss.backward()

            head_optim.step()
            if upsampler_optim:
                upsampler_optim.step()

            train_losses.append(loss.item())
            predictions.append(pred.detach().cpu())

    return np.array(train_losses), predictions



def evaluate(val_dataloader: DataLoader,
             head_model,
             criterion: Any,
             ssl_model: Optional[Any] = None,
             upsampler_model: ESPCN1d = None,
             device: Optional[str] = "cuda"
             ) -> tuple[np.ndarray, list[torch.Tensor]]:
    '''Evaluate model in the donwstream task.
    
    Args:
        ssl_model: Pretrained feature extractor (ssl model).
        val_dataloader: Dataloader of the validation data.
        criterion: Loss function used to train the model.
    
    Returns:
        validation loss (np.ndarray), batch predictions (list[Tensor])
    '''
    head_model.eval()
    
    val_loss = []
    predictions = []

    # self-supervised tuning --------------------------------------
    if ssl_model:
        ssl_model.eval()

        if upsampler_model:
               upsampler_model.eval() 

        with torch.no_grad():
            for (inputs, labels) in val_dataloader:
                inputs = inputs.to(device)
                labels = labels.to(device)

                if upsampler_model:
                    inputs = upsampler_model(inputs) 

                repr = ssl_model.encode(inputs)
                pred = head_model(repr)

                loss = criterion(pred, labels)

                val_loss.append(loss.item())
                predictions.append(pred.detach().cpu())
    
    # supervised tuning evaluation -----------------------------------
    else: 
        if upsampler_model:
               upsampler_model.eval() 

        with torch.no_grad():
            for (inputs, labels) in val_dataloader:
                inputs = inputs.to(device)
                labels = labels.to(device)

                if upsampler_model:
                    inputs = upsampler_model(inputs) 

                pred = head_model(inputs)

                loss = criterion(pred, labels)

                val_loss.append(loss.item())
                predictions.append(pred.detach().cpu())

    return np.array(val_loss), predictions