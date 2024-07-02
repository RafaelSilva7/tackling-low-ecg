from datetime import datetime
import numpy as np

import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.dataloader import Dataset
from src.upsampler import ESPCN1d


def _interp_data(data: Dataset, model_args: dict) -> Dataset:
    "Create a new Dataset by downsampling the `data` to be used as `samples` and use its original version as `label`."
    resampled = nn.functional.interpolate(data.samples, model_args['size_in'], mode='linear')
    dataset = Dataset(samples=resampled, label=data.samples)

    return dataset


def pretrain(dataset: tuple[Dataset, Dataset, Dataset],
             train_params: dict,
             model_args: dict,
             optim_args: dict,
             device: str = "cuda"):
    
    train_data, val_data, test_data = dataset

    # Interpolate (downsample) the dataset to be the input samples, 
    # and use the original data as label
    train_dataset = _interp_data(train_data, model_args)
    train_dl = DataLoader(train_dataset, train_params['batch'], shuffle=True)

    val_dataset = _interp_data(val_data, model_args)
    val_dl = DataLoader(val_dataset, train_params['batch'], shuffle=True)

    test_dataset = _interp_data(test_data, model_args)
    test_dl = DataLoader(test_dataset, train_params['batch'], shuffle=False)

    upsampler = ESPCN1d(**model_args)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(upsampler.parameters(), **optim_args)
    scheduler = optim.lr_scheduler.LinearLR(optimizer, start_factor=0.3, 
                                            total_iters=train_params['epochs']//5)

    train_losses = []
    val_losses = []

    upsampler.to(device)

    # TRAIN PHASE -----------------------------------------------
    since = datetime.now()
    for epoch in range(train_params['epochs']):

        # Training ----------------------------------------------
        upsampler.train()
        epoch_train_losses = []

        for (data, labels) in tqdm(train_dl, desc=f"{epoch+1:2.0f}/{train_params['epochs']}", total=len(train_dl)):
            data, labels = data.to(device), labels.to(device)

            optimizer.zero_grad()

            preds = upsampler(data)

            loss = criterion(preds, labels)
            epoch_train_losses.append(loss.item())

            loss.backward()
            optimizer.step()

        train_loss = np.mean(epoch_train_losses)
        train_losses.append(train_loss)
        scheduler.step()
        
        # Validation ----------------------------------------------
        upsampler.eval()
        epoch_val_losses = []

        with torch.no_grad():
            for (data, labels) in val_dl:
                data, labels = data.to(device), labels.to(device)

                preds = upsampler(data)

                loss = criterion(preds, labels)
                epoch_val_losses.append(loss.item())

        val_loss = np.mean(epoch_val_losses)
        val_losses.append(val_loss)

        print(f"[Train {train_loss} | Val {val_loss}]")

    train_time = datetime.now() - since

    print(f"\nFinished! Training time {train_time}")


    # TEST PHASE -----------------------------------------------
    upsampler.eval()
    test_loss = []

    with torch.no_grad():
        for (data, labels) in test_dl:
            data, labels = data.to(device), labels.to(device)

            preds = upsampler(data)

            loss = criterion(preds, labels)
            test_loss.append(loss.item())

    test_loss = np.array(test_loss)

    print(f"Test loss: {test_loss.mean():.4f}  ({test_loss.std():.4f} std)")

    upsampler.cpu()

    checkpoint = {'upsampler': upsampler.state_dict()}
    logs = {
        'train_time': train_time.total_seconds(),
        'train_loss': np.array(train_losses),
        'val_loss': np.array(val_losses),
        'test_loss': test_loss.mean(),
    }

    return checkpoint, logs