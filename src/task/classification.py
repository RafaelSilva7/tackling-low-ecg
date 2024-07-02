import numpy as np
from typing import Any, Optional
from datetime import datetime
import torch
import tqdm

import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.optimizer import Optimizer

from src.upsampler import ESPCN1d
from src.dataloader import Dataset
from src.head_model import trainer
from .metrics import MulticlassMetrics



def fit_cls(cls_model: Any,
            cls_optim: Optimizer, 
            data: tuple[Dataset, Dataset],
            train_params: dict, 
            ssl_model: Optional[Any] = None, 
            ssl_optim: Optional[Optimizer] = None, 
            upsampler_model: ESPCN1d = None,
            upsampler_optim: Optimizer = None,
            train_mode: str = "supervised", 
            device: Optional[str] = "cuda"
            ) -> tuple[dict, dict]:
    '''Fit the classifier model on the multiclass classification task.
    
    Args:
        cls_model: Classifier model.
        ssl_model: Feature extractor model.
        data (tuple[Dataset, Dataset]): Train and validation dataset.
        train_params (dict): Dictionary with the training parameters.
        ssl_optim (Optimizer): Optimizer of the ``ssl_model``.
        cls_optim (Optimizer): Optimizer of the ``cls_model``.
        train_mode (str): Training mode [``'linear_probing'`` ``'fine_tuning'``].
        device (str, optional): Device used to run the models.

    Returns:
        trained parameters (dict), train logs (dict) 
    '''
    # send the models to device
    cls_model.to(device)
    if ssl_model:
        ssl_model.to(device)
    if upsampler_model:
        upsampler_model.to(device)


    train_dataset, val_dataset = data
    train_loader = DataLoader(train_dataset, train_params['batch'], shuffle=True)
    val_loader = DataLoader(val_dataset, train_params['batch'], shuffle=False)

    # Labels used to compute the accuracy of the model
    if val_loader.drop_last:
        last_idx = len(val_loader) * val_loader.batch_size
        val_labels = val_dataset.label[:last_idx]
    else:
        val_labels = val_dataset.label

    criterion = nn.CrossEntropyLoss()
    scheduler = optim.lr_scheduler.LinearLR(cls_optim, 
                                            start_factor=0.3, 
                                            total_iters=train_params['epochs']//5)

    cls_metric = MulticlassMetrics(num_classes=train_dataset.metadata['num_classes'])

    train_loss = []
    val_loss = []
    val_metrics = []

    print(f"\nFit model [Multiclass classification]")
    pbar = tqdm.trange(train_params['epochs'])

    since = datetime.now()
    for _ in pbar:

        # Train phase -----------------
        epoch_train_loss, _ = trainer.fit(train_loader, cls_model, cls_optim, criterion,
                                          ssl_model, ssl_optim, upsampler_model, 
                                          upsampler_optim, train_mode, device)
        
        scheduler.step()

        train_loss_value = epoch_train_loss.mean()
        train_loss.append(train_loss_value)

        # Validation phase -----------------
        epoch_val_loss, val_pred = trainer.evaluate(val_loader, cls_model, 
                                                    criterion, ssl_model, 
                                                    upsampler_model, device)

        val_loss_value = epoch_val_loss.mean()
        val_loss.append(val_loss_value)

        val_metric = cls_metric.compute(torch.cat(val_pred), val_labels)
        val_metrics.append(val_metric)

        # print the mean of the epoch train and validation loss
        pbar.set_description(f"[Train {train_loss_value:.4f} | Val {val_loss_value:.4f}]")
    

    train_time = datetime.now() - since

    # send the models to CPU
    cls_model.cpu()
    if ssl_model:
        ssl_model.cpu()
    if upsampler_model:
        upsampler_model.cpu()

    # save models' parameters
    checkpoint = {'sl_model_params': cls_model.state_dict(),}
    if ssl_model:
        checkpoint['ssl_model_params'] = ssl_model.state_dict()
    if upsampler_model:
        checkpoint['upsampler_model_params'] = upsampler_model.state_dict()

    # save the trainning logs
    logs = {
        'train_time': train_time.total_seconds(),
        'train_loss': np.array(train_loss),
        'val_loss': np.array(val_loss),
        'val_metrics': val_metrics,
    }

    print(f"\nFinished! Training time {train_time}")
    print(f"Validation loss: {logs['val_loss'].mean():.4f} (std {logs['val_loss'].std():.4f})")

    return checkpoint, logs



def evaluate_cls(cls_model: Any, 
                 test_data: Dataset, 
                 train_params: dict, 
                 ssl_model: Optional[Any] = None, 
                 upsampler_model: ESPCN1d = None,
                 device: Optional[str]= "cuda"
                 ) -> tuple[np.ndarray, dict, np.ndarray]:
    '''Evaluate the classifier model on the multiclass classification task.
    
    Args:
        ssl_model: Feature extractor model.
        cls_model: Classifier model.
        test_data (Dataset): Test dataset used to evalute the models.
        train_params (dict): Dictionary with the training parameters.
        device (str, optional): Device used to run the models.

    Returns:
        evaluation loss (np.ndarray), accuracy (float), predicted classes (np.ndarray)
    '''
    cls_model.to(device)
    cls_model.eval()

    if ssl_model:
        ssl_model.to(device)
        ssl_model.eval()

    if upsampler_model:
        upsampler_model.to(device)
        upsampler_model.eval()

    test_dataloader = DataLoader(test_data, train_params['batch'], 
                                 shuffle=False, drop_last=False)

    criterion = nn.CrossEntropyLoss()
    loss, predictions = trainer.evaluate(test_dataloader, cls_model,
                                         criterion, ssl_model,
                                         upsampler_model, device)

    metric = MulticlassMetrics(test_data.metadata['num_classes'], 
                               compute_matrix=True)

    pred_classes = torch.cat(predictions)
    metrics = metric.compute(pred_classes, test_data.label)

    print(f"Loss: {loss.mean():.4f}  ({loss.std():.4f} std)")
    print(f"Metrics: {metrics}")

    cls_model.cpu()
    if ssl_model:
        ssl_model.cpu()

    if upsampler_model:
        upsampler_model.cpu()

    logs = {
        'loss': loss, 
        'metrics': metrics, 
        'pred_classes': pred_classes.numpy()
    }

    return logs