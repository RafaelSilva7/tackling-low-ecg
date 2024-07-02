from typing import Optional
from numpy import dtype
import torch
from pathlib import Path

from src.utils import load_json
from .dataset import Dataset


def _load_pt(path: str):

    train_data = torch.load(path)
    samples, labels = train_data['samples'], train_data['labels']

    # convert [batch, signal] to [batch, channel, signal]
    if samples.ndim == 2: 
        samples = samples.unsqueeze(dim=1)

    return samples, labels



def downstream_dataset(path_dir: str, 
                       verbose: Optional[bool] = True,
                       sample_dtype: dtype = torch.float32
                       ) -> tuple[Dataset, Dataset, Dataset]:
    '''Load a Downstream task dataset splited into train, validation and test.
    
    Args:
        path_dir (str): Path of the directory dataset.
        sample_dtype (dtype): Convert `dtype` of the samples. Default `torch.float32`.

    Returns:
        train (Dataset), validation (Dataset), test (Dataset)
    '''
    path_dir = Path(path_dir)

    metadata_path = Path(path_dir / "metadata.json")
    metadata = load_json(metadata_path) if metadata_path.is_file() else None

    train_samples, train_labels = _load_pt(path_dir / 'train.pt')
    train_dataset = Dataset(train_samples.to(sample_dtype), train_labels, metadata)

    val_samples, val_labels = _load_pt(path_dir / 'val.pt')
    val_dataset = Dataset(val_samples.to(sample_dtype), val_labels, metadata)

    test_samples, test_labels = _load_pt(path_dir / 'test.pt')
    test_dataset = Dataset(test_samples.to(sample_dtype), test_labels, metadata)

    if verbose:
        print(f"Dataset: {path_dir}")
        print(f"Info: train - {train_dataset.samples.shape}", end=' | ')
        print(f"val - {val_dataset.samples.shape}", end=' | ')
        print(f"test - {test_dataset.samples.shape}\n")

    return train_dataset, val_dataset, test_dataset



def pretrain_dataset(path_dir: str, 
                     verbose: Optional[bool] = True,
                     sample_dtype: dtype = torch.float32
                     ) -> tuple[Dataset, Dataset]:
    '''Load a pretrain dataset splited into train and validation.
    
    Args:
        path_dir (str): Path of the directory dataset.
        sample_dtype (dtype): Convert `dtype` of the samples. Default `torch.float32`.

    Returns:
        train (Dataset), validation (Dataset)
    '''
    path_dir = Path(path_dir)

    metadata_path = Path(path_dir / "metadata.json")
    metadata = load_json(metadata_path) if metadata_path.is_file() else None

    train_samples, train_labels = _load_pt(path_dir / 'train.pt')
    train_dataset = Dataset(train_samples.to(sample_dtype), train_labels, metadata)

    val_samples, val_labels = _load_pt(path_dir / 'val.pt')
    val_dataset = Dataset(val_samples.to(sample_dtype), val_labels, metadata)

    if verbose:
        print(f"Dataset: {path_dir}")
        print(f"Info: train - {train_dataset.samples.shape}", end=' | ')
        print(f"val - {val_dataset.samples.shape}\n")

    return train_dataset, val_dataset