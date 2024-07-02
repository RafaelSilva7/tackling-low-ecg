import numpy as np
from typing import Optional, Sequence

import torch
from torch import Tensor
from torch.utils.data import Dataset



class Dataset(Dataset):
    '''Custom class representing a ``Dataset``.
    '''
    def __init__(self, 
                 samples: Tensor, 
                 label: Tensor, 
                 metadata: Optional[dict] = None) -> None:
        '''
        Args:
            samples (Tensor): Sample of the dataset.
            label (Tensor): Labels of the samples.
            metadata (dict, optional): Metadata of the dataset. Default = None.
        '''
        super().__init__()

        self.samples = samples
        self.label = label
        self.metadata = metadata


    def __getitem__(self, idx):
        return self.samples[idx], self.label[idx]
    
    
    def __len__(self):
        return len(self.samples)
    

    def permute(self, dims):
        self.samples = torch.permute(self.samples, dims)


    def get_subset(self, 
                   size: Optional[int], 
                   subset: Optional[Sequence[int]],
                   replace: Optional[bool] = False) -> Dataset:
        '''Extract a subset of the current dataset.

        Args:
            size (int, optional): Size of the subset.
            subset (Sequence[int], optional): Samples indices to be selected as subset. If informed `size` is ignored.
            replace (bool, optional): Whether the sample is with or without replacement. Default = False.

        Returns:
            subset of the current dataset (Dataset).

        Notes:
            The Dataset `metadata` property is not passed to the subset.
        '''
        assert size is None and subset is None, "At least one, 'size' or 'subset', must be not None."

        if subset is None:
            all_idx = np.arange(len(self.samples))
            subset = np.random.choice(all_idx, size, replace)

        samples = self.samples[subset].detach()
        label = self.label[subset].detach()

        return Dataset(samples, label)

    
    def to_dict(self) -> dict:
        '''Dictionary representation of ``Dataset``'''
        return {"samples": self.samples, "label": self.label}