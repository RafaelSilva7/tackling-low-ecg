import torch
import torch.nn as nn


class Linear(nn.Module):
    '''Linear model'''

    def __init__(self, 
                 in_dim: int,
                 out_dim: int,
                 sample_size: int,
                 **kwargs) -> None:
        '''Linear model.
        
        Args:
            in_dim: The model input dimension.
            out_dim: Output's dimension of the model.
            sample_size (int): Size of the input sample.
        '''
        super().__init__(**kwargs)
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(in_features=in_dim * sample_size, out_features=out_dim)

    def forward(self, x: torch.Tensor):
        x_ = self.flatten(x)
        return self.linear(x_)



class LinearClassifier(Linear):
    '''Linear classifier model'''

    def __init__(self, 
                 in_dim: int, 
                 out_dim: int, 
                 sample_size: int, 
                 **kwargs) -> None:
        '''Linear classifier model.
        
        Args:
            in_dim: The model input dimension.
            out_dim: Output's dimension of the model.
            sample_size (int): Size of the input sample.
        '''
        super().__init__(in_dim, out_dim, sample_size, **kwargs)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x: torch.Tensor):
        x_ = super().forward(x)
        return self.softmax(x_)
    

class LinearRegressor(Linear):
    '''Linear regressor model'''

    def __init__(self, 
                 in_dim: int, 
                 out_dim: int, 
                 sample_size: int, 
                 **kwargs) -> None:
        '''Linear regressor model.
        
        Args:
            in_dim: The model input dimension.
            out_dim: Output's dimension of the model.
            sample_size (int): Size of the input sample.
        '''
        super().__init__(in_dim, out_dim, sample_size, **kwargs)

    def forward(self, x: torch.Tensor):
        return super().forward(x)