import torch.nn as nn
from torch import Tensor



class MLP(nn.Module):
    def __init__(self,  
                 in_dim: int, 
                 sample_size: int,
                 hidden_dim: int,
                 **kwargs) -> None:
        '''Multilayer model.
        
        Args:
            in_dim (int): The model input dimension.
            out_dim (int): Output's dimension of the model.
            sample_size (int, optional): Size of the input sample.
            hidden_dim (int, optional): The model hidden dimension.
        '''
        super().__init__(**kwargs)
    
        self.net = nn.Sequential(
            nn.Linear(in_dim * sample_size, hidden_dim),
            nn.ReLU(),

            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

    def forward(self, x: Tensor) -> Tensor:
        x_ = x.reshape(x.shape[0], -1)
        return self.net(x_)


class MLPClassifier(MLP):
    '''Multilayer classifier'''
    def __init__(self, 
                 in_dim: int, 
                 sample_size: int, 
                 hidden_dim: int, 
                 out_dim: int, 
                 **kwargs) -> None:
        '''Multilayer classifier model.
        
        Args:
            in_dim (int): The model input dimension.
            sample_size (int, optional): Size of the input sample.
            hidden_dim (int, optional): The model hidden dimension.
            out_dim (int): Output's dimension of the model.
        '''
        super().__init__(in_dim, sample_size, hidden_dim, **kwargs)

        self.output_layer = nn.Linear(hidden_dim, out_dim)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x: Tensor) -> Tensor:
        x_ = super().forward(x)
        x_ = self.output_layer(x_)

        return self.softmax(x_)
    

class MLPRegressor(MLP):
    '''Multilayer regressor'''
    def __init__(self, 
                 in_dim: int, 
                 sample_size: int, 
                 hidden_dim: int, 
                 out_dim: int = 1, 
                 **kwargs) -> None:
        '''Multilayer regressor model.
        
        Args:
            in_dim (int): The model input dimension.
            sample_size (int, optional): Size of the input sample.
            hidden_dim (int, optional): The model hidden dimension.
            out_dim (int): Output's dimension of the model. Default 1.
        '''
        super().__init__(in_dim, sample_size, hidden_dim, **kwargs)
        self.output_layer = nn.Linear(hidden_dim, out_dim)

    def forward(self, x: Tensor) -> Tensor:
        x_ = super().forward(x)
        return self.output_layer(x_)