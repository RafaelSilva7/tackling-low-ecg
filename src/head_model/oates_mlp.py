import torch
import torch.nn as nn



class OatesMLP(nn.Module):
    '''Oates MLP'''

    def __init__(self, 
                 in_dim: int, 
                 sample_size: int,
                 **kwargs) -> None:
        '''
        Args:
            in_dim (int): The model input dimension.
            sample_size (int, optional): Size of the input sample.
        '''
        super().__init__(**kwargs)

        self.flatten = nn.Flatten()

        self.layers = nn.ModuleList([
            nn.Dropout(p=0.1),
            nn.Linear(in_features=in_dim * sample_size, out_features=500),
            nn.ReLU(),
            
            nn.Dropout(p=0.2),
            nn.Linear(in_features=500, out_features=500),
            nn.ReLU(),
            
            nn.Dropout(p=0.2),
            nn.Linear(in_features=500, out_features=500),
            nn.ReLU(),
               
            nn.Dropout(p=0.3)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_ = self.flatten(x)

        for layer in self.layers:
            x_ = layer(x_)

        return x_
    


class OatesMLPClassifier(OatesMLP):
    '''MLP Classifier - Strong baseline for time series.'''
    def __init__(self, 
                 in_dim: int, 
                 out_dim: int, 
                 sample_size: int, 
                 **kwargs) -> None:
        '''
        Args:
            in_dim (int): The model input dimension.
            out_dim (int): Number of classes.
            sample_size (int, optional): Size of the input sample.
        '''
        super().__init__(in_dim, sample_size, **kwargs)

        self.output_layer = nn.Sequential(
            nn.Linear(in_features=500, out_features=out_dim),
            nn.Softmax(dim=1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_ = super().forward(x)
        return self.output_layer(x_)
    


class OatesMLPRegressor(OatesMLP):
    '''MLP Regressor - Strong baseline for time series.'''
    def __init__(self, 
                 in_dim: int, 
                 sample_size: int, 
                 out_dim: int = 1, 
                 **kwargs) -> None:
        '''
        Args:
            in_dim (int): The model input dimension.
            sample_size (int, optional): Size of the input sample.
            out_dim (int): Dimension of the repression. Default 1.
        '''
        super().__init__(in_dim, sample_size, **kwargs)
        self.output_layer = nn.Linear(in_features=500, out_features=out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_ = super().forward(x)
        return self.output_layer(x_)
    