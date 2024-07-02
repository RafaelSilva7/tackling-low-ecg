import torch
import torch.nn as nn

class GAP1d(nn.Module):
    '''Global Average Pooling Layer'''
    
    def __init__(self, output_size: int = 1) -> None:
        '''Global Average Pooling Layer'''
        super().__init__()
        self.gap = nn.AdaptiveAvgPool1d(output_size)
        self.flatten = nn.Flatten()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.flatten(self.gap(x))


class FCN(nn.Module):
    '''Fully Convolutional Networks - FCN'''

    def __init__(self, in_dim: int, **kwargs) -> None:
        '''Fully Convolutional Networks - FCN
        
        Args:
            in_dim (int): Input dimension.
        '''
        super().__init__(**kwargs)

        self.layers = nn.ModuleList([
            nn.Conv1d(in_channels=in_dim,
                      out_channels=128,
                      kernel_size=8,
                      padding='same'),
            nn.BatchNorm1d(128),
            nn.ReLU(),

            nn.Conv1d(in_channels=128,
                      out_channels=256,
                      kernel_size=5,
                      padding='same'),
            nn.BatchNorm1d(256),
            nn.ReLU(),

            nn.Conv1d(in_channels=256,
                      out_channels=128,
                      kernel_size=3,
                      padding='same'),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            GAP1d()
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x
    

class FCNClassifier(FCN):
    '''FCN Classifier'''

    def __init__(self, in_dim: int, out_dim: int, **kwargs) -> None:
        '''FCN Classifier
        
        Args:
            in_dim (int): Input dimension.
            out_dim (int): Number of classes.
        '''
        super().__init__(in_dim, **kwargs)
        self.output_layer = nn.Linear(in_features=128, out_features=out_dim)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_ = super().forward(x)
        x_ = self.output_layer(x_)
        return self.softmax(x_)


class FCNRegressor(FCN):
    '''FCN Regressor'''

    def __init__(self, in_dim: int, out_dim: int = 1, **kwargs) -> None:
        '''FCN Regressor
        
        Args:
            in_dim (int): Input dimension.
            out_dim (int): Output dimension. Default = 1.
        '''
        super().__init__(in_dim,  **kwargs)
        self.output_layer = nn.Linear(in_features=128, out_features=out_dim)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_ = super().forward(x)
        return self.output_layer(x_)