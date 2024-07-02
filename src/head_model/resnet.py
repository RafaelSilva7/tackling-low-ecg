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


class ResNetBlock(nn.Module):

    def __init__(self, in_dim, out_dim, **kwargs) -> None:
        '''ResNetBlock Module
        
        Args:
            in_dim (int): Input dimension.
            out_dim (int): Output dimension.
        '''
        super().__init__(**kwargs)

        # Convolutios by kernel num
        self.conv_8 = nn.Conv1d(in_dim, out_dim,
                                kernel_size=8,
                                padding='same')
        
        self.conv_5 = nn.Conv1d(out_dim, out_dim,
                                kernel_size=5,
                                padding='same')
        
        self.conv_3 = nn.Conv1d(out_dim, out_dim,
                                kernel_size=8,
                                padding='same')

        self.conv_shortcut = nn.Conv1d(in_dim, out_dim,
                                       kernel_size=1,
                                       padding='same')

        self.bn_8 = nn.BatchNorm1d(out_dim)
        self.bn_5 = nn.BatchNorm1d(out_dim)
        self.bn_3 = nn.BatchNorm1d(out_dim)

        self.activation = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # First convolution with kernel 8
        conv_x = self.conv_8(x)
        conv_x = self.bn_8(conv_x)
        conv_x = self.activation(conv_x)

        # Second convolution with kernel 5
        conv_y = self.conv_5(conv_x)
        conv_y = self.bn_5(conv_y)
        conv_y = self.activation(conv_y)

        # Third convolution with kernel 3
        conv_z = self.conv_3(conv_y)
        conv_z = self.bn_3(conv_z)

        # Expand channels for the sum with shortcut
        shortcut_ = self.conv_shortcut(x)
        shortcut_ = self.bn_8(shortcut_)

        # Prepare the output summing the shortcut
        out = shortcut_ + conv_z
        out = self.activation(out)
        return out


class ResNet(nn.Module):

    def __init__(self, 
                 in_dim: int, 
                 hidden_dim: int, 
                 **kwargs) -> None:
        '''ResNet Module
        
        Args:
            in_dim (int): Input dimension.
            hidden_dim (int): Hidden dimension.
        '''
        super().__init__(**kwargs)

        self.block_1 = ResNetBlock(in_dim, hidden_dim)
        self.block_2 = ResNetBlock(hidden_dim, hidden_dim * 2)
        self.block_3 = ResNetBlock(hidden_dim * 2, hidden_dim * 2)

        self.global_avg_pooling = GAP1d()


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out_1 = self.block_1(x)
        out_2 = self.block_2(out_1)
        out_3 = self.block_3(out_2)

        gap_ = self.global_avg_pooling(out_3)
        # gap_ = gap_layer.squeeze()
        return gap_
    

class ResNetClassifier(ResNet):
    def __init__(self, 
                 in_dim: int, 
                 out_dim: int, 
                 hidden_dim: int = 64, 
                 **kwargs) -> None:
        '''ResNet Classifier
        
        Args:
            in_dim (int): Input dimension.
            out_dim (int): Number of classes.
            hidden_dim (int): Hidden dimension. Default 64.
        '''
        super().__init__(in_dim, hidden_dim, **kwargs)
        self.output_layer = nn.Linear(hidden_dim * 2, out_features=out_dim)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_ = super().forward(x)
        x_ = self.output_layer(x_)
        return self.softmax(x_)


class ResNetRegressor(ResNet):
    def __init__(self, 
                 in_dim: int, 
                 out_dim: int = 1, 
                 hidden_dim: int = 64, 
                 **kwargs) -> None:
        '''ResNet Regressor
        
        Args:
            in_dim (int): Input dimension.
            out_dim (int): Number of classes.
            hidden_dim (int): Hidden dimension. Default 64.
        '''
        super().__init__(in_dim, hidden_dim, **kwargs)
        self.output_layer = nn.Linear(in_features=hidden_dim * 2, out_features=out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_ = super().forward(x)
        return self.output_layer(x_)