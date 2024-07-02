from math import ceil
from torch import nn


# "long" and "short" denote longer and shorter samples
class PixelShuffle1D(nn.Module):
    """
    1D pixel shuffler. Upscales sample length by downscaling channel dimension.

    Github: https://github.com/serkansulun/pytorch-pixelshuffle1d/tree/master 
      
    Reference: Shi, W., Caballero, J., Huszár, F., Totz, J., Aitken, A.P., Bishop, R., Rueckert, D. and Wang, Z., 2016. 
    Real-time single image and video super-resolution using an efficient sub-pixel convolutional neural network. 
    In Proceedings of the IEEE conference on computer vision and pattern recognition (pp. 1874-1883).
    """
    def __init__(self, upscale_factor):
        super(PixelShuffle1D, self).__init__()
        self.upscale_factor = upscale_factor

    def forward(self, x):
        batch_size = x.shape[0]
        short_channel_len = x.shape[1]
        short_width = x.shape[2]

        long_channel_len = short_channel_len // self.upscale_factor
        long_width = self.upscale_factor * short_width

        x = x.contiguous().view([batch_size, self.upscale_factor, long_channel_len, short_width])
        x = x.permute(0, 2, 3, 1).contiguous()
        x = x.view(batch_size, long_channel_len, long_width)

        return x



class PixelUnshuffle1D(nn.Module):
    """
    Inverse 1D pixel shuffler. Upscales channel length by downscaling sample length.

    Github: https://github.com/serkansulun/pytorch-pixelshuffle1d/tree/master 
      
    Reference: Shi, W., Caballero, J., Huszár, F., Totz, J., Aitken, A.P., Bishop, R., Rueckert, D. and Wang, Z., 2016. 
    Real-time single image and video super-resolution using an efficient sub-pixel convolutional neural network. 
    In Proceedings of the IEEE conference on computer vision and pattern recognition (pp. 1874-1883).
    """
    def __init__(self, downscale_factor):
        super(PixelUnshuffle1D, self).__init__()
        self.downscale_factor = downscale_factor

    def forward(self, x):
        batch_size = x.shape[0]
        long_channel_len = x.shape[1]
        long_width = x.shape[2]

        short_channel_len = long_channel_len * self.downscale_factor
        short_width = long_width // self.downscale_factor

        x = x.contiguous().view([batch_size, long_channel_len, short_width, self.downscale_factor])
        x = x.permute(0, 3, 1, 2).contiguous()
        x = x.view([batch_size, short_channel_len, short_width])
        return x



class ESPCN1d(nn.Module):
    '''ESPCN for univariated time series.
    
    Adapted from: https://github.com/yjn870/ESPCN-pytorch/tree/master

    Reference:
    Shi, W., Caballero, J., Huszár, F., Totz, J., Aitken, A.P., Bishop, R., Rueckert, D. and Wang, Z., 2016. 
    Real-time single image and video super-resolution using an efficient sub-pixel convolutional neural network. 
    In Proceedings of the IEEE conference on computer vision and pattern recognition (pp. 1874-1883).
    '''

    def __init__(self, size_in: int, size_out: int):
        '''ESPCN for univariated time series.

        Args:
            size_in (int): Length of the input.
            size_out (int): Length of the output.
        '''
        super().__init__()
        self.size_in = size_in
        self.size_out = size_out
        self.scale_factor = ceil(size_out / size_in)

        # feature extractor
        self.first_part = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=5, padding=5//2),
            nn.Tanh(),
            nn.Conv1d(64, 32, kernel_size=3, padding=3//2),
            nn.Tanh(),
        )

        # upsampler module
        self.last_part = nn.Sequential(
            nn.Conv1d(32, self.scale_factor, kernel_size=3, padding=3 // 2),
            PixelShuffle1D(self.scale_factor),
            nn.Linear(size_in * self.scale_factor, size_out),
        )

    def forward(self, x):
        x = self.first_part(x)
        x = self.last_part(x)

        return x