# Copyright 2021 Dakewe Biotech Corporation. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from torch import nn, Tensor


class SubpixelConvolutionLayer(nn.Module):
    """
    Module to perform subpixel convolutions in the generator.

    Parameters
    ----------
    channels : int
        An ``int`` of the number of channels to use in the convolutional layer.
    """
    def __init__(self, channels: int = 64) -> None:
        super(SubpixelConvolutionLayer, self).__init__()
        self.conv = nn.Conv2d(channels, channels * 4, kernel_size=3, stride=1, padding=1)
        self.pixel_shuffle = nn.PixelShuffle(upscale_factor=2)
        self.prelu = nn.PReLU()

    def forward(self, x: Tensor) -> Tensor:
        """
        Complete a forward pass of the subpixel convolution block.

        Parameters
        ----------
        x : Tensor
            A ``tensor`` of the previous convolution layers in the generator.

        Returns
        -------
        Tensor
            Returns a ``tensor`` after the convolutions and pixel shuffles.
        """
        out = self.conv(x)
        out = self.pixel_shuffle(out)
        out = self.prelu(out)
        return out


class ResidualBlock(nn.Module):
    """
    A Residual Block as part of the generator.

    Parameters
    ----------
    channels : int
        An ``int`` of the number of channels to use in the convolutional and
        batch normalization layers.
    """
    def __init__(self, channels: int = 64) -> None:
        super(ResidualBlock, self).__init__()

        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.prelu = nn.PReLU()
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x: Tensor) -> Tensor:
        """
        Complete a forward pass of the Residual Block.

        Parameters
        ----------
        x : Tensor
            A ``tensor`` of output from the convolutional layers in the
            generator.

        Returns
        -------
        Tensor
            Returns a ``tensor`` of the batch normalized output from the
            Residual Block.
        """
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.prelu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = out + x
        return out
