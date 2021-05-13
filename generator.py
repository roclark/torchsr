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
import math
import torch
from torch import nn, Tensor

from residual import ResidualBlock, SubpixelConvolutionLayer


NUM_RESIDUAL = 16


class Generator(nn.Module):
    """
    Generator to create a new image with the requested upscaling.

    Parameters
    ----------
    scale_factor : int
        An ``int`` of the amount the image should be upscaled in each
        direction.
    """
    def __init__(self, scale_factor: int = 4) -> None:
        super(Generator, self).__init__()
        num_conv_layers = int(math.log(scale_factor, 2))

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=9, stride=1, padding=4),
            nn.PReLU()
        )

        blocks = []
        for _ in range(NUM_RESIDUAL):
            blocks.append(ResidualBlock(channels=64))
        self.blocks = nn.Sequential(*blocks)

        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64)
        )

        conv_layers = []

        for _ in range(num_conv_layers):
            conv_layers.append(SubpixelConvolutionLayer(64))
        self.conv_layers = nn.Sequential(*conv_layers)

        self.conv3 = nn.Conv2d(64, 3, kernel_size=9, stride=1, padding=4)

    def forward(self, x: Tensor) -> Tensor:
        """
        Complete a forward pass of the generator.

        Parameters
        ----------
        x : Tensor
            A ``tensor`` representing a single batch of images.

        Returns
        -------
        Tensor
            Returns a ``tensor`` of the final output from the last
            convolutional layer.
        """
        conv1 = self.conv1(x)
        block = self.blocks(conv1)
        conv2 = self.conv2(block)
        out = torch.add(conv1, conv2)
        out = self.conv_layers(out)
        out = self.conv3(out)
        return out
