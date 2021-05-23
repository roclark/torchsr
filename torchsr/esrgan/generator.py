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
import torch
import torch.nn.functional as F
from torch import nn, Tensor

from torchsr.esrgan.residual import ResidualInResidualDenseBlock


NUM_RESIDUAL = 23


class Generator(nn.Module):
    """
    Generator to create a new image with the requested upscaling.

    Parameters
    ----------
    num_rrdb_blocks : int
        An ``int`` of the number of Residual in Residual Blocks to use.
    """
    def __init__(self, num_rrdb_blocks: int = NUM_RESIDUAL) -> None:
        super(Generator, self).__init__()

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)

        blocks = []
        for _ in range(num_rrdb_blocks):
            blocks += [ResidualInResidualDenseBlock(channels=64, growth_channels=32, scale_ratio=0.2)]
        self.blocks = nn.Sequential(*blocks)

        self.conv2 =nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)

        self.upsample1 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.upsample2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)

        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
        )

        self.conv4 = nn.Conv2d(64, 3, kernel_size=3, stride=1, padding=1)

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
        out = F.leaky_relu(self.upsample1(F.interpolate(out, scale_factor=2, mode='nearest')),
                           negative_slope=0.2,
                           inplace=True)
        out = F.leaky_relu(self.upsample2(F.interpolate(out, scale_factor=2, mode='nearest')),
                           negative_slope=0.2,
                           inplace=True)
        out = self.conv3(out)
        out = self.conv4(out)
        return out
