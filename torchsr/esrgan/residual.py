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
from torch import nn, Tensor


class ResidualDenseBlock(nn.Module):
    """
    A Residual Dense Block as part of the generator.

    Parameters
    ----------
    channels : int
        An ``int`` of the number of channels to use in the convolutional layers.
    growth_channels : int
        An ``int`` of the number of growth channels to use in the convolutional
        layers.
    scale_ratio : float
        A ``float`` of the scale to use in the convolutional layers.
    """
    def __init__(self, channels: int = 64, growth_channels: int = 32,
                 scale_ratio: float = 0.2):
        super(ResidualDenseBlock, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(channels + 0 * growth_channels, growth_channels, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(channels + 1 * growth_channels, growth_channels, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(channels + 2 * growth_channels, growth_channels, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
        )

        self.conv4 = nn.Sequential(
            nn.Conv2d(channels + 3 * growth_channels, growth_channels, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
        )

        self.conv5 = nn.Conv2d(channels + 4 * growth_channels, channels, kernel_size=3, stride=1, padding=1)

        self.scale_ratio = scale_ratio

        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight)
                module.weight.data *= 0.1
                if module.bias is not None:
                    module.bias.data.zero_()

    def forward(self, x: Tensor) -> Tensor:
        """
        Complete a forward pass of the Residual Dense Block.

        Parameters
        ----------
        x : Tensor
            A ``tensor`` of output from the convolutional layers in the
            generator.

        Returns
        -------
        Tensor
            Returns a ``tensor`` of the batch normalized output from the
            Residual Dense Block.
        """
        conv1 = self.conv1(x)
        conv2 = self.conv2(torch.cat((x, conv1), dim=1))
        conv3 = self.conv3(torch.cat((x, conv1, conv2), dim=1))
        conv4 = self.conv4(torch.cat((x, conv1, conv2, conv3), dim=1))
        conv5 = self.conv5(torch.cat((x, conv1, conv2, conv3, conv4), dim=1))
        return conv5 * self.scale_ratio + x


class ResidualInResidualDenseBlock(nn.Module):
    """
    A block of multiple Residual Dense Blocks as part of the generator.

    Parameters
    ----------
    channels : int
        An ``int`` of the number of channels to use in the convolutional layers.
    growth_channels : int
        An ``int`` of the number of growth channels to use in the convolutional
        layers.
    scale_ratio : float
        A ``float`` of the scale to use in the convolutional layers.
    """
    def __init__(self, channels: int = 64, growth_channels: int = 32,
                 scale_ratio: float = 0.2):
        super(ResidualInResidualDenseBlock, self).__init__()
        self.RDB1 = ResidualDenseBlock(channels, growth_channels, scale_ratio)
        self.RDB2 = ResidualDenseBlock(channels, growth_channels, scale_ratio)
        self.RDB3 = ResidualDenseBlock(channels, growth_channels, scale_ratio)

    def forward(self, x: Tensor) -> Tensor:
        """
        Complete a forward pass of the Residual In Residual Dense Block.

        Parameters
        ----------
        x : Tensor
            A ``tensor`` of output from the convolutional layers in the
            generator.

        Returns
        -------
        Tensor
            Returns a ``tensor`` of the batch normalized output from the
            Residual In Residual Dense Block.
        """
        out = self.RDB1(x)
        out = self.RDB2(out)
        out = self.RDB3(out)
        return out * 0.2 + x
