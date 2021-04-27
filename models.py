import torch
from torch import nn


NUM_RESIDUAL = 8


class UpsampleBlock(nn.Module):
    def __init__(self, in_channels, scale_factor, kernel_size=3, padding=1):
        super(UpsampleBlock, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels,
                      in_channels * (scale_factor ** 2),
                      kernel_size=kernel_size,
                      padding=padding),
            nn.PixelShuffle(scale_factor),
            nn.PReLU()
        )

    def forward(self, x):
        return self.net(x)


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
        super(ResidualBlock, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size,
                      padding=padding),
            nn.BatchNorm2d(out_channels),
            nn.PReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size,
                      padding=padding),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, x):
        return x + self.net(x)


class Generator(nn.Module):
    def __init__(self, num_residual=NUM_RESIDUAL):
        super(Generator, self).__init__()
        self.num_residual = num_residual
        self.conv1 = nn.Sequential(nn.Conv2d(3, 64, kernel_size=9, padding=4),
                                   nn.PReLU())
        for block in range(num_residual):
            self.add_module(f'residual{block + 1}', ResidualBlock(64, 64))
        self.conv2 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, padding=1),
                                   nn.PReLU())
        self.upsample = nn.Sequential(
            UpsampleBlock(64, 2),
            UpsampleBlock(64, 2),
            nn.Conv2d(64, 3, kernel_size=9, padding=4)
        )

    def forward(self, x):
        y = self.conv1(x)
        identity = y.clone()

        for block in range(self.num_residual):
            y = self.__getattr__(f'residual{block + 1}')(y)
        y = self.conv2(y)
        y = self.upsample(y + identity)
        return (torch.tanh(y) + 1.0) / 2.0


class Discriminator(nn.Module):
    def __init__(self, slope=0.2):
        super(Discriminator, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.LeakyReLU(slope),

            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(slope),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(slope),

            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(slope),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(slope),

            nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(slope),

            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(slope),

            nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(slope),

            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(512, 1024, kernel_size=1),
            nn.LeakyReLU(slope),
            nn.Conv2d(1024, 1, kernel_size=1)
        )

    def forward(self, x):
        y = self.net(x)
        si = torch.sigmoid(y).view(y.size()[0])
        return si
