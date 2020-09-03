import torch.nn as nn
import torch

from block import ResidualBlockDown
from block import ResidualBlockUp


class Generator(nn.Module):
    """Generator network. Conv : W = (W - F + 2P) /S + 1 / TransPosed : W = (Win - 1) * S - 2P + F + OutP"""
    def __init__(self, spec_norm=False):
        super(Generator, self).__init__()
        self.main = list()
        self.main.append(ResidualBlockDown(3, 32, spec_norm))  # 512 -> 256
        self.main.append(ResidualBlockDown(32, 64, spec_norm))  # 256 -> 128
        self.main.append(ResidualBlockDown(64, 128, spec_norm))  # 128 -> 64
        self.main.append(ResidualBlockDown(128, 256, spec_norm))  # 64 -> 32

        self.main.append(ResidualBlockUp(256, 128, spec_norm))  # 32 -> 64
        self.main.append(ResidualBlockUp(128, 64, spec_norm))  # 64 -> 128
        self.main.append(ResidualBlockUp(64, 32, spec_norm))  # 128 -> 256
        self.main.append(ResidualBlockUp(32, 16, spec_norm))  # 256 -> 512

        self.main.append(nn.Conv2d(16, 3, kernel_size=3, stride=1, padding=1))  # 512 -> 512
        self.main.append(nn.Tanh())
        self.main = nn.Sequential(*self.main)

    def forward(self, x):
        return self.main(x)


class Discriminator(nn.Module):
    """Discriminator network with PatchGAN.
    W = (W - F + 2P) /S + 1"""

    def __init__(self, spec_norm=True):
        super(Discriminator, self).__init__()
        self.main = list()
        self.main.append(ResidualBlockDown(3, 32, spec_norm)) # 512 -> 256
        self.main.append(ResidualBlockDown(32, 64, spec_norm)) # 256 -> 128
        self.main.append(ResidualBlockDown(64, 128, spec_norm)) # 128 -> 64
        self.main.append(ResidualBlockDown(128, 256, spec_norm)) # 64 -> 32
        self.main.append(nn.Conv2d(256, 1, kernel_size=3, stride=1, padding=1))
        self.main = nn.Sequential(*self.main)

    def forward(self, x):
        return self.main(x)