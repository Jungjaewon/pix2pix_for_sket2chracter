import torch
import torch.nn as nn

from spectral_normalization import SpectralNorm


class ConvDownBlock(nn.Module):
    def __init__(self, dim_in, dim_out, spec_norm=False):
        super(ConvDownBlock, self).__init__()

        if spec_norm:
            self.main = nn.Sequential(
                nn.Conv2d(dim_in, dim_out, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(dim_out, affine=True, track_running_stats=True),
                nn.LeakyReLU(0.01, inplace=False),
                nn.Conv2d(dim_out, dim_out, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(dim_out, affine=True, track_running_stats=True),
                nn.LeakyReLU(0.01, inplace=False),
                nn.Conv2d(dim_out, dim_out, kernel_size=4, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(dim_out, affine=True, track_running_stats=True),
            )
        else:
            self.main = nn.Sequential(
                SpectralNorm(nn.Conv2d(dim_in, dim_out, kernel_size=3, stride=1, padding=1, bias=False)),
                nn.BatchNorm2d(dim_out, affine=True, track_running_stats=True),
                nn.LeakyReLU(0.01, inplace=False),
                SpectralNorm(nn.Conv2d(dim_out, dim_out, kernel_size=3, stride=1, padding=1, bias=False)),
                nn.BatchNorm2d(dim_out, affine=True, track_running_stats=True),
                nn.LeakyReLU(0.01, inplace=False),
                SpectralNorm(nn.Conv2d(dim_out, dim_out, kernel_size=4, stride=2, padding=1, bias=False)),
                nn.BatchNorm2d(dim_out, affine=True, track_running_stats=True),
            )

    def forward(self, x):
        return self.main(x)


class ConvUpBlock(nn.Module):
    def __init__(self, dim_in, dim_out, spec_norm=False):
        super(ConvUpBlock, self).__init__()

        if spec_norm:
            self.main = nn.Sequential(
                nn.Conv2d(dim_in, dim_out, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(dim_out, affine=True, track_running_stats=True),
                nn.LeakyReLU(0.01, inplace=False),
                nn.Conv2d(dim_out, dim_out, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(dim_out, affine=True, track_running_stats=True),
                nn.LeakyReLU(0.01, inplace=False),
                nn.UpsamplingBilinear2d(scale_factor=2),
                nn.BatchNorm2d(dim_out, affine=True, track_running_stats=True),
            )
        else:
            self.main = nn.Sequential(
                SpectralNorm(nn.Conv2d(dim_in, dim_out, kernel_size=3, stride=1, padding=1, bias=False)),
                nn.BatchNorm2d(dim_out, affine=True, track_running_stats=True),
                nn.LeakyReLU(0.01, inplace=False),
                SpectralNorm(nn.Conv2d(dim_out, dim_out, kernel_size=3, stride=1, padding=1, bias=False)),
                nn.BatchNorm2d(dim_out, affine=True, track_running_stats=True),
                nn.LeakyReLU(0.01, inplace=False),
                nn.UpsamplingBilinear2d(scale_factor=2),
                nn.BatchNorm2d(dim_out, affine=True, track_running_stats=True),
            )

    def forward(self, x):
        return self.main(x)


class ResidualBlock(nn.Module):
    """Residual Block with some normalization. Conv : W = (W - F + 2P) /S + 1 / TransPosed : W = (Win - 1) * S - 2P + F + OutP"""
    def __init__(self, dim_in, dim_out):
        super(ResidualBlock, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(dim_out, affine=True, track_running_stats=True),
            nn.LeakyReLU(0.01, inplace=False),
            nn.Conv2d(dim_out, dim_out, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(dim_out, affine=True, track_running_stats=True))

    def forward(self, x):
        return x + self.main(x)


class ResidualBlockUp(nn.Module):
    """Residual Block with some normalization. Conv : W = (W - F + 2P) /S + 1 / TransPosed : W = (Win - 1) * S - 2P + F + OutP"""
    def __init__(self, dim_in, dim_out, spec_norm=False):
        super(ResidualBlockUp, self).__init__()

        if spec_norm:
            self.stream1 = nn.Sequential(
                nn.UpsamplingBilinear2d(scale_factor=2),
                SpectralNorm(nn.Conv2d(dim_in, dim_out, kernel_size=1, stride=1, bias=False))
            )
            self.stream2 = nn.Sequential(
                nn.BatchNorm2d(dim_in, affine=True, track_running_stats=True),
                nn.ReLU(),
                nn.UpsamplingBilinear2d(scale_factor=2),
                SpectralNorm(nn.Conv2d(dim_in, dim_in, kernel_size=3, stride=1, padding=1, bias=False)),
                nn.BatchNorm2d(dim_in, affine=True, track_running_stats=True),
                nn.ReLU(),
                SpectralNorm(nn.Conv2d(dim_in, dim_out, kernel_size=3, stride=1, padding=1, bias=False)),
            )
        else:
            self.stream1 = nn.Sequential(
                nn.UpsamplingBilinear2d(scale_factor=2),
                nn.Conv2d(dim_in, dim_out, kernel_size=1, stride=1, bias=False)
            )
            self.stream2 = nn.Sequential(
                nn.BatchNorm2d(dim_in, affine=True, track_running_stats=True),
                nn.ReLU(),
                nn.UpsamplingBilinear2d(scale_factor=2),
                nn.Conv2d(dim_in, dim_in, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(dim_in, affine=True, track_running_stats=True),
                nn.ReLU(),
                nn.Conv2d(dim_in, dim_out, kernel_size=3, stride=1, padding=1, bias=False),
            )

    def forward(self, x):
        return self.stream1(x) + self.stream2(x)


class ResidualBlockDown(nn.Module):
    """Residual Block with some normalization. Conv : W = (W - F + 2P) /S + 1 / TransPosed : W = (Win - 1) * S - 2P + F + OutP"""
    def __init__(self, dim_in, dim_out, spec_norm=False):
        super(ResidualBlockDown, self).__init__()

        if spec_norm:
            self.stream1 = nn.Sequential(
                SpectralNorm(nn.Conv2d(dim_in, dim_out, kernel_size=1, stride=1, bias=False)),
                nn.AvgPool2d(kernel_size=2, stride=2),
            )
            self.stream2 = nn.Sequential(
                nn.ReLU(),
                nn.Conv2d(dim_in, dim_in, kernel_size=3, stride=1, padding=1, bias=False),
                nn.ReLU(),
                SpectralNorm(nn.Conv2d(dim_in, dim_out, kernel_size=3, stride=1, padding=1, bias=False)),
                nn.AvgPool2d(kernel_size=2, stride=2),
            )
        else:
            self.stream1 = nn.Sequential(
                nn.Conv2d(dim_in, dim_out, kernel_size=1, stride=1, bias=False),
                nn.AvgPool2d(kernel_size=2, stride=2),
            )
            self.stream2 = nn.Sequential(
                nn.ReLU(),
                nn.Conv2d(dim_in, dim_in, kernel_size=3, stride=1, padding=1, bias=False),
                nn.ReLU(),
                nn.Conv2d(dim_in, dim_out, kernel_size=3, stride=1, padding=1, bias=False),
                nn.AvgPool2d(kernel_size=2, stride=2),
            )

    def forward(self, x):
        return self.stream1(x) + self.stream2(x)


class NoiseBlock(nn.Module):
    """https://pytorch.org/docs/stable/torch.html?highlight=rand#torch.randn"""
    def __init__(self):
        super(NoiseBlock, self).__init__()
    def forward(self, input):
        return input + torch.rand_like(input)