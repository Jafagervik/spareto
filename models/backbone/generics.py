import torch
from torch import nn


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, stride: int = 1, padding: int = 0, pool_size:int = 2, bias: bool = True):
        super().__init__()

        self.c1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)
        self.act = nn.GELU()
        self.pool = nn.MaxPool2d(kernel_size=pool_size)

    def forward(self, x):
        x = self.c1(x)
        x = self.act(x)
        x = self.pool(x)

        return x
