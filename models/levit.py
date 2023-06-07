"""
Implementation based on this paper: https://arxiv.org/pdf/2104.01136.pdf
"""

import torch
from torch import convolution, nn
import torchvision
import os
import numpy as np


"""
Cycle goes like this:

224 x 224 x 3 --> 256 x 14 x 14 --> 384 x 7 x 7 --> 512 x 4 x 4 --> 512


Conv
Conv
Conv
Conv
STAGE 1 -> 4 attention, 4  MLP
Shrink

STAGE 2 -> 5 MLP, 4 attention
Shrink

STAGE 3 -> 5 MLP, 4 attention

AvgPool

Return to Supervised classifier and Distallation classifier
"""


def shrink(heads: int):
    pass


class Attention(nn.Module):

    def __init__(self, num_heads: int = 8) -> None:
        super().__init__()

        self.num_heads = num_heads

    def forward(self, x):

        return x


class Stage(nn.Module):
    """
    Represents one of the stages in the LeViT Net
    """

    def __init__(
            self,
            stage: int,
            heads: int,
            num_mlps: int = 5,
            num_attention: int = 4):
        """
        Attr:
            heads, int: number od heads to use in the attention
        """
        super().__init__()

        self.heads = heads
        self.mlps = num_mlps
        self.attentions = num_attention
        self.stage = stage

    def forward(self, x):
        """
        Forward pass through a single stage
        """
        if self.stage == 1:
            for _ in range(self.attentions):
                x = attention(self.heads)
                x = MLP(x)

        else:
            x = MLP(x)
            for _ in range(self.attentions):
                x = attention(self.heads)
                x = MLP(x)

        return x


class MLP(nn.Module):
    """
    Represents the Multi Layer Perceptron blocks that will be used
    """

    def __init__(self, x):
        super().__init__()
        self.x = x

    def forward(self):

        return self.x


class StartConvolutions(nn.Module):
    """
    Represents the first 4 convolutions in the LeViT Net
    """

    def __init__(self, in_features: int):
        super().__init__()

        self.c1 = nn.Conv2d(in_features, 1000, kernel_size=(
            3, 3), stride=1, padding=1)
        self.c2 = nn.Conv2d(1000, 2000, kernel_size=(
            3, 3), stride=1, padding=1)
        self.c3 = nn.Conv2d(2000, 500, kernel_size=(
            3, 3), stride=1, padding=1)
        self.c4 = nn.Conv2d(500, 256*14*14, kernel_size=(
            3, 3), stride=1, padding=1)

        # Activation will be GeLU
        self.act = nn.GELU()

    def forward(self, x):
        """
        First for convolutions of the LeViT net
        """
        x = self.act(self.c1(x))
        x = self.act(self.c2(x))
        x = self.act(self.c3(x))
        x = self.act(self.c4(x))

        return x


class LeViT256(nn.Module):
    """
    TODO: Decide if this should be the architecture or not
    """

    def __init__(self, in_features: int):
        super().__init__()

        # Start
        self.start_convolutions = StartConvolutions(in_features)

        self.s1 = Stage(stage=1, num_mlps=4, num_attention=4, heads=4)
        self.s2 = Stage(stage=2, num_mlps=5, num_attention=4, heads=6)
        self.s3 = Stage(stage=3, num_mlps=5, num_attention=4, heads=8)

        self.shrink1 = shrink(8)
        self.shrink2 = shrink(12)

        self.avgpool = nn.AvgPool2d(kernel_size=(2, 2))

    def forward(self, x):
        x = StartConvolutions(x)

        # Stage 1
        x = self.s1(x)

        x = self.shrink1(x)

        # Stage 2
        x = self.s2(x)

        x = self.shrink2(x)

        # Stage 3
        x = self.s3(x)

        # Pooling before dividing result
        x = self.avgpool(x)

        return x
