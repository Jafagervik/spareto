from typing import List
import torch 
from torch import nn 
import torch.nn.functional as F


class DoubleConvBlock(nn.Module):
    def __init__(self, inp: int, out: int) -> None:
        super().__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(inp, out, kernel_size=3, bias=False),
            nn.BatchNorm2d(out),
            nn.ReLU(inplace=True),
            nn.Conv2d(out, out, kernel_size=3, bias=False),
            nn.BatchNorm2d(out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.layer(x)


class Downsample(nn.Module):
    def __init__(self, inp: int, out: int):
        super().__init__()
        self.layer = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConvBlock(inp, out),
        )

    def forward(self, x):
        return self.layer(x)


class Upsample(nn.Module):
    def __init__(self, inp: int, out: int):
        super().__init__()
        self.up = nn.ConvTranspose2d(inp, inp // 2, kernel_size=2, stride=2)
        self.conv = DoubleConvBlock(inp, out)

    def forward(self, x1, x2):
        x1 = self.up(x1)

        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])

        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class Encoder(nn.Module):
    def __init__(self, layers: List[int]):
        super().__init__()
        self.layer = nn.Sequential(
            DoubleConvBlock(layers[0], layers[1]),
            Downsample(layers[1], layers[2]),
            Downsample(layers[2], layers[3]),
            Downsample(layers[3], layers[4]),
            Downsample(layers[4], layers[5]),
        )

    def forward(self, x):
        return self.layer(x)


class Decoder(nn.Module):
    def __init__(self, layers: List[int]):
        super().__init__()
        self.layer = nn.Sequential(
            Upsample(layers[0], layers[1]),
            Upsample(layers[1], layers[2]),
            Upsample(layers[2], layers[3]),
            Upsample(layers[3], layers[4]),
            nn.Conv2d(layers[4], layers[5], kernel_size=1)
        )

    def forward(self, x):
        return self.layer(x)



class Unet(nn.Module):
    def __init__(self, layers: List[int],) -> None:
        super().__init__()
        self.enc = Encoder(layers)

        # [3, 64, 128, 256, 512, 1024, num_classes=10]
        # [10, 1024, 512, 256, 128, 64, 3] 
        # [3, 1024, 512, 256, 128, 64, 10] 
        rev = list(reversed(layers))
        rev[0], rev[-1] = rev[-1], rev[0]
        rev.pop(0)

        self.dec= Decoder(rev)


    def forward(self, x):
        x = self.enc(x)
        x = self.dec(x)
        return x


if __name__ == "__main__":
    num_classes = 10
    layers = [3,64,128,256,512,1024,num_classes]
    model = Unet(layers)
    assert(1 == 1)
