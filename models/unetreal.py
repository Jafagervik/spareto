from typing import List
import torch 
from torch import nn 
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, inp: int, out: int) -> None:
        super().__init__()
        self.c = nn.Conv2d(inp, out, 3, 1, 1)
        self.bn = nn.BatchNorm2d(out)

    def forward(self, x):
        return F.relu(self.bn(self.c(x)))


class Encoder(nn.Module):
    def __init__(self, layers: List[int]) -> None:
        super().__init__()
        self.l1 = nn.Sequential( 
            ConvBlock(layers[0], layers[1]),
            ConvBlock(layers[1], layers[2]),
        )
        self.l2 = nn.Sequential( 
            ConvBlock(layers[1], layers[2]),
            ConvBlock(layers[2], layers[3]),
        )
        self.l3 = nn.Sequential( 
            ConvBlock(layers[2], layers[3]),
            ConvBlock(layers[3], layers[4]),
        )
        self.l4 = nn.Sequential( 
            ConvBlock(layers[3], layers[4]),
            ConvBlock(layers[4], layers[5]),
        )
        self.pool = nn.MaxPool2d(2,2)


    def forward(self, x):
        x1 = self.pool(self.l1(x))
        x2 = self.pool(self.l2(x1))
        x3 = self.pool(self.l3(x2))
        x4 = self.pool(self.l4(x3))

        return x4, x3, x2, x1


class Decoder(nn.Module):
    def __init__(self, layers: List[int], x4, x3, x2, x1) -> None:
        super().__init__()
        self.l1 = nn.Sequential( 
            torch.cat((x3, ConvBlock(layers[0], layers[1])), dim=1),
            ConvBlock(layers[1], layers[2]),
        )
        self.l2 = nn.Sequential( 
            torch.cat((x2, ConvBlock(layers[1], layers[2])), dim=1),
            ConvBlock(layers[2], layers[3]),
        )
        self.l3 = nn.Sequential( 
            torch.cat((x1, ConvBlock(layers[2], layers[3])), dim=1),
            ConvBlock(layers[2], layers[3]),
            ConvBlock(layers[3], layers[4]),
        )
        self.l4 = nn.Sequential( 
            ConvBlock(layers[3], layers[4]),
            ConvBlock(layers[4], layers[5]),
        )

        self.upconv = nn.MaxUnpool2d(2,2)


    def forward(self, x):
        x = self.upconv(self.l1(x))
        x = self.upconv(self.l2(x)) # add concat
        x = self.upconv(self.l3(x)) # add concat
        x = self.upconv(self.l4(x)) # add concat

        return x


class Unet(nn.Module):
    def __init__(self, layers: List[int]) -> None:
        # [3,32,64,128,256,512]
        super().__init__()
        self.encoder = Encoder(layers)
        self.decoder = Decoder(list(reversed(layers)))

        self.conv = nn.Conv2d(64, 3, 3, 1, 1)


    def forward(self, x):
        x4, x3, x2, x1 = self.encoder(x)
        x = self.decoder(x4,x3,x2,x1)

        x = self.conv(x)

        return x


if __name__ == "__main__":
    layers = [3,32,64,128,256,512]
    model = Unet(layers)
    assert(1 == 1)
