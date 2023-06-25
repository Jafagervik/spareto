from torch import nn
import torch.nn.functional as F
from typing import List


class ConvBlock(nn.Module):
    """
    https://learnopencv.com/understanding-convolutional-neural-networks-cnn/

    This is a more concise version of this:
    https://github.com/chongwar/vgg16-pytorch/blob/master/vgg16.py


    Utilizing the power of reuse, we can easily set up a general block and 
    just reuse this many times
    """
    def __init__(self, in_channels: int, out_channels: int, convs: int, kernel_size: int = 3, stride: int = 1, padding: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        self.conv2 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        # Batch_norm is here the same as out_channels, so we don't need to add an extra parameter
        # See the github link for explenations
        self.batch_norm = nn.BatchNorm2d(out_channels)
        self.a = nn.ReLU(True)
        self.mp = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        self.convs = convs

    def forward(self, x):
        x = self.conv1(x)
        x = self.batch_norm(x)
        x = self.a(x)

        # Some layers do 2 convolutions first, some do 3
        for _ in range(self.convs):
            x = self.conv2(x)
            x = self.batch_norm(x)
            x = self.a(x)

        x = self.mp(x)

        return x


class ClassifierLayer(nn.Module):
    """
    Classifier layer. this is actually redundant due to how little it does,
    but it shows how you don't need to keep all nn info 
    in a single class
    """

    def __init__(self, inp: int, out: int):
        super().__init__()
        self.dense = nn.Linear(inp, out) # DENSE LAYER is called Linear in pytorch
        self.act = nn.ReLU(True)
        self.drop = nn.Dropout(p=0.65)

    def forward(self, x):
        x = self.dense(x)
        x = self.act(x)
        x = self.drop(x)

        return x


class UNet(nn.Module):
    def __init__(self, channels: List[int]):
        super().__init__()

        assert(len(channels) == 9), "Dimension of channels does not match VGG16 net"

        # conv_layers = [1, 64, 128, 256, 512, 512, 4096, 4096, 10]

        self.conv1 = ConvBlock(in_channels=channels[0], out_channels=channels[1], convs=2)
        self.conv2 = ConvBlock(in_channels=channels[1], out_channels=channels[2], convs=2)

        self.conv3 = ConvBlock(in_channels=channels[2], out_channels=channels[3], convs=3)
        self.conv4 = ConvBlock(in_channels=channels[3], out_channels=channels[4], convs=3)
        self.conv5 = ConvBlock(in_channels=channels[4], out_channels=channels[5], convs=3)

        self.classifier = nn.Sequential(
            ClassifierLayer(channels[5], channels[6]),
            ClassifierLayer(channels[6], channels[7]), 
            nn.Linear(channels[7], channels[-1]), # DENSE LAYER
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, mode='fan_in', nonlinearity='leaky_relu')
                if m.bias is not None:
                    m.bias.detach().zero_()

        
    def forward(self, x):
        # FEATURE EXTRACTOR 
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        
        # Flatten
        x = x.view(x.size(0), -1) 
        
        # Classifier
        x = self.classifier(x)
        
        # NOTE: Softmax before output, check if this is even needed due to loss
        # x = F.softmax(x, dim=1) 

        return x


if __name__ == "__main__":
    model = UNet([1,2,3, 42, 69, 10])

    p = model.parameters()

    assert(1 == 1), "Failed model test"
