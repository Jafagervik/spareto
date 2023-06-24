import torch 
from torch import nn

"""
Backbones for maskrcnn could be all of the following:

ResNet, FPN ..

"""
class MaskRCNN(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        pass 


if __name__ == "__main__":
    model = MaskRCNN()

    model.train() 


    print(model.parameters())
