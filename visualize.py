import PIL 
import matplotlib.pyplot as plt 
import numpy as np 

from enum import Enum

class MediaType(Enum):
    IMAGE = 0,
    VIDEO = 1,

class Model(Enum):
    UNET = 0,
    LEVIT = 1,

"""
1. setup and run model in eval mode 
2. segment or videos based on input
3. Show stats on screen if wanted 
"""


def run_image(path: str, amount: int):
    """Run segmentation on a single image"""
    a = 2

    # Open image
    
    # Run through model in eval mode 

    # Color based on outputs


def run_frame():
    # Basically the same as run_image, but now on a single frame for a video
    # Output a screen with overlay containing mIoU, fps and so on
    pass


def run_video(video_path: str):
    """Run segmentation on a video"""
    fps = 0.0
    mIoU = 0.0


def run():
    model = Model.UNET
    t = MediaType.IMAGE

    match t:
        case MediaType.IMAGE:
            match model:
                case Model.UNET:
                    print("@")
                case Model.LEVIT:
                    print("#")

        case MediaType.VIDEO:
            match model:
                case Model.UNET:
                    print("@")
                case Model.LEVIT:
                    print("#")


if __name__ == "__main__":
    run()
