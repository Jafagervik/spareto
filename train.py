"""Main file for running and setting up training loop"""
import torch
from models.levit import LeViT256
from torch.optim.lr_scheduler import StepLR
from configs import commons, levit256

import engine
from tqdm import tqdm


device = "cuda" if torch.cuda.is_available() else "cpu"


def main():
    """
    Entry point for training loop
    """

    # TODO: store convolution sizes in levit config

    model = LeViT256().to(device)

    if commons.LOAD_MODEL:
        model.load_state_dict(torch.load(commons.CHECKPOINT_PATH))

    criterion = torch.nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(params=model.parameters())

    # To find the very best LR
    scheduler = StepLR(optimizer, step_size=levit256.STEP_SIZE)

    print("DONE!")


if __name__ == "__main__":
    main()
