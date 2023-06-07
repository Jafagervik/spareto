"""Main file for running and setting up training loop"""
import torch
from models.levit import LeViT256
from torch.optim.lr_scheduler import StepLR
from configs import commons, levit256

import engine


def main():
    """
    Entry point for training loop
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Get the dataloaders with prepared data ready for use
    train_dataloader = None
    test_dataloader = None

    # TODO: store convolution sizes in levit config

    model = LeViT256(in_features=(levit256.IMG_SIZE ** 2)
                     * levit256.NUM_CHANNELS).to(device)

    if commons.LOAD_MODEL:
        model.load_state_dict(torch.load(commons.CHECKPOINT_PATH))

    criterion = torch.nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(params=model.parameters())

    # To find the very best LR
    scheduler = StepLR(optimizer, step_size=levit256.STEP_SIZE)

    engine.train(
        model,
        train_dataloader=train_dataloader,
        test_dataloader=test_dataloader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        epochs=levit256.NUM_EPOCHS,
        device=torch.device(device))

    print("Finished training!")


if __name__ == "__main__":
    main()
