"""Main file for running and setting up training loop"""
import torch
import torch.nn as nn
from torch import optim
from helpers.datasetup import create_dataloaders
from models.unet import UNet
from torch.optim.lr_scheduler import StepLR
from torch.utils.tensorboard.writer import SummaryWriter
from helpers import utils
from helpers import graphs
import time
import yaml

import engine

def main():
    args = utils.parse_args()

    with open("configs/config.yaml", "r") as stream:
        config = yaml.safe_load(stream)

    utils.seed_all(args.seed)

    use_cuda = not args.no_cuda and torch.cuda.is_available()

    device = torch.device("cuda" if use_cuda else "cpu")

    print(f"{device=}")

    if use_cuda:
        for i in range(torch.cuda.device_count()):
            print(torch.cuda.get_device_properties(i))


    train_dataloader, test_dataloader, class_names = create_dataloaders(config)

    model = UNet(channels=[1, 64, 128, 256, 512, 512, 4096, 4096, 10])

    model = nn.DataParallel(model)
    compiled = torch.compile(model)

    # Transfer learning
    if args.load_model:
        compiled.load_state_dict(torch.load(config['checkpoint_best']))

    compiled.to(device)

    optimizer = optim.Adadelta(compiled.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    scheduler = StepLR(optimizer, step_size=config['step_size'], gamma=args.gamma)

    start = time.time()

    writer = None 

    if args.debug:
        writer = SummaryWriter()

    results = engine.train(
        model,
        train_dataloader=train_dataloader,
        test_dataloader=test_dataloader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        config=config,
        writer=writer)

    end = time.time()

    if args.debug:
        graphs.plot_acc_loss(results)

        elapsed = end  - start 
        print(f"Training took: {elapsed} seconds using {device}")

    print("Finished training!")

    # Save model to file if selected
    if args.save_model:
        torch.save(model.state_dict(), config['checkpoint_last'])


if __name__ == "__main__":
    main()
