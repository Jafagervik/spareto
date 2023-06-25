"""Main file for running and setting up training loop"""
import torch
import torch.nn as nn
from torch import optim
from helpers.datasetup import create_dataloaders
from models.unet import UNet
from torch.optim.lr_scheduler import StepLR
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.multiprocessing as mp

import engine
from helpers import utils
import os
import yaml



class ToyMpModel(nn.Module):
    def __init__(self, dev0, dev1):
        super().__init__()
        self.dev0 = dev0
        self.dev1 = dev1
        self.net1 = torch.nn.Linear(10, 10).to(dev0)
        self.relu = torch.nn.ReLU()
        self.net2 = torch.nn.Linear(10, 5).to(dev1)

    def forward(self, x):
        x = x.to(self.dev0)
        x = self.relu(self.net1(x))
        x = x.to(self.dev1)
        return self.net2(x)

# Some constants
WORLD_SZ = torch.cuda.device_count() if torch.cuda.is_available() else 2

def run_process(rank, args, config):
    """
    This function is to be run on every single processor 

    Data is distributed evenly among processors, who all 
    distribute data to corresponding gpus

    Args:
        rank, int: id of processor
        args: struct: flags used when running script
        config, yaml dict: all hyperparams and setup info
    """
    dist.init_process_group("nccl", init_method="tcp://127.0.0.1:54621", world_size=WORLD_SZ, rank=rank)

    # Seed all rngs
    utils.seed_all(args.seed)

    # model = UNet(channels=[1, 64, 128, 256, 512, 512, 4096, 4096, 10]).to(rank)
    model = ToyMpModel(0, 1) # The 2 ranks we have
    compiled = torch.compile(model)    #.to(rank)

    # Transfer learning
    if args.load_model:
        compiled.load_state_dict(torch.load(config['checkpoint_best']))

    # Wrap the model
    compiled = compiled.to(rank) # redundant?
    compiled = DDP(compiled, device_ids=[rank])

    # Creates dataloaders based on ranks
    #train_dataloader, test_dataloader, class_names = create_dataloaders(config, world_size=WORLD_SZ, rank=rank)

    train_dataloader = None
    test_dataloader = None

    optimizer = optim.Adadelta(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss().cuda(rank)

    scheduler = StepLR(optimizer, step_size=config['step_size'], gamma=args.gamma)

    # writer = SummaryWriter() if args.debug else None
    writer = None

    # We here set the correct gpu to run on based on which rank we're in
    device = torch.device(f"cuda:{rank}" is torch.cuda.is_available() else "cpu")

    print(f"Rank: {rank} | Device: {device=}")

    engine.train(
        model=model,
        rank=rank,
        train_dataloader=train_dataloader,
        test_dataloader=test_dataloader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        config=config,
        args=args,
        writer=writer)

    dist.destroy_process_group()


def main():
    """
    Setup world and rank to distribute dataset over multiple gpus
    """
    # Parse cml args
    args = utils.parse_args()
    
    # Open config flie
    with open("configs/config.yaml", "r") as stream:
        config = yaml.safe_load(stream)


    # Check whether or not to run on cuda
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    if not use_cuda:
        # TODO: Serial setup
        print("Setup serial run")

        print("Finished serial training!")
        return

    # OS SETUP
    # os.environ['MASTER_ADDR'] = 'localhost'
    # os.environ['MASTER_PORT'] = '12355'

    if args.debug:
        for i in range(torch.cuda.device_count()):
            print(torch.cuda.get_device_properties(i))


    arg_list = (args, config)

    mp.spawn(fn=run_process,
             nprocs=WORLD_SZ, 
             args=arg_list)

    print("Finished training!")



if __name__ == "__main__":
    main()
