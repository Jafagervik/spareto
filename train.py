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

from torchvision import datasets, transforms
from torch.utils.data.distributed import DistributedSampler
import torch.nn.functional as F

import engine
from helpers import utils
import os
import yaml


# OS SETUP
os.environ['MASTER_ADDR'] = 'localhost'
os.environ['MASTER_PORT'] = '12356'
torch.set_float32_matmul_precision('high')


class ToyMpModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        # output = F.log_softmax(x, dim=1)
        return x



# Some constants
WORLD_SZ = torch.cuda.device_count() if torch.cuda.is_available() else 2


def run_process(rank, args, config, dataset1, dataset2):
    """
    This function is to be run on every single processor 

    Data is distributed evenly among processors, who all 
    distribute data to corresponding gpus

    Args:
        rank, int: id of processor
        args: struct: flags used when running script
        config, yaml dict: all hyperparams and setup info
    """
    dist.init_process_group("nccl", world_size=WORLD_SZ, rank=rank)
    # init_method="tcp://127.0.0.1:54621"

    # Seed all rngs
    utils.seed_all(args.seed)

    # model = UNet(channels=[1, 64, 128, 256, 512, 512, 4096, 4096, 10]).to(rank)
    model = ToyMpModel()
    compiled = torch.compile(model).to(rank)

    #dist.barrier()

    # Wrap the model
    compiled = DDP(compiled, device_ids=[rank])

    # Transfer learning
    if args.load_model:
        map_location = {'cuda:%d' % 0: 'cuda:%d' % rank}
        compiled.load_state_dict(torch.load(config['checkpoint_best'], map_location=map_location))

    # Creates dataloaders based on ranks
    #train_dataloader, test_dataloader, class_names = create_dataloaders(config, world_size=WORLD_SZ, rank=rank)

    train_kwargs = {'batch_size': 32}
    test_kwargs = {'batch_size': 1000}
                   
    cuda_kwargs = {'num_workers': config['use_cpu'],
                    'pin_memory': True}

    train_kwargs.update(cuda_kwargs)
    test_kwargs.update(cuda_kwargs)

    train_sampler = DistributedSampler(
        dataset1,
        num_replicas=WORLD_SZ,
        rank=rank,
        shuffle=config['shuffle']
    )

    test_sampler = DistributedSampler(
        dataset2,
        num_replicas=WORLD_SZ,
        rank=rank,
        shuffle=config['shuffle']
    )

    train_dataloader = torch.utils.data.DataLoader(dataset1, sampler=train_sampler, **train_kwargs)
    test_dataloader = torch.utils.data.DataLoader(dataset2, sampler=test_sampler, **test_kwargs)

    optimizer = optim.Adadelta(compiled.parameters(), lr=float(config['learning_rate']))
    criterion = nn.CrossEntropyLoss().cuda(rank)

    scheduler = StepLR(optimizer, step_size=config['step_size'], gamma=float(config['gamma']))

    # writer = SummaryWriter() if args.debug else None
    writer = None

    # We here set the correct gpu to run on based on which rank we're in
    device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")

    print(f"Rank: {rank} | Device: {device=}")

    engine.train(
        model=compiled,
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


def setup_dataset():
    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    # Handle downloads in it's own folder
    dataset1 = datasets.MNIST('./data', train=True, download=True,
                       transform=transform)
    dataset2 = datasets.MNIST('./data', train=False,
                       transform=transform)

    return dataset1, dataset2


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


    if args.debug:
        for i in range(torch.cuda.device_count()):
            print(torch.cuda.get_device_properties(i))

    dataset1, dataset2 = setup_dataset()

    arg_list = (args, config, dataset1, dataset2)

    mp.spawn(fn=run_process,
             nprocs=WORLD_SZ, 
             args=arg_list)

    print("Finished training!")


if __name__ == "__main__":
    main()
