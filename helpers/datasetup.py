from pathlib import Path
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from PIL import Image


class CustomDataSet(Dataset):
    """
    We create our own custom datasets, and use this in `create_dataloaders`
    in this file. Then we can iterate over a transformed and normalized dataset 
    in batches
    """
    def __init__(self, root, data_dir: str, transform=None, target_transform=None):
        super().__init__()
        with open(root + data_dir, 'r') as f:
            imgs = []
            for line in f:
                line = line.rstrip()
                words = line.split()
                imgs.append((words[0], int(words[1])))

        self.root = root
        self.imgs = imgs
        self.transform = transform
        self.target_transform = target_transform
        self.classes = ["city"]

    def __getitem__(self, idx):
        f, label = self.imgs[idx]  
        img = Image.open(self.root + f).convert('RGB')

        if self.transform is not None:
            img = self.transform(img)

        return img, label

    def __len__(self):
        return len(self.imgs)


def create_dataloaders(config):
    """
    Initializes data from datasets, creates dataloaders and return them 
    for use
    """

    transform = transforms.Compose([
        transforms.Resize((config['img_size'], config['img_size'])),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])

    # train_data = datasets.ImageFolder(train_dir, transform=transform)
    # test_data = datasets.ImageFolder(test_dir, transform=transform)
    
    train_data = CustomDataSet("./", config['train_dir'], transform=transform)
    test_data = CustomDataSet("./", config['test_dir'], transform=transform)

    class_names = train_data.classes

    train_dl = DataLoader(
        train_data,
        batch_size=config['batch_size'],
        shuffle=config['shuffle'],
        num_workers=config['use_cpu'],
        pin_memory=config['pin_memory']
    )

    test_dl = DataLoader(
        test_data,
        batch_size=config['batch_size'],
        shuffle=config['shuffle'],
        num_workers=config['use_cpu'],
        pin_memory=config['pin_memory']
    )

    return train_dl, test_dl, class_names
