"""
engine.py basically defines what a train and a test step is
"""
from tqdm import tqdm
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR
from typing import Tuple


def train_step(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    writer = None
) -> Tuple[float, float]:
    """Performs one epoch worth of training

    Returns:
        train loss and accuracy
    """
    model.train() 

    train_loss, train_acc = 0, 0

    # Go through the batches in current epoch
    for batch_idx, (X, y) in dataloader:
        X = X.to(device)
        y = y.to(device)

        y_hat = model(X)

        loss = criterion(y_hat, y)
        train_loss += loss.item()

        if writer:
            writer.add_scalar("Loss/train", train_loss, epoch)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        y_pred_class = torch.argmax(torch.softmax(y_hat, dim=1), dim=1)
        train_acc += (y_pred_class == y).sum().item()/len(y_hat)

    train_loss /= len(dataloader)
    train_acc /= len(dataloader)

    return train_loss, train_acc


@torch.no_grad()
def test_step(
    model: nn.Module,
    highest_acc: float,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    config
) -> Tuple[float, float, float]:
    """ Performs one epoch worth of testing

    Returns:
        test loss and accuracy as well as highest accuracy
    """
    model.eval()

    test_loss, test_acc = 0, 0

    # Go through the batches in current epoch
    for _, (X, y) in dataloader:
        X = X.to(device)
        y = y.to(device)

        test_pred_logits = model(X)

        loss = criterion(test_pred_logits, y)
        test_loss += loss.item()

        test_pred_labels = test_pred_logits.argmax(dim=1)
        test_acc += ((test_pred_labels == y).sum().item() /
                        len(test_pred_labels))

        if test_acc > highest_acc:
            highest_acc = test_acc
            torch.save(model.state_dict(), config.checkpoint_best)

    test_loss /= len(dataloader)
    test_acc /= len(dataloader)

    return test_loss, test_acc, highest_acc 


def train(
    model: nn.Module,
    train_dataloader,
    test_dataloader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: StepLR,
    device: torch.device,
    config, # YAML
    writer = None
):
    results = {
        "train_loss": [],
        "train_acc": [],
        "test_loss": [],
        "test_acc": [],
    }

    highest_acc = 0.0

    for epoch in tqdm(range(1, config['epochs']+ 1)):
        train_loss, train_acc = train_step(
            model, train_dataloader, criterion, optimizer, device, epoch, writer if writer else None)

        test_loss, test_acc, highest_acc = test_step(
            model, highest_acc, test_dataloader, criterion,  device, config)

        scheduler.step()

        print(
            f"Epoch: {epoch} | "
            f"train_loss: {train_loss:.4f} | "
            f"train_acc: {train_acc:.4f} | "
            f"test_loss: {test_loss:.4f} | "
            f"test_acc: {test_acc:.4f}"
        )

        results["train_loss"].append(train_loss)
        results["train_acc"].append(train_acc)
        results["test_loss"].append(test_loss)
        results["test_acc"].append(test_acc)

    if writer:
        writer.flush()

    return results
