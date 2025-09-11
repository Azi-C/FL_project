# utils.py
from typing import List, Tuple
import torch
from torch.utils.data import DataLoader, random_split, Subset
from torchvision import datasets, transforms
import numpy as np

DATA_DIR = "./data"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------- Transforms ----------------
def _mnist_tf():
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])


# ---------------- Splits 90/10 ----------------
def _get_train_val_sets(val_split: float = 0.10, seed: int = 42):
    """Retourne (train_set_90, val_set_10) de façon déterministe."""
    full = datasets.MNIST(DATA_DIR, train=True, download=True, transform=_mnist_tf())
    val_size = int(len(full) * val_split)
    train_size = len(full) - val_size
    gen = torch.Generator().manual_seed(seed)
    train_set, val_set = random_split(full, [train_size, val_size], generator=gen)
    return train_set, val_set


def load_validation_loader(batch_size: int = 32, seed: int = 42) -> DataLoader:
    """Loader partagé pour V (10% du train)."""
    _, val_set = _get_train_val_sets(seed=seed)
    return DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=2)


def load_partition_for_client(
    client_id: int,
    num_clients: int,
    batch_size: int = 32,
    seed: int = 42,
) -> DataLoader:
    """
    Donne au client 'client_id' une partition IID de *seulement* les 90% du train.
    Partition déterministe (seed). Le dernier client récupère le reste.
    """
    train_set, _ = _get_train_val_sets(seed=seed)

    gen = torch.Generator().manual_seed(seed)
    # train_set est un Subset -> indices vers le dataset de base
    perm = torch.randperm(len(train_set), generator=gen).tolist()

    base = len(perm) // num_clients
    start = client_id * base
    end = len(perm) if client_id == num_clients - 1 else start + base

    # Mapper vers indices du dataset de base (full)
    base_ds = train_set.dataset
    selected_base_idxs = [train_set.indices[i] for i in perm[start:end]]

    client_subset = Subset(base_ds, selected_base_idxs)
    return DataLoader(client_subset, batch_size=batch_size, shuffle=True, num_workers=2)


def load_test_loader(batch_size: int = 32) -> DataLoader:
    """Loader du test officiel MNIST (pour suivi de perf)."""
    testset = datasets.MNIST(DATA_DIR, train=False, download=True, transform=_mnist_tf())
    return DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)


# ---------------- Train / Eval ----------------
def train_one_epoch(model: torch.nn.Module, loader: DataLoader, lr: float = 0.01, device=DEVICE):
    import torch.nn as nn, torch.optim as optim
    model.train().to(device)
    opt = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    crit = nn.CrossEntropyLoss()
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        opt.zero_grad()
        loss = crit(model(x), y)
        loss.backward()
        opt.step()


@torch.no_grad()
def accuracy(model: torch.nn.Module, loader: DataLoader, device=DEVICE) -> float:
    model.eval().to(device)
    correct, total = 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        pred = model(x).argmax(dim=1)
        correct += (pred == y).sum().item()
        total += y.size(0)
    return correct / total if total > 0 else 0.0
