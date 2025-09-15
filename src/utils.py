from __future__ import annotations

from typing import Iterable, List, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset, random_split
from torchvision import datasets, transforms

# ------------------ Device ------------------

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ------------------ Data ------------------

def _mnist_transform():
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])


def load_validation_loader(batch_size: int = 64) -> DataLoader:
    """Shared validation loader (MNIST test set)."""
    testset = datasets.MNIST(root="./data", train=False, download=True, transform=_mnist_transform())
    return DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)


def load_partition_for_client(
    client_id: int,
    num_clients: int,
    batch_size: int = 32,
    train_fraction: float = 0.90,
    seed: int = 42,
) -> DataLoader:
    """
    Deterministic IID partition:
      - Take `train_fraction` of the MNIST train split for training (default 90%).
      - Evenly slice that region into `num_clients` contiguous shards and return client `client_id` shard.
    """
    assert 0 <= client_id < num_clients, "client_id out of range"
    full = datasets.MNIST(root="./data", train=True, download=True, transform=_mnist_transform())

    # Use only the first `train_fraction` of the dataset as the training pool
    train_size = int(len(full) * train_fraction)
    train_indices = np.arange(train_size)

    # Deterministic split into equal shards
    shard_size = train_size // num_clients
    start = client_id * shard_size
    end = train_size if client_id == num_clients - 1 else (start + shard_size)
    indices = train_indices[start:end]

    subset = Subset(full, indices.tolist())
    return DataLoader(subset, batch_size=batch_size, shuffle=True, num_workers=2)


# ------------------ Training / Eval ------------------

def train_one_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    criterion: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
) -> Tuple[float, float]:
    """Train for one epoch; returns (avg_loss, avg_accuracy)."""
    model.train()
    total, correct, total_loss = 0, 0, 0.0
    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)

        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        total_loss += float(loss.detach().cpu().item()) * x.size(0)
        pred = torch.argmax(logits, dim=1)
        correct += int((pred == y).sum().item())
        total += int(x.size(0))

    avg_loss = total_loss / max(1, total)
    avg_acc = correct / max(1, total)
    return avg_loss, avg_acc


def accuracy(model: torch.nn.Module, loader: DataLoader) -> float:
    """Compute accuracy on a DataLoader."""
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            logits = model(x)
            pred = torch.argmax(logits, dim=1)
            correct += int((pred == y).sum().item())
            total += int(x.size(0))
    return correct / max(1, total)


# ------------------ Param conversion ------------------

def params_to_numpy(model: torch.nn.Module) -> List[np.ndarray]:
    """Extract parameters (same order every time) as float32 numpy arrays."""
    arrays: List[np.ndarray] = []
    for p in model.parameters():
        arrays.append(p.detach().cpu().numpy().astype(np.float32, copy=True))
    return arrays


def numpy_to_params(model: torch.nn.Module, params: List[np.ndarray]) -> None:
    """Load numpy arrays into model parameters (order must match)."""
    with torch.no_grad():
        for p, arr in zip(model.parameters(), params):
            tensor = torch.from_numpy(arr).to(p.device, dtype=p.dtype)
            p.copy_(tensor)
