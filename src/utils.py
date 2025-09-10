# src/utils.py
from typing import Tuple
import torch
from torch.utils.data import DataLoader, Subset, random_split
from torchvision import datasets, transforms

DATA_DIR = "./data"
BATCH_SIZE = 16
NUM_WORKERS = 2
VAL_SPLIT = 0.10
SEED = 42  # fixed seed for deterministic split

# Shared transforms (same normalize for train/val)
_train_tf = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])
_val_tf = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

def create_validation_loader() -> DataLoader:
    """Create a deterministic 10% validation loader shared by all clients."""
    full_train = datasets.MNIST(DATA_DIR, train=True, download=True, transform=_train_tf)

    # Random, deterministic split into train + val (90/10)
    val_size = int(len(full_train) * VAL_SPLIT)
    train_size = len(full_train) - val_size
    gen = torch.Generator().manual_seed(SEED)
    _, val_set = random_split(full_train, [train_size, val_size], generator=gen)

    # Use the same normalization for val (define transform at dataset creation)
    # (We already set transform on full_train; random_split keeps it.)
    valloader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
    return valloader

def load_client_trainloader(partition_id: int, num_clients: int = 10) -> DataLoader:
    """
    Return a train DataLoader for a given client partition, using only the 90% train portion.
    Partitions are IID by shuffling indices with a fixed seed, then slicing equally.
    """
    full_train = datasets.MNIST(DATA_DIR, train=True, download=True, transform=_train_tf)

    # First, make the same 90/10 split as above (so train set excludes validation)
    val_size = int(len(full_train) * VAL_SPLIT)
    train_size = len(full_train) - val_size
    gen = torch.Generator().manual_seed(SEED)
    train_set, _ = random_split(full_train, [train_size, val_size], generator=gen)

    # Create IID partitions by shuffling indices deterministically
    indices = torch.randperm(len(train_set), generator=gen).tolist()
    part_size = len(indices) // num_clients
    start = partition_id * part_size
    end = len(indices) if partition_id == num_clients - 1 else start + part_size
    part_indices = indices[start:end]

    client_subset = Subset(train_set, part_indices)
    trainloader = DataLoader(client_subset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    return trainloader
