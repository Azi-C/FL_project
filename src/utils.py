# utils.py
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split, Subset
import torch
import numpy as np
from typing import Optional, Sequence
from collections import defaultdict

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# -------------------- Shared helpers --------------------
def _get_train_val_sets(val_split: float = 0.10, seed: int = 42):
    tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    full = datasets.MNIST("./data", train=True, download=True, transform=tf)
    val_size = int(len(full) * val_split)
    train_size = len(full) - val_size
    gen = torch.Generator().manual_seed(seed)
    return random_split(full, [train_size, val_size], generator=gen)


def load_validation_loader(batch_size: int = 32, seed: int = 42):
    _, val_set = _get_train_val_sets(seed=seed)
    return DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=2)


# -------------------- Imbalanced IID Partition --------------------
def _partition_indices_iid_imbalanced(
    total_size: int,
    num_clients: int,
    seed: int = 42,
    proportions: Optional[Sequence[float]] = None,
    dirichlet_alpha: Optional[float] = None,
):
    rng = np.random.default_rng(seed)

    if proportions is not None:
        p = np.array(proportions, dtype=float)
        if len(p) != num_clients:
            raise ValueError("proportions length must equal num_clients")
        if p.sum() <= 0:
            raise ValueError("proportions must sum to a positive value")
        p = p / p.sum()
    else:
        if dirichlet_alpha is not None:
            p = rng.dirichlet(alpha=[dirichlet_alpha] * num_clients)
        else:
            p = np.full(num_clients, 1.0 / num_clients)

    raw_sizes = np.floor(p * total_size).astype(int)
    shortfall = total_size - raw_sizes.sum()
    if shortfall > 0:
        order = np.argsort(-p)
        for i in range(shortfall):
            raw_sizes[order[i % num_clients]] += 1

    slices = []
    start = 0
    for sz in raw_sizes:
        end = start + int(sz)
        slices.append((start, end))
        start = end
    assert slices[-1][1] == total_size
    return slices


def load_partition_for_client(
    client_id: int,
    num_clients: int,
    batch_size: int = 32,
    seed: int = 42,
    proportions: Optional[Sequence[float]] = None,
    dirichlet_alpha: Optional[float] = None,
) -> DataLoader:
    train_set, _ = _get_train_val_sets(seed=seed)
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(len(train_set), generator=g).tolist()

    slices = _partition_indices_iid_imbalanced(
        total_size=len(perm),
        num_clients=num_clients,
        seed=seed,
        proportions=proportions,
        dirichlet_alpha=dirichlet_alpha,
    )
    start, end = slices[client_id]
    chosen_in_trainset = perm[start:end]

    base_ds = train_set.dataset
    base_indices = [train_set.indices[i] for i in chosen_in_trainset]

    subset = Subset(base_ds, base_indices)
    return DataLoader(subset, batch_size=batch_size, shuffle=True, num_workers=2)


# -------------------- Non-IID Label Skew --------------------
def assigned_labels_for_client(client_id: int, labels_per_client: int = 2) -> list[int]:
    all_labels = list(range(10))
    start = (client_id * labels_per_client) % len(all_labels)
    return [all_labels[(start + j) % len(all_labels)] for j in range(labels_per_client)]


def load_partition_for_client_non_iid(
    client_id: int,
    num_clients: int,
    labels_per_client: int = 2,
    batch_size: int = 32,
    seed: int = 42,
) -> DataLoader:
    train_set, _ = _get_train_val_sets(seed=seed)
    base_ds = train_set.dataset

    label_to_idxs = defaultdict(list)
    for idx in train_set.indices:
        _, y = base_ds[idx]
        label_to_idxs[int(y)].append(idx)

    assigned = assigned_labels_for_client(client_id, labels_per_client)
    chosen_idxs = []
    for lbl in assigned:
        chosen_idxs.extend(label_to_idxs[lbl])

    g = torch.Generator().manual_seed(seed + client_id)
    perm = torch.randperm(len(chosen_idxs), generator=g).tolist()
    chosen_idxs = [chosen_idxs[i] for i in perm]

    subset = Subset(base_ds, chosen_idxs)
    return DataLoader(subset, batch_size=batch_size, shuffle=True, num_workers=2)


# -------------------- Train / Eval --------------------
def train_one_epoch(model, loader, criterion, optimizer):
    model.train()
    total_loss = 0.0
    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        optimizer.zero_grad()
        loss = criterion(model(x), y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


def accuracy(model, loader):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            pred = model(x).argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.size(0)
    return correct / total
