# utils.py
from typing import Optional, Sequence, List, Tuple
from collections import defaultdict
import numpy as np
import torch
from torch.utils.data import DataLoader, random_split, Subset
from torchvision import datasets, transforms

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# -------------------- Train/Val split (90/10) --------------------
def _get_train_val_sets(val_split: float = 0.10, seed: int = 42):
    """Return (train_set_90, val_set_10) deterministically."""
    tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    full = datasets.MNIST("./data", train=True, download=True, transform=tf)
    val_size = int(len(full) * val_split)
    train_size = len(full) - val_size
    gen = torch.Generator().manual_seed(seed)
    train_set, val_set = random_split(full, [train_size, val_size], generator=gen)
    return train_set, val_set


def load_validation_loader(batch_size: int = 32, seed: int = 42):
    """Shared validation loader (the 10% split)."""
    _, val_set = _get_train_val_sets(seed=seed)
    return DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=2)


# -------------------- IID but IMBALANCED partition --------------------
def _partition_indices_iid_imbalanced(
    total_size: int,
    num_clients: int,
    seed: int = 42,
    proportions: Optional[Sequence[float]] = None,
    dirichlet_alpha: Optional[float] = None,
) -> List[Tuple[int, int]]:
    """
    Compute (start,end) slices over [0,total_size) for each client.
    - proportions: explicit list of floats (sum ≈ 1) of length num_clients.
    - dirichlet_alpha: if provided, sample proportions from Dirichlet(alpha).
    If neither is given, falls back to equal sizes.
    """
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

    # integer sizes summing exactly to total_size
    raw_sizes = np.floor(p * total_size).astype(int)
    shortfall = total_size - raw_sizes.sum()
    if shortfall > 0:
        order = np.argsort(-p)  # give extras to the largest proportions
        for i in range(shortfall):
            raw_sizes[order[i % num_clients]] += 1

    slices: List[Tuple[int, int]] = []
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
    """
    IID labels/features but IMBALANCED sizes across clients.
    We:
      1) take the 90% train split,
      2) shuffle deterministically,
      3) cut according to proportions or Dirichlet alpha (or equal if none).
    """
    train_set, _ = _get_train_val_sets(seed=seed)

    # deterministic shuffle of indices within the 90% pool
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(len(train_set), generator=g).tolist()

    # compute this client's slice
    slices = _partition_indices_iid_imbalanced(
        total_size=len(perm),
        num_clients=num_clients,
        seed=seed,
        proportions=proportions,
        dirichlet_alpha=dirichlet_alpha,
    )
    start, end = slices[client_id]
    chosen_in_trainset = perm[start:end]

    # map train_set-relative indices -> base dataset indices
    base_ds = train_set.dataset  # underlying full MNIST train
    base_indices = [train_set.indices[i] for i in chosen_in_trainset]

    subset = Subset(base_ds, base_indices)
    return DataLoader(subset, batch_size=batch_size, shuffle=True, num_workers=2)


# -------------------- Non-IID label skew (kept for later use) --------------------
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
    model.train().to(DEVICE)
    total_loss = 0.0
    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        optimizer.zero_grad()
        loss = criterion(model(x), y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / max(1, len(loader))


@torch.no_grad()
def accuracy(model, loader):
    model.eval().to(DEVICE)
    correct, total = 0, 0
    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        pred = model(x).argmax(dim=1)
        correct += (pred == y).sum().item()
        total += y.size(0)
    return correct / total if total > 0 else 0.0
