from __future__ import annotations
from typing import List, Tuple
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def _tf():
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])

def load_validation_loader(batch_size: int = 64) -> DataLoader:
    tf = _tf()
    testset = datasets.MNIST("./data", train=False, download=True, transform=tf)
    return DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)

def load_partition_for_client(cid: int, num_clients: int, batch_size: int = 32, imbalanced: bool = False, seed: int = 42) -> DataLoader:
    tf = _tf()
    full = datasets.MNIST("./data", train=True, download=True, transform=tf)
    n = len(full)
    rng = np.random.default_rng(seed)
    indices = np.arange(n)
    rng.shuffle(indices)
    if not imbalanced:
        per = n // num_clients
        start = cid * per
        end = n if cid == num_clients - 1 else start + per
        part = indices[start:end]
    else:
        weights = np.arange(1, num_clients + 1, dtype=float)
        weights = (weights / weights.sum()) * n
        sizes = np.floor(weights).astype(int)
        sizes[-1] += n - sizes.sum()
        start = int(np.sum(sizes[:cid])); end = int(np.sum(sizes[:cid+1]))
        part = indices[start:end]
    return DataLoader(Subset(full, part.tolist()), batch_size=batch_size, shuffle=True, num_workers=2)

def train_one_epoch(model: torch.nn.Module, loader: DataLoader, criterion: torch.nn.Module, optimizer: torch.optim.Optimizer) -> float:
    model.train()
    total, count = 0.0, 0
    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward(); optimizer.step()
        total += float(loss.item()) * x.size(0); count += x.size(0)
    return total / max(1, count)

@torch.no_grad()
def accuracy(model: torch.nn.Module, loader: DataLoader) -> float:
    model.eval()
    correct, total = 0, 0
    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        pred = model(x).argmax(1)
        correct += int((pred == y).sum().item()); total += x.size(0)
    return correct / max(1, total)

def _ordered_state(model: torch.nn.Module) -> List[Tuple[str, torch.Tensor]]:
    sd = model.state_dict()
    return [(k, sd[k]) for k in sorted(sd.keys())]

def params_to_numpy(model: torch.nn.Module):
    return [t.detach().cpu().numpy().copy() for _, t in _ordered_state(model)]

def numpy_to_params(model: torch.nn.Module, params):
    items = _ordered_state(model)
    if len(items) != len(params):
        raise ValueError("Param length mismatch")
    new_sd = {}
    for (k, t), a in zip(items, params):
        new_sd[k] = torch.from_numpy(a).to(t.dtype)
    model.load_state_dict(new_sd, strict=True)
    model.to(DEVICE)
