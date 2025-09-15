# src/utils.py
from typing import Optional, List
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------- Data Partitioning ----------------------

def load_partition_for_client(
    cid: int,
    num_clients: int,
    batch_size: int = 32,
    non_iid: bool = False,
    labels_per_client: int = 2,
    proportions: Optional[List[float]] = None,
    dirichlet_alpha: Optional[float] = None,
) -> DataLoader:
    """Return DataLoader for client cid with options for IID, non-IID, and imbalanced."""
    tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    dataset = datasets.MNIST("./data", train=True, download=True, transform=tf)
    N = len(dataset)

    # ---------- IID but allow IMBALANCED via proportions ----------
    if not non_iid:
        if proportions is not None:
            if len(proportions) != num_clients:
                raise ValueError(f"'proportions' must have length {num_clients}")
            if sum(proportions) <= 0:
                raise ValueError("'proportions' must sum to > 0")

            norm = float(sum(proportions))
            ps = [float(p) / norm for p in proportions]
            counts = [int(round(p * N)) for p in ps]

            # Fix rounding drift
            drift = N - sum(counts)
            if drift != 0:
                j = int(np.argmax(counts))
                counts[j] += drift

            # Build cumulative split
            starts = [0]
            for k in range(num_clients - 1):
                starts.append(starts[-1] + counts[k])
            ends = [starts[i] + counts[i] for i in range(num_clients)]

            start, end = starts[cid], ends[cid]
            idxs = list(range(start, end))
            partition = torch.utils.data.Subset(dataset, idxs)
            return DataLoader(partition, batch_size=batch_size, shuffle=True)

        # Fallback: balanced IID
        partition_size = N // num_clients
        start = cid * partition_size
        end = (cid + 1) * partition_size if cid < num_clients - 1 else N
        partition = torch.utils.data.Subset(dataset, list(range(start, end)))
        return DataLoader(partition, batch_size=batch_size, shuffle=True)

    # ---------- Non-IID: disjoint labels ----------
    if non_iid and dirichlet_alpha is None:
        labels = np.array(dataset.targets)
        chosen_labels = list(range((cid * labels_per_client) % 10,
                                   (cid * labels_per_client) % 10 + labels_per_client))
        mask = np.isin(labels, chosen_labels)
        idxs = np.where(mask)[0]
        partition = torch.utils.data.Subset(dataset, idxs)
        return DataLoader(partition, batch_size=batch_size, shuffle=True)

    # ---------- Non-IID: Dirichlet ----------
    if non_iid and dirichlet_alpha is not None:
        labels = np.array(dataset.targets)
        num_classes = 10
        idx_by_class = [np.where(labels == i)[0] for i in range(num_classes)]
        sizes = np.random.dirichlet([dirichlet_alpha] * num_clients, num_classes)
        client_indices = [[] for _ in range(num_clients)]
        for c in range(num_classes):
            idxs = idx_by_class[c]
            np.random.shuffle(idxs)
            splits = (sizes[c] * len(idxs)).astype(int)
            start = 0
            for client_id in range(num_clients):
                end = start + splits[client_id]
                client_indices[client_id].extend(idxs[start:end])
                start = end
        partition = torch.utils.data.Subset(dataset, client_indices[cid])
        return DataLoader(partition, batch_size=batch_size, shuffle=True)

    raise ValueError("Invalid partitioning setup.")


# ---------------------- Validation Loader ----------------------

def load_validation_loader(batch_size: int = 32):
    tf = transforms.Compose([transforms.ToTensor(),
                             transforms.Normalize((0.1307,), (0.3081,))])
    dataset = datasets.MNIST("./data", train=False, download=True, transform=tf)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False)


# ---------------------- Training Helpers ----------------------

def train_one_epoch(model, dataloader, criterion, optimizer):
    model.train()
    total_loss = 0.0
    for (x, y) in dataloader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * x.size(0)
    return total_loss / len(dataloader.dataset)


def accuracy(model, dataloader):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for (x, y) in dataloader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            logits = model(x)
            preds = logits.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)
    return correct / total if total > 0 else 0.0


# ---------------------- Param Conversion ----------------------

def params_to_numpy(model) -> List[np.ndarray]:
    return [p.detach().cpu().numpy().copy() for p in model.parameters()]

def numpy_to_params(model, params: List[np.ndarray]) -> None:
    with torch.no_grad():
        for p, np_val in zip(model.parameters(), params):
            p.copy_(torch.from_numpy(np_val).to(p.device))
