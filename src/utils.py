import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
import numpy as np
from typing import List, Tuple, Optional

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --------------------------
# Dataset loading utilities
# --------------------------

def load_data(batch_size: int = 32) -> Tuple[DataLoader, DataLoader]:
    """Load MNIST train/test datasets."""
    tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    trainset = datasets.MNIST("./data", train=True, download=True, transform=tf)
    testset = datasets.MNIST("./data", train=False, download=True, transform=tf)
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)
    return trainloader, testloader


def load_train_val(batch_size: int = 32, val_split: float = 0.10, seed: int = 42) -> Tuple[DataLoader, DataLoader]:
    """Split MNIST train into train/validation loaders."""
    tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    full = datasets.MNIST("./data", train=True, download=True, transform=tf)
    val_size = int(len(full) * val_split)
    train_size = len(full) - val_size
    gen = torch.Generator().manual_seed(seed)
    train_set, val_set = random_split(full, [train_size, val_size], generator=gen)
    trainloader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=2)
    valloader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=2)
    return trainloader, valloader


def load_validation_loader(batch_size: int = 32) -> DataLoader:
    """Return only the validation dataset (10% of MNIST train)."""
    _, valloader = load_train_val(batch_size=batch_size, val_split=0.10)
    return valloader


# --------------------------
# Partitioning for clients
# --------------------------

def load_partition_for_client(
    cid: int,
    num_clients: int,
    batch_size: int = 32,
    non_iid: bool = False,
    labels_per_client: int = 2,
    proportions: Optional[List[float]] = None,
    dirichlet_alpha: Optional[float] = None,
) -> DataLoader:
    """Return the training partition for a specific client."""
    tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    dataset = datasets.MNIST("./data", train=True, download=True, transform=tf)

    # IID partition
    if not non_iid:
        partition_size = len(dataset) // num_clients
        start = cid * partition_size
        end = start + partition_size
        partition = torch.utils.data.Subset(dataset, list(range(start, end)))
        return DataLoader(partition, batch_size=batch_size, shuffle=True)

    # Simple Non-IID (label restriction)
    if non_iid and dirichlet_alpha is None:
        labels = np.array(dataset.targets)
        chosen_labels = list(range((cid * labels_per_client) % 10,
                                   (cid * labels_per_client) % 10 + labels_per_client))
        mask = np.isin(labels, chosen_labels)
        idxs = np.where(mask)[0]
        partition = torch.utils.data.Subset(dataset, idxs)
        return DataLoader(partition, batch_size=batch_size, shuffle=True)

    # Dirichlet-based partitioning (heterogeneous splits)
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


# --------------------------
# Training helpers
# --------------------------

def train_one_epoch(model, trainloader, criterion, optimizer) -> float:
    """Run one epoch of training and return average loss."""
    model.train()
    total_loss = 0.0
    for x, y in trainloader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)   # ✅ correct: use criterion, not optimizer
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(trainloader)


def accuracy(model, loader) -> float:
    """Compute accuracy of a model on a given dataset loader."""
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            logits = model(x)
            preds = logits.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)
    return correct / total if total > 0 else 0.0


# --------------------------
# Param <-> numpy conversions
# --------------------------

def params_to_numpy(model: torch.nn.Module) -> List[np.ndarray]:
    """Convert model parameters to list of numpy arrays."""
    return [p.detach().cpu().numpy().copy() for p in model.parameters()]

def numpy_to_params(model: torch.nn.Module, params: List[np.ndarray]) -> None:
    """Load numpy arrays back into model parameters."""
    for p, np_arr in zip(model.parameters(), params):
        p.data = torch.tensor(np_arr, dtype=p.data.dtype, device=p.data.device)
