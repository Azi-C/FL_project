# utils.py
from typing import Tuple, Optional
import torch
from torch.utils.data import DataLoader, random_split, Subset
from torchvision import datasets, transforms

DATA_DIR = "./data"

def load_train_val(
    batch_size: int = 32,
    val_split: float = 0.10,
    num_workers: int = 2,
    seed: int = 42,
) -> Tuple[DataLoader, DataLoader]:
    """Return trainloader (90%) and valloader (10%) with deterministic split."""
    tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    full_train = datasets.MNIST(DATA_DIR, train=True, download=True, transform=tf)

    val_size = int(len(full_train) * val_split)  # 6000
    train_size = len(full_train) - val_size      # 54000
    gen = torch.Generator().manual_seed(seed)
    train_set, val_set = random_split(full_train, [train_size, val_size], generator=gen)

    trainloader = DataLoader(train_set, batch_size=batch_size, shuffle=True,  num_workers=num_workers)
    valloader  = DataLoader(val_set,  batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return trainloader, valloader

def load_partition_for_client(
    client_id: int, num_clients: int = 10, batch_size: int = 32, num_workers: int = 2, seed: int = 42
) -> DataLoader:
    """Create an IID partition of the 90% training split for a given client."""
    trainloader, valloader = load_train_val(batch_size=batch_size, num_workers=num_workers, seed=seed)
    train_subset: Subset = trainloader.dataset  # type: ignore
    indices = torch.randperm(len(train_subset), generator=torch.Generator().manual_seed(seed)).tolist()
    part_size = len(indices) // num_clients
    start = client_id * part_size
    end = len(indices) if client_id == num_clients - 1 else start + part_size
    part_indices = indices[start:end]
    client_set = Subset(train_subset, part_indices)
    return DataLoader(client_set, batch_size=batch_size, shuffle=True, num_workers=num_workers)

@torch.no_grad()
def evaluate_accuracy(model: torch.nn.Module, loader: DataLoader, device: Optional[torch.device] = None) -> float:
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval().to(device)
    correct, total = 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        preds = model(x).argmax(dim=1)
        correct += (preds == y).sum().item()
        total += y.size(0)
    return correct / total if total > 0 else 0.0
