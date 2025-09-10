# src/utils.py
import io
import random
from typing import Tuple, List, Dict, Any, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset, random_split
from torchvision import datasets, transforms


def get_device() -> torch.device:
    """Return a CUDA device if available, else CPU."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed: int = 42) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # safe even if no CUDA
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_transforms(dataset: str = "MNIST", augment: bool = False):
    """Return (train_transform, test_transform) for dataset."""
    if dataset.upper() == "MNIST":
        base = [
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ]
        aug = [transforms.RandomRotation(10)] if augment else []
        train_tf = transforms.Compose(aug + base)
        test_tf = transforms.Compose(base)
        return train_tf, test_tf
    raise ValueError(f"Unsupported dataset: {dataset}")


def _maybe_subset(ds: torch.utils.data.Dataset, subset: Optional[int]) -> torch.utils.data.Dataset:
    """Optionally take a small subset (useful for quick tests)."""
    if subset is None or subset <= 0 or subset >= len(ds):
        return ds
    indices = list(range(subset))
    return Subset(ds, indices)


def load_data_with_validation(
    data_dir: str = "./data",
    dataset: str = "MNIST",
    batch_size: int = 32,
    num_workers: int = 2,
    subset_train: Optional[int] = None,
    subset_val: Optional[int] = None,
    subset_test: Optional[int] = None,
    augment: bool = False,
    val_split: float = 0.1
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Load data and split training into (train, validation).

    Returns:
        trainloader, valloader, testloader
    """
    train_tf, test_tf = get_transforms(dataset, augment)

    if dataset.upper() == "MNIST":
        full_train = datasets.MNIST(data_dir, train=True, download=True, transform=train_tf)
        testset = datasets.MNIST(data_dir, train=False, download=True, transform=test_tf)
    else:
        raise ValueError(f"Unsupported dataset: {dataset}")

    # Optional subset for quick tests
    full_train = _maybe_subset(full_train, subset_train)
    testset = _maybe_subset(testset, subset_test)

    # Train/val split
    val_size = int(len(full_train) * val_split)
    train_size = len(full_train) - val_size
    trainset, valset = random_split(full_train, [train_size, val_size])

    # Optional subset for validation (if needed)
    valset = _maybe_subset(valset, subset_val)

    # Create loaders
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    valloader = DataLoader(valset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return trainloader, valloader, testloader


@torch.no_grad()
def evaluate_accuracy(model: torch.nn.Module, loader: DataLoader, device: Optional[torch.device] = None) -> float:
    """Compute accuracy."""
    if device is None:
        device = get_device()
    model.eval()
    model.to(device)
    correct, total = 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        preds = logits.argmax(dim=1)
        correct += (preds == y).sum().item()
        total += y.size(0)
    return correct / total if total > 0 else 0.0


def state_to_bytes(state_dict: Dict[str, torch.Tensor]) -> bytes:
    """Serialize state_dict to bytes."""
    buffer = io.BytesIO()
    torch.save(state_dict, buffer)
    return buffer.getvalue()


def bytes_to_state(blob: bytes) -> Dict[str, torch.Tensor]:
    """Deserialize bytes to state_dict."""
    buffer = io.BytesIO(blob)
    state = torch.load(buffer, map_location="cpu")
    return state


def params_to_numpy(model: torch.nn.Module) -> List[np.ndarray]:
    """Model parameters to NumPy list."""
    return [p.detach().cpu().numpy() for _, p in model.state_dict().items()]


def numpy_to_params(model: torch.nn.Module, parameters: List[np.ndarray]) -> None:
    """Load NumPy list into model."""
    keys = list(model.state_dict().keys())
    if len(keys) != len(parameters):
        raise ValueError("Mismatch between model parameters and provided parameters")
    tensor_params = {k: torch.tensor(v) for k, v in zip(keys, parameters)}
    model.load_state_dict(tensor_params, strict=True)
