# utils.py
from typing import List, Optional, Sequence, Tuple
import numpy as np
import torch
from torch.utils.data import DataLoader, random_split, Subset
from torchvision import datasets, transforms

# Device used everywhere
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------- Core dataset helpers ----------------

def _mnist_dataset(transform=None):
    tf = transform or transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),  # MNIST stats
    ])
    full = datasets.MNIST("./data", train=True, download=True, transform=tf)
    return full

def _train_val_indices(val_split: float = 0.10, seed: int = 42) -> Tuple[List[int], List[int]]:
    """Deterministically split MNIST train into train(90%)/val(10%)."""
    full = _mnist_dataset()
    n = len(full)
    val_size = int(n * val_split)
    train_size = n - val_size
    gen = torch.Generator().manual_seed(seed)
    train_subset, val_subset = random_split(range(n), [train_size, val_size], generator=gen)
    return list(train_subset), list(val_subset)

# ---------------- Public loaders ----------------

def load_validation_loader(batch_size: int = 64, val_split: float = 0.10, seed: int = 42) -> DataLoader:
    """Deterministic validation loader shared by all peers (10% default)."""
    full = _mnist_dataset()
    _, val_idx = _train_val_indices(val_split=val_split, seed=seed)
    val_set = Subset(full, val_idx)
    # Deterministic (no shuffle, one worker)
    return DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=0)

@torch.no_grad()
def accuracy(model: torch.nn.Module, loader: DataLoader) -> float:
    """Top-1 accuracy on DEVICE."""
    model.eval()
    correct = 0
    total = 0
    for x, y in loader:
        x = x.to(DEVICE)
        y = y.to(DEVICE)
        logits = model(x)
        preds = logits.argmax(dim=1)
        correct += (preds == y).sum().item()
        total += y.numel()
    return correct / max(1, total)

# ---------------- Non-IID label helper (if you need it) ----------------

def assigned_labels_for_client(
    cid: int,
    num_clients: int,
    labels_per_client: int = 2,
    seed: int = 42,
) -> List[int]:
    """
    Deterministically assign a small set of labels to a client (for non-IID setups).
    Seed-based shuffle of [0..9] and a cyclic window per client.
    """
    assert labels_per_client >= 1
    rng = np.random.default_rng(seed)
    labels = list(range(10))
    rng.shuffle(labels)  # deterministic per seed
    start = (cid * labels_per_client) % len(labels)
    chosen = [labels[(start + i) % len(labels)] for i in range(labels_per_client)]
    return sorted(set(chosen))

# ---------------- Client partitioning ----------------

def load_partition_for_client(
    cid: Optional[int] = None,
    num_clients: int = 1,
    batch_size: int = 32,
    *,
    # alias/knobs
    client_id: Optional[int] = None,     # alias for cid (to match client.py)
    non_iid: bool = False,               # accepted for API compatibility
    labels_per_client: int = 2,          # (not used in this simple split)
    proportions: Optional[Sequence[float]] = None,
    dirichlet_alpha: Optional[float] = None,
    seed: int = 42,
    val_split: float = 0.10,
) -> DataLoader:
    """
    Return a DataLoader for client over the 90% training split.

    Accepts either `cid` or `client_id`. If `proportions` is provided (len == num_clients),
    creates an imbalanced IID split; otherwise equal split.

    non_iid/labels_per_client/dirichlet_alpha are ignored in this simple version
    (kept for API compatibility).
    """
    if cid is None and client_id is not None:
        cid = client_id
    assert cid is not None, "load_partition_for_client: provide cid or client_id"
    assert 0 <= cid < num_clients, "cid out of range"

    full = _mnist_dataset()
    train_idx, _ = _train_val_indices(val_split=val_split, seed=seed)

    # Deterministic shuffle of training indices
    rng = np.random.default_rng(seed)
    train_idx = np.array(train_idx)
    rng.shuffle(train_idx)

    n_train = len(train_idx)
    if proportions is not None:
        assert len(proportions) == num_clients, "proportions must match num_clients"
        p = np.asarray(proportions, dtype=np.float64)
        p = np.maximum(p, 0)
        if p.sum() == 0:
            p = np.ones_like(p)
        p = p / p.sum()
        raw = (p * n_train).astype(int)
        diff = n_train - raw.sum()
        frac = (p * n_train) - raw
        order = np.argsort(-frac)
        for i in range(diff):
            raw[order[i]] += 1
        sizes = raw.tolist()
    else:
        base = n_train // num_clients
        sizes = [base] * num_clients
        sizes[-1] = n_train - base * (num_clients - 1)

    starts = np.cumsum([0] + sizes[:-1])
    ends = np.cumsum(sizes)
    s, e = int(starts[cid]), int(ends[cid])
    my_idx = train_idx[s:e].tolist()

    train_set = Subset(full, my_idx)
    return DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=2, drop_last=False)

# ---------------- Training helper ----------------

def train_one_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion = torch.nn.CrossEntropyLoss(),
) -> float:
    """
    Minimal training loop for one epoch. Returns average loss.
    """
    model.train()
    total_loss = 0.0
    total_batches = 0
    for x, y in loader:
        x = x.to(DEVICE)
        y = y.to(DEVICE)
        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        total_loss += float(loss.item())
        total_batches += 1
    return total_loss / max(1, total_batches)

# ---------------- Parameter helpers (shared by client/aggregator/strategy) ----------------

def params_to_numpy(model: torch.nn.Module) -> List[np.ndarray]:
    """Convert model parameters to a list of numpy arrays."""
    return [p.detach().cpu().numpy().copy() for p in model.parameters()]

def numpy_to_params(model: torch.nn.Module, params: List[np.ndarray]) -> None:
    """Load numpy arrays back into model parameters."""
    for p, np_val in zip(model.parameters(), params):
        p.data = torch.tensor(np_val, dtype=p.data.dtype, device=p.data.device)
