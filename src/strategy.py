import torch
import numpy as np
from typing import List, Tuple

from model import create_model
from utils import (
    load_data,                
    train_one_epoch,
    accuracy,
    DEVICE,
)

# ---- Helpers: torch <-> numpy for state_dict ----
def params_to_numpy(model: torch.nn.Module) -> List[np.ndarray]:
    return [p.detach().cpu().numpy() for _, p in model.state_dict().items()]

def numpy_to_params(model: torch.nn.Module, params: List[np.ndarray]) -> None:
    keys = list(model.state_dict().keys())
    state = {k: torch.tensor(v) for k, v in zip(keys, params)}
    model.load_state_dict(state, strict=True)

def fedavg(updates: List[Tuple[int, List[np.ndarray]]]) -> List[np.ndarray]:
    """Weighted average of params by num_examples."""
    total = sum(n for n, _ in updates)
    if total == 0:
        raise ValueError("No samples to aggregate.")
    _, first = updates[0]
    agg = [np.zeros_like(arr) for arr in first]
    for n, params in updates:
        w = n / total
        for i, arr in enumerate(params):
            agg[i] += w * arr
    return agg

class Peer:
    """A client that can also act as aggregator when selected."""
    def __init__(self, cid: int, num_clients: int, lr: float = 0.01, local_epochs: int = 1):
        self.cid = cid
        self.num_clients = num_clients
        self.lr = lr
        self.local_epochs = local_epochs

        self.model = create_model().to(DEVICE)
        try:
            # Try to load client-specific data partition
            self.trainloader = load_partition_for_client(client_id=cid, num_clients=num_clients)
        except Exception:
            # Fallback: use full dataset if partitioning not yet implemented
            self.trainloader, _ = load_data()

        # Shared test set for evaluation
        _, self.testloader = load_data()

    def get_params(self) -> List[np.ndarray]:
        return params_to_numpy(self.model)

    def set_params(self, params: List[np.ndarray]) -> None:
        numpy_to_params(self.model, params)

    def num_examples(self) -> int:
        return len(self.trainloader.dataset)

    def train_local(self):
        for _ in range(self.local_epochs):
            train_one_epoch(self.model, self.trainloader, lr=self.lr, device=DEVICE)

    def evaluate(self) -> float:
        return accuracy(self.model, self.testloader, device=DEVICE)

def run_decentralized(
    num_clients: int = 2,
    num_rounds: int = 3,
    local_epochs: int = 1,
    lr: float = 0.01,
):
    # Initialize peers
    peers = [Peer(cid=i, num_clients=num_clients, lr=lr, local_epochs=local_epochs) for i in range(num_clients)]

    # Start with a global model
    global_model = create_model().to(DEVICE)
    global_params = params_to_numpy(global_model)

    print(f"Starting decentralized FL with {num_clients} clients for {num_rounds} rounds")

    for rnd in range(1, num_rounds + 1):
        print(f"\n=== Round {rnd} ===")

        # Broadcast global params
        for p in peers:
            p.set_params(global_params)

        # Local training
        updates: List[Tuple[int, List[np.ndarray]]] = []
        for p in peers:
            p.train_local()
            updates.append((p.num_examples(), p.get_params()))

        # Select aggregator (rotation)
        agg_id = (rnd - 1) % num_clients
        print(f"Aggregator this round: client {agg_id}")

        # Aggregation
        aggregated = fedavg(updates)
        global_params = aggregated

        # Evaluate global model
        accs = []
        for p in peers:
            p.set_params(global_params)
            accs.append(p.evaluate())
        avg_acc = sum(accs) / len(accs)
        print(f"Avg accuracy after round {rnd}: {avg_acc:.4f}")

    # Save final model
    numpy_to_params(global_model, global_params)
    torch.save(global_model.state_dict(), "global_mnist_cnn.pt")
    print("Saved final global model to global_mnist_cnn.pt")


if __name__ == "__main__":
    run_decentralized(num_clients=2, num_rounds=3, local_epochs=1, lr=0.01)