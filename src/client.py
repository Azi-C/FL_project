# client.py
import torch
from model import create_model
from utils import load_data, train_one_epoch, accuracy, DEVICE
from typing import List
import numpy as np

# ---------- Helpers: model params <-> numpy ----------
def params_to_numpy(model: torch.nn.Module) -> List[np.ndarray]:
    return [p.detach().cpu().numpy() for _, p in model.state_dict().items()]

def numpy_to_params(model: torch.nn.Module, params: List[np.ndarray]) -> None:
    keys = list(model.state_dict().keys())
    state = {k: torch.tensor(v) for k, v in zip(keys, params)}
    model.load_state_dict(state, strict=True)


class Client:
    """
    Basic client:
    - holds its own model and data partition
    - can set/get global params
    - can train locally and evaluate
    """
    def __init__(self, cid: int, num_clients: int, lr: float = 0.01, local_epochs: int = 1):
        self.cid = cid
        self.num_clients = num_clients
        self.lr = lr
        self.local_epochs = local_epochs

        self.model = create_model().to(DEVICE)

        # Partitioned train set per client (fallback to full train if not available)
        try:
            self.trainloader = load_partition_for_client(client_id=cid, num_clients=num_clients)
        except Exception:
            self.trainloader, _ = load_data()

        # Shared test set for reporting
        _, self.testloader = load_data()

    # ---- synchronization ----
    def set_params(self, params: List[np.ndarray]) -> None:
        numpy_to_params(self.model, params)

    def get_params(self) -> List[np.ndarray]:
        return params_to_numpy(self.model)

    def num_examples(self) -> int:
        return len(self.trainloader.dataset)

    # ---- local work ----
    def train_local(self) -> None:
        for _ in range(self.local_epochs):
            train_one_epoch(self.model, self.trainloader, lr=self.lr, device=DEVICE)

    def evaluate(self) -> float:
        return accuracy(self.model, self.testloader, device=DEVICE)
