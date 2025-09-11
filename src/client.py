# client.py
from typing import List
import numpy as np
import torch

from model import create_model
from utils import (
    load_partition_for_client,
    load_test_loader,
    train_one_epoch,
    accuracy,
    DEVICE,
)

# --------- helpers params <-> numpy ----------
def params_to_numpy(model: torch.nn.Module) -> List[np.ndarray]:
    return [p.detach().cpu().numpy() for _, p in model.state_dict().items()]

def numpy_to_params(model: torch.nn.Module, params: List[np.ndarray]) -> None:
    keys = list(model.state_dict().keys())
    state = {k: torch.tensor(v) for k, v in zip(keys, params)}
    model.load_state_dict(state, strict=True)


class Client:
    """
    Un client :
    - a son modèle local
    - s'entraîne sur sa partition (dans les 90% du train)
    - peut évaluer (test partagé ou V)
    - sait envoyer/recevoir des poids (numpy)
    """
    def __init__(self, cid: int, num_clients: int, lr: float = 0.01, local_epochs: int = 1):
        self.cid = cid
        self.num_clients = num_clients
        self.lr = lr
        self.local_epochs = local_epochs

        self.model = create_model().to(DEVICE)

        # Partition *dans les 90% du train*
        self.trainloader = load_partition_for_client(client_id=cid, num_clients=num_clients)

        # Test partagé (set officiel MNIST)
        self.testloader = load_test_loader()

    # --- sync ---
    def set_params(self, params: List[np.ndarray]) -> None:
        numpy_to_params(self.model, params)

    def get_params(self) -> List[np.ndarray]:
        return params_to_numpy(self.model)

    def num_examples(self) -> int:
        return len(self.trainloader.dataset)

    # --- local work ---
    def train_local(self) -> None:
        for _ in range(self.local_epochs):
            train_one_epoch(self.model, self.trainloader, lr=self.lr, device=DEVICE)

    def evaluate(self) -> float:
        return accuracy(self.model, self.testloader, device=DEVICE)

    def evaluate_on(self, loader) -> float:
        return accuracy(self.model, loader, device=DEVICE)
