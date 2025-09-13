# client.py
from typing import List, Optional, Sequence
import numpy as np
import torch

from model import create_model
from utils import (
    load_partition_for_client,
    load_validation_loader,
    train_one_epoch,
    accuracy,
    assigned_labels_for_client,
    DEVICE,
)

# ---------- helpers to convert params ----------
def params_to_numpy(model: torch.nn.Module) -> List[np.ndarray]:
    return [p.detach().cpu().numpy() for _, p in model.state_dict().items()]

def numpy_to_params(model: torch.nn.Module, params: List[np.ndarray]) -> None:
    keys = list(model.state_dict().keys())
    state = {k: torch.tensor(v) for k, v in zip(keys, params)}
    model.load_state_dict(state, strict=True)


class Client:
    """
    Federated client:
    - trains on its own partition (IID or Non-IID) within the 90% train pool
    - evaluates on shared validation (here used as 'test' metric)
    - can send/receive parameters as numpy lists
    """
    def __init__(
        self,
        cid: int,
        num_clients: int,
        lr: float = 0.01,
        local_epochs: int = 1,
        non_iid: bool = False,
        labels_per_client: int = 2,
        proportions: Optional[Sequence[float]] = None,
        dirichlet_alpha: Optional[float] = None,
    ):
        self.cid = cid
        self.num_clients = num_clients
        self.lr = lr
        self.local_epochs = local_epochs

        self.model = create_model().to(DEVICE)

        if non_iid:
            self.trainloader = load_partition_for_client(
                client_id=cid, num_clients=num_clients,
                labels_per_client=labels_per_client
            )
            self.assigned_labels = assigned_labels_for_client(cid, labels_per_client)
        else:
            self.trainloader = load_partition_for_client(
                client_id=cid, num_clients=num_clients,
                proportions=proportions, dirichlet_alpha=dirichlet_alpha
            )
            self.assigned_labels = None

        # We reuse the 10% validation split as a shared evaluation set
        self.testloader = load_validation_loader()

        # Debug: show client dataset size
        print(f"[Client {self.cid}] num_examples = {len(self.trainloader.dataset)}")

    # ---- sync ----
    def get_params(self) -> List[np.ndarray]:
        return params_to_numpy(self.model)

    def set_params(self, params: List[np.ndarray]) -> None:
        numpy_to_params(self.model, params)

    def num_examples(self) -> int:
        return len(self.trainloader.dataset)

    # ---- local work ----
    def train_local(self) -> None:
        optimizer = torch.optim.SGD(self.model.parameters(), lr=self.lr, momentum=0.9)
        criterion = torch.nn.CrossEntropyLoss()
        for _ in range(self.local_epochs):
            train_one_epoch(self.model, self.trainloader, criterion, optimizer)

    def evaluate(self) -> float:
        return accuracy(self.model, self.testloader)

    def evaluate_on(self, loader) -> float:
        return accuracy(self.model, loader)
