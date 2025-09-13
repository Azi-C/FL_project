# client.py
from typing import List, Optional
import torch
import numpy as np

from model import create_model
from utils import (
    DEVICE,
    load_partition_for_client,
    train_one_epoch,
    accuracy,
    params_to_numpy,
    numpy_to_params,
)

class Client:
    def __init__(
        self,
        cid: int,
        num_clients: int,
        lr: float = 0.01,
        local_epochs: int = 1,
        batch_size: int = 32,
        non_iid: bool = False,
        labels_per_client: int = 2,
        proportions: Optional[List[float]] = None,
        dirichlet_alpha: Optional[float] = None,
    ):
        self.cid = cid
        self.lr = lr
        self.local_epochs = local_epochs

        # Create model
        self.model = create_model().to(DEVICE)

        # Each client gets its own partition of the training data
        self.trainloader = load_partition_for_client(
            client_id=cid,                # alias supported in utils
            num_clients=num_clients,
            batch_size=batch_size,
            non_iid=non_iid,
            labels_per_client=labels_per_client,
            proportions=proportions,
            dirichlet_alpha=dirichlet_alpha,
        )

    def num_examples(self) -> int:
        return len(self.trainloader.dataset)

    def set_params(self, params: List[np.ndarray]) -> None:
        numpy_to_params(self.model, params)

    def get_params(self) -> List[np.ndarray]:
        return params_to_numpy(self.model)

    def train_local(self) -> None:
        optimizer = torch.optim.SGD(self.model.parameters(), lr=self.lr, momentum=0.9)
        criterion = torch.nn.CrossEntropyLoss()
        for _ in range(self.local_epochs):
            train_one_epoch(
                model=self.model,
                loader=self.trainloader,
                optimizer=optimizer,
                criterion=criterion,
            )

    def evaluate(self) -> float:
        return accuracy(self.model, self.trainloader)

