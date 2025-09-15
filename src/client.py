from __future__ import annotations

from typing import List

import numpy as np
import torch

from src.model import create_model
from src.utils import (
    DEVICE,
    load_partition_for_client,
    params_to_numpy,
    numpy_to_params,
    train_one_epoch,
)


class Client:
    """Simple local-training client."""

    def __init__(
        self,
        cid: int,
        num_clients: int,
        lr: float = 0.01,
        local_epochs: int = 1,
        batch_size: int = 32,
        non_iid: bool = False,          # kept for compatibility (not used here)
        labels_per_client: int = 2,     # kept for compatibility (not used here)
        proportions=None,               # kept for compatibility (not used here)
        dirichlet_alpha=None,           # kept for compatibility (not used here)
    ) -> None:
        self.cid = cid
        self.num_clients = num_clients
        self.lr = lr
        self.local_epochs = local_epochs

        # Model and data
        self.model = create_model().to(DEVICE)
        self.trainloader = load_partition_for_client(
            client_id=cid, num_clients=num_clients, batch_size=batch_size
        )

    # ----- Param helpers -----

    def get_params(self) -> List[np.ndarray]:
        """Return model parameters as a list of numpy arrays."""
        return params_to_numpy(self.model)

    def set_params(self, params: List[np.ndarray]) -> None:
        """Load numpy parameters into the local model."""
        numpy_to_params(self.model, params)

    # ----- Training -----

    def train_local(self) -> None:
        """Run local training for `local_epochs` epochs."""
        optimizer = torch.optim.SGD(self.model.parameters(), lr=self.lr, momentum=0.9)
        criterion = torch.nn.CrossEntropyLoss()
        for _ in range(self.local_epochs):
            train_one_epoch(self.model, self.trainloader, criterion, optimizer)
