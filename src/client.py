from __future__ import annotations
from typing import List
import torch
import numpy as np

from src.model import create_model
from src.utils import DEVICE, load_partition_for_client, train_one_epoch, params_to_numpy, numpy_to_params

class Client:
    def __init__(self, cid: int, num_clients: int, lr: float = 0.01, local_epochs: int = 1,
                 non_iid: bool = False, labels_per_client: int = 2, proportions=None,
                 dirichlet_alpha=None, batch_size: int = 32, imbalanced: bool = False):
        self.cid = cid
        self.lr = lr
        self.local_epochs = local_epochs
        self.model = create_model().to(DEVICE)
        self.trainloader = load_partition_for_client(cid, num_clients, batch_size=batch_size, imbalanced=imbalanced)

    def get_params(self):
        return params_to_numpy(self.model)

    def set_params(self, params):
        numpy_to_params(self.model, params)

    def train_local(self) -> None:
        opt = torch.optim.SGD(self.model.parameters(), lr=self.lr, momentum=0.9)
        crit = torch.nn.CrossEntropyLoss()
        for _ in range(self.local_epochs):
            train_one_epoch(self.model, self.trainloader, crit, opt)
