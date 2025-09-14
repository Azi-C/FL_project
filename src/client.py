from __future__ import annotations
from typing import List, Optional, Sequence
import torch
import numpy as np

from model import create_model
from utils import DEVICE, train_one_epoch, params_to_numpy, numpy_to_params, load_partition_for_client
from storage_chain import FLStorageChain, unpack_params_float32


class Client:
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
        self.cid = int(cid)
        self.num_clients = int(num_clients)
        self.lr = float(lr)
        self.local_epochs = int(local_epochs)

        # Local model
        self.model = create_model().to(DEVICE)

        # Local data partition (IID / non-IID)
        self.trainloader = load_partition_for_client(
            cid=self.cid,
            num_clients=self.num_clients,
            batch_size=32,
            non_iid=non_iid,
            labels_per_client=labels_per_client,
            proportions=proportions,
            dirichlet_alpha=dirichlet_alpha,
        )

    # ---------- Parameters I/O ----------

    def get_params(self) -> List[np.ndarray]:
        """Return model parameters as list of numpy arrays."""
        return params_to_numpy(self.model)

    def set_params(self, params: List[np.ndarray]) -> None:
        """Load numpy parameters into local model."""
        numpy_to_params(self.model, params)

    def sync_from_storage(
        self,
        store: FLStorageChain,
        round_id: int,
        writer_id: int,
        template: List[np.ndarray],
        chunk_size: int = 4 * 1024,
    ) -> None:
        """
        Pull the authoritative global params from FLStorage and load them locally.

        Args:
            store: FLStorageChain instance (connected to RPC and contract).
            round_id: On-chain round id where the global model is stored.
            writer_id: Writer namespace (e.g., GLOBAL_NS_OFFSET + winner_agg_id).
            template: Parameter list used only for shapes/dtypes.
            chunk_size: Bytes per chunk when downloading.
        """
        blob = store.download_blob(round_id, writer_id, chunk_size=chunk_size)
        params = unpack_params_float32(blob, template)
        self.set_params(params)

    # ---------- Local training ----------

    def train_local(self) -> None:
        """One local training session (local_epochs) on this client's partition."""
        optimizer = torch.optim.SGD(self.model.parameters(), lr=self.lr, momentum=0.9)
        criterion = torch.nn.CrossEntropyLoss()
        for _ in range(self.local_epochs):
            train_one_epoch(self.model, self.trainloader, criterion, optimizer)

    # ---------- Meta ----------

    def num_examples(self) -> int:
        """Return number of local training examples."""
        return len(self.trainloader.dataset)
