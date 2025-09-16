from __future__ import annotations
from typing import List, Optional, Sequence
import torch
import numpy as np
import hashlib

from model import create_model
from utils import DEVICE, train_one_epoch, params_to_numpy, numpy_to_params, load_partition_for_client
from storage_chain import FLStorageChain, unpack_params_float32


def _hash_params_rounded(params: List[np.ndarray], decimals: int = 6) -> str:
    """Replicates strategy.params_hash: float32, round, then SHA-256 over bytes."""
    h = hashlib.sha256()
    for a in params:
        arr = np.asarray(a, dtype=np.float32)
        if decimals is not None:
            arr = np.round(arr, decimals=decimals)
        h.update(arr.tobytes(order="C"))
    return h.hexdigest()


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
        Pull the authoritative params from FLStorage and load them locally.
        """
        blob = store.download_blob(round_id, writer_id, chunk_size=chunk_size)
        params = unpack_params_float32(blob, template)
        self.set_params(params)

    def fetch_baseline_from_chain(
        self,
        chain,
        store: FLStorageChain,
        template: List[np.ndarray],
        chunk_size: int = 4 * 1024,
        verify_hash: bool = True,
    ) -> None:
        """
        Fetch the baseline model from the coordinator + storage.
        - Reads the baseline pointer from FLCoordinator.
        - Downloads from FLStorage.
        - Optionally verifies the hash against the on-chain baselineHash
          using the same rounding-based hashing as strategy.params_hash.
        """
        set_, h_hex, rid, wid, _ = chain.get_baseline()
        if not set_:
            raise RuntimeError("Baseline not assigned on-chain yet.")

        # Download baseline bytes
        blob = store.download_blob(rid, wid, chunk_size=chunk_size)
        # Unpack into params using the provided template (for shapes)
        params = unpack_params_float32(blob, template)
        # Load into local model
        self.set_params(params)

        # Optional integrity check (match strategy’s hashing)
        if verify_hash:
            local_hash = _hash_params_rounded(params, decimals=6)
            # h_hex from contract comes without 0x and already lowercase via .hex()
            if local_hash != h_hex.lower().lstrip("0x"):
                raise RuntimeError(f"Baseline hash mismatch for client {self.cid}")

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
