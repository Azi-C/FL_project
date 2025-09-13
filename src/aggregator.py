# aggregator.py
from typing import List, Tuple, Dict, Sequence, Optional
import numpy as np
import torch

from utils import DEVICE, accuracy
from model import create_model
from client import Client, params_to_numpy, numpy_to_params
from storage_chain import unpack_params_float32

class Aggregator(Client):
    """Client that can also aggregate proposals from on-chain client blobs."""

    # Legacy path (kept for compatibility/testing)
    def aggregate(
        self,
        updates: List[Tuple[int, List[np.ndarray]]],
        valloader,
        clients: List[Client],
        tau: float = 1.0,
    ):
        base_acc = accuracy(self.model, valloader)
        passed: List[Tuple[int, float, List[np.ndarray]]] = []
        failed: List[Tuple[int, float]] = []

        for (n_i, p_i), c in zip(updates, clients):
            tmp = create_model().to(DEVICE)
            numpy_to_params(tmp, p_i)
            acc_i = accuracy(tmp, valloader)
            if acc_i + 1e-12 >= base_acc * tau:
                passed.append((c.cid, acc_i, p_i))
            else:
                failed.append((c.cid, acc_i))

        if not passed:
            if failed:
                best_idx = int(np.argmax([a for (_, a) in failed]))
                cid_f, acc_f = failed[best_idx]
                p_f = updates[best_idx][1]
                passed = [(cid_f, acc_f, p_f)]
                failed = []
            else:
                p_f = params_to_numpy(self.model)
                passed = [(int(self.cid), float(base_acc), p_f)]

        # Weights by |D_i|
        n_map = {c.cid: n for (n, _), c in zip(updates, clients)}
        weights = np.asarray([float(n_map[cid]) for (cid, _, _) in passed], dtype=np.float64)
        total_w = float(np.sum(weights)) if np.sum(weights) > 0 else 1.0

        acc_params = [np.zeros_like(p, dtype=np.float32) for p in passed[0][2]]
        for (cid, _, p_i), w in zip(passed, weights):
            for j in range(len(acc_params)):
                acc_params[j] += (w * np.asarray(p_i[j], dtype=np.float32))

        aggregated = [p / total_w for p in acc_params]
        report = {
            "passed": [(int(cid), float(acc)) for (cid, acc, _) in passed],
            "failed": [(int(cid), float(acc)) for (cid, acc) in failed],
        }
        return aggregated, report

    # Deterministic path: use a precomputed valid_ids list provided by strategy
    def aggregate_from_chain_deterministic(
        self,
        store,                       # FLStorageChain
        round_id: int,
        template_params: List[np.ndarray],
        valid_ids: List[int],        # <- fixed set chosen by strategy
        client_sizes: Dict[int, int],
    ):
        """Average only the specified valid_ids. No local validation here."""
        if not valid_ids:
            # Fall back to self only
            params_f = params_to_numpy(self.model)
            weights = np.asarray([1.0], dtype=np.float64)
            total_w = 1.0
            return params_f, {"passed": [(int(self.cid), 0.0)], "failed": []}

        # Download and prepare weights
        weights = np.asarray([float(client_sizes.get(int(cid), 0)) for cid in valid_ids], dtype=np.float64)
        total_w = float(np.sum(weights)) if np.sum(weights) > 0 else 1.0

        first_blob = store.download_blob(round_id, int(valid_ids[0]))
        first_params = unpack_params_float32(first_blob, template_params)
        acc_params = [np.zeros_like(p, dtype=np.float32) for p in first_params]

        for cid, w in zip(valid_ids, weights):
            blob = store.download_blob(round_id, int(cid))
            params_i = unpack_params_float32(blob, template_params)
            for j in range(len(acc_params)):
                acc_params[j] += (w * np.asarray(params_i[j], dtype=np.float32))

        aggregated = [p / total_w for p in acc_params]
        report = {
            "passed": [(int(cid), 0.0) for cid in valid_ids],  # accuracy not computed here
            "failed": [],
        }
        return aggregated, report
