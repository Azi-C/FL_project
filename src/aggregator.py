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

    def aggregate(
        self,
        updates: List[Tuple[int, List[np.ndarray]]],
        valloader,
        clients: List[Client],
        tau: float = 1.0,
    ):
        """(Legacy) Aggregate from in-memory updates list (kept for compatibility)."""
        # Baseline accuracy with current global (self.model already set to global)
        base_acc = accuracy(self.model, valloader)

        passed: List[Tuple[int, float, List[np.ndarray]]] = []
        failed: List[Tuple[int, float]] = []

        # Evaluate each client model
        for (n_i, p_i), c in zip(updates, clients):
            tmp = create_model().to(DEVICE)
            numpy_to_params(tmp, p_i)
            acc_i = accuracy(tmp, valloader)
            # pass if non-decreasing (or relaxed by tau factor)
            if acc_i >= base_acc * tau:
                passed.append((c.cid, acc_i, p_i))
            else:
                failed.append((c.cid, acc_i))

        if not passed:
            # fallback: include best client
            best_idx = int(np.argmax([a for (_, a) in [(cid, acc) for cid, acc in failed]])) if failed else 0
            cid_f, acc_f = failed[best_idx] if failed else (clients[0].cid, base_acc)
            p_f = updates[best_idx][1] if failed else updates[0][1]
            passed = [(cid_f, acc_f, p_f)]
            failed = []

        # FedAvg over PASSED, weighted by |D_i| (use provided n_i weights)
        # Need weights for the subset; reconstruct map cid->n_i
        n_map = {c.cid: n for (n, _), c in zip(updates, clients)}
        weights = np.asarray([float(n_map[cid]) for (cid, _, _) in passed], dtype=np.float64)
        total_w = float(np.sum(weights)) if np.sum(weights) > 0 else 1.0

        # Initialize accumulator with zeros
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

    # --- NEW: aggregate by downloading updates from FLStorage on-chain ---
    def aggregate_from_chain(
        self,
        store,                       # FLStorageChain
        round_id: int,
        template_params: List[np.ndarray],
        client_ids: List[int],
        client_sizes: List[int],
        valloader,
        tau: float = 1.0,
    ):
        """Download each client's blob from chain, validate vs V, then FedAvg over passed."""
        # Baseline accuracy with current global (self.model already set)
        base_acc = accuracy(self.model, valloader)

        # Build size map
        size_map: Dict[int, int] = {int(cid): int(sz) for cid, sz in zip(client_ids, client_sizes)}

        passed: List[Tuple[int, float, List[np.ndarray]]] = []
        failed: List[Tuple[int, float]] = []

        for cid in client_ids:
            # Download and unpack this client's params
            blob = store.download_blob(round_id, int(cid))
            params_i = unpack_params_float32(blob, template_params)

            # Evaluate
            tmp = create_model().to(DEVICE)
            numpy_to_params(tmp, params_i)
            acc_i = accuracy(tmp, valloader)

            if acc_i >= base_acc * tau:
                passed.append((int(cid), float(acc_i), params_i))
            else:
                failed.append((int(cid), float(acc_i)))

        if not passed:
            # fallback: take best failed one
            if failed:
                best_idx = int(np.argmax([a for (_, a) in failed]))
                cid_f, acc_f = failed[best_idx]
                blob = store.download_blob(round_id, int(cid_f))
                params_f = unpack_params_float32(blob, template_params)
                passed = [(int(cid_f), float(acc_f), params_f)]
                failed = []
            else:
                # extreme fallback: use own current params
                params_f = params_to_numpy(self.model)
                passed = [(int(self.cid), float(base_acc), params_f)]

        # FedAvg over passed with |D_i| weights
        weights = np.asarray([float(size_map.get(cid, 0)) for (cid, _, _) in passed], dtype=np.float64)
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
