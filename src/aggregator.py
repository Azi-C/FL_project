# aggregator.py
from typing import List, Tuple, Dict, Any
import numpy as np
from client import Client

def _fedavg(updates: List[Tuple[int, List[np.ndarray]]]) -> List[np.ndarray]:
    """
    Weighted FedAvg:
      w_t+1 = (Σ |D_i| * m'_i) / (Σ |D_i|)
    """
    total = sum(n for n, _ in updates)
    if total == 0:
        raise ValueError("No samples to aggregate.")
    _, first = updates[0]
    agg = [np.zeros_like(arr) for arr in first]
    for n, params in updates:
        w = n / total
        for i, arr in enumerate(params):
            agg[i] += w * arr
    return agg

class Aggregator(Client):
    """
    Aggregator extends Client with aggregation.
    Returns (aggregated_params, report) where report contains pass/fail on V.
    """
    def aggregate(
        self,
        updates: List[Tuple[int, List[np.ndarray]]],
        valloader=None,
        clients: List[Client] | None = None,
        tau: float = 0.90,
    ) -> Tuple[List[np.ndarray], Dict[str, Any]]:
        if valloader is None or clients is None:
            return _fedavg(updates), {"passed": [], "failed": []}

        eligible: List[Tuple[int, List[np.ndarray]]] = []
        passed, failed = [], []

        for (n, params), c in zip(updates, clients):
            old = c.get_params()
            c.set_params(params)
            val_acc = c.evaluate_on(valloader)
            c.set_params(old)

            if val_acc >= tau:
                eligible.append((n, params))
                passed.append((c.cid, float(val_acc)))
            else:
                failed.append((c.cid, float(val_acc)))

        if not eligible:  # fallback to keep training progressing
            eligible = updates

        aggregated = _fedavg(eligible)
        report = {"passed": passed, "failed": failed}
        return aggregated, report
