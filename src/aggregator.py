# aggregator.py
from typing import List, Tuple
import numpy as np
from client import Client

def _fedavg(updates: List[Tuple[int, List[np.ndarray]]]) -> List[np.ndarray]:
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
    def aggregate(
        self,
        updates: List[Tuple[int, List[np.ndarray]]],
        valloader=None,
        clients: List[Client] | None = None,
        tau: float = 0.90,
    ) -> List[np.ndarray]:
        # If no validation provided, aggregate all
        if valloader is None or clients is None:
            return _fedavg(updates)

        eligible: List[Tuple[int, List[np.ndarray]]] = []
        for (n, params), c in zip(updates, clients):
            # Save client params, test candidate params on V, then restore
            old = c.get_params()
            c.set_params(params)
            val_acc = c.evaluate_on(valloader)   # <-- now exists
            c.set_params(old)

            if val_acc >= tau:
                eligible.append((n, params))

        if not eligible:  # fallback to keep training progressing
            eligible = updates

        return _fedavg(eligible)
