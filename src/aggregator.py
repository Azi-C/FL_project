from typing import List, Tuple
import numpy as np
from client import Client

class Aggregator(Client):
    """
    An Aggregator is a Client that can also aggregate others' updates.
    """
    def aggregate(self, updates: List[Tuple[int, List[np.ndarray]]]) -> List[np.ndarray]:
        """
        FedAvg:
        new model = (Σ |D_i| * m'_i) / (Σ |D_i|)
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
