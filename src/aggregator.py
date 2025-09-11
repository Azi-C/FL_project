from client import Client
from utils import DEVICE, accuracy
import torch

class Aggregator(Client):
    def aggregate(self, updates, valloader=None, clients=None, tau=0.90):
        total_examples = sum(n for n, _ in updates)
        new_params = []
        for i in range(len(updates[0][1])):
            weighted_sum = sum(n * updates[j][1][i] for j, (n, _) in enumerate(updates))
            new_params.append(weighted_sum / total_examples)

        report = {"passed": [], "failed": []}
        if valloader is not None and clients is not None:
            for c in clients:
                old_params = c.get_params()
                c.set_params(new_params)
                val_acc = c.evaluate_on(valloader)
                if val_acc >= tau:
                    report["passed"].append((c.cid, val_acc))
                else:
                    report["failed"].append((c.cid, val_acc))
                c.set_params(old_params)

        return new_params, report
