# strategy.py
from typing import List, Tuple
import torch
from model import create_model
from client import Client
from aggregator import Aggregator
from utils import DEVICE

def select_aggregators(round_idx: int, aggregators: List[Aggregator], k: int = 1, policy: str = "rotate") -> List[Aggregator]:
    n = len(aggregators)
    if n == 0:
        return []
    if policy == "rotate":
        start = (round_idx - 1) % n
        return [aggregators[(start + i) % n] for i in range(min(k, n))]
    else:
        return aggregators[:min(k, n)]

def run_rounds(
    num_clients: int = 4,
    num_aggregators: int = 1,
    num_rounds: int = 3,
    local_epochs: int = 1,
    lr: float = 0.01,
    agg_policy: str = "rotate",
):
    clients: List[Client] = []
    aggregators: List[Aggregator] = []

    for cid in range(num_clients):
        if cid < num_aggregators:
            agg = Aggregator(cid, num_clients, lr=lr, local_epochs=local_epochs)
            aggregators.append(agg)
            clients.append(agg)
        else:
            clients.append(Client(cid, num_clients, lr=lr, local_epochs=local_epochs))

    global_model = create_model().to(DEVICE)
    from client import params_to_numpy, numpy_to_params
    global_params = params_to_numpy(global_model)

    print(f"Starting serverless FL with {num_clients} clients, {len(aggregators)} aggregator-capable nodes")
    for rnd in range(1, num_rounds + 1):
        print(f"\n=== Round {rnd} ===")

        for c in clients:
            c.set_params(global_params)

        updates: List[Tuple[int, List[np.ndarray]]] = []
        for c in clients:
            c.train_local()
            updates.append((c.num_examples(), c.get_params()))

        chosen = select_aggregators(rnd, aggregators, k=1, policy=agg_policy)
        chosen_ids = [a.cid for a in chosen]
        print(f"Chosen aggregator(s): {chosen_ids}")

        if not chosen:
            raise RuntimeError("No aggregators available")
        aggregated = chosen[0].aggregate(updates)

        global_params = aggregated

        accs = []
        for c in clients:
            c.set_params(global_params)
            accs.append(c.evaluate())
        print(f"Average accuracy after round {rnd}: {sum(accs)/len(accs):.4f}")

    numpy_to_params(global_model, global_params)
    torch.save(global_model.state_dict(), "global_mnist_cnn.pt")
    print("Saved final global model to global_mnist_cnn.pt")


if __name__ == "__main__":
    run_rounds(
        num_clients=4,
        num_aggregators=2,
        num_rounds=3,
        local_epochs=1,
        lr=0.01,
        agg_policy="rotate",
    )
