# strategy.py
from typing import List, Tuple
import torch

from model import create_model
from client import Client, params_to_numpy, numpy_to_params
from aggregator import Aggregator
from utils import DEVICE, load_validation_loader

def select_aggregators(
    round_idx: int,
    aggregators: List[Aggregator],
    k: int = 1,
    policy: str = "rotate",
) -> List[Aggregator]:
    """
    Choisit >= 1 agrégateur(s) par round.
    'rotate' : round-robin simple.
    """
    n = len(aggregators)
    if n == 0:
        return []
    k = max(1, min(k, n))
    if policy == "rotate":
        start = (round_idx - 1) % n
        return [aggregators[(start + i) % n] for i in range(k)]
    return aggregators[:k]

def run_rounds(
    num_clients: int = 4,
    num_aggregators: int = 1,
    num_rounds: int = 3,
    local_epochs: int = 1,
    lr: float = 0.01,
    agg_policy: str = "rotate",
    k_aggregators: int = 1,
    tau: float = 0.90,
):
    if num_clients < 1:
        raise ValueError("num_clients must be >= 1")
    if num_aggregators < 1:
        raise ValueError("num_aggregators must be >= 1")

    # Population
    clients: List[Client] = []
    aggregators: List[Aggregator] = []
    for cid in range(num_clients):
        if cid < num_aggregators:
            agg = Aggregator(cid, num_clients, lr=lr, local_epochs=local_epochs)
            aggregators.append(agg)
            clients.append(agg)
        else:
            clients.append(Client(cid, num_clients, lr=lr, local_epochs=local_epochs))

    # Validation partagée V (10% du train)
    valloader = load_validation_loader()

    # Modèle global
    global_model = create_model().to(DEVICE)
    global_params = params_to_numpy(global_model)

    print(f"Starting serverless FL with {num_clients} clients and {len(aggregators)} aggregator-capable node(s)")
    for rnd in range(1, num_rounds + 1):
        print(f"\n=== Round {rnd} ===")

        # 1) Broadcast
        for c in clients:
            c.set_params(global_params)

        # 2) Local train + collecte des updates
        updates: List[Tuple[int, List]] = []
        for c in clients:
            c.train_local()
            updates.append((c.num_examples(), c.get_params()))
        if not updates:
            raise RuntimeError("No client updates collected.")

        # 3) Sélection d'agrégateur(s)
        chosen = select_aggregators(rnd, aggregators, k=k_aggregators, policy=agg_policy)
        if not chosen:
            raise RuntimeError("No aggregators available; ensure num_aggregators >= 1.")
        print(f"Chosen aggregator(s): {[a.cid for a in chosen]}")

        # 4) Agrégation (avec filtrage V si fourni)
        aggregated = chosen[0].aggregate(updates, valloader=valloader, clients=clients, tau=tau)

        # 5) Nouveau global + évaluation
        global_params = aggregated

        accs = []
        for c in clients:
            c.set_params(global_params)
            accs.append(c.evaluate())
        avg_acc = sum(accs) / len(accs)
        print(f"Average accuracy after round {rnd}: {avg_acc:.4f}")

    # Sauvegarde finale
    numpy_to_params(global_model, global_params)
    torch.save(global_model.state_dict(), "global_mnist_cnn.pt")
    print("Saved final global model to global_mnist_cnn.pt")


if __name__ == "__main__":
    run_rounds(
        num_clients=4,
        num_aggregators=1,
        num_rounds=3,
        local_epochs=1,
        lr=0.01,
        agg_policy="rotate",
        k_aggregators=1,
        tau=0.90,
    )
