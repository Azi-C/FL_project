# strategy.py
from typing import List, Tuple, Dict
import torch

from model import create_model
from client import Client
from aggregator import Aggregator
from utils import DEVICE, load_validation_loader
from persistence import save_json, load_json, save_model_state_dict


def select_by_reputation(reputation: Dict[int, float], gamma: float, aggregators: List[Aggregator], k: int):
    eligible = [a for a in aggregators if reputation.get(a.cid, 0.0) >= gamma]
    if not eligible:
        ranked = sorted(aggregators, key=lambda a: reputation.get(a.cid, 0.0), reverse=True)
        return ranked[: max(1, min(k, len(ranked)))]
    elig_ranked = sorted(eligible, key=lambda a: reputation.get(a.cid, 0.0), reverse=True)
    return elig_ranked[: max(1, min(k, len(elig_ranked)))]


def run_rounds(
    num_clients=4,
    num_aggregators=2,
    num_rounds=3,
    local_epochs=1,
    lr=0.01,
    tau=0.90,
    gamma=0.20,
    k_aggregators=1,
    non_iid=False,
    labels_per_client=2,
    proportions=None,
    dirichlet_alpha=None
):
    clients: List[Client] = []
    aggregators: List[Aggregator] = []

    for cid in range(num_clients):
        if cid < num_aggregators:
            agg = Aggregator(
                cid, num_clients, lr=lr, local_epochs=local_epochs,
                non_iid=non_iid, labels_per_client=labels_per_client,
                proportions=proportions, dirichlet_alpha=dirichlet_alpha
            )
            aggregators.append(agg)
            clients.append(agg)
        else:
            clients.append(Client(
                cid, num_clients, lr=lr, local_epochs=local_epochs,
                non_iid=non_iid, labels_per_client=labels_per_client,
                proportions=proportions, dirichlet_alpha=dirichlet_alpha
            ))

    valloader = load_validation_loader()

    global_model = create_model().to(DEVICE)
    global_params = clients[0].get_params()

    reputation_path = "artifacts/reputation.json"
    ckpt_dir = "artifacts/checkpoints"

    sizes = [c.num_examples() for c in clients]
    total_k = float(sum(sizes)) if sum(sizes) > 0 else 1.0
    reputation: Dict[int, float] = {c.cid: (sizes[i] / total_k) for i, c in enumerate(clients)}

    loaded_rep = load_json(reputation_path)
    if loaded_rep:
        reputation.update({int(k): float(v) for k, v in loaded_rep.items()})
        print("Loaded existing reputation from disk.")

    for rnd in range(1, num_rounds + 1):
        print(f"\n=== Round {rnd} ===")
        print(f"Reputations: { {cid: round(reputation[cid], 4) for cid in reputation} }")

        for c in clients:
            c.set_params(global_params)

        updates: List[Tuple[int, List]] = []
        for c in clients:
            c.train_local()
            updates.append((c.num_examples(), c.get_params()))

        chosen = select_by_reputation(reputation, gamma, aggregators, k_aggregators)
        aggregated, report = chosen[0].aggregate(updates, valloader=valloader, clients=clients, tau=tau)

        passed = set(cid for cid, _ in report.get("passed", []))
        failed = set(cid for cid, _ in report.get("failed", []))

        for cid in passed:
            reputation[cid] = reputation.get(cid, 0.0) + 1.0
        for cid in failed:
            reputation[cid] = reputation.get(cid, 0.0) - 1.0

        global_params = aggregated

        accs = []
        for c in clients:
            c.set_params(global_params)
            accs.append(c.evaluate())
        print(f"Avg accuracy after round {rnd}: {sum(accs)/len(accs):.4f}")

        save_json(reputation_path, {str(cid): float(r) for cid, r in reputation.items()})
        save_model_state_dict(f"{ckpt_dir}/round_{rnd:03d}.pt", global_model.state_dict())

    torch.save(global_model.state_dict(), "global_mnist_cnn.pt")
    print("Saved final model to global_mnist_cnn.pt")


if __name__ == "__main__":
    run_rounds(
        num_clients=4,
        num_aggregators=2,
        num_rounds=3,
        proportions=[0.5, 0.2, 0.2, 0.1]  # Example imbalance
    )
