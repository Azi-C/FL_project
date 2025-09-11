# strategy.py
from typing import List, Tuple, Dict, Sequence, Optional
import torch

from model import create_model
from client import Client, params_to_numpy, numpy_to_params
from aggregator import Aggregator
from utils import DEVICE, load_validation_loader
from persistence import save_json, load_json, save_model_state_dict


def select_by_reputation(
    reputation: Dict[int, float], gamma: float, aggregators: List[Aggregator], k: int
) -> List[Aggregator]:
    """Pick up to k aggregators with r_i >= gamma. Fallback: top-k by r."""
    if not aggregators:
        return []
    eligible = [a for a in aggregators if reputation.get(a.cid, 0.0) >= gamma]
    if not eligible:
        ranked = sorted(aggregators, key=lambda a: reputation.get(a.cid, 0.0), reverse=True)
        return ranked[: max(1, min(k, len(ranked)))]
    elig_ranked = sorted(eligible, key=lambda a: reputation.get(a.cid, 0.0), reverse=True)
    return elig_ranked[: max(1, min(k, len(elig_ranked)))]


def run_rounds(
    num_clients: int = 4,
    num_aggregators: int = 2,
    num_rounds: int = 3,
    local_epochs: int = 1,
    lr: float = 0.01,
    tau: float = 0.90,
    gamma: float = 0.20,
    k_aggregators: int = 1,
    non_iid: bool = False,                   # keep IID by default
    labels_per_client: int = 2,              # used only if non_iid=True
    proportions: Optional[Sequence[float]] = None,  # imbalance via explicit proportions
    dirichlet_alpha: Optional[float] = None,        # or via Dirichlet(alpha)
    reset_reputation: bool = False,                 # force fresh r_i^(0)
):
    if num_clients < 1:
        raise ValueError("num_clients must be >= 1")
    if num_aggregators < 1:
        raise ValueError("num_aggregators must be >= 1")

    # ---- build population ----
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

    # ---- shared validation loader (10% of train) ----
    valloader = load_validation_loader()

    # ---- global model/params ----
    global_model = create_model().to(DEVICE)
    global_params = params_to_numpy(global_model)

    # ---- persistence paths ----
    reputation_path = "artifacts/reputation.json"
    ckpt_dir = "artifacts/checkpoints"

    # ---- initial reputation r_i^(0) = |D_i| / Σ|D_j| ----
    sizes = [c.num_examples() for c in clients]
    total_k = float(sum(sizes)) if sum(sizes) > 0 else 1.0
    reputation: Dict[int, float] = {c.cid: (sizes[i] / total_k) for i, c in enumerate(clients)}

    # Debug: show sizes and init reps
    print("Client sizes:", {c.cid: sizes[i] for i, c in enumerate(clients)})
    print("Init reputations:", {c.cid: round(reputation[c.cid], 4) for c in clients})

    # Try to resume from disk unless resetting
    if not reset_reputation:
        loaded_rep = load_json(reputation_path)
        if loaded_rep:
            reputation.update({int(k): float(v) for k, v in loaded_rep.items()})
            print("Loaded existing reputation from disk.")

    print(f"Starting serverless FL with {num_clients} clients, {len(aggregators)} aggregator-capable node(s)")
    print(f"Reputation threshold gamma = {gamma}, validation tau = {tau}, non_iid={non_iid}")

    for rnd in range(1, num_rounds + 1):
        print(f"\n=== Round {rnd} ===")
        print(f"Reputations (start): { {cid: round(reputation[cid], 4) for cid in reputation} }")

        # 1) broadcast
        for c in clients:
            c.set_params(global_params)

        # 2) local train + collect updates
        updates: List[Tuple[int, List]] = []
        for c in clients:
            c.train_local()
            updates.append((c.num_examples(), c.get_params()))
        if not updates:
            raise RuntimeError("No client updates collected.")

        # 3) choose aggregator(s) by reputation threshold
        chosen = select_by_reputation(reputation, gamma, aggregators, k_aggregators)
        if not chosen:
            raise RuntimeError("No aggregators available; check num_aggregators.")
        print(f"Chosen aggregator(s): {[a.cid for a in chosen]}")

        # 4) aggregate with validation report (use the first chosen)
        aggregated, report = chosen[0].aggregate(
            updates, valloader=valloader, clients=clients, tau=tau
        )

        # 5) update reputations (+1 if passed V, -1 otherwise)
        passed = set(cid for cid, _ in report.get("passed", []))
        failed = set(cid for cid, _ in report.get("failed", []))

        for cid in passed:
            reputation[cid] = reputation.get(cid, 0.0) + 1.0
        for cid in failed:
            reputation[cid] = reputation.get(cid, 0.0) - 1.0

        print(f"Passed on V: {sorted(list(passed))} | Failed on V: {sorted(list(failed))}")
        print(f"Reputations (updated): { {cid: round(reputation[cid], 4) for cid in reputation} }")

        # 6) update global params and evaluate
        global_params = aggregated

        accs = []
        for c in clients:
            c.set_params(global_params)
            accs.append(c.evaluate())
        avg_acc = sum(accs) / len(accs)
        print(f"Average accuracy after round {rnd}: {avg_acc:.4f}")

        # 7) persist reputation and checkpoint for this round
        save_json(reputation_path, {str(cid): float(r) for cid, r in reputation.items()})
        # save checkpoint: load global_params -> global_model -> save state_dict
        numpy_to_params(global_model, global_params)
        save_model_state_dict(f"{ckpt_dir}/round_{rnd:03d}.pt", global_model.state_dict())

    # ---- save final model ----
    numpy_to_params(global_model, global_params)
    torch.save(global_model.state_dict(), "global_mnist_cnn.pt")
    print("Saved final global model to global_mnist_cnn.pt")


if __name__ == "__main__":
    # Example run with explicit imbalance (proportions must match num_clients)
    run_rounds(
        num_clients=4,
        num_aggregators=2,
        num_rounds=3,
        local_epochs=1,
        lr=0.01,
        tau=0.90,
        gamma=0.20,
        k_aggregators=1,
        non_iid=False,                            # keep IID
        proportions=[0.5, 0.2, 0.2, 0.1],         # <-- imbalanced sizes
        dirichlet_alpha=None,
        reset_reputation=True,                    # ignore old reputation.json
    )
