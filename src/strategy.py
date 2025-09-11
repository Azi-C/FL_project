# strategy.py
from typing import List, Tuple
import torch

from model import create_model
from client import Client, params_to_numpy, numpy_to_params
from aggregator import Aggregator
from utils import DEVICE, load_train_val

# ----------------------------------------------------
# Helper: always return at least one aggregator
# ----------------------------------------------------
def select_aggregators(
    round_idx: int,
    aggregators: List[Aggregator],
    k: int = 1,
    policy: str = "rotate",
) -> List[Aggregator]:
    """
    Choose which aggregators act this round.
    - 'rotate': simple round-robin
    Always returns a non-empty list if aggregators is non-empty.
    """
    n = len(aggregators)
    if n == 0:
        return []  # caller handles this (we fail fast there)
    k = max(1, min(k, n))  # at least 1, at most n

    if policy == "rotate":
        start = (round_idx - 1) % n
        return [aggregators[(start + i) % n] for i in range(k)]

    # default: first k
    return aggregators[:k]


# ----------------------------------------------------
# Main loop
# ----------------------------------------------------
def run_rounds(
    num_clients: int = 4,
    num_aggregators: int = 1,
    num_rounds: int = 3,
    local_epochs: int = 1,
    lr: float = 0.01,
    agg_policy: str = "rotate",
    k_aggregators: int = 1,
    tau: float = 0.90,  # validation threshold for V (if your Aggregator uses it)
):
    # --- sanity checks ---
    if num_clients < 1:
        raise ValueError("num_clients must be >= 1")
    if num_aggregators < 1:
        raise ValueError("You must have at least one aggregator (num_aggregators >= 1)")

    # --- build population ---
    clients: List[Client] = []
    aggregators: List[Aggregator] = []

    for cid in range(num_clients):
        if cid < num_aggregators:
            agg = Aggregator(cid, num_clients, lr=lr, local_epochs=local_epochs)
            aggregators.append(agg)
            clients.append(agg)  # Aggregator is also a Client (trains locally)
        else:
            clients.append(Client(cid, num_clients, lr=lr, local_epochs=local_epochs))

    # --- shared validation loader V (deterministic split created inside) ---
    # If you didn't add load_train_val to utils.py, create it per our previous message.
    _, valloader = load_train_val()

    # --- initialize global model/params ---
    global_model = create_model().to(DEVICE)
    global_params = params_to_numpy(global_model)

    print(f"Starting serverless FL with {num_clients} clients and {len(aggregators)} aggregator-capable node(s)")
    for rnd in range(1, num_rounds + 1):
        print(f"\n=== Round {rnd} ===")

        # 1) broadcast current global params to everyone
        for c in clients:
            c.set_params(global_params)

        # 2) local training and collect updates
        updates: List[Tuple[int, List]] = []
        for c in clients:
            c.train_local()
            updates.append((c.num_examples(), c.get_params()))

        if not updates:
            raise RuntimeError("No client updates collected; ensure num_clients >= 1.")

        # 3) select aggregator(s); enforce non-empty
        chosen = select_aggregators(rnd, aggregators, k=k_aggregators, policy=agg_policy)
        if not chosen:
            # Fail fast with a clear message (prevents UnboundLocalError)
            raise RuntimeError(
                "No aggregators available this round. "
                "Ensure num_aggregators >= 1 and your selection policy returns at least one."
            )

        chosen_ids = [a.cid for a in chosen]
        print(f"Chosen aggregator(s): {chosen_ids}")

        # 4) aggregation (use the first chosen for the actual combine step)
        # If your Aggregator.aggregate only accepts (updates), use:
        # aggregated = chosen[0].aggregate(updates)
        aggregated = chosen[0].aggregate(
            updates,
            valloader=valloader,
            clients=clients,
            tau=tau,
        )

        # 5) update global params and evaluate on each client (shared test)
        global_params = aggregated

        accs = []
        for c in clients:
            c.set_params(global_params)
            accs.append(c.evaluate())
        avg_acc = sum(accs) / len(accs)
        print(f"Average accuracy after round {rnd}: {avg_acc:.4f}")

    # --- save final global model ---
    numpy_to_params(global_model, global_params)
    torch.save(global_model.state_dict(), "global_mnist_cnn.pt")
    print("Saved final global model to global_mnist_cnn.pt")


if __name__ == "__main__":
    # Example quick run; adjust as needed
    run_rounds(
        num_clients=4,
        num_aggregators=2,
        num_rounds=4,
        local_epochs=1,
        lr=0.01,
        agg_policy="rotate",
        k_aggregators=1,
        tau=0.90,
    )
