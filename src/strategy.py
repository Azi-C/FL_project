# strategy.py
from typing import List, Tuple, Dict, Sequence, Optional
import json
import hashlib
import numpy as np
import torch

from model import create_model
from client import Client, params_to_numpy, numpy_to_params
from aggregator import Aggregator
from utils import DEVICE, load_validation_loader, accuracy
from persistence import (
    save_json,
    load_json,
    save_model_state_dict,
    ensure_csv,
    append_csv_row,
)

# -------------------- Hashing / consensus helpers --------------------
def params_hash(params: List[np.ndarray], decimals: int = 6) -> str:
    """
    Robust hash of parameter list:
      - cast to float32
      - round to 'decimals' to avoid tiny numeric diffs
      - SHA-256 over concatenated bytes
    """
    h = hashlib.sha256()
    for a in params:
        arr = np.asarray(a, dtype=np.float32)
        if decimals is not None:
            arr = np.round(arr, decimals=decimals)
        h.update(arr.tobytes(order="C"))
    return h.hexdigest()

def majority_by_hash(hashes: List[str]) -> Tuple[Optional[str], int]:
    """Return (winning_hash, count). Winning hash must be > 50% of proposals."""
    if not hashes:
        return None, 0
    counts: Dict[str, int] = {}
    for h in hashes:
        counts[h] = counts.get(h, 0) + 1
    winner, cnt = max(counts.items(), key=lambda kv: kv[1])
    if cnt > len(hashes) // 2:
        return winner, cnt
    return None, cnt

# -------------------- Uniform random selection among eligible --------------------
def select_aggregators_uniform(
    reputation: Dict[int, float],
    gamma: float,
    aggregators: List[Aggregator],
    k: int,
    rng: np.random.Generator,
) -> List[Aggregator]:
    """
    1) Build eligible set: r_i >= gamma. If empty, fallback to all aggregators.
    2) Uniformly sample k aggregators without replacement from that set.
    """
    if not aggregators:
        return []
    eligible = [a for a in aggregators if reputation.get(a.cid, 0.0) >= gamma]
    pool = eligible if len(eligible) > 0 else aggregators

    k = max(1, min(k, len(pool)))
    idxs = rng.choice(len(pool), size=k, replace=False)
    chosen = [pool[int(i)] for i in idxs]
    return chosen

# -------------------- Main training loop --------------------
def run_rounds(
    num_clients: int = 6,
    num_aggregators: int = 4,            # total nodes capable of aggregation
    max_rounds: int = 100,               # safety cap (we stop earlier on convergence)
    local_epochs: int = 1,
    lr: float = 0.01,
    tau: float = 0.90,                   # validation threshold on V
    gamma: float = 0.20,                 # START value for gamma (will grow multiplicatively)
    k_aggregators: int = 2,              # number of aggregators to SELECT each round
    non_iid: bool = False,
    labels_per_client: int = 2,
    proportions: Optional[Sequence[float]] = None,
    dirichlet_alpha: Optional[float] = None,
    reset_reputation: bool = False,
    alpha: float = 0.6,                  # product rule weights
    beta: float = 0.4,
    epsilon: float = 1e-3,               # convergence: |A_t+1 - A_t| < epsilon
    random_seed: int = 42,               # for reproducibility of random selection

    # ---- NEW: multiplicative gamma schedule ----
    gamma_growth: float = 1.05,          # multiply gamma each round (e.g., 1.05 -> +5%)
    gamma_cap: float = 0.90,             # cap so it never exceeds this value
):
    """
    Reputation update (PRODUCT form):
        r_i^{t+1} = r_i^{t} + (alpha * beta) * a_i * s_part_i

    Multi-aggregator + hash consensus:
        - Select k_aggregators **uniformly at random** among eligible (r_i >= gamma).
        - Each runs FedAvg -> proposal (params, report).
        - Hash proposals; if strict majority (>50%) on a hash, adopt it.
        - Else fallback to proposal with best V-accuracy.

    Early stopping:
        Stop when |A(w_{t+1}, V) - A(w_t, V)| < epsilon.

    Gamma schedule:
        Start at `gamma` and after each round do:
            gamma = min(gamma_cap, gamma * gamma_growth)
    """
    if num_clients < 1:
        raise ValueError("num_clients must be >= 1")
    if num_aggregators < 1:
        raise ValueError("num_aggregators must be >= 1")

    rng = np.random.default_rng(random_seed)

    # Normalize alpha/beta to sum to 1 (interpretability)
    s = alpha + beta
    if s <= 0:
        raise ValueError("alpha + beta must be > 0")
    if abs(s - 1.0) > 1e-8:
        alpha, beta = alpha / s, beta / s

    # ---- build population ----
    clients: List[Client] = []
    aggregs: List[Aggregator] = []
    for cid in range(num_clients):
        if cid < num_aggregators:
            agg = Aggregator(
                cid, num_clients, lr=lr, local_epochs=local_epochs,
                non_iid=non_iid, labels_per_client=labels_per_client,
                proportions=proportions, dirichlet_alpha=dirichlet_alpha
            )
            aggregs.append(agg)
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
    csv_path = "artifacts/metrics.csv"
    csv_fields = [
        "round", "avg_acc", "chosen_aggregators", "proposal_hashes", "consensus_hash",
        "consensus_votes", "consensus_ok", "fallback_used", "passed", "failed",
        "reputations", "client_sizes", "alpha", "beta", "tau", "gamma_used",
        "val_acc_prev", "val_acc_new", "delta", "converged"
    ]
    ensure_csv(csv_path, csv_fields)

    # ---- initial reputation r_i^(0) = |D_i| / Σ|D_j| ----
    sizes = [c.num_examples() for c in clients]
    total_k = float(sum(sizes)) if sum(sizes) > 0 else 1.0
    reputation: Dict[int, float] = {c.cid: (sizes[i] / total_k) for i, c in enumerate(clients)}

    print("Client sizes:", {c.cid: sizes[i] for i, c in enumerate(clients)})
    print("Init reputations:", {c.cid: round(reputation[c.cid], 4) for c in clients})

    # Try to resume from disk unless resetting
    if not reset_reputation:
        loaded_rep = load_json(reputation_path)
        if loaded_rep:
            reputation.update({int(k): float(v) for k, v in loaded_rep.items()})
            print("Loaded existing reputation from disk.")

    # ---- gamma schedule state ----
    current_gamma = float(gamma)

    print(f"Starting serverless FL with {num_clients} clients, {len(aggregs)} aggregator-capable node(s)")
    print(f"Initial gamma={current_gamma}, tau={tau}, alpha={alpha}, beta={beta}, "
          f"epsilon={epsilon}, k_aggregators={k_aggregators}, "
          f"gamma_growth={gamma_growth}, gamma_cap={gamma_cap}")

    # Evaluate initial global model on V
    numpy_to_params(global_model, global_params)
    prev_val_acc = accuracy(global_model, valloader)

    converged = False
    rnd = 0
    while rnd < max_rounds and not converged:
        rnd += 1
        print(f"\n=== Round {rnd} ===")
        print(f"Reputations (start): { {cid: round(reputation[cid], 4) for cid in reputation} }")
        print(f"Prev global V-accuracy: {prev_val_acc:.4f}")
        print(f"Gamma used this round: {current_gamma:.4f}")

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

        # 3) UNIFORM random selection among eligible aggregators with current_gamma
        chosen = select_aggregators_uniform(reputation, current_gamma, aggregs, k_aggregators, rng)
        if not chosen:
            raise RuntimeError("No aggregators available; check num_aggregators/k_aggregators.")
        chosen_ids = [a.cid for a in chosen]
        print(f"Chosen aggregator(s): {chosen_ids}")

        # 4) each aggregator produces a proposal (params, report) and hash; also evaluate proposal on V
        proposals = []
        for agg in chosen:
            agg_params, report = agg.aggregate(updates, valloader=valloader, clients=clients, tau=tau)
            h = params_hash(agg_params, decimals=6)

            tmp_model = create_model().to(DEVICE)
            numpy_to_params(tmp_model, agg_params)
            prop_val_acc = accuracy(tmp_model, valloader)

            proposals.append({
                "agg_id": agg.cid,
                "params": agg_params,
                "report": report,
                "hash": h,
                "val_acc": prop_val_acc,
            })
            print(f"Aggregator {agg.cid} -> hash={h[:12]}..  V-acc={prop_val_acc:.4f}")

        # 5) consensus by hash (50% + 1)
        hashes = [p["hash"] for p in proposals]
        win_hash, votes = majority_by_hash(hashes)
        consensus_ok = win_hash is not None

        fallback_used = False
        if consensus_ok:
            winning = next(p for p in proposals if p["hash"] == win_hash)
            aggregated, report = winning["params"], winning["report"]
            print(f"Consensus SUCCESS: hash={win_hash[:12]}.. votes={votes}/{len(hashes)}")
        else:
            winning = max(proposals, key=lambda p: p["val_acc"])
            aggregated, report = winning["params"], winning["report"]
            fallback_used = True
            print(f"Consensus FAILED (max votes={votes}/{len(hashes)}). "
                  f"Fallback to best V-acc from aggregator {winning['agg_id']} (hash={winning['hash'][:12]}..)")

        # 6) update reputations (product rule) using the winning report
        acc_map: Dict[int, float] = {}
        s_part_map: Dict[int, int] = {}
        for cid, a in report.get("passed", []):
            acc_map[int(cid)] = float(a)
            s_part_map[int(cid)] = +1
        for cid, a in report.get("failed", []):
            acc_map[int(cid)] = float(a)
            s_part_map[int(cid)] = -1

        scale = alpha * beta
        for cid, a_i in acc_map.items():
            s_part = s_part_map.get(cid, 0)
            delta_rep = scale * a_i * float(s_part)
            reputation[cid] = reputation.get(cid, 0.0) + delta_rep

        print(f"Passed on V: {sorted([int(cid) for cid, _ in report.get('passed', [])])} | "
              f"Failed on V: {sorted([int(cid) for cid, _ in report.get('failed', [])])}")
        print(f"Reputations (updated): { {cid: round(reputation[cid], 4) for cid in reputation} }")

        # 7) update global params and evaluate
        global_params = aggregated
        numpy_to_params(global_model, global_params)
        new_val_acc = accuracy(global_model, valloader)
        delta = abs(new_val_acc - prev_val_acc)
        print(f"New global V-accuracy: {new_val_acc:.4f} | Δ = {delta:.6f}")

        accs = []
        for c in clients:
            c.set_params(global_params)
            accs.append(c.evaluate())
        avg_acc = sum(accs) / len(accs)
        print(f"Average client accuracy after round {rnd}: {avg_acc:.4f}")

        # 8) convergence check
        converged = (delta < epsilon)
        if converged:
            print(f"Converged (|A_t+1 - A_t| < {epsilon}) at round {rnd}.")

        # 9) log CSV
        row = {
            "round": rnd,
            "avg_acc": avg_acc,
            "chosen_aggregators": json.dumps(chosen_ids),
            "proposal_hashes": json.dumps(hashes),
            "consensus_hash": win_hash if win_hash else "",
            "consensus_votes": votes,
            "consensus_ok": int(consensus_ok),
            "fallback_used": int(fallback_used),
            "passed": json.dumps(sorted([int(cid) for cid, _ in report.get("passed", [])])),
            "failed": json.dumps(sorted([int(cid) for cid, _ in report.get("failed", [])])),
            "reputations": json.dumps({int(cid): float(r) for cid, r in reputation.items()}),
            "client_sizes": json.dumps({int(c.cid): int(sizes[i]) for i, c in enumerate(clients)}),
            "alpha": alpha,
            "beta": beta,
            "tau": tau,
            "gamma_used": current_gamma,
            "val_acc_prev": prev_val_acc,
            "val_acc_new": new_val_acc,
            "delta": delta,
            "converged": int(converged),
        }
        append_csv_row(csv_path, row, csv_fields)

        # 10) persist reputation and checkpoint
        save_json(reputation_path, {str(cid): float(r) for cid, r in reputation.items()})
        save_model_state_dict(f"{ckpt_dir}/round_{rnd:03d}.pt", global_model.state_dict())

        # ---- update for next round ----
        prev_val_acc = new_val_acc
        # multiplicative gamma schedule
        current_gamma = min(gamma_cap, current_gamma * gamma_growth)
        print(f"Updated gamma for next round: {current_gamma:.4f}")

        if converged:
            break

    # ---- save final model ----
    torch.save(global_model.state_dict(), "global_mnist_cnn.pt")
    if converged:
        print(f"Training stopped due to convergence at round {rnd}. Saved final model to global_mnist_cnn.pt")
    else:
        print(f"Reached max_rounds={max_rounds} without convergence. Saved final model to global_mnist_cnn.pt")


if __name__ == "__main__":
    # Example: 4 aggregators exist; pick 2 uniformly at random each round
    run_rounds(
        num_clients=6,
        num_aggregators=4,           # total aggregators in the system
        k_aggregators=2,             # randomly choose 2 of them each round
        max_rounds=100,
        local_epochs=1,
        lr=0.01,
        tau=0.90,
        gamma=0.20,                  # start gamma
        non_iid=False,
        proportions=[0.4, 0.2, 0.2, 0.1, 0.06, 0.04],  # example sizes for 6 clients
        dirichlet_alpha=None,
        reset_reputation=True,
        alpha=0.6,
        beta=0.4,
        epsilon=1e-3,
        random_seed=123,
        gamma_growth=1.10,           # +10% per round
        gamma_cap=0.90,              # cap at 0.90
    )
