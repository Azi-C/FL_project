# strategy.py
from typing import List, Tuple, Dict, Sequence, Optional
import json
import hashlib
import numpy as np
import torch
import os

from dotenv import load_dotenv

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

# On-chain bridge
from onchain import FLChain

# -------------------- Helpers --------------------
def params_hash(params: List[np.ndarray], decimals: int = 6) -> str:
    """Return a hex hash (without 0x) over rounded float32 parameter bytes."""
    h = hashlib.sha256()
    for a in params:
        arr = np.asarray(a, dtype=np.float32)
        if decimals is not None:
            arr = np.round(arr, decimals=decimals)
        h.update(arr.tobytes(order="C"))
    return h.hexdigest()  # e.g., 'a1b2...'

def normalize_hex(hs: str) -> str:
    """Normalize hex string to lowercase without 0x prefix."""
    if hs.startswith(("0x", "0X")):
        hs = hs[2:]
    return hs.lower()

def select_aggregators_uniform(
    reputation: Dict[int, float],
    gamma: float,
    aggregators: List[Aggregator],
    k: int,
    rng: np.random.Generator,
) -> List[Aggregator]:
    """Uniformly sample k aggregators among those with r_i >= gamma (fallback to all if none)."""
    if not aggregators:
        return []
    eligible = [a for a in aggregators if reputation.get(a.cid, 0.0) >= gamma]
    pool = eligible if len(eligible) > 0 else aggregators
    k = max(1, min(k, len(pool)))
    idxs = rng.choice(len(pool), size=k, replace=False)
    return [pool[int(i)] for i in idxs]

def first_free_round(chain: FLChain, start: int = 1, max_scan: int = 100000) -> int:
    """Return the first round id >= start that is not finalized on-chain."""
    r = start
    for _ in range(max_scan):
        finalized, _ = chain.get_round(r)
        if not finalized:
            return r
        r += 1
    raise RuntimeError("No free round found in scan range")

# -------------------- Main training loop --------------------
def run_rounds(
    num_clients: int = 6,
    num_aggregators: int = 4,
    max_rounds: int = 100,
    local_epochs: int = 1,
    lr: float = 0.01,
    tau: float = 0.90,
    gamma: float = 0.20,             # starting gamma (will grow)
    k_aggregators: int = 2,
    non_iid: bool = False,
    labels_per_client: int = 2,
    proportions: Optional[Sequence[float]] = None,
    dirichlet_alpha: Optional[float] = None,
    reset_reputation: bool = False,
    alpha: float = 0.6,              # product rule weights (alpha+beta will be normalized to 1)
    beta: float = 0.4,
    epsilon: float = 1e-3,           # |A_{t+1}-A_t| < eps stops
    random_seed: int = 42,
    gamma_growth: float = 1.05,      # multiplicative schedule
    gamma_cap: float = 0.90,
):
    """
    Serverless FL with on-chain consensus:
      - Selected aggregators compute proposals (FedAvg) and submit their hashes on-chain.
      - Contract finalizes if a strict majority (>50%) agrees; otherwise we fallback to best V-acc.
      - Reputation updates by product rule: r_i <- r_i + (alpha*beta)*a_i*s_i, s_i in {+1,-1}.
      - Gamma (eligibility threshold) grows multiplicatively each round.
      - Early stop when |A_{t+1}-A_t| < epsilon.
    """
    if num_clients < 1 or num_aggregators < 1:
        raise ValueError("num_clients and num_aggregators must be >= 1")

    rng = np.random.default_rng(random_seed)

    # Normalize alpha/beta to sum to 1
    s = alpha + beta
    if s <= 0:
        raise ValueError("alpha + beta must be > 0")
    if abs(s - 1.0) > 1e-8:
        alpha, beta = alpha / s, beta / s

    # -------------------- Build population --------------------
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

    # shared validation loader (10% of train)
    valloader = load_validation_loader()

    # global model/params
    global_model = create_model().to(DEVICE)
    global_params = params_to_numpy(global_model)

    # persistence
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

    # initial reputation r_i^(0) = |D_i| / Σ|D_j|
    sizes = [c.num_examples() for c in clients]
    total_k = float(sum(sizes)) if sum(sizes) > 0 else 1.0
    reputation: Dict[int, float] = {c.cid: (sizes[i] / total_k) for i, c in enumerate(clients)}

    print("Client sizes:", {c.cid: sizes[i] for i, c in enumerate(clients)})
    print("Init reputations:", {c.cid: round(reputation[c.cid], 4) for c in clients})

    if not reset_reputation:
        loaded_rep = load_json(reputation_path)
        if loaded_rep:
            reputation.update({int(k): float(v) for k, v in loaded_rep.items()})
            print("Loaded existing reputation from disk.")

    # -------------------- On-chain connector --------------------
    load_dotenv()
    rpc_url = os.getenv("RPC_URL", "http://127.0.0.1:8545")
    contract_address = os.getenv("CONTRACT_ADDRESS")
    privkey = os.getenv("PRIVKEY")
    if not contract_address or not privkey:
        raise RuntimeError("Set CONTRACT_ADDRESS and PRIVKEY in .env before running.")

    chain = FLChain(rpc_url=rpc_url, contract_address=contract_address, privkey=privkey)

    # Pick a fresh base round to avoid 'Round finalized' reverts
    round_base = first_free_round(chain, start=1) - 1
    print(f"On-chain first free round: {round_base + 1} (using base={round_base})")

    # gamma schedule state
    current_gamma = float(gamma)

    print(f"Starting serverless FL with {num_clients} clients, {len(aggregs)} aggregator-capable node(s)")
    print(f"Initial gamma={current_gamma}, tau={tau}, alpha={alpha}, beta={beta}, "
          f"epsilon={epsilon}, k_aggregators={k_aggregators}, "
          f"gamma_growth={gamma_growth}, gamma_cap={gamma_cap}")

    # evaluate initial global V-acc
    numpy_to_params(global_model, global_params)
    prev_val_acc = accuracy(global_model, valloader)

    converged = False
    rnd = 0
    while rnd < max_rounds and not converged:
        rnd += 1
        round_id = round_base + rnd  # <-- use a fresh on-chain round id

        print(f"\n=== Round {rnd} (on-chain id {round_id}) ===")
        print(f"Reputations (start): { {cid: round(reputation[cid], 4) for cid in reputation} }")
        print(f"Prev global V-accuracy: {prev_val_acc:.4f}")
        print(f"Gamma used this round: {current_gamma:.4f}")

        # 1) broadcast
        for c in clients:
            c.set_params(global_params)

        # 2) local training
        updates: List[Tuple[int, List]] = []
        for c in clients:
            c.train_local()
            updates.append((c.num_examples(), c.get_params()))
        if not updates:
            raise RuntimeError("No client updates collected.")

        # 3) choose aggregators (uniform among eligible)
        chosen = select_aggregators_uniform(reputation, current_gamma, aggregs, k_aggregators, rng)
        if not chosen:
            raise RuntimeError("No aggregators available; check num_aggregators/k_aggregators.")
        chosen_ids = [a.cid for a in chosen]
        print(f"Chosen aggregator(s): {chosen_ids}")

        # 4) each aggregator aggregates -> proposal params + report; compute hash and V-acc
        proposals = []
        for agg in chosen:
            agg_params, report = agg.aggregate(updates, valloader=valloader, clients=clients, tau=tau)
            h = params_hash(agg_params, decimals=6)  # hex without 0x
            tmp_model = create_model().to(DEVICE)
            numpy_to_params(tmp_model, agg_params)
            prop_val_acc = accuracy(tmp_model, valloader)
            proposals.append({
                "agg_id": agg.cid,
                "params": agg_params,
                "report": report,
                "hash": h,              # store without 0x
                "val_acc": prop_val_acc,
            })
            print(f"Aggregator {agg.cid} -> hash={h[:12]}..  V-acc={prop_val_acc:.4f}")

        # 5) ON-CHAIN: submit proposals + finalize
        for p in proposals:
            chain.submit_proposal(round_id=round_id, agg_id=int(p["agg_id"]), hash_hex="0x" + p["hash"])

        chain.finalize(round_id=round_id, total_selected=len(proposals))
        finalized, consensus_hex = chain.get_round(round_id=round_id)
        consensus_hex_norm = normalize_hex(consensus_hex)

        consensus_ok = finalized and (consensus_hex_norm != "0" * 64)
        fallback_used = False

        if consensus_ok:
            # find winning proposal by hash
            winner = None
            for p in proposals:
                if normalize_hex(p["hash"]) == consensus_hex_norm:
                    winner = p
                    break
            if winner is None:
                winner = max(proposals, key=lambda q: q["val_acc"])
                fallback_used = True
                print("Consensus hash not found among proposals, falling back to best V-acc.")
            else:
                print(f"Consensus SUCCESS on-chain: hash={consensus_hex[:12]}..")
            aggregated, report = winner["params"], winner["report"]
            votes = -1  # (optional) query per-hash votes via contract if you need them
        else:
            winner = max(proposals, key=lambda p: p["val_acc"])
            aggregated, report = winner["params"], winner["report"]
            fallback_used = True
            votes = -1
            print("Consensus FAILED on-chain. Fallback to best V-acc proposal.")

        # 6) update reputations (product rule)
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
            reputation[cid] = reputation.get(cid, 0.0) + (scale * a_i * float(s_part))

        print(f"Passed on V: {sorted([int(cid) for cid, _ in report.get('passed', [])])} | "
              f"Failed on V: {sorted([int(cid) for cid, _ in report.get('failed', [])])}")
        print(f"Reputations (updated): { {cid: round(reputation[cid], 4) for cid in reputation} }")

        # 7) update global params and evaluate
        global_params = aggregated
        numpy_to_params(global_model, global_params)
        new_val_acc = accuracy(global_model, valloader)
        delta = abs(new_val_acc - prev_val_acc)
        print(f"New global V-accuracy: {new_val_acc:.4f} | Δ = {delta:.6f}")

        # client-side eval (optional)
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
            "proposal_hashes": json.dumps([p["hash"] for p in proposals]),
            "consensus_hash": consensus_hex if consensus_ok else "",
            "consensus_votes": votes,
            "consensus_ok": int(bool(consensus_ok)),
            "fallback_used": int(bool(fallback_used)),
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

        # update for next round
        prev_val_acc = new_val_acc
        current_gamma = min(gamma_cap, current_gamma * gamma_growth)
        print(f"Updated gamma for next round: {current_gamma:.4f}")

        if converged:
            break

    torch.save(global_model.state_dict(), "global_mnist_cnn.pt")
    if converged:
        print(f"Training stopped due to convergence at round {rnd}. Saved final model to global_mnist_cnn.pt")
    else:
        print(f"Reached max_rounds={max_rounds} without convergence. Saved final model to global_mnist_cnn.pt")


if __name__ == "__main__":
    # Example run
    run_rounds(
        num_clients=6,
        num_aggregators=4,
        k_aggregators=2,
        max_rounds=50,
        local_epochs=1,
        lr=0.01,
        tau=0.90,
        gamma=0.20,
        non_iid=False,
        proportions=[0.4, 0.2, 0.2, 0.1, 0.06, 0.04],
        dirichlet_alpha=None,
        reset_reputation=True,
        alpha=0.6,
        beta=0.4,
        epsilon=1e-3,
        random_seed=123,
        gamma_growth=1.05,
        gamma_cap=0.90,
    )
