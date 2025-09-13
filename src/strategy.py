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

from onchain import FLChain
from storage_chain import FLStorageChain, pack_params_float32, unpack_params_float32

def params_hash(params: List[np.ndarray], decimals: int = 6) -> str:
    h = hashlib.sha256()
    for a in params:
        arr = np.asarray(a, dtype=np.float32)
        if decimals is not None:
            arr = np.round(arr, decimals=decimals)
        h.update(arr.tobytes(order="C"))
    return h.hexdigest()

def normalize_hex(hs: str) -> str:
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
    if not aggregators:
        return []
    eligible = [a for a in aggregators if reputation.get(a.cid, 0.0) >= gamma]
    pool = eligible if len(eligible) > 0 else aggregators
    k = max(1, min(k, len(pool)))
    idxs = rng.choice(len(pool), size=k, replace=False)
    return [pool[int(i)] for i in idxs]

def first_free_round(chain: FLChain, start: int = 1, max_scan: int = 100000) -> int:
    r = start
    for _ in range(max_scan):
        finalized, _ = chain.get_round(r)
        if not finalized:
            return r
        r += 1
    raise RuntimeError("No free round found in scan range")

def run_rounds(
    num_clients: int = 6,
    num_aggregators: int = 4,
    max_rounds: int = 100,
    local_epochs: int = 1,
    lr: float = 0.01,
    tau: float = 1.0,          # 1.0: strict non-decrease; 0.90: allow 10% relative drop
    gamma: float = 0.20,
    k_aggregators: int = 2,
    non_iid: bool = False,
    labels_per_client: int = 2,
    proportions: Optional[Sequence[float]] = None,
    dirichlet_alpha: Optional[float] = None,
    reset_reputation: bool = False,
    alpha: float = 0.6,
    beta: float = 0.4,
    epsilon: float = 1e-3,
    random_seed: int = 42,
    gamma_growth: float = 1.05,
    gamma_cap: float = 0.90,
    client_chunk_size: int = 4*1024,
    global_chunk_size: int = 4*1024,

    total_token_pool: Optional[float] = 10_000.0,
    tokens_per_round: float = 100.0,
    rewards_path: str = "artifacts/rewards.json",
):
    if num_clients < 1 or num_aggregators < 1:
        raise ValueError("num_clients and num_aggregators must be >= 1")

    rng = np.random.default_rng(random_seed)

    s = alpha + beta
    if s <= 0:
        raise ValueError("alpha + beta must be > 0")
    if abs(s - 1.0) > 1e-8:
        alpha, beta = alpha / s, beta / s

    # Build population
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

    valloader = load_validation_loader()

    global_model = create_model().to(DEVICE)
    global_params = params_to_numpy(global_model)

    reputation_path = "artifacts/reputation.json"
    ckpt_dir = "artifacts/checkpoints"
    csv_path = "artifacts/metrics.csv"
    csv_fields = [
        "round", "avg_acc", "chosen_aggregators", "proposal_hashes", "consensus_hash",
        "consensus_votes", "consensus_ok", "fallback_used", "passed", "failed",
        "reputations", "client_sizes", "alpha", "beta", "tau", "gamma_used",
        "val_acc_prev", "val_acc_new", "delta", "converged", "hash_votes",
        "client_chunks", "winner_global_chunks",
        "tokens_per_round_used", "rewards_this_round", "balances_after", "remaining_pool",
    ]
    ensure_csv(csv_path, csv_fields)

    sizes = [c.num_examples() for c in clients]
    total_k = float(sum(sizes)) if sum(sizes) > 0 else 1.0
    reputation: Dict[int, float] = {c.cid: (sizes[i] / total_k) for i, c in enumerate(clients)}

    print("Client sizes:", {c.cid: sizes[i] for i, c in enumerate(clients)})
    print("Init reputations:", {cid: round(reputation[cid], 4) for cid in sorted(reputation)})

    if not reset_reputation:
        loaded_rep = load_json(reputation_path)
        if loaded_rep:
            reputation.update({int(k): float(v) for k, v in loaded_rep.items()})
            print("Loaded existing reputation from disk.")

    # Rewards state
    rewards_state = load_json(rewards_path) or {}
    balances: Dict[int, float] = {int(k): float(v) for k, v in rewards_state.get("balances", {}).items()}
    remaining_pool: Optional[float] = float(rewards_state.get("remaining_pool", total_token_pool)) if total_token_pool is not None else None
    for c in clients:
        if int(c.cid) not in balances:
            balances[int(c.cid)] = 0.0

    load_dotenv()
    rpc_url = os.getenv("RPC_URL", "http://127.0.0.1:8545")
    coord_addr = os.getenv("CONTRACT_ADDRESS")
    privkey = os.getenv("PRIVKEY")
    storage_addr = os.getenv("FLSTORAGE_ADDRESS")
    if not coord_addr or not privkey:
        raise RuntimeError("Set CONTRACT_ADDRESS and PRIVKEY in .env before running.")
    if not storage_addr:
        raise RuntimeError("Set FLSTORAGE_ADDRESS in .env (deploy FLStorage first).")

    chain = FLChain(rpc_url=rpc_url, contract_address=coord_addr, privkey=privkey)
    store = FLStorageChain(rpc_url=rpc_url, contract_address=storage_addr, privkey=privkey)

    round_base = first_free_round(chain, start=1) - 1
    print(f"On-chain first free round: {round_base + 1} (using base={round_base})")

    current_gamma = float(gamma)

    numpy_to_params(global_model, global_params)
    prev_val_acc = accuracy(global_model, valloader)

    converged = False
    rnd = 0
    while rnd < max_rounds and not converged:
        rnd += 1
        round_id = round_base + rnd

        print(f"\n=== Round {rnd} (on-chain id {round_id}) ===")
        print(f"Reputations (start): { {cid: round(reputation[cid], 4) for cid in reputation} }")
        print(f"Prev global V-accuracy: {prev_val_acc:.4f}")
        print(f"Gamma used this round: {current_gamma:.4f}")

        # Broadcast global to all
        for c in clients:
            c.set_params(global_params)

        # Local training (off-chain)
        for c in clients:
            c.train_local()

        # Upload each client's local update to FLStorage (chunked)
        client_chunks_info: Dict[int, int] = {}
        for c in clients:
            blob = pack_params_float32(c.get_params())
            n_chunks, _ = store.upload_blob(round_id, c.cid, blob, chunk_size=client_chunk_size)
            client_chunks_info[int(c.cid)] = int(n_chunks)
        print(f"Uploaded {len(client_chunks_info)} client updates to FLStorage.")

        # --- NEW: One-time deterministic V-filter (strategy decides valid_ids) ---
        # Evaluate each client's model vs V using the SAME code path once.
        base_acc = accuracy(global_model, valloader)
        valid_ids: List[int] = []
        failed_ids: List[int] = []

        for c in clients:
            blob = store.download_blob(round_id, int(c.cid))
            params_i = unpack_params_float32(blob, global_params)
            tmp = create_model().to(DEVICE)
            numpy_to_params(tmp, params_i)
            acc_i = accuracy(tmp, valloader)
            if acc_i + 1e-12 >= base_acc * tau:
                valid_ids.append(int(c.cid))
            else:
                failed_ids.append(int(c.cid))

        if not valid_ids:
            # fallback: include best single client
            best_acc = -1.0
            best_id = int(clients[0].cid)
            for c in clients:
                blob = store.download_blob(round_id, int(c.cid))
                params_i = unpack_params_float32(blob, global_params)
                tmp = create_model().to(DEVICE)
                numpy_to_params(tmp, params_i)
                acc_i = accuracy(tmp, valloader)
                if acc_i > best_acc:
                    best_acc = acc_i
                    best_id = int(c.cid)
            valid_ids = [best_id]
            failed_ids = [cid for cid in failed_ids if cid != best_id]

        print(f"Deterministic V-filter → valid_ids={sorted(valid_ids)}, failed_ids={sorted(failed_ids)}")

        # Select aggregators
        chosen = select_aggregators_uniform(reputation, current_gamma, aggregs, k_aggregators, rng)
        if not chosen:
            raise RuntimeError("No aggregators available; check num_aggregators/k_aggregators.")
        chosen_ids = [a.cid for a in chosen]
        print(f"Chosen aggregator(s): {chosen_ids}")

        # Aggregator proposals using the SAME valid_ids list
        client_sizes = {int(c.cid): int(c.num_examples()) for c in clients}

        proposals = []
        for agg in chosen:
            agg_params, report = agg.aggregate_from_chain_deterministic(
                store=store,
                round_id=round_id,
                template_params=global_params,
                valid_ids=valid_ids,
                client_sizes=client_sizes,
            )
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

        # On-chain voting (hashes should now match)
        for p in proposals:
            chain.submit_proposal(round_id=round_id, agg_id=int(p["agg_id"]), hash_hex="0x" + p["hash"])
        chain.finalize(round_id=round_id, total_selected=len(proposals))
        finalized, consensus_hex = chain.get_round(round_id=round_id)
        consensus_hex_norm = normalize_hex(consensus_hex)

        hash_votes: Dict[str, int] = {}
        for p in proposals:
            vcount = chain.get_votes(round_id, "0x" + p["hash"])
            hash_votes[p["hash"]] = int(vcount)
            print(f"Votes for {p['hash'][:12]}.. : {vcount}")

        consensus_ok = finalized and (consensus_hex_norm != "0" * 64)
        fallback_used = False
        consensus_votes = -1

        if consensus_ok:
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
                consensus_votes = hash_votes.get(winner["hash"], -1)
                print(f"Consensus SUCCESS on-chain: hash={consensus_hex[:12]}.. votes={consensus_votes}")
            aggregated, report = winner["params"], winner["report"]
            winner_agg_id = int(winner["agg_id"])
        else:
            winner = max(proposals, key=lambda p: p["val_acc"])
            aggregated, report = winner["params"], winner["report"]
            winner_agg_id = int(winner["agg_id"])
            fallback_used = True
            print("Consensus FAILED on-chain. Fallback to best V-acc proposal.")

        # Store ONLY the winner's global model to FLStorage (chunked)
        GLOBAL_NS_OFFSET = 1_000_000
        global_ns_id = GLOBAL_NS_OFFSET + winner_agg_id
        winner_blob = pack_params_float32(aggregated)
        winner_chunks, _ = store.upload_blob(round_id, global_ns_id, winner_blob, chunk_size=global_chunk_size)
        print(f"Stored winner global model on-chain: agg={winner_agg_id}, chunks={winner_chunks}")

        # Reputation update (still uses the V-filter result)
        acc_map: Dict[int, float] = {}
        s_part_map: Dict[int, int] = {}
        for cid in valid_ids:
            acc_map[int(cid)] = 0.0  # not recording per-client V-acc here
            s_part_map[int(cid)] = +1
        for cid in failed_ids:
            acc_map[int(cid)] = 0.0
            s_part_map[int(cid)] = -1

        scale = alpha * beta
        # If you want to use actual accuracies in the product rule, compute and store them above
        for cid in set(list(acc_map.keys()) + list(s_part_map.keys())):
            s_part = s_part_map.get(cid, 0)
            # Here we use base_acc as proxy; you can compute per-client acc_i earlier and cache.
            # Keep it simple: reward/penalize by s_part only
            reputation[cid] = reputation.get(cid, 0.0) + (scale * (1.0) * float(s_part))

        print(f"Reputations (updated): { {cid: round(reputation[cid], 4) for cid in reputation} }")

        # Rewards distribution (unchanged)
        k_this_round = float(tokens_per_round)
        if remaining_pool is not None:
            if remaining_pool <= 0.0:
                k_this_round = 0.0
            else:
                k_this_round = float(min(remaining_pool, tokens_per_round))

        rep_sum = float(sum(reputation.values()))
        rewards_this_round: Dict[int, float] = {}
        if k_this_round > 0 and rep_sum > 0:
            for cid in reputation:
                p_i = k_this_round * (reputation[cid] / rep_sum)
                rewards_this_round[int(cid)] = float(p_i)
                balances[int(cid)] = float(balances.get(int(cid), 0.0) + p_i)
            if remaining_pool is not None:
                remaining_pool = float(max(0.0, remaining_pool - k_this_round))

        # New global
        global_params = aggregated
        numpy_to_params(global_model, global_params)
        new_val_acc = accuracy(global_model, valloader)
        delta = abs(new_val_acc - prev_val_acc)
        print(f"New global V-accuracy: {new_val_acc:.4f} | Δ = {delta:.6f}")

        # Client eval
        accs = []
        for c in clients:
            c.set_params(global_params)
            accs.append(c.evaluate())
        avg_acc = sum(accs) / len(accs)
        print(f"Average client accuracy after round {rnd}: {avg_acc:.4f}")

        # Convergence
        converged = (delta < epsilon)
        if converged:
            print(f"Converged (|A_t+1 - A_t| < {epsilon}) at round {rnd}.")

        # CSV log
        row = {
            "round": rnd,
            "avg_acc": avg_acc,
            "chosen_aggregators": json.dumps(chosen_ids),
            "proposal_hashes": json.dumps([p["hash"] for p in proposals]),
            "consensus_hash": consensus_hex if consensus_ok else "",
            "consensus_votes": consensus_votes,
            "consensus_ok": int(bool(consensus_ok)),
            "fallback_used": int(bool(fallback_used)),
            "passed": json.dumps(sorted(valid_ids)),
            "failed": json.dumps(sorted(failed_ids)),
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
            "hash_votes": json.dumps(hash_votes),
            "client_chunks": json.dumps(client_chunks_info),
            "winner_global_chunks": int(winner_chunks),
            "tokens_per_round_used": k_this_round,
            "rewards_this_round": json.dumps({int(k): float(v) for k, v in rewards_this_round.items()}),
            "balances_after": json.dumps({int(k): float(v) for k, v in balances.items()}),
            "remaining_pool": (None if remaining_pool is None else float(remaining_pool)),
        }
        append_csv_row(csv_path, row, csv_fields)

        # Persist
        save_json(reputation_path, {str(cid): float(r) for cid, r in reputation.items()})
        save_model_state_dict(f"{ckpt_dir}/round_{rnd:03d}.pt", global_model.state_dict())
        rewards_out = {
            "balances": {str(k): float(v) for k, v in balances.items()},
            **({ "remaining_pool": float(remaining_pool) } if remaining_pool is not None else {}),
        }
        save_json(rewards_path, rewards_out)

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
    run_rounds(
        num_clients=6,
        num_aggregators=4,
        k_aggregators=2,
        max_rounds=50,
        local_epochs=1,
        lr=0.01,
        tau=1.0,
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
        client_chunk_size=4*1024,
        global_chunk_size=4*1024,
        total_token_pool=10_000.0,
        tokens_per_round=100.0,
    )
