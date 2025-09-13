from typing import List, Dict, Sequence, Optional
import json
import hashlib
import numpy as np
import torch
import os
from dotenv import load_dotenv

from model import create_model
from client import Client
from aggregator import Aggregator
from utils import DEVICE, load_validation_loader, accuracy, params_to_numpy, numpy_to_params
from persistence import save_json, load_json, save_model_state_dict, ensure_csv, append_csv_row
from onchain import FLChain
from storage_chain import FLStorageChain, pack_params_float32

def params_hash(params: List[np.ndarray], decimals: int = 6) -> str:
    h = hashlib.sha256()
    for a in params:
        arr = np.asarray(a, dtype=np.float32)
        if decimals is not None:
            arr = np.round(arr, decimals=decimals)
        h.update(arr.tobytes(order="C"))
    return h.hexdigest()

def normalize_hex(hs) -> str:
    if isinstance(hs, (bytes, bytearray)):
        hs = hs.hex()
    else:
        hs = str(hs)
        if hs.startswith(("0x", "0X")):
            hs = hs[2:]
    return hs.lower()

def select_aggregators_uniform(reputation: Dict[int, float], gamma: float,
                               aggregators: List[Aggregator], k: int,
                               rng: np.random.Generator) -> List[Aggregator]:
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
    raise RuntimeError("No free round found")

def fedavg_weighted(client_params: Dict[int, List[np.ndarray]],
                    client_sizes: Dict[int, int],
                    valid_ids: List[int],
                    template: List[np.ndarray]) -> List[np.ndarray]:
    if not valid_ids:
        raise ValueError("fedavg_weighted: valid_ids is empty")
    out = [np.zeros_like(p, dtype=np.float32) for p in template]
    total = float(sum(client_sizes[i] for i in valid_ids)) or 1.0
    for cid in valid_ids:
        w = client_sizes[cid] / total
        params = client_params[cid]
        for j in range(len(out)):
            out[j] += (params[j].astype(np.float32) * w)
    return out

def run_rounds(
    num_clients=6, num_aggregators=4, max_rounds=50, local_epochs=1, lr=0.01,
    tau=1.0, gamma=0.20, k_aggregators=2, non_iid=False, labels_per_client=2,
    proportions: Optional[Sequence[float]] = None, dirichlet_alpha: Optional[float] = None,
    reset_reputation=False, alpha=0.6, beta=0.4, epsilon=1e-3, random_seed=42,
    gamma_growth=1.05, gamma_cap=0.90, client_chunk_size=4*1024, global_chunk_size=4*1024,
    total_token_pool=10_000.0, tokens_per_round=100.0,
):
    rng = np.random.default_rng(random_seed)
    # Build population
    clients: List[Client] = []
    aggregs: List[Aggregator] = []
    for cid in range(num_clients):
        if cid < num_aggregators:
            agg = Aggregator(cid, num_clients, lr=lr, local_epochs=local_epochs,
                             non_iid=non_iid, labels_per_client=labels_per_client,
                             proportions=proportions, dirichlet_alpha=dirichlet_alpha)
            aggregs.append(agg); clients.append(agg)
        else:
            clients.append(Client(cid, num_clients, lr=lr, local_epochs=local_epochs,
                                  non_iid=non_iid, labels_per_client=labels_per_client,
                                  proportions=proportions, dirichlet_alpha=dirichlet_alpha))

    valloader = load_validation_loader()
    global_model = create_model().to(DEVICE)
    global_params = params_to_numpy(global_model)

    # Logs
    os.makedirs("artifacts/checkpoints", exist_ok=True)
    ensure_csv("artifacts/metrics.csv", [
        "round","avg_acc","chosen_aggregators","proposal_hashes","consensus_hash",
        "consensus_votes","consensus_ok","fallback_used","passed","failed",
        "reputations","client_sizes","alpha","beta","tau","gamma_used",
        "val_acc_prev","val_acc_new","delta","converged","hash_votes",
        "client_chunks","winner_global_chunks","tokens_per_round_used",
        "rewards_this_round","balances_after","remaining_pool",
    ])

    # Initial reputation ∝ data size
    sizes = [c.num_examples() for c in clients]
    total_k = float(sum(sizes)) or 1.0
    reputation: Dict[int, float] = {c.cid: (sizes[i] / total_k) for i, c in enumerate(clients)}
    print("Client sizes:", {c.cid: sizes[i] for i, c in enumerate(clients)})
    print("Init reputations:", {cid: round(reputation[cid], 4) for cid in sorted(reputation)})

    # On-chain
    load_dotenv()
    rpc_url = os.getenv("RPC_URL", "http://127.0.0.1:8545")
    coord_addr = os.getenv("CONTRACT_ADDRESS")
    storage_addr = os.getenv("FLSTORAGE_ADDRESS")
    privkey = os.getenv("PRIVKEY")
    if not (coord_addr and storage_addr and privkey):
        raise RuntimeError("Set CONTRACT_ADDRESS, FLSTORAGE_ADDRESS, PRIVKEY in .env")

    chain = FLChain(rpc_url=rpc_url, contract_address=coord_addr, privkey=privkey)
    store = FLStorageChain(rpc_url=rpc_url, contract_address=storage_addr, privkey=privkey)
    round_base = first_free_round(chain, start=1) - 1
    prev_val_acc = accuracy(global_model, valloader)

    converged, rnd, current_gamma = False, 0, float(gamma)
    while rnd < max_rounds and not converged:
        rnd += 1
        round_id = round_base + rnd
        print(f"\n=== Round {rnd} (on-chain id {round_id}) ===")
        print(f"Prev global V-accuracy: {prev_val_acc:.4f}")

        for c in clients: c.set_params(global_params)
        for c in clients: c.train_local()

        client_params = {int(c.cid): c.get_params() for c in clients}
        client_sizes  = {int(c.cid): int(c.num_examples()) for c in clients}

        base_acc = accuracy(global_model, valloader)
        valid_ids, failed_ids = [], []
        for c in clients:
            tmp = create_model().to(DEVICE)
            numpy_to_params(tmp, client_params[int(c.cid)])
            acc_i = accuracy(tmp, valloader)
            (valid_ids if acc_i + 1e-12 >= base_acc * tau else failed_ids).append(int(c.cid))
        if not valid_ids:
            best_id, best_acc = int(clients[0].cid), -1.0
            for c in clients:
                tmp = create_model().to(DEVICE)
                numpy_to_params(tmp, client_params[int(c.cid)])
                acc_i = accuracy(tmp, valloader)
                if acc_i > best_acc: best_acc, best_id = acc_i, int(c.cid)
            valid_ids = [best_id]; failed_ids = [cid for cid in failed_ids if cid != best_id]

        chosen = select_aggregators_uniform(reputation, current_gamma, aggregs, k_aggregators, rng)
        chosen_ids = [a.cid for a in chosen]

        proposals = []
        for agg in chosen:
            agg_params = fedavg_weighted(client_params, client_sizes, valid_ids, global_params)
            h = params_hash(agg_params, decimals=6)
            tmp = create_model().to(DEVICE); numpy_to_params(tmp, agg_params)
            prop_val_acc = accuracy(tmp, valloader)
            proposals.append({"agg_id": agg.cid, "params": agg_params, "hash": h, "val_acc": prop_val_acc})

        for p in proposals:
            chain.submit_proposal(round_id=round_id, agg_id=int(p["agg_id"]), hash_hex="0x" + p["hash"])
        chain.finalize(round_id=round_id, total_selected=len(proposals))

        finalized, consensus_hex = chain.get_round(round_id)
        consensus_hex_norm = normalize_hex(consensus_hex)

        hash_votes: Dict[str, int] = {}
        for p in proposals:
            hash_votes[p["hash"]] = int(chain.get_votes(round_id, "0x" + p["hash"]))

        consensus_ok = finalized and (consensus_hex_norm != "0"*64)
        fallback_used, consensus_votes = False, -1
        if consensus_ok:
            winner = next((p for p in proposals if normalize_hex(p["hash"]) == consensus_hex_norm), None)
            if winner is None:
                winner = max(proposals, key=lambda q: q["val_acc"]); fallback_used = True
            else:
                consensus_votes = hash_votes.get(winner["hash"], -1)
        else:
            winner = max(proposals, key=lambda p: p["val_acc"]); fallback_used = True

        aggregated = winner["params"]; winner_agg_id = int(winner["agg_id"])

        # Upload updates & winner model to storage
        client_chunks_info: Dict[int, int] = {}
        for cid, params in client_params.items():
            blob = pack_params_float32(params)
            n_chunks, _ = store.upload_blob(round_id, cid, blob, chunk_size=client_chunk_size)
            client_chunks_info[cid] = int(n_chunks)

        GLOBAL_NS_OFFSET = 1_000_000
        global_ns_id = GLOBAL_NS_OFFSET + winner_agg_id
        winner_blob = pack_params_float32(aggregated)
        winner_chunks, _ = store.upload_blob(round_id, global_ns_id, winner_blob, chunk_size=global_chunk_size)

        # Reputation update (sign-like)
        scale = (alpha / (alpha+beta)) * (beta / (alpha+beta))  # same product as before after norm
        for cid in valid_ids:  reputation[cid] = reputation.get(cid, 0.0) + scale
        for cid in failed_ids: reputation[cid] = reputation.get(cid, 0.0) - scale

        # Rewards (proportional to reputation; keep simple as before)
        rep_sum = float(sum(reputation.values())) or 1.0
        rewards_this_round = {int(cid): float(tokens_per_round * (reputation[cid]/rep_sum)) for cid in reputation}

        # Update global and evaluate
        global_params = aggregated
        numpy_to_params(global_model, global_params)
        new_val_acc = accuracy(global_model, valloader)
        delta = abs(new_val_acc - prev_val_acc)
        converged = (delta < epsilon)

        row = {
            "round": rnd,
            "avg_acc": 0.0,
            "chosen_aggregators": json.dumps(chosen_ids),
            "proposal_hashes": json.dumps([p["hash"] for p in proposals]),
            "consensus_hash": consensus_hex_norm if consensus_ok else "",
            "consensus_votes": consensus_votes,
            "consensus_ok": int(bool(consensus_ok)),
            "fallback_used": int(bool(fallback_used)),
            "passed": json.dumps(sorted(valid_ids)),
            "failed": json.dumps(sorted(failed_ids)),
            "reputations": json.dumps({int(cid): float(r) for cid, r in reputation.items()}),
            "client_sizes": json.dumps({int(c.cid): int(sizes[i]) for i, c in enumerate(clients)}),
            "alpha": float(alpha), "beta": float(beta), "tau": float(tau),
            "gamma_used": float(current_gamma),
            "val_acc_prev": float(prev_val_acc), "val_acc_new": float(new_val_acc),
            "delta": float(delta), "converged": int(converged),
            "hash_votes": json.dumps(hash_votes),
            "client_chunks": json.dumps(client_chunks_info),
            "winner_global_chunks": int(winner_chunks),
            "tokens_per_round_used": float(tokens_per_round),
            "rewards_this_round": json.dumps({int(k): float(v) for k, v in rewards_this_round.items()}),
            "balances_after": json.dumps({}),
            "remaining_pool": None,
        }
        append_csv_row("artifacts/metrics.csv", row, list(row.keys()))

        save_json("artifacts/reputation.json", {str(cid): float(r) for cid, r in reputation.items()})
        save_model_state_dict(f"artifacts/checkpoints/round_{rnd:03d}.pt", global_model.state_dict())

        prev_val_acc = new_val_acc
        current_gamma = min(gamma_cap, current_gamma * gamma_growth)
        if converged:
            break

    torch.save(global_model.state_dict(), "global_mnist_cnn.pt")
    print("Done.")
    
if __name__ == "__main__":
    run_rounds(
        num_clients=6, num_aggregators=4, k_aggregators=2, max_rounds=50,
        local_epochs=1, lr=0.01, tau=1.0, gamma=0.20, non_iid=False,
        proportions=[0.4,0.2,0.2,0.1,0.06,0.04], dirichlet_alpha=None,
        reset_reputation=True, alpha=0.6, beta=0.4, epsilon=1e-3,
        random_seed=123, gamma_growth=1.05, gamma_cap=0.90,
        client_chunk_size=4*1024, global_chunk_size=4*1024,
        total_token_pool=10_000.0, tokens_per_round=100.0,
    )
