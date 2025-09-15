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
from persistence import save_json, save_model_state_dict, ensure_csv, append_csv_row
from onchain import FLChain
from storage_chain import FLStorageChain, pack_params_float32, unpack_params_float32

# ---------------- Utilities ----------------

def params_hash(params: List[np.ndarray], decimals: int = 6) -> str:
    """Compute SHA256 hash of rounded model parameters."""
    h = hashlib.sha256()
    for a in params:
        arr = np.asarray(a, dtype=np.float32)
        if decimals is not None:
            arr = np.round(arr, decimals=decimals)
        h.update(arr.tobytes(order="C"))
    return h.hexdigest()

def normalize_hex(hs) -> str:
    """Normalize a hex string (strip 0x, lower)."""
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
    """Select k aggregators uniformly from those above gamma (or fallback to all)."""
    if not aggregators:
        return []
    eligible = [a for a in aggregators if reputation.get(a.cid, 0.0) >= gamma]
    pool = eligible if len(eligible) > 0 else aggregators
    k = max(1, min(k, len(pool)))
    idxs = rng.choice(len(pool), size=k, replace=False)
    return [pool[int(i)] for i in idxs]

def first_free_round(chain: FLChain, start: int = 1, max_scan: int = 100000) -> int:
    """Find the first round id not yet finalized on chain."""
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
    """Weighted FedAvg aggregator."""
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

# ---------------- Main Orchestrator ----------------

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
    global_params = params_to_numpy(global_model)  # template for shapes

    # Logs
    os.makedirs("artifacts/checkpoints", exist_ok=True)
    csv_fields = [
        "round","chosen_aggregators","proposal_hashes","consensus_hash",
        "consensus_votes","consensus_ok","fallback_used","passed","failed",
        "reputations","client_sizes","alpha","beta","tau","gamma_used",
        "val_acc_prev","val_acc_new","delta","converged","hash_votes",
        "client_chunks","winner_global_chunks","tokens_per_round_used",
        "rewards_this_round","remaining_pool",
    ]
    ensure_csv("artifacts/metrics.csv", csv_fields)

    # Initial reputation ∝ data size
    sizes = [c.num_examples() for c in clients]
    total_k = float(sum(sizes)) or 1.0
    reputation: Dict[int, float] = {c.cid: (sizes[i] / total_k) for i, c in enumerate(clients)}
    print("Client sizes:", {c.cid: sizes[i] for c in enumerate(clients)})
    print("Init reputations:", {cid: round(reputation[cid], 4) for cid in sorted(reputation)})

    # On-chain setup
    load_dotenv()
    rpc_url = os.getenv("RPC_URL", "http://127.0.0.1:8545")
    coord_addr = os.getenv("CONTRACT_ADDRESS")
    storage_addr = os.getenv("FLSTORAGE_ADDRESS")
    privkey = os.getenv("PRIVKEY")
    if not (coord_addr and storage_addr and privkey):
        raise RuntimeError("Set CONTRACT_ADDRESS, FLSTORAGE_ADDRESS, PRIVKEY in .env")

    chain = FLChain(rpc_url=rpc_url, contract_address=coord_addr, privkey=privkey)
    store = FLStorageChain(rpc_url=rpc_url, contract_address=storage_addr, privkey=privkey)

    # Baseline upload (Round 0)
    round_base = first_free_round(chain, start=1) - 1
    print(f"On-chain first free round will be: {round_base + 1}")
    print(f"Storing baseline (Round 0) under round_id={round_base}")

    baseline_hash = params_hash(global_params, decimals=6)
    GLOBAL_NS_OFFSET = 1_000_000
    baseline_ns_id = GLOBAL_NS_OFFSET + 0
    baseline_blob = pack_params_float32(global_params)
    baseline_chunks, _ = store.upload_blob(round_base, baseline_ns_id, baseline_blob, chunk_size=global_chunk_size)
    print(f"[Baseline] hash={baseline_hash[:12]}..  chunks={baseline_chunks}  ns_id={baseline_ns_id}")

    # Clients + manager pull baseline
    for c in clients:
        c.sync_from_storage(store, round_id=round_base, writer_id=baseline_ns_id,
                            template=global_params, chunk_size=global_chunk_size)
    blob0 = store.download_blob(round_base, baseline_ns_id, chunk_size=global_chunk_size)
    downloaded_global_params = unpack_params_float32(blob0, template=global_params)
    numpy_to_params(global_model, downloaded_global_params)
    global_params = downloaded_global_params

    save_model_state_dict("artifacts/checkpoints/round_000.pt", global_model.state_dict())
    prev_val_acc = accuracy(global_model, valloader)

    # Track authoritative source for next pulls
    current_global_round_id = round_base
    current_global_writer_ns = baseline_ns_id

    converged, rnd, current_gamma = False, 0, float(gamma)
    while rnd < max_rounds and not converged:
        rnd += 1
        round_id = round_base + rnd
        next_gamma = min(gamma_cap, current_gamma * gamma_growth)

        print(f"\n=== Round {rnd} (on-chain id {round_id}) ===")
        print(f"Gamma now: {current_gamma:.4f} → next: {next_gamma:.4f}")
        print(f"Reputations (start): { {cid: round(reputation[cid],4) for cid in sorted(reputation)} }")
        print(f"Prev global V-accuracy: {prev_val_acc:.6f} | ε = {epsilon}")

        # --- All pull current global from chain ---
        mgr_blob = store.download_blob(current_global_round_id, current_global_writer_ns, chunk_size=global_chunk_size)
        mgr_global_params = unpack_params_float32(mgr_blob, template=global_params)
        numpy_to_params(global_model, mgr_global_params)
        global_params = mgr_global_params
        for c in clients:
            c.sync_from_storage(store, round_id=current_global_round_id,
                                writer_id=current_global_writer_ns,
                                template=global_params, chunk_size=global_chunk_size)

        # --- Local training ---
        for c in clients: c.train_local()

        # Collect updates
        client_params = {int(c.cid): c.get_params() for c in clients}
        client_sizes  = {int(c.cid): int(c.num_examples()) for c in clients}

        # Validation gate
        base_acc = accuracy(global_model, valloader)
        valid_ids, failed_ids = [], []
        for c in clients:
            tmp = create_model().to(DEVICE)
            numpy_to_params(tmp, client_params[int(c.cid)])
            acc_i = accuracy(tmp, valloader)
            if acc_i + 1e-12 >= base_acc * tau:
                valid_ids.append(int(c.cid))
            else:
                failed_ids.append(int(c.cid))
        if not valid_ids:
            best_id, best_acc = int(clients[0].cid), -1.0
            for c in clients:
                tmp = create_model().to(DEVICE)
                numpy_to_params(tmp, client_params[int(c.cid)])
                acc_i = accuracy(tmp, valloader)
                if acc_i > best_acc: best_acc, best_id = acc_i, int(c.cid)
            valid_ids.append(best_id)
            failed_ids = [cid for cid in failed_ids if cid != best_id]
        print(f"[Validation τ={tau}] passed={sorted(valid_ids)} failed={sorted(failed_ids)}")

        # Aggregator eligibility + selection
        eligible = [a.cid for a in aggregs if reputation.get(a.cid,0.0) >= current_gamma]
        print(f"[Gamma] eligible={eligible}")
        chosen = select_aggregators_uniform(reputation, current_gamma, aggregs, k_aggregators, rng)
        chosen_ids = [a.cid for a in chosen]
        print(f"Chosen aggregators: {chosen_ids}")

        # Proposals
        proposals = []
        for agg in chosen:
            agg_params = fedavg_weighted(client_params, client_sizes, valid_ids, global_params)
            h = params_hash(agg_params, decimals=6)
            tmp = create_model().to(DEVICE); numpy_to_params(tmp, agg_params)
            prop_val_acc = accuracy(tmp, valloader)
            proposals.append({"agg_id": agg.cid, "params": agg_params, "hash": h, "val_acc": prop_val_acc})
            print(f" • Agg {agg.cid}: hash={h[:12]} V-acc={prop_val_acc:.4f}")

        # On-chain finalize
        for p in proposals:
            chain.submit_proposal(round_id=round_id, agg_id=int(p["agg_id"]), hash_hex="0x"+p["hash"])
        chain.finalize(round_id=round_id, total_selected=len(proposals))

        finalized, consensus_hex = chain.get_round(round_id)
        consensus_hex_norm = normalize_hex(consensus_hex)
        hash_votes: Dict[str,int] = {p["hash"]: int(chain.get_votes(round_id,"0x"+p["hash"])) for p in proposals}
        if finalized and consensus_hex_norm != "0"*64:
            winner = next((p for p in proposals if normalize_hex(p["hash"]) == consensus_hex_norm), None)
            if winner is None:
                winner = max(proposals, key=lambda q:q["val_acc"])
                print("Consensus hash not found → fallback to best V-acc")
        else:
            winner = max(proposals, key=lambda p:p["val_acc"])
            print("Consensus FAILED → fallback to best V-acc")
        aggregated = winner["params"]; winner_agg_id = int(winner["agg_id"])

        # Upload updates + winner
        for cid, params in client_params.items():
            blob = pack_params_float32(params)
            store.upload_blob(round_id, cid, blob, chunk_size=client_chunk_size)
        global_ns_id = GLOBAL_NS_OFFSET + winner_agg_id
        winner_blob = pack_params_float32(aggregated)
        store.upload_blob(round_id, global_ns_id, winner_blob, chunk_size=global_chunk_size)

        # Reputation update
        s_part_map = {cid:+1 for cid in valid_ids}; s_part_map.update({cid:-1 for cid in failed_ids})
        s = alpha+beta; a_n = alpha/s; b_n = beta/s; scale = a_n*b_n
        for cid, s_part in s_part_map.items():
            reputation[cid] = reputation.get(cid,0.0) + (scale*float(s_part))
        print(f"Reputations updated: { {cid:round(reputation[cid],4) for cid in sorted(reputation)} }")

        # Rewards
        rep_sum = float(sum(reputation.values())) or 1.0
        rewards_this_round = {int(cid): float(tokens_per_round * (reputation[cid]/rep_sum)) for cid in reputation}

        # Convergence
        global_params = aggregated
        numpy_to_params(global_model, global_params)
        new_val_acc = accuracy(global_model, valloader)
        delta = abs(new_val_acc - prev_val_acc)
        converged = (delta < epsilon)
        print(f"Δ={delta:.6f}, new V-acc={new_val_acc:.4f}, converged={converged}")

        # Advance pointer for next round
        current_global_round_id = round_id
        current_global_writer_ns = global_ns_id
        prev_val_acc = new_val_acc
        current_gamma = next_gamma

        # Log CSV
        row = {
            "round": rnd, "chosen_aggregators": json.dumps(chosen_ids),
            "proposal_hashes": json.dumps([p["hash"] for p in proposals]),
            "consensus_hash": consensus_hex_norm,
            "consensus_votes": None,
            "consensus_ok": int(finalized),
            "fallback_used": 0,
            "passed": json.dumps(valid_ids),
            "failed": json.dumps(failed_ids),
            "reputations": json.dumps({int(cid):float(r) for cid,r in reputation.items()}),
            "client_sizes": json.dumps(client_sizes),
            "alpha": float(alpha), "beta": float(beta), "tau": float(tau),
            "gamma_used": float(current_gamma),
            "val_acc_prev": float(prev_val_acc), "val_acc_new": float(new_val_acc),
            "delta": float(delta), "converged": int(converged),
            "hash_votes": json.dumps(hash_votes),
            "client_chunks": json.dumps({}),
            "winner_global_chunks": 0,
            "tokens_per_round_used": float(tokens_per_round),
            "rewards_this_round": json.dumps(rewards_this_round),
            "remaining_pool": None,
        }
        append_csv_row("artifacts/metrics.csv", row, csv_fields)

        save_json("artifacts/reputation.json", {str(cid):float(r) for cid,r in reputation.items()})
        save_model_state_dict(f"artifacts/checkpoints/round_{rnd:03d}.pt", global_model.state_dict())

        if converged: break

    torch.save(global_model.state_dict(),"global_mnist_cnn.pt")
    print("Done.")

if __name__ == "__main__":
    run_rounds()
