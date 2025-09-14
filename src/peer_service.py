from __future__ import annotations
import os
import time
import json
import hashlib
from typing import Dict, List, Optional

import numpy as np
import torch
from dotenv import load_dotenv

from client import Client
from model import create_model
from utils import DEVICE, load_validation_loader, accuracy, params_to_numpy, numpy_to_params
from onchain import FLChain
from storage_chain import FLStorageChain, pack_params_float32, unpack_params_float32

GLOBAL_NS_OFFSET = 1_000_000  # namespace for winner globals: GLOBAL_NS_OFFSET + agg_id


# ---------- Small helpers ----------

def sha256_params(params: List[np.ndarray], decimals: int = 6) -> str:
    h = hashlib.sha256()
    for a in params:
        arr = np.asarray(a, dtype=np.float32)
        if decimals is not None:
            arr = np.round(arr, decimals=decimals)
        h.update(arr.tobytes(order="C"))
    return h.hexdigest()

def normalize_hex(hs) -> str:
    s = hs.hex() if isinstance(hs, (bytes, bytearray)) else str(hs)
    if s.startswith(("0x", "0X")):
        s = s[2:]
    return s.lower()

def first_free_round(chain: FLChain, start: int = 1, max_scan: int = 100000) -> int:
    r = start
    for _ in range(max_scan):
        finalized, _ = chain.get_round(r)
        if not finalized:
            return r
        r += 1
    raise RuntimeError("No free round found")

def try_download_params(store: FLStorageChain, round_id: int, writer_id: int, template: List[np.ndarray], chunk_size: int) -> Optional[List[np.ndarray]]:
    try:
        blob = store.download_blob(round_id, writer_id, chunk_size=chunk_size)
        return unpack_params_float32(blob, template)
    except Exception:
        return None

def fedavg_equal_weight(param_list: List[List[np.ndarray]], template: List[np.ndarray]) -> List[np.ndarray]:
    """Simple FedAvg with equal weights (works without knowing |D_i| of others)."""
    if not param_list:
        raise ValueError("No params to average")
    out = [np.zeros_like(p, dtype=np.float32) for p in template]
    n = float(len(param_list))
    for params in param_list:
        for j in range(len(out)):
            out[j] += params[j].astype(np.float32) / n
    return out

# If you later publish |D_i| sizes per client (on-chain or via index),
# you can switch to weighted FedAvg by sizes.


# ---------- Peer main loop ----------

def main():
    load_dotenv()

    # --- Env/config ---
    rpc_url   = os.getenv("RPC_URL", "http://127.0.0.1:8545")
    coord_addr = os.getenv("CONTRACT_ADDRESS")
    store_addr = os.getenv("FLSTORAGE_ADDRESS")
    privkey    = os.getenv("PRIVKEY")  # this peer's signer
    if not (coord_addr and store_addr and privkey):
        raise RuntimeError("Set CONTRACT_ADDRESS, FLSTORAGE_ADDRESS, PRIVKEY in .env")

    # Peer identity / dataset
    CLIENT_ID    = int(os.getenv("CLIENT_ID", "0"))
    NUM_CLIENTS  = int(os.getenv("NUM_CLIENTS", "6"))
    LOCAL_EPOCHS = int(os.getenv("LOCAL_EPOCHS", "1"))
    LR           = float(os.getenv("LR", "0.01"))

    # Policy / thresholds
    TAU          = float(os.getenv("TAU", "1.0"))    # validation gate
    GAMMA        = float(os.getenv("GAMMA", "0.20")) # aggregator eligibility
    K_AGG        = int(os.getenv("K_AGGREGATORS", "2"))
    EPSILON      = float(os.getenv("EPSILON", "1e-3"))

    # Chunk sizes for FLStorage
    CLIENT_CHUNK = int(os.getenv("CLIENT_CHUNK", str(4 * 1024)))
    GLOBAL_CHUNK = int(os.getenv("GLOBAL_CHUNK", str(4 * 1024)))

    # Instantiate chain handles
    chain = FLChain(rpc_url=rpc_url, contract_address=coord_addr, privkey=privkey)
    store = FLStorageChain(rpc_url=rpc_url, contract_address=store_addr, privkey=privkey)

    # Local data/model
    client = Client(cid=CLIENT_ID, num_clients=NUM_CLIENTS, lr=LR, local_epochs=LOCAL_EPOCHS,
                    non_iid=False, labels_per_client=2, proportions=None, dirichlet_alpha=None)

    valloader = load_validation_loader()
    manager_model = create_model().to(DEVICE)  # for local evaluation; same as global template
    template_params = params_to_numpy(manager_model)

    # --- Baseline pull (Round 0) ---
    # We assume Round 0 baseline is already stored by someone under writer_id=(GLOBAL_NS_OFFSET+0)
    # If not present, the cluster needs one bootstrapper (e.g., CLIENT_ID==0) to upload it.
    round_free = first_free_round(chain, start=1)      # first not-finalized round on-chain
    round_base = round_free - 1                        # previous slot is where baseline or last winner lives

    # Try to pull baseline (writer_id = GLOBAL_NS_OFFSET + 0)
    baseline_params = try_download_params(store, round_base, GLOBAL_NS_OFFSET + 0, template_params, GLOBAL_CHUNK)
    if baseline_params is None and CLIENT_ID == 0:
        # bootstrap: CLIENT 0 creates a deterministic baseline and uploads
        print(f"[peer {CLIENT_ID}] Baseline missing → bootstrapping Round 0 baseline")
        baseline_params = template_params  # model initialized already
        blob = pack_params_float32(baseline_params)
        store.upload_blob(round_base, GLOBAL_NS_OFFSET + 0, blob, chunk_size=GLOBAL_CHUNK)
    elif baseline_params is None:
        print(f"[peer {CLIENT_ID}] Waiting for baseline on-chain... (another peer should upload)")
        # Wait loop until baseline appears
        while baseline_params is None:
            time.sleep(2)
            baseline_params = try_download_params(store, round_base, GLOBAL_NS_OFFSET + 0, template_params, GLOBAL_CHUNK)

    # Load baseline into our eval model and local client model
    for m in (manager_model, client.model):
        numpy_to_params(m, baseline_params)

    prev_val_acc = accuracy(manager_model, valloader)
    print(f"[peer {CLIENT_ID}] Pulled baseline (Round 0). Prev V-acc = {prev_val_acc:.4f}")

    # ---- Main peer loop ----
    # This loop processes one on-chain round at a time.
    # It is safe if multiple peers run concurrently: the contracts are the source of truth.
    while True:
        try:
            # Determine the current work round
            r = first_free_round(chain, start=1)  # treat this as the round we’ll propose/finalize
            pull_round = r - 1                    # authoritative global lives in previous round
            # Find which writer_id hosts the winner global of (r-1).
            # Convention: writer_id = GLOBAL_NS_OFFSET + agg_id
            # We discover it by matching hash with consensus hash if (r-1) was finalized.
            finalized_prev, consensus_hex = chain.get_round(pull_round)
            consensus_hex_norm = normalize_hex(consensus_hex) if finalized_prev else None

            # Attempt to pull the authoritative global for pull_round:
            pulled = None
            if finalized_prev and consensus_hex_norm and consensus_hex_norm != "0"*64:
                # Try all possible writer namespaces (GLOBAL_NS_OFFSET + client_id)
                for cid in range(NUM_CLIENTS):
                    params = try_download_params(store, pull_round, GLOBAL_NS_OFFSET + cid, template_params, GLOBAL_CHUNK)
                    if params is None:
                        continue
                    if sha256_params(params, decimals=6) == consensus_hex_norm:
                        pulled = params
                        print(f"[peer {CLIENT_ID}] Pulled previous winner from writer_ns={GLOBAL_NS_OFFSET+cid}")
                        break
            # Fallback: use baseline if previous round not finalized (first round)
            if pulled is None:
                pulled = try_download_params(store, pull_round, GLOBAL_NS_OFFSET + 0, template_params, GLOBAL_CHUNK)
                if pulled is None:
                    print(f"[peer {CLIENT_ID}] Waiting for authoritative global at round {pull_round}...")
                    time.sleep(2)
                    continue

            # Load pulled global into local models
            numpy_to_params(manager_model, pulled)
            client.set_params(pulled)

            # Local training
            client.train_local()

            # τ gate: compare local update vs pulled global on V
            base_acc = accuracy(manager_model, valloader)
            tmp = create_model().to(DEVICE)
            numpy_to_params(tmp, client.get_params())
            acc_i = accuracy(tmp, valloader)
            passed_tau = (acc_i + 1e-12) >= (base_acc * TAU)
            print(f"[peer {CLIENT_ID}] τ-gate: base={base_acc:.4f} | mine={acc_i:.4f} | passed={passed_tau}")

            # Upload local update to FLStorage for this round
            my_params = client.get_params()
            my_blob = pack_params_float32(my_params)
            store.upload_blob(r, client.cid, my_blob, chunk_size=CLIENT_CHUNK)
            print(f"[peer {CLIENT_ID}] Uploaded local update for round {r}")

            # If eligible aggregator (simple check: reputation/γ not tracked on-chain here),
            # you can approximate eligibility off-chain or allow anyone to propose.
            # For now, allow anyone to try proposing; the chain's consensus will pick majority.
            # Discover other updates (best-effort: try all client ids)
            candidate_params: List[List[np.ndarray]] = []
            submitters = []
            for cid in range(NUM_CLIENTS):
                p = try_download_params(store, r, cid, template_params, CLIENT_CHUNK)
                if p is None:
                    continue
                # τ check for others (optional): skip if they fail; we only have our local V
                # Here we accept all present updates to keep things moving.
                candidate_params.append(p)
                submitters.append(cid)

            if candidate_params:
                agg_params = fedavg_equal_weight(candidate_params, template_params)
                h = sha256_params(agg_params, decimals=6)
                print(f"[peer {CLIENT_ID}] Proposing hash={h[:12]}.. using {len(candidate_params)} updates")
                chain.submit_proposal(round_id=r, agg_id=client.cid, hash_hex="0x" + h)
                # Try finalize (idempotent; ok if someone else does it too)
                try:
                    chain.finalize(round_id=r, total_selected=1)  # we propose 1; many peers may also propose
                except Exception:
                    pass

                # If our hash wins, upload winner global (so next round can pull it)
                finalized, consensus_hex = chain.get_round(r)
                if finalized and normalize_hex(consensus_hex) == h:
                    g_blob = pack_params_float32(agg_params)
                    store.upload_blob(r, GLOBAL_NS_OFFSET + client.cid, g_blob, chunk_size=GLOBAL_CHUNK)
                    print(f"[peer {CLIENT_ID}] Our proposal WON. Uploaded winner global (writer_ns={GLOBAL_NS_OFFSET+client.cid}).")

                # Local convergence tracking (optional)
                numpy_to_params(manager_model, agg_params)
                new_acc = accuracy(manager_model, valloader)
                delta = abs(new_acc - prev_val_acc)
                print(f"[peer {CLIENT_ID}] ΔV-acc={delta:.6f} (ε={EPSILON}) → {'STOP' if delta < EPSILON else 'CONTINUE'}")
                prev_val_acc = new_acc

            # Small pause before next iteration
            time.sleep(2)

        except KeyboardInterrupt:
            print(f"[peer {CLIENT_ID}] stopping...")
            break
        except Exception as e:
            print(f"[peer {CLIENT_ID}] error:", e)
            time.sleep(3)


if __name__ == "__main__":
    main()
