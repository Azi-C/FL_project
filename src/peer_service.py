from __future__ import annotations
import os
import time
import hashlib
import random
import argparse
from typing import List, Optional

import numpy as np
import torch
from dotenv import load_dotenv

from src.client import Client
from src.model import create_model
from src.utils import DEVICE, load_validation_loader, accuracy, params_to_numpy, numpy_to_params
from src.onchain import FLChain
from src.storage_chain import FLStorageChain, pack_params_float32, unpack_params_float32

GLOBAL_NS_OFFSET = 1_000_000  # namespace for winner globals: GLOBAL_NS_OFFSET + agg_id


# ---------- Helpers ----------

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

def try_download_params(
    store: FLStorageChain, round_id: int, writer_id: int,
    template: List[np.ndarray], chunk_size: int
) -> Optional[List[np.ndarray]]:
    try:
        blob = store.download_blob(round_id, writer_id, chunk_size=chunk_size)
        return unpack_params_float32(blob, template)
    except Exception:
        return None

def fedavg_equal_weight(param_list: List[List[np.ndarray]], template: List[np.ndarray]) -> List[np.ndarray]:
    """Simple FedAvg with equal weights."""
    if not param_list:
        raise ValueError("No params to average")
    out = [np.zeros_like(p, dtype=np.float32) for p in template]
    n = float(len(param_list))
    for params in param_list:
        for j in range(len(out)):
            out[j] += params[j].astype(np.float32) / n
    return out

def deterministic_proposers(round_id: int, num_clients: int, k_proposers: int) -> List[int]:
    """Pick K proposers deterministically using round_id as RNG seed."""
    k = max(1, min(k_proposers, num_clients))
    rng = np.random.default_rng(seed=round_id)
    choices = rng.choice(num_clients, size=k, replace=False)
    return sorted(int(x) for x in choices)


# ---------- CLI ARG PARSER ----------

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--id", type=int, help="Client ID (0..NUM_CLIENTS-1)")
    return parser.parse_args()


# ---------- Peer main loop ----------

def main():
    args = parse_args()

    # If --id is provided, auto-pick .env.clientX
    if args.id is not None:
        env_file = f".env.client{args.id}"
    else:
        env_file = os.getenv("ENV_FILE") or ".env"

    load_dotenv(env_file)
    print(f"[peer] Loaded config from {env_file}")

    # --- Env/config ---
    rpc_url    = os.getenv("RPC_URL", "http://127.0.0.1:8545")
    coord_addr = os.getenv("CONTRACT_ADDRESS")
    store_addr = os.getenv("FLSTORAGE_ADDRESS")
    privkey    = os.getenv("PRIVKEY")
    if not (coord_addr and store_addr and privkey):
        raise RuntimeError("Set CONTRACT_ADDRESS, FLSTORAGE_ADDRESS, PRIVKEY")

    CLIENT_ID    = int(os.getenv("CLIENT_ID", args.id or 0))
    NUM_CLIENTS  = int(os.getenv("NUM_CLIENTS", "6"))
    LOCAL_EPOCHS = int(os.getenv("LOCAL_EPOCHS", "1"))
    LR           = float(os.getenv("LR", "0.01"))

    # Policy
    TAU         = float(os.getenv("TAU", "1.0"))
    EPSILON     = float(os.getenv("EPSILON", "1e-3"))
    K_PROPOSERS = int(os.getenv("K_PROPOSERS", "2"))

    ROUND_WINDOW_SEC    = int(os.getenv("ROUND_WINDOW_SEC", "5"))
    PROPOSAL_JITTER_SEC = float(os.getenv("PROPOSAL_JITTER_SEC", "0.5"))
    FINALIZE_DELAY_SEC  = float(os.getenv("FINALIZE_DELAY_SEC", "1.5"))
    IDLE_SLEEP_SEC      = float(os.getenv("IDLE_SLEEP_SEC", "2.0"))

    MAX_ROUNDS = int(os.getenv("MAX_ROUNDS", "0"))

    CLIENT_CHUNK = int(os.getenv("CLIENT_CHUNK", str(4 * 1024)))
    GLOBAL_CHUNK = int(os.getenv("GLOBAL_CHUNK", str(4 * 1024)))

    # Chain handles
    chain = FLChain(rpc_url=rpc_url, contract_address=coord_addr, privkey=privkey)
    store = FLStorageChain(rpc_url=rpc_url, contract_address=store_addr, privkey=privkey)

    # Local client
    client = Client(
        cid=CLIENT_ID,
        num_clients=NUM_CLIENTS,
        lr=LR,
        local_epochs=LOCAL_EPOCHS,
        non_iid=False,
        labels_per_client=2,
        proportions=None,
        dirichlet_alpha=None,
    )

    valloader = load_validation_loader()
    manager_model = create_model().to(DEVICE)
    template_params = params_to_numpy(manager_model)

    # --- Baseline (Round 0) ---
    round_free = first_free_round(chain, start=1)
    round_base = round_free - 1

    baseline_params = try_download_params(store, round_base, GLOBAL_NS_OFFSET + 0, template_params, GLOBAL_CHUNK)
    if baseline_params is None and CLIENT_ID == 0:
        print(f"[peer {CLIENT_ID}] Baseline missing → bootstrapping")
        baseline_params = template_params
        blob = pack_params_float32(baseline_params)
        store.upload_blob(round_base, GLOBAL_NS_OFFSET + 0, blob, chunk_size=GLOBAL_CHUNK)
    elif baseline_params is None:
        print(f"[peer {CLIENT_ID}] Waiting for baseline...")
        while baseline_params is None:
            time.sleep(2)
            baseline_params = try_download_params(store, round_base, GLOBAL_NS_OFFSET + 0, template_params, GLOBAL_CHUNK)

    for m in (manager_model, client.model):
        numpy_to_params(m, baseline_params)

    prev_val_acc = accuracy(manager_model, valloader)
    print(f"[peer {CLIENT_ID}] Baseline V-acc = {prev_val_acc:.4f}")

    rounds_done = 0

    # --- Main loop ---
    while True:
        try:
            if MAX_ROUNDS > 0 and rounds_done >= MAX_ROUNDS:
                print(f"[peer {CLIENT_ID}] Reached MAX_ROUNDS={MAX_ROUNDS} — stopping.")
                break

            r = first_free_round(chain, start=1)
            pull_round = r - 1

            finalized_prev, consensus_hex = chain.get_round(pull_round)
            consensus_hex_norm = normalize_hex(consensus_hex) if finalized_prev else None

            pulled = None
            if finalized_prev and consensus_hex_norm and consensus_hex_norm != "0" * 64:
                for cid in range(NUM_CLIENTS):
                    params = try_download_params(store, pull_round, GLOBAL_NS_OFFSET + cid, template_params, GLOBAL_CHUNK)
                    if params is not None and sha256_params(params, 6) == consensus_hex_norm:
                        pulled = params
                        print(f"[peer {CLIENT_ID}] Pulled winner from writer_ns={GLOBAL_NS_OFFSET + cid}")
                        break
            if pulled is None:
                pulled = try_download_params(store, pull_round, GLOBAL_NS_OFFSET + 0, template_params, GLOBAL_CHUNK)
                if pulled is None:
                    print(f"[peer {CLIENT_ID}] Waiting for global at round {pull_round}...")
                    time.sleep(IDLE_SLEEP_SEC)
                    continue

            numpy_to_params(manager_model, pulled)
            client.set_params(pulled)

            # Local training
            client.train_local()

            # τ-gate
            base_acc = accuracy(manager_model, valloader)
            tmp = create_model().to(DEVICE)
            numpy_to_params(tmp, client.get_params())
            acc_i = accuracy(tmp, valloader)
            print(f"[peer {CLIENT_ID}] τ-gate: base={base_acc:.4f} | mine={acc_i:.4f}")

            # Upload update
            my_params = client.get_params()
            my_blob = pack_params_float32(my_params)
            store.upload_blob(r, client.cid, my_blob, chunk_size=CLIENT_CHUNK)
            print(f"[peer {CLIENT_ID}] Uploaded update for round {r}")

            time.sleep(ROUND_WINDOW_SEC)

            # Collect updates
            candidate_params = []
            for cid in range(NUM_CLIENTS):
                p = try_download_params(store, r, cid, template_params, CLIENT_CHUNK)
                if p is not None:
                    candidate_params.append(p)

            if not candidate_params:
                time.sleep(IDLE_SLEEP_SEC)
                continue

            # Deterministic proposers
            proposers = deterministic_proposers(r, NUM_CLIENTS, K_PROPOSERS)
            if CLIENT_ID in proposers:
                agg_params = fedavg_equal_weight(candidate_params, template_params)
                h = sha256_params(agg_params, decimals=6)

                time.sleep(random.uniform(0.0, PROPOSAL_JITTER_SEC))
                chain.submit_proposal(round_id=r, agg_id=client.cid, hash_hex="0x" + h)
                print(f"[peer {CLIENT_ID}] Submitted proposal {h[:12]}..")

                if CLIENT_ID == proposers[0]:
                    time.sleep(FINALIZE_DELAY_SEC)
                    try:
                        chain.finalize(round_id=r, total_selected=len(proposers))
                        print(f"[peer {CLIENT_ID}] Finalized round {r}")
                    except Exception as e:
                        print(f"[peer {CLIENT_ID}] finalize error: {e}")

                finalized, consensus_hex = chain.get_round(r)
                if finalized and normalize_hex(consensus_hex) == h:
                    g_blob = pack_params_float32(agg_params)
                    store.upload_blob(r, GLOBAL_NS_OFFSET + client.cid, g_blob, chunk_size=GLOBAL_CHUNK)
                    print(f"[peer {CLIENT_ID}] Our proposal WON. Uploaded winner global.")

                # Convergence check
                numpy_to_params(manager_model, agg_params)
                new_acc = accuracy(manager_model, valloader)
                delta = abs(new_acc - prev_val_acc)
                print(f"[peer {CLIENT_ID}] ΔV-acc={delta:.6f} (ε={EPSILON})")
                prev_val_acc = new_acc

                if delta < EPSILON:
                    print(f"[peer {CLIENT_ID}] Converged — stopping.")
                    break

            rounds_done += 1
            time.sleep(IDLE_SLEEP_SEC)

        except KeyboardInterrupt:
            print(f"[peer {CLIENT_ID}] stopping...")
            break
        except Exception as e:
            print(f"[peer {CLIENT_ID}] error:", e)
            time.sleep(3)


if __name__ == "__main__":
    main()
