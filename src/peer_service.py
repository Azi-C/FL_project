from __future__ import annotations
import os
import time
import hashlib
import random
import argparse
import threading
from typing import List, Optional, Sequence

import numpy as np
import torch
from dotenv import load_dotenv

# IMPORTANT: package imports (requires src/__init__.py to exist, even empty)
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
    store: FLStorageChain,
    round_id: int,
    writer_id: int,
    template: List[np.ndarray],
    chunk_size: int,
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


# ---------- Core peer loop (runs in its own thread) ----------

def run_peer(
    client_id: int,
    privkey: str,
    rpc_url: str,
    coord_addr: str,
    store_addr: str,
    num_clients: int,
    local_epochs: int,
    lr: float,
    tau: float,
    epsilon: float,
    k_proposers: int,
    round_window_sec: int,
    proposal_jitter_sec: float,
    finalize_delay_sec: float,
    idle_sleep_sec: float,
    max_rounds: int,
    client_chunk: int,
    global_chunk: int,
):
    try:
        chain = FLChain(rpc_url=rpc_url, contract_address=coord_addr, privkey=privkey)
        store = FLStorageChain(rpc_url=rpc_url, contract_address=store_addr, privkey=privkey)

        client = Client(
            cid=client_id,
            num_clients=num_clients,
            lr=lr,
            local_epochs=local_epochs,
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

        baseline_params = try_download_params(store, round_base, GLOBAL_NS_OFFSET + 0, template_params, global_chunk)
        if baseline_params is None and client_id == 0:
            print(f"[peer {client_id}] Baseline missing → bootstrapping")
            baseline_params = template_params
            blob = pack_params_float32(baseline_params)
            store.upload_blob(round_base, GLOBAL_NS_OFFSET + 0, blob, chunk_size=global_chunk)
        elif baseline_params is None:
            print(f"[peer {client_id}] Waiting for baseline...")
            while baseline_params is None:
                time.sleep(2)
                baseline_params = try_download_params(store, round_base, GLOBAL_NS_OFFSET + 0, template_params, global_chunk)

        for m in (manager_model, client.model):
            numpy_to_params(m, baseline_params)

        prev_val_acc = accuracy(manager_model, valloader)
        print(f"[peer {client_id}] Baseline V-acc = {prev_val_acc:.4f}")

        rounds_done = 0

        # --- Main loop ---
        while True:
            if max_rounds > 0 and rounds_done >= max_rounds:
                print(f"[peer {client_id}] Reached MAX_ROUNDS={max_rounds} — stopping.")
                break

            r = first_free_round(chain, start=1)
            pull_round = r - 1

            finalized_prev, consensus_hex = chain.get_round(pull_round)
            consensus_hex_norm = normalize_hex(consensus_hex) if finalized_prev else None

            pulled = None
            if finalized_prev and consensus_hex_norm and consensus_hex_norm != "0" * 64:
                for cid in range(num_clients):
                    params = try_download_params(store, pull_round, GLOBAL_NS_OFFSET + cid, template_params, global_chunk)
                    if params is not None and sha256_params(params, 6) == consensus_hex_norm:
                        pulled = params
                        print(f"[peer {client_id}] Pulled winner from writer_ns={GLOBAL_NS_OFFSET + cid}")
                        break
            if pulled is None:
                pulled = try_download_params(store, pull_round, GLOBAL_NS_OFFSET + 0, template_params, global_chunk)
                if pulled is None:
                    print(f"[peer {client_id}] Waiting for global at round {pull_round}...")
                    time.sleep(idle_sleep_sec)
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
            print(f"[peer {client_id}] τ-gate: base={base_acc:.4f} | mine={acc_i:.4f}")

            # Upload update
            my_params = client.get_params()
            my_blob = pack_params_float32(my_params)
            store.upload_blob(r, client.cid, my_blob, chunk_size=client_chunk)
            print(f"[peer {client_id}] Uploaded update for round {r}")

            # Let others upload before proposing
            time.sleep(round_window_sec)

            # Collect updates
            candidate_params: List[List[np.ndarray]] = []
            for cid in range(num_clients):
                p = try_download_params(store, r, cid, template_params, client_chunk)
                if p is not None:
                    candidate_params.append(p)
            if not candidate_params:
                time.sleep(idle_sleep_sec)
                continue

            # Deterministic proposers
            proposers = deterministic_proposers(r, num_clients, k_proposers)
            if client_id in proposers:
                agg_params = fedavg_equal_weight(candidate_params, template_params)
                h = sha256_params(agg_params, decimals=6)

                time.sleep(random.uniform(0.0, proposal_jitter_sec))
                chain.submit_proposal(round_id=r, agg_id=client.cid, hash_hex="0x" + h)
                print(f"[peer {client_id}] Submitted proposal {h[:12]}..")

                if client_id == proposers[0]:
                    time.sleep(finalize_delay_sec)
                    try:
                        chain.finalize(round_id=r, total_selected=len(proposers))
                        print(f"[peer {client_id}] Finalized round {r}")
                    except Exception as e:
                        print(f"[peer {client_id}] finalize error: {e}")

                finalized, consensus_hex = chain.get_round(r)
                if finalized and normalize_hex(consensus_hex) == h:
                    g_blob = pack_params_float32(agg_params)
                    store.upload_blob(r, GLOBAL_NS_OFFSET + client.cid, g_blob, chunk_size=global_chunk)
                    print(f"[peer {client_id}] Our proposal WON. Uploaded winner global.")

                # Convergence check
                numpy_to_params(manager_model, agg_params)
                new_acc = accuracy(manager_model, valloader)
                delta = abs(new_acc - prev_val_acc)
                print(f"[peer {client_id}] ΔV-acc={delta:.6f} (ε={epsilon})")
                prev_val_acc = new_acc
                if delta < epsilon:
                    print(f"[peer {client_id}] Converged — stopping.")
                    break

            rounds_done += 1
            time.sleep(idle_sleep_sec)

    except KeyboardInterrupt:
        print(f"[peer {client_id}] stopping...")
    except Exception as e:
        print(f"[peer {client_id}] error: {e}")
        # Let the thread end; main will continue


# ---------- CLI / Launcher ----------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--id", type=int, help="Run a single client with this ID (0..NUM_CLIENTS-1)")
    p.add_argument("--spawn", type=int, help="Spawn N clients (0..N-1) in this process")
    return p.parse_args()


def main():
    # Base env (shared settings)
    env_file = os.getenv("ENV_FILE") or ".env"
    load_dotenv(env_file)
    print(f"[launcher] Loaded base env from {env_file}")

    args = parse_args()

    # Shared config (from base .env)
    rpc_url    = os.getenv("RPC_URL", "http://127.0.0.1:8545")
    coord_addr = os.getenv("CONTRACT_ADDRESS")
    store_addr = os.getenv("FLSTORAGE_ADDRESS")

    # Training/FL config
    num_clients         = int(os.getenv("NUM_CLIENTS", "6"))
    local_epochs        = int(os.getenv("LOCAL_EPOCHS", "1"))
    lr                  = float(os.getenv("LR", "0.01"))
    tau                 = float(os.getenv("TAU", "1.0"))
    epsilon             = float(os.getenv("EPSILON", "1e-3"))
    k_proposers         = int(os.getenv("K_PROPOSERS", "2"))
    round_window_sec    = int(os.getenv("ROUND_WINDOW_SEC", "5"))
    proposal_jitter_sec = float(os.getenv("PROPOSAL_JITTER_SEC", "0.5"))
    finalize_delay_sec  = float(os.getenv("FINALIZE_DELAY_SEC", "1.5"))
    idle_sleep_sec      = float(os.getenv("IDLE_SLEEP_SEC", "2.0"))
    max_rounds          = int(os.getenv("MAX_ROUNDS", "0"))
    client_chunk        = int(os.getenv("CLIENT_CHUNK", str(4 * 1024)))
    global_chunk        = int(os.getenv("GLOBAL_CHUNK", str(4 * 1024)))

    # PRIVKEYS for multi-client runs:
    # Provide as CSV in env: PRIVKEYS_CSV="0xaaa,0xbbb,0xccc,..."
    pk_csv = os.getenv("PRIVKEYS_CSV", "").strip()
    privkeys: List[str] = [s.strip() for s in pk_csv.split(",") if s.strip()]

    if args.id is not None and args.spawn is not None:
        raise SystemExit("Use either --id or --spawn, not both.")

    # --- Single client mode ---
    if args.id is not None:
        client_id = args.id
        # Prefer PRIVKEYS_CSV if present
        if privkeys and len(privkeys) > client_id:
            privkey = privkeys[client_id]
        else:
            # fallback: single PRIVKEY in env
            privkey = os.getenv("PRIVKEY")
        if not (coord_addr and store_addr and privkey):
            raise RuntimeError("Set CONTRACT_ADDRESS, FLSTORAGE_ADDRESS and either PRIVKEY or PRIVKEYS_CSV")

        run_peer(
            client_id=client_id,
            privkey=privkey,
            rpc_url=rpc_url,
            coord_addr=coord_addr,
            store_addr=store_addr,
            num_clients=num_clients,
            local_epochs=local_epochs,
            lr=lr,
            tau=tau,
            epsilon=epsilon,
            k_proposers=k_proposers,
            round_window_sec=round_window_sec,
            proposal_jitter_sec=proposal_jitter_sec,
            finalize_delay_sec=finalize_delay_sec,
            idle_sleep_sec=idle_sleep_sec,
            max_rounds=max_rounds,
            client_chunk=client_chunk,
            global_chunk=global_chunk,
        )
        return

    # --- Spawn N clients mode ---
    if args.spawn:
        N = args.spawn
        if N > num_clients:
            print(f"[launcher] Warning: --spawn {N} > NUM_CLIENTS {num_clients}, adjusting.")
            N = num_clients

        # Need N privkeys
        if len(privkeys) < N:
            raise RuntimeError(
                f"Provide at least {N} private keys in PRIVKEYS_CSV for multi-client spawn."
            )

        threads: List[threading.Thread] = []
        for cid in range(N):
            t = threading.Thread(
                target=run_peer,
                kwargs=dict(
                    client_id=cid,
                    privkey=privkeys[cid],
                    rpc_url=rpc_url,
                    coord_addr=coord_addr,
                    store_addr=store_addr,
                    num_clients=num_clients,
                    local_epochs=local_epochs,
                    lr=lr,
                    tau=tau,
                    epsilon=epsilon,
                    k_proposers=k_proposers,
                    round_window_sec=round_window_sec,
                    proposal_jitter_sec=proposal_jitter_sec,
                    finalize_delay_sec=finalize_delay_sec,
                    idle_sleep_sec=idle_sleep_sec,
                    max_rounds=max_rounds,
                    client_chunk=client_chunk,
                    global_chunk=global_chunk,
                ),
                name=f"peer-{cid}",
                daemon=False,
            )
            t.start()
            threads.append(t)

        # Wait for all peers to finish
        for t in threads:
            t.join()

        print("[launcher] All peers finished.")
        return

    # If neither provided, show usage
    print("Usage:")
    print("  python -m src.peer_service --id 3          # run single client 3")
    print("  PRIVKEYS_CSV=0xA,0xB,... python -m src.peer_service --spawn 6  # run 6 clients")
    raise SystemExit(2)


if __name__ == "__main__":
    main()
