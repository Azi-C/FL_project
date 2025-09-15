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
from src.onchain import FLChain, FLChainV2
from src.storage_chain import FLStorageChain, pack_params_float32, unpack_params_float32

GLOBAL_NS_OFFSET = 1_000_000
FP = 1_000_000


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

def try_download_params(store: FLStorageChain, round_id: int, writer_id: int,
                        template: List[np.ndarray], chunk_size: int) -> Optional[List[np.ndarray]]:
    try:
        blob = store.download_blob(round_id, writer_id, chunk_size=chunk_size)
        return unpack_params_float32(blob, template)
    except Exception:
        return None

def fedavg_equal_weight(param_list: List[List[np.ndarray]], template: List[np.ndarray]) -> List[np.ndarray]:
    if not param_list:
        raise ValueError("No params to average")
    out = [np.zeros_like(p, dtype=np.float32) for p in template]
    n = float(len(param_list))
    for params in param_list:
        for j in range(len(out)):
            out[j] += params[j].astype(np.float32) / n
    return out

def deterministic_proposers(round_id: int, eligible: List[int], k_proposers: int) -> List[int]:
    if not eligible:
        return []
    k = max(1, min(k_proposers, len(eligible)))
    rng = np.random.default_rng(seed=round_id)
    choices = rng.choice(eligible, size=k, replace=False)
    return sorted(int(x) for x in choices)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--id", type=int, help="Run a single client with this ID (0..NUM_CLIENTS-1)")
    return p.parse_args()


def run_peer(client_id: int,
             privkey: str,
             rpc_url: str,
             coord_addr: str,
             store_addr: str,
             coord_v2_addr: str,
             num_clients: int,
             local_epochs: int,
             lr: float,
             tau: float,
             epsilon: float,
             gamma: float,
             k_proposers: int,
             round_window_sec: int,
             proposal_jitter_sec: float,
             finalize_delay_sec: float,
             idle_sleep_sec: float,
             max_rounds: int,
             client_chunk: int,
             global_chunk: int):

    chain  = FLChain(rpc_url=rpc_url, contract_address=coord_addr, privkey=privkey)
    chain2 = FLChainV2(rpc_url=rpc_url, contract_address=coord_v2_addr, privkey=privkey)
    store  = FLStorageChain(rpc_url=rpc_url, contract_address=store_addr,   privkey=privkey)

    # quick sanity prints
    print(f"[peer {client_id}] V2 address:", coord_v2_addr)
    code = chain2.w3.eth.get_code(chain2.addr).hex()
    print(f"[peer {client_id}] V2 bytecode present? ", code != "0x")

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

    # Register with |D_i|
    data_size = len(client.trainloader.dataset) if hasattr(client.trainloader, "dataset") else 0
    try:
        chain2.register_client(data_size)
        print(f"[peer {client_id}] Registered (|D|={data_size})")
    except Exception as e:
        print(f"[peer {client_id}] register_client skipped: {e}")

    # Pull/Bootstrap baseline
    valloader = load_validation_loader()
    manager_model = create_model().to(DEVICE)
    template_params = params_to_numpy(manager_model)

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

    while True:
        if max_rounds > 0 and rounds_done >= max_rounds:
            print(f"[peer {client_id}] Reached MAX_ROUNDS={max_rounds} — stopping.")
            break

        r = first_free_round(chain, start=1)
        pull_round = r - 1

        finalized_prev, consensus_hex = chain.get_round(pull_round)
        consensus_hex_norm = normalize_hex(consensus_hex) if finalized_prev else None

        pulled = None
        if finalized_prev and consensus_hex_norm and consensus_hex_norm != "0"*64:
            for cid in range(num_clients):
                params = try_download_params(store, pull_round, GLOBAL_NS_OFFSET + cid, template_params, global_chunk)
                if params is not None and sha256_params(params, 6) == consensus_hex_norm:
                    pulled = params
                    break
        if pulled is None:
            pulled = try_download_params(store, pull_round, GLOBAL_NS_OFFSET + 0, template_params, global_chunk)
            if pulled is None:
                time.sleep(idle_sleep_sec)
                continue

        numpy_to_params(manager_model, pulled)
        client.set_params(pulled)

        # Local train
        client.train_local()

        # τ-gate & submitEval (on-chain rep update)
        base_acc = accuracy(manager_model, valloader)
        tmp = create_model().to(DEVICE)
        numpy_to_params(tmp, client.get_params())
        acc_i = accuracy(tmp, valloader)

        spart = 1 if (acc_i + 1e-12) >= (base_acc * tau) else -1
        a_i_fp = int(max(0.0, min(1.0, float(acc_i))) * FP)

        try:
            chain2.submit_eval(round_id=r, who=chain2.acct.address, a_i_fp=a_i_fp, s_i_part=spart)
            print(f"[peer {client_id}] submitEval ok (a={acc_i:.4f}, s={spart})")
        except Exception as e:
            print(f"[peer {client_id}] submitEval error: {e}")

        # Upload my local update to FLStorage
        my_blob = pack_params_float32(client.get_params())
        store.upload_blob(r, client.cid, my_blob, chunk_size=client_chunk)
        print(f"[peer {client_id}] Uploaded local update @ round {r}")

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

        # γ-gated proposers (read on-chain reputation)
        addrs = chain2.get_participants()
        reps_fp = {}
        for a in addrs:
            _, _, rep_fp = chain2.get_client(a)
            reps_fp[a.lower()] = rep_fp
        gamma_fp = int(max(0.0, min(1.0, gamma)) * FP)

        elig_cids: List[int] = []
        if len(addrs) >= num_clients:
            for idx in range(num_clients):
                a = addrs[idx].lower()
                if reps_fp.get(a, 0) >= gamma_fp:
                    elig_cids.append(idx)
        else:
            elig_cids = list(range(num_clients))  # fallback

        proposers = deterministic_proposers(r, elig_cids, k_proposers)
        if client_id in proposers:
            agg_params = fedavg_equal_weight(candidate_params, template_params)
            h = sha256_params(agg_params, 6)
            time.sleep(random.uniform(0.0, proposal_jitter_sec))

            chain.submit_proposal(round_id=r, agg_id=client.cid, hash_hex="0x" + h)
            if client_id == proposers[0]:
                time.sleep(finalize_delay_sec)
                try:
                    chain.finalize(round_id=r, total_selected=len(proposers))
                except Exception:
                    pass

            finalized, consensus_hex = chain.get_round(r)
            if finalized and normalize_hex(consensus_hex) == h:
                g_blob = pack_params_float32(agg_params)
                store.upload_blob(r, GLOBAL_NS_OFFSET + client.cid, g_blob, chunk_size=global_chunk)

            numpy_to_params(manager_model, agg_params)
            new_acc = accuracy(manager_model, valloader)
            delta = abs(new_acc - prev_val_acc)
            prev_val_acc = new_acc
            print(f"[peer {client_id}] ΔV-acc={delta:.6f} (ε={epsilon})")

            if delta < epsilon:
                try:
                    chain2.mark_converged(r, bytes.fromhex(h))
                except Exception:
                    pass
                print(f"[peer {client_id}] Converged — stopping.")
                return

        rounds_done += 1
        time.sleep(idle_sleep_sec)


def main():
    env_file = os.getenv("ENV_FILE") or ".env"
    load_dotenv(env_file)
    print(f"[launcher] Loaded env from {env_file}")

    args = parse_args()

    rpc_url     = os.getenv("RPC_URL", "http://127.0.0.1:8545")
    coord_addr  = os.getenv("CONTRACT_ADDRESS")         # v1 coordinator
    store_addr  = os.getenv("FLSTORAGE_ADDRESS")
    coord_v2    = os.getenv("COORDINATOR_V2_ADDRESS")   # v2 coordinator

    num_clients         = int(os.getenv("NUM_CLIENTS", "6"))
    local_epochs        = int(os.getenv("LOCAL_EPOCHS", "1"))
    lr                  = float(os.getenv("LR", "0.01"))
    tau                 = float(os.getenv("TAU", "1.0"))
    epsilon             = float(os.getenv("EPSILON", "1e-3"))
    gamma               = float(os.getenv("GAMMA", "0.20"))
    k_proposers         = int(os.getenv("K_PROPOSERS", "2"))
    round_window_sec    = int(os.getenv("ROUND_WINDOW_SEC", "5"))
    proposal_jitter_sec = float(os.getenv("PROPOSAL_JITTER_SEC", "0.5"))
    finalize_delay_sec  = float(os.getenv("FINALIZE_DELAY_SEC", "1.5"))
    idle_sleep_sec      = float(os.getenv("IDLE_SLEEP_SEC", "2.0"))
    max_rounds          = int(os.getenv("MAX_ROUNDS", "0"))
    client_chunk        = int(os.getenv("CLIENT_CHUNK", str(4 * 1024)))
    global_chunk        = int(os.getenv("GLOBAL_CHUNK", str(4 * 1024)))

    if args.id is None:
        print("Use: python -m src.peer_service --id 0")
        raise SystemExit(2)

    pk_csv = os.getenv("PRIVKEYS_CSV", "").strip()
    privkeys = [s.strip() for s in pk_csv.split(",") if s.strip()]
    if privkeys and len(privkeys) > args.id:
        privkey = privkeys[args.id]
    else:
        privkey = os.getenv("PRIVKEY")
    if not (coord_addr and store_addr and coord_v2 and privkey):
        raise RuntimeError("Set CONTRACT_ADDRESS, FLSTORAGE_ADDRESS, COORDINATOR_V2_ADDRESS and key(s)")

    run_peer(
        client_id=args.id,
        privkey=privkey,
        rpc_url=rpc_url,
        coord_addr=coord_addr,
        store_addr=store_addr,
        coord_v2_addr=coord_v2,
        num_clients=num_clients,
        local_epochs=local_epochs,
        lr=lr,
        tau=tau,
        epsilon=epsilon,
        gamma=gamma,
        k_proposers=k_proposers,
        round_window_sec=round_window_sec,
        proposal_jitter_sec=proposal_jitter_sec,
        finalize_delay_sec=finalize_delay_sec,
        idle_sleep_sec=idle_sleep_sec,
        max_rounds=max_rounds,
        client_chunk=client_chunk,
        global_chunk=global_chunk,
    )


if __name__ == "__main__":
    main()
