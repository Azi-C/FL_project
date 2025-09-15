import os, time, hashlib, random, argparse
import numpy as np
from dotenv import load_dotenv

from src.client import Client
from src.model import create_model
from src.utils import DEVICE, load_validation_loader, accuracy, params_to_numpy, numpy_to_params
from src.onchain import FLChainV2, FLStorageChain
from src.storage_utils import pack_params_float32, unpack_params_float32

GLOBAL_NS_OFFSET = 1_000_000
FP = 1_000_000

def sha256_params(params, decimals=6):
    import numpy as _np, hashlib as _h
    h = _h.sha256()
    for a in params:
        arr = _np.asarray(a, dtype=_np.float32)
        if decimals is not None:
            arr = _np.round(arr, decimals=decimals)
        h.update(arr.tobytes(order="C"))
    return h.hexdigest()

def deterministic_proposers(round_id, eligible, k):
    rng = np.random.default_rng(seed=round_id)
    k = max(1, min(k, len(eligible)))
    return sorted(int(x) for x in rng.choice(eligible, size=k, replace=False))

def run_peer(client_id: int, privkey: str, rpc_url: str, coord_v2_addr: str, store_addr: str,
             num_clients: int, local_epochs: int, lr: float, tau: float, epsilon: float,
             k_proposers: int, round_window_sec: int, finalize_delay_sec: float,
             idle_sleep_sec: float, max_rounds: int, client_chunk: int, global_chunk: int):

    chain = FLChainV2(rpc_url, coord_v2_addr, privkey)
    store = FLStorageChain(rpc_url, store_addr, privkey)

    client = Client(cid=client_id, num_clients=num_clients, lr=lr, local_epochs=local_epochs)

    valloader = load_validation_loader()
    manager_model = create_model().to(DEVICE)
    template = params_to_numpy(manager_model)

    # Registration (|D_i|)
    try:
        dsize = len(client.trainloader.dataset)
    except Exception:
        dsize = 0
    try:
        chain.register_client(dsize)
        print(f"[peer {client_id}] registered |D|={dsize}")
    except Exception as e:
        print(f"[peer {client_id}] register skipped: {e}")

    # Wait for init (owner must run scripts/init_v2.js once)
    for _ in range(30):
        try:
            if chain.contract.functions.initialized().call(): break
            print(f"[peer {client_id}] waiting init...")
        except Exception: pass
        time.sleep(1)

    # Baseline bootstrap (peer 0 uploads model 0 at round 0)
    baseline = template
    if client_id == 0:
        blob0 = pack_params_float32(baseline)
        store.upload_blob(0, GLOBAL_NS_OFFSET + 0, blob0, chunk_size=global_chunk)
    numpy_to_params(manager_model, baseline)
    numpy_to_params(client.model, baseline)

    prev_acc = accuracy(manager_model, valloader)
    print(f"[peer {client_id}] baseline V-acc={prev_acc:.4f}")

    rounds_done = 0
    r = 1
    while True:
        if max_rounds > 0 and rounds_done >= max_rounds:
            print(f"[peer {client_id}] done (max rounds)"); return

        # Ensure round begun on-chain
        begun, finalized, _ = chain.get_round(r)
        if not begun:
            # deterministic: peer 0 opens it
            if client_id == 0:
                try:
                    chain.begin_round(r)
                    print(f"[peer {client_id}] beginRound({r})")
                except Exception as e:
                    print(f"[peer {client_id}] beginRound skipped: {e}")
            # small wait to let others see it as begun
            time.sleep(1)

        # Pull previous round’s committed global if any; else baseline
        pulled = None
        try:
            # try winner (we use namespace GLOBAL_NS_OFFSET + cid who uploaded)
            # in this simplified flow we just use baseline for r==1
            if r == 1:
                pulled = store.download_blob(0, GLOBAL_NS_OFFSET + 0, chunk_size=global_chunk)
                pulled = unpack_params_float32(pulled, template)
            else:
                # fallback: baseline
                pulled = unpack_params_float32(store.download_blob(0, GLOBAL_NS_OFFSET + 0, global_chunk), template)
        except Exception:
            pulled = template

        numpy_to_params(manager_model, pulled)
        client.set_params(pulled)

        # Local train
        client.train_local()

        # τ-gate / submit eval update to on-chain reputation (your formula)
        base_acc = accuracy(manager_model, valloader)
        tmp_params = client.get_params()
        tmp_model = create_model().to(DEVICE); numpy_to_params(tmp_model, tmp_params)
        acc_i = accuracy(tmp_model, valloader)

        s_part = 1 if (acc_i + 1e-12) >= (base_acc * tau) else -1
        a_i_fp = int(max(0.0, min(1.0, float(acc_i))) * FP)
        try:
            chain.submit_eval(r, chain.acct.address, a_i_fp, s_part)
            print(f"[peer {client_id}] submitEval ok (a={acc_i:.4f}, s={s_part})")
        except Exception as e:
            print(f"[peer {client_id}] submitEval err: {e}")

        # Upload local update to storage
        my_blob = pack_params_float32(tmp_params)
        store.upload_blob(r, client.cid, my_blob, chunk_size=client_chunk)
        print(f"[peer {client_id}] uploaded local update r={r}")

        # Wait window for other uploads
        time.sleep(round_window_sec)

        # Collect updates for aggregation
        cand = []
        for cid in range(num_clients):
            try:
                b = store.download_blob(r, cid, chunk_size=client_chunk)
                p = unpack_params_float32(b, template)
                cand.append(p)
            except Exception:
                pass

        if not cand:
            print(f"[peer {client_id}] no updates found; retry")
            time.sleep(idle_sleep_sec)
            continue

        proposers = deterministic_proposers(r, list(range(num_clients)), k_proposers)
        if client_id in proposers:
            agg = [np.zeros_like(x, dtype=np.float32) for x in template]
            for params in cand:
                for j in range(len(agg)):
                    agg[j] += params[j].astype(np.float32) / float(len(cand))
            h = sha256_params(agg, 6)
            try:
                # we reuse V1-like anchoring calls inside V2
                chain._send(chain.contract.functions.submitProposal, r, client.cid, bytes.fromhex(h))
            except Exception:
                pass

            if client_id == proposers[0]:
                time.sleep(finalize_delay_sec)
                try:
                    chain.finalize(r, len(proposers))
                except Exception:
                    pass

            # Upload proposer’s global to storage namespace
            gblob = pack_params_float32(agg)
            store.upload_blob(r, GLOBAL_NS_OFFSET + client.cid, gblob, chunk_size=global_chunk)

            # Convergence check (shared V)
            numpy_to_params(manager_model, agg)
            new_acc = accuracy(manager_model, valloader)
            delta = abs(new_acc - prev_acc)
            prev_acc = new_acc
            print(f"[peer {client_id}] ΔV-acc={delta:.6f}  (ε={epsilon})")
            if delta < epsilon:
                try:
                    chain.mark_converged(r, bytes.fromhex(h))
                except Exception:
                    pass
                print(f"[peer {client_id}] converged — stop.")
                return

        rounds_done += 1
        r += 1
        time.sleep(idle_sleep_sec)

def parse_args():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--id", type=int, required=True)
    return p.parse_args()

def main():
    load_dotenv(os.getenv("ENV_FILE") or ".env")
    args = parse_args()

    rpc_url   = os.getenv("RPC_URL", "http://127.0.0.1:8545")
    coord_v2  = os.getenv("COORDINATOR_V2_ADDRESS")
    storage   = os.getenv("FLSTORAGE_ADDRESS")
    privkeys  = [s.strip() for s in os.getenv("PRIVKEYS_CSV", "").split(",") if s.strip()]

    num_clients      = int(os.getenv("NUM_CLIENTS", "3"))
    local_epochs     = int(os.getenv("LOCAL_EPOCHS", "1"))
    lr               = float(os.getenv("LR", "0.01"))
    tau              = float(os.getenv("TAU", "1.0"))
    epsilon          = float(os.getenv("EPSILON", "1e-3"))
    k_proposers      = int(os.getenv("K_PROPOSERS", "2"))
    round_window_sec = int(os.getenv("ROUND_WINDOW_SEC", "5"))
    finalize_delay   = float(os.getenv("FINALIZE_DELAY_SEC", "1.5"))
    idle_sleep       = float(os.getenv("IDLE_SLEEP_SEC", "2.0"))
    max_rounds       = int(os.getenv("MAX_ROUNDS", "0"))
    client_chunk     = int(os.getenv("CLIENT_CHUNK", "4096"))
    global_chunk     = int(os.getenv("GLOBAL_CHUNK", "4096"))

    if privkeys and len(privkeys) > args.id:
        privkey = privkeys[args.id]
    else:
        privkey = os.getenv("PRIVKEY")

    if not (coord_v2 and storage and privkey):
        raise RuntimeError("Set COORDINATOR_V2_ADDRESS, FLSTORAGE_ADDRESS and keys in .env")

    run_peer(args.id, privkey, rpc_url, coord_v2, storage,
             num_clients, local_epochs, lr, tau, epsilon,
             k_proposers, round_window_sec, finalize_delay,
             idle_sleep, max_rounds, client_chunk, global_chunk)

if __name__ == "__main__":
    main()
