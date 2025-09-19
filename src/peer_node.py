# src/peer_node.py
import os, time, hashlib, argparse
import numpy as np
from dotenv import load_dotenv

from src.client import Client
from src.model import create_model
from src.utils import DEVICE, load_validation_loader, accuracy, params_to_numpy, numpy_to_params
from src.onchain import FLChainV2, FLStorageChain
from src.storage_utils import pack_params_float32, unpack_params_float32

GLOBAL_NS_OFFSET = 1_000_000
FP = 1_000_000  # fixed-point for accuracy on-chain (if needed)


# ------------------ helpers ------------------

def sha256_params(params, decimals=6):
    import numpy as _np, hashlib as _h
    h = _h.sha256()
    for a in params:
        arr = _np.asarray(a, dtype=_np.float32)
        if decimals is not None:
            arr = _np.round(arr, decimals=decimals)
        h.update(arr.tobytes(order="C"))
    return h.hexdigest()

def to_bytes32(hex_no_0x: str) -> bytes:
    """Ensure a 32-byte value from hex (without 0x)."""
    h = hex_no_0x.lower().lstrip("0x")
    if len(h) != 64:
        # pad or trim defensively
        if len(h) < 64:
            h = h.rjust(64, "0")
        else:
            h = h[:64]
    return bytes.fromhex(h)

def deterministic_proposers(round_id, eligible, k):
    rng = np.random.default_rng(seed=int(round_id))
    k = max(1, min(k, len(eligible)))
    return sorted(int(x) for x in rng.choice(eligible, size=k, replace=False))

def wait_tx_settle(s: float = 0.4):
    time.sleep(s)


# ------------------ main peer loop ------------------

def run_peer(client_id: int, privkey: str, rpc_url: str, coord_v2_addr: str, store_addr: str,
             num_clients: int, local_epochs: int, lr: float, tau: float, epsilon: float,
             k_proposers: int, round_window_sec: int, finalize_delay_sec: float,
             idle_sleep_sec: float, max_rounds: int, client_chunk: int, global_chunk: int):

    print(f"[peer {client_id}] starting …", flush=True)
    chain = FLChainV2(rpc_url, coord_v2_addr, privkey)
    store = FLStorageChain(rpc_url, store_addr, privkey)

    # local client/train
    client = Client(cid=client_id, num_clients=num_clients, lr=lr, local_epochs=local_epochs)
    valloader = load_validation_loader()
    manager_model = create_model().to(DEVICE)  # “global” que le peer utilise pour vérifier/publier
    template = params_to_numpy(manager_model)

    # Registration (|D_i|)
    try:
        dsize = len(client.trainloader.dataset)
    except Exception:
        dsize = 0
    try:
        chain.register_client(dsize)
        print(f"[peer {client_id}] registered |D|={dsize}", flush=True)
        wait_tx_settle()
    except Exception as e:
        print(f"[peer {client_id}] register skipped: {e}", flush=True)

    # Attendre l'initialisation (scripts/init_v2.js ou équiv.)
    for _ in range(60):
        try:
            if chain.contract.functions.initialized().call():
                break
            print(f"[peer {client_id}] waiting init...", flush=True)
        except Exception as e:
            print(f"[peer {client_id}] init check err: {e}", flush=True)
        time.sleep(1.0)

    # Baseline: peer 0 dépose round 0, writer GLOBAL_NS_OFFSET+0
    baseline = template
    if client_id == 0:
        try:
            blob0 = pack_params_float32(baseline)
            store.upload_blob(0, GLOBAL_NS_OFFSET + 0, blob0, chunk_size=global_chunk)
            print(f"[peer {client_id}] baseline uploaded to (round=0, writer={GLOBAL_NS_OFFSET+0})", flush=True)
        except Exception as e:
            print(f"[peer {client_id}] baseline upload skipped: {e}", flush=True)
    # Tous bootstrappent la baseline
    try:
        blob_boot = store.download_blob(0, GLOBAL_NS_OFFSET + 0, chunk_size=global_chunk)
        base_params = unpack_params_float32(blob_boot, template)
    except Exception:
        base_params = baseline
    numpy_to_params(manager_model, base_params)
    numpy_to_params(client.model, base_params)

    prev_acc = accuracy(manager_model, valloader)
    print(f"[peer {client_id}] baseline V-acc={prev_acc:.4f}", flush=True)

    rounds_done = 0
    r = 1
    while True:
        if max_rounds > 0 and rounds_done >= max_rounds:
            print(f"[peer {client_id}] done (max rounds)", flush=True)
            return

        # S’assurer que le round r est “begun”
        try:
            begun, finalized, _ = chain.get_round(r)
        except Exception:
            begun, finalized = False, False

        if not begun:
            # règle simple: peer 0 ouvre
            if client_id == 0:
                try:
                    chain.begin_round(r)
                    print(f"[peer {client_id}] beginRound({r})", flush=True)
                    wait_tx_settle()
                except Exception as e:
                    print(f"[peer {client_id}] beginRound skipped: {e}", flush=True)
            time.sleep(0.5)
        else:
            print(f"[peer {client_id}] round {r} already begun", flush=True)

        # Pull du global finalisé précédent si dispo
        pulled = None
        try:
            if r == 1:
                # round 1 → baseline de r=0
                blob = store.download_blob(0, GLOBAL_NS_OFFSET + 0, chunk_size=global_chunk)
                pulled = unpack_params_float32(blob, template)
            else:
                # essai: global du round r-1 (writer=GLOBAL_NS_OFFSET + leader_cid)
                # si ta logique ne garde pas trace du “winner”, on peut fallback baseline
                pulled = unpack_params_float32(
                    store.download_blob(0, GLOBAL_NS_OFFSET + 0, global_chunk), template
                )
        except Exception as e:
            print(f"[peer {client_id}] pull global fallback: {e}", flush=True)
            pulled = template

        numpy_to_params(manager_model, pulled)
        client.set_params(pulled)

        # -------- Entraînement local --------
        start_hash = sha256_params(params_to_numpy(client.model))[:12]
        client.train_local()
        after_hash = sha256_params(params_to_numpy(client.model))[:12]
        print(f"[peer {client_id}] r={r} trained: {start_hash} → {after_hash}", flush=True)

        # τ-gate / score local (diagnostic)
        base_acc = accuracy(manager_model, valloader)
        tmp_params = client.get_params()
        tmp_model = create_model().to(DEVICE); numpy_to_params(tmp_model, tmp_params)
        acc_i = accuracy(tmp_model, valloader)
        s_part = 1 if (acc_i + 1e-12) >= (base_acc * tau) else -1
        print(f"[peer {client_id}] r={r} acc_i={acc_i:.4f} base={base_acc:.4f} τ={tau} s={s_part}", flush=True)

        # Remontée du score (si ton contrat l’expose)
        try:
            a_i_fp = int(max(0.0, min(1.0, float(acc_i))) * FP)
            chain.submit_eval(r, chain.acct.address, a_i_fp, s_part)
            print(f"[peer {client_id}] submitEval ok", flush=True)
            wait_tx_settle()
        except Exception as e:
            print(f"[peer {client_id}] submitEval err: {e}", flush=True)

        # Upload de la mise à jour locale (poids complets ici)
        try:
            my_blob = pack_params_float32(tmp_params)
            store.upload_blob(r, client.cid, my_blob, chunk_size=client_chunk)
            print(f"[peer {client_id}] uploaded local update r={r}", flush=True)
        except Exception as e:
            print(f"[peer {client_id}] upload local fail: {e}", flush=True)

        # Laisser une fenêtre de commits
        time.sleep(max(0.1, float(round_window_sec)))

        # Collecter les updates pour agrégation
        cand = []
        present = []
        for cid in range(num_clients):
            try:
                b = store.download_blob(r, cid, chunk_size=client_chunk)
                if b:
                    p = unpack_params_float32(b, template)
                    cand.append(p)
                    present.append(cid)
            except Exception:
                pass
        print(f"[peer {client_id}] r={r} collected updates from cids={present}", flush=True)

        if not cand:
            print(f"[peer {client_id}] r={r} no updates found; retry later", flush=True)
            time.sleep(idle_sleep_sec)
            # ne pas incrémenter le round, on réessaie
            continue

        # Sélection des proposeurs déterministe
        proposers = deterministic_proposers(r, list(range(num_clients)), k_proposers)
        print(f"[peer {client_id}] r={r} proposers={proposers}", flush=True)

        # Si ce peer est proposeur → agrège et propose
        if client_id in proposers:
            # FedAvg simple sur poids complets (mouvement fort)
            agg = [np.zeros_like(x, dtype=np.float32) for x in template]
            for params in cand:
                for j in range(len(agg)):
                    agg[j] += params[j].astype(np.float32) / float(len(cand))

            h_hex = sha256_params(agg, 6)  # sans 0x
            print(f"[peer {client_id}] r={r} proposal hash={h_hex[:12]}..", flush=True)

            # ancrage de la proposition (V1-like dans V2)
            try:
                chain._send(chain.contract.functions.submitProposal, int(r), int(client.cid), to_bytes32(h_hex))
                print(f"[peer {client_id}] r={r} submitProposal ok", flush=True)
                wait_tx_settle()
            except Exception as e:
                print(f"[peer {client_id}] r={r} submitProposal err: {e}", flush=True)

            # le premier proposeur finalise après un petit délai
            if client_id == proposers[0]:
                time.sleep(max(0.1, float(finalize_delay_sec)))
                try:
                    chain.finalize(int(r), int(len(proposers)))
                    print(f"[peer {client_id}] r={r} finalize ok", flush=True)
                    wait_tx_settle()
                except Exception as e:
                    print(f"[peer {client_id}] r={r} finalize err: {e}", flush=True)

            # uploader le global proposé (espace writer propre au proposeur)
            try:
                gblob = pack_params_float32(agg)
                store.upload_blob(r, GLOBAL_NS_OFFSET + client.cid, gblob, chunk_size=global_chunk)
                print(f"[peer {client_id}] r={r} uploaded aggregated global (writer={GLOBAL_NS_OFFSET+client.cid})", flush=True)
            except Exception as e:
                print(f"[peer {client_id}] r={r} upload aggregated fail: {e}", flush=True)

            # Convergence (sur V commun)
            numpy_to_params(manager_model, agg)
            new_acc = accuracy(manager_model, valloader)
            delta = abs(new_acc - prev_acc)
            print(f"[peer {client_id}] r={r} ΔV-acc={delta:.6f} (prev={prev_acc:.4f} new={new_acc:.4f}, ε={epsilon})", flush=True)
            prev_acc = new_acc

            if delta < epsilon:
                try:
                    chain.mark_converged(int(r), to_bytes32(h_hex))
                except Exception as e:
                    print(f"[peer {client_id}] mark_converged err: {e}", flush=True)
                print(f"[peer {client_id}] converged — stop.", flush=True)
                return

        rounds_done += 1
        r += 1
        time.sleep(max(0.0, float(idle_sleep_sec)))


# ------------------ CLI ------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--id", type=int, required=True)
    return p.parse_args()

def main():
    load_dotenv(os.getenv("ENV_FILE") or ".env")
    args = parse_args()

    rpc_url   = os.getenv("RPC_URL", "http://127.0.0.1:8545")
    # supporte deux noms d'env
    coord_v2  = os.getenv("COORDINATOR_V2_ADDRESS") or os.getenv("CONTRACT_ADDRESS_V2")
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
        raise RuntimeError("Set COORDINATOR_V2_ADDRESS/CONTRACT_ADDRESS_V2, FLSTORAGE_ADDRESS and keys in .env")

    print(f"[peer {args.id}] env OK: RPC={rpc_url}, COORD={coord_v2}, STORE={storage}", flush=True)

    run_peer(args.id, privkey, rpc_url, coord_v2, storage,
             num_clients, local_epochs, lr, tau, epsilon,
             k_proposers, round_window_sec, finalize_delay,
             idle_sleep, max_rounds, client_chunk, global_chunk)

if __name__ == "__main__":
    main()
