# src/aggregator_agent.py
from __future__ import annotations
import os, json, time, hashlib
import numpy as np
import torch
from typing import Dict, List, Optional
from dotenv import load_dotenv

from .model import create_model
from .utils import DEVICE, load_validation_loader, accuracy, params_to_numpy, numpy_to_params
from .storage_chain import FLStorageChain, unpack_params_float32, pack_params_float32
from .onchain_v2 import FLChainV2


def hash_params_rounded(params, decimals: int = 6) -> str:
    h = hashlib.sha256()
    for a in params:
        arr = np.asarray(a, dtype=np.float32)
        arr = np.round(arr, decimals=decimals)
        h.update(arr.tobytes(order="C"))
    return h.hexdigest()


def fedavg_weighted(client_params: Dict[int, List[np.ndarray]],
                    client_sizes: Dict[int, int],
                    valid_ids: List[int],
                    template: List[np.ndarray]) -> List[np.ndarray]:
    out = [np.zeros_like(p, dtype=np.float32) for p in template]
    total = float(sum(client_sizes[i] for i in valid_ids)) or 1.0
    for cid in valid_ids:
        w = client_sizes[cid] / total
        params = client_params[cid]
        for j in range(len(out)):
            out[j] += (params[j].astype(np.float32) * w)
    return out


class AggregatorAgent:
    """
    M1: multi-agrégateurs off-chain, un seul finalise.
    - Tous observent commits et calculent la proposition.
    - Seul le 'leader' publie le global (upload) + finalize.
    - begin_round / finalize sont idempotents (guards).
    """
    def __init__(self, agg_id: int):
        load_dotenv(dotenv_path=".env")
        self.agg_id = int(agg_id)

        self.rpc_url      = os.getenv("RPC_URL", "http://127.0.0.1:8545")
        self.coord_addr   = os.getenv("CONTRACT_ADDRESS_V2")
        self.storage_addr = os.getenv("FLSTORAGE_ADDRESS")
        self.priv         = os.getenv(f"PRIVKEY_{self.agg_id}") or os.getenv("PRIVKEY")
        if not (self.coord_addr and self.storage_addr and self.priv):
            raise RuntimeError("Missing CONTRACT_ADDRESS_V2 / FLSTORAGE_ADDRESS / PRIVKEY_*")

        # Timing/config
        self.commit_window_s  = int(float(os.getenv("ROUND_COMMIT_WINDOW_SEC", "30")))
        self.propose_window_s = int(float(os.getenv("PROPOSE_WINDOW_SEC", "30")))
        self.client_chunk_sz  = int(os.getenv("CLIENT_CHUNK", "4096"))
        self.global_chunk_sz  = int(os.getenv("GLOBAL_CHUNK", "4096"))
        self.epsilon          = float(os.getenv("EPSILON", "0.001"))
        self.tau              = float(os.getenv("TAU", "1.0"))
        self.max_rounds       = int(os.getenv("MAX_ROUNDS", "20"))

        # Multi-aggregator config
        # 1) On essaie d'utiliser la liste on-chain (si beginRound l'a fournie)
        # 2) Sinon, on tombe sur une liste statique via .env: AGG_SET_JSON ou AGG_COUNT
        self.use_onchain_agglist = int(os.getenv("USE_ONCHAIN_AGG_LIST", "1")) == 1
        self.agg_count_fallback  = int(os.getenv("AGG_COUNT", "3"))
        self.agg_set_json        = os.getenv("AGG_SET_JSON", "")  # ex: "[0,1,2]"

        # Chain + storage
        self.chain = FLChainV2(self.rpc_url, self.coord_addr, self.priv)
        self.store = FLStorageChain(self.rpc_url, self.storage_addr, self.priv)

        # Validation dataset + model
        self.valloader = load_validation_loader()
        self.global_model = create_model().to(DEVICE)
        self.template = params_to_numpy(self.global_model)

        # Bootstrap from baseline
        set_, h_hex, rid, wid, _ = self.chain.get_baseline()
        if not set_:
            raise RuntimeError("Baseline not assigned on-chain.")
        blob = self.store.download_blob(rid, wid, chunk_size=self.global_chunk_sz)
        base_params = unpack_params_float32(blob, self.template)
        numpy_to_params(self.global_model, base_params)
        self.template = base_params
        self.prev_val_acc = accuracy(self.global_model, self.valloader)
        print(f"[Agg {self.agg_id}] bootstrapped baseline, prev_val_acc={self.prev_val_acc:.4f}", flush=True)

    # ---------- helpers: aggregator set & leader election ----------

    def _get_aggregator_set(self, round_id: int) -> List[int]:
        if self.use_onchain_agglist:
            try:
                lst = self.chain.get_aggregators(round_id)  # may be empty if non-paramétré
                if lst and all(isinstance(x, int) for x in lst):
                    return lst
            except Exception:
                pass
        if self.agg_set_json:
            try:
                return [int(x) for x in json.loads(self.agg_set_json)]
            except Exception:
                pass
        return list(range(self.agg_count_fallback))

    def _is_leader(self, round_id: int) -> bool:
        A_t = self._get_aggregator_set(round_id)
        if not A_t:
            # fallback: tout le monde pense que le min est 0..agg_count-1; leader = min(range)
            A_t = list(range(self.agg_count_fallback))
        leader_id = min(A_t)
        is_leader = (self.agg_id == leader_id)
        print(f"[Agg {self.agg_id}] round {round_id} aggregators={A_t} leader={leader_id} -> leader? {is_leader}", flush=True)
        return is_leader

    # ---------- round orchestration (idempotent) ----------

    def _begin_round_if_needed(self, round_id: int):
        begun, finalized, *_ = self.chain.get_round(round_id)
        if begun:
            return
        now = int(time.time())
        try:
            self.chain.begin_round(
                round_id,
                now + self.commit_window_s,
                now + self.commit_window_s + self.propose_window_s,
                self._get_aggregator_set(round_id)
            )
            print(f"[Agg {self.agg_id}] began round {round_id}", flush=True)
            # petit délai pour laisser les clients/agrégateurs observer l'état
            time.sleep(0.5)
        except Exception as e:
            print(f"[Agg {self.agg_id}] begin_round skipped: {e}", flush=True)

    def _wait_for_commits_or_timeout(self, round_id: int, expected_clients: List[int]) -> List[int]:
        deadline = int(time.time()) + self.commit_window_s
        committed: List[int] = []
        seen = set()
        while True:
            for cid in expected_clients:
                if cid in seen:
                    continue
                hh = self.chain.get_client_commit(round_id, cid)
                if hh and int(hh, 16) != 0:
                    seen.add(cid)
                    committed.append(cid)
            if len(seen) == len(expected_clients):
                print(f"[Agg {self.agg_id}] all {len(expected_clients)} commits received.", flush=True)
                return committed
            if time.time() >= deadline:
                print(f"[Agg {self.agg_id}] commit window elapsed. commits={sorted(list(seen))}", flush=True)
                return committed
            time.sleep(0.5)

    def _download_client_params(self, round_id: int, cids: List[int]):
        out = {}
        for cid in cids:
            try:
                blob = self.store.download_blob(round_id, cid, chunk_size=self.client_chunk_sz)
                if not blob:
                    continue
                params = unpack_params_float32(blob, self.template)
                out[cid] = params
            except Exception as e:
                print(f"[Agg {self.agg_id}] download error for cid={cid}: {e}", flush=True)
        return out

    def _upload_and_maybe_finalize(self, round_id: int, params, prop_hash: str, is_leader: bool):
        """
        M1 rule:
        - Tous peuvent submit_proposal (utile pour M3).
        - Seul le leader upload le global et tente finalize (les autres s'abstiennent).
        - Si finalize échoue car déjà finalisé -> on log et on continue.
        """
        # Tous soumettent la proposition (hash)
        try:
            self.chain.submit_proposal(round_id, self.agg_id, prop_hash)
        except Exception as e:
            print(f"[Agg {self.agg_id}] submit_proposal skipped: {e}", flush=True)

        if not is_leader:
            print(f"[Agg {self.agg_id}] non-leader -> skip upload/finalize", flush=True)
            return

        # Leader: upload + finalize
        writer_id = 1_000_000 + self.agg_id
        blob = pack_params_float32(params)
        n_chunks, _ = self.store.upload_blob(round_id, writer_id, blob, chunk_size=self.global_chunk_sz)
        try:
            self.chain.finalize(round_id, self.agg_id, prop_hash, writer_id, n_chunks)
            print(f"[Agg {self.agg_id}] finalized r={round_id} writer={writer_id} chunks={n_chunks}", flush=True)
        except Exception as e:
            print(f"[Agg {self.agg_id}] finalize skipped: {e}", flush=True)

    # ---------- main loop ----------

    def run_until_convergence(self, start_round_id: int, client_sizes: Dict[int, int]):
        current_params = self.template
        rid = int(start_round_id)
        rounds_done = 0

        while rounds_done < self.max_rounds:
            rounds_done += 1
            print(f"\n=== Round {rid} (Agg {self.agg_id}) ===", flush=True)

            # 0) leader election view (log only)
            is_leader = self._is_leader(rid)

            # 1) open round (idempotent across aggregators)
            self._begin_round_if_needed(rid)

            # 2) collect commits
            expected_clients = list(client_sizes.keys())
            committed_cids = self._wait_for_commits_or_timeout(rid, expected_clients)

            if not committed_cids:
                print(f"[Agg {self.agg_id}] no commits; publishing current global to finalize round {rid}.", flush=True)
                prop_hash = hash_params_rounded(current_params)
                self._upload_and_maybe_finalize(rid, current_params, prop_hash, is_leader=is_leader)
                rid += 1
                continue

            # 3) download updates
            client_params = self._download_client_params(rid, committed_cids)
            if not client_params:
                print(f"[Agg {self.agg_id}] no params downloadable; publishing current global to finalize round {rid}.", flush=True)
                prop_hash = hash_params_rounded(current_params)
                self._upload_and_maybe_finalize(rid, current_params, prop_hash, is_leader=is_leader)
                rid += 1
                continue

            # 4) validation gate
            base_acc = self.prev_val_acc
            valid_ids, failed_ids = [], []
            for cid, params in client_params.items():
                tmp = create_model().to(DEVICE)
                numpy_to_params(tmp, params)
                acc_i = accuracy(tmp, self.valloader)
                if acc_i + 1e-12 >= base_acc * self.tau:
                    valid_ids.append(cid)
                else:
                    failed_ids.append(cid)
            if not valid_ids:
                best_c, best_acc = None, -1.0
                for cid, params in client_params.items():
                    tmp = create_model().to(DEVICE)
                    numpy_to_params(tmp, params)
                    ai = accuracy(tmp, self.valloader)
                    if ai > best_acc:
                        best_acc, best_c = ai, cid
                if best_c is not None:
                    valid_ids = [best_c]

            print(f"[Agg {self.agg_id}] valid={sorted(valid_ids)} failed={sorted(failed_ids)}", flush=True)

            # 5) aggregate
            aggregated = fedavg_weighted(client_params, client_sizes, valid_ids, current_params)

            # 6) proposal + (leader) finalize
            tmp = create_model().to(DEVICE)
            numpy_to_params(tmp, aggregated)
            val_acc = accuracy(tmp, self.valloader)
            prop_hash = hash_params_rounded(aggregated)
            print(f"[Agg {self.agg_id}] proposal hash={prop_hash[:12]}.. A={val_acc:.4f}", flush=True)

            self._upload_and_maybe_finalize(rid, aggregated, prop_hash, is_leader=is_leader)

            # 7) local update & convergence check (même si non-leader, pour garder l'état aligné)
            numpy_to_params(self.global_model, aggregated)
            current_params = params_to_numpy(self.global_model)
            delta = abs(val_acc - self.prev_val_acc)
            print(f"[Agg {self.agg_id}] Δ={delta:.6f}, prev={self.prev_val_acc:.4f}, new={val_acc:.4f}", flush=True)
            self.prev_val_acc = val_acc

            if delta < self.epsilon:
                print(f"[Agg {self.agg_id}] Converged (Δ < ε={self.epsilon}). Stopping.", flush=True)
                break

            rid += 1


if __name__ == "__main__":
    load_dotenv(dotenv_path=".env")
    agg_id = int(os.getenv("AGG_ID", "0"))
    n = int(os.getenv("NUM_CLIENTS", "6"))

    sizes_env = os.getenv("CLIENT_SIZES_JSON", "")
    if sizes_env:
        client_sizes = {int(k): int(v) for k, v in json.loads(sizes_env).items()}
    else:
        client_sizes = {i: 1000 for i in range(n)}

    start_round = int(os.getenv("START_ROUND_ID", "1"))

    agent = AggregatorAgent(agg_id)
    agent.run_until_convergence(start_round_id=start_round, client_sizes=client_sizes)
