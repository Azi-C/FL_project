# src/aggregator_agent.py
from __future__ import annotations
import os, json, time, hashlib
import numpy as np
import torch
from typing import Dict, List
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
    def __init__(self, agg_id: int):
        load_dotenv()
        self.agg_id = int(agg_id)

        # Env config
        self.rpc_url      = os.getenv("RPC_URL", "http://127.0.0.1:8545")
        self.coord_addr   = os.getenv("CONTRACT_ADDRESS_V2")
        self.storage_addr = os.getenv("FLSTORAGE_ADDRESS")
        self.priv         = os.getenv(f"PRIVKEY_{self.agg_id}") or os.getenv("PRIVKEY")

        if not (self.coord_addr and self.storage_addr and self.priv):
            raise RuntimeError("Missing CONTRACT_ADDRESS_V2 / FLSTORAGE_ADDRESS / PRIVKEY_*")

        # Timing/config knobs
        self.commit_window_s  = int(float(os.getenv("ROUND_COMMIT_WINDOW_SEC", "30")))
        self.propose_window_s = int(float(os.getenv("PROPOSE_WINDOW_SEC", "30")))
        self.client_chunk_sz  = int(os.getenv("CLIENT_CHUNK", "4096"))
        self.global_chunk_sz  = int(os.getenv("GLOBAL_CHUNK", "4096"))
        self.epsilon          = float(os.getenv("EPSILON", "0.001"))
        self.tau              = float(os.getenv("TAU", "1.0"))
        self.max_rounds       = int(os.getenv("MAX_ROUNDS", "20"))

        # Chain + storage
        self.chain = FLChainV2(self.rpc_url, self.coord_addr, self.priv)
        self.store = FLStorageChain(self.rpc_url, self.storage_addr, self.priv)

        # Validation dataset + model template
        self.valloader = load_validation_loader()
        self.global_model = create_model().to(DEVICE)
        self.template = params_to_numpy(self.global_model)

        # Bootstrap from baseline (authoritative pointer)
        set_, h_hex, rid, wid, _ = self.chain.get_baseline()
        if not set_:
            raise RuntimeError("Baseline not assigned on-chain.")
        blob = self.store.download_blob(rid, wid, chunk_size=self.global_chunk_sz)
        base_params = unpack_params_float32(blob, self.template)
        numpy_to_params(self.global_model, base_params)
        self.template = base_params
        self.prev_val_acc = accuracy(self.global_model, self.valloader)

    def _begin_round_if_needed(self, round_id: int) -> None:
        begun, finalized, *_ = self.chain.get_round(round_id)
        if begun:
            return
        now = int(time.time())
        try:
            self.chain.begin_round(
                round_id,
                now + self.commit_window_s,            # commit deadline (informational)
                now + self.commit_window_s + self.propose_window_s,  # propose deadline
                []                                     # elected set optional
            )
            print(f"[Agg {self.agg_id}] began round {round_id}")
        except Exception as e:
            # If another aggregator began it first, ignore
            print(f"[Agg {self.agg_id}] begin_round skipped: {e}")

    def _wait_for_commits_or_timeout(self, round_id: int, expected_clients: List[int]) -> List[int]:
        """Poll on-chain commit map until all expected clients have committed OR commit window has elapsed."""
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
                print(f"[Agg {self.agg_id}] all {len(expected_clients)} commits received.")
                return committed
            if time.time() >= deadline:
                print(f"[Agg {self.agg_id}] commit window elapsed. commits={sorted(list(seen))}")
                return committed
            time.sleep(0.5)

    def _download_client_params(self, round_id: int, cids: List[int]) -> Dict[int, List[np.ndarray]]:
        out: Dict[int, List[np.ndarray]] = {}
        for cid in cids:
            try:
                blob = self.store.download_blob(round_id, cid, chunk_size=self.client_chunk_sz)
                if not blob:
                    continue
                params = unpack_params_float32(blob, self.template)
                out[cid] = params
            except Exception:
                pass
        return out

    def _finalize_and_publish(self, round_id: int, winner_params: List[np.ndarray], winner_hash: str) -> None:
        writer_id = 1_000_000 + self.agg_id
        blob = pack_params_float32(winner_params)
        n_chunks, _ = self.store.upload_blob(round_id, writer_id, blob, chunk_size=self.global_chunk_sz)
        self.chain.finalize(round_id, self.agg_id, winner_hash, writer_id, n_chunks)
        print(f"[Agg {self.agg_id}] finalized r={round_id} writer={writer_id} chunks={n_chunks}")

    def run_until_convergence(self, start_round_id: int, client_sizes: Dict[int, int]) -> None:
        """Main loop: rounds continue until Δ < ε or max_rounds hit."""
        current_params = self.template
        rid = int(start_round_id)
        rounds_done = 0

        while rounds_done < self.max_rounds:
            rounds_done += 1
            print(f"\n=== Round {rid} (Agg {self.agg_id}) ===")
            # 1) Start the round if needed
            self._begin_round_if_needed(rid)

            # 2) Wait for commits or timeout
            expected_clients = list(client_sizes.keys())
            committed_cids = self._wait_for_commits_or_timeout(rid, expected_clients)
            if not committed_cids:
                print(f"[Agg {self.agg_id}] no commits; skipping round {rid}.")
                rid += 1
                continue

            # 3) Pull committed updates from storage
            client_params = self._download_client_params(rid, committed_cids)
            if not client_params:
                print(f"[Agg {self.agg_id}] no params downloadable; skipping round {rid}.")
                rid += 1
                continue

            # 4) Validation gate vs current global
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
                # If none passed, take best single client
                best_c, best_acc = None, -1.0
                for cid, params in client_params.items():
                    tmp = create_model().to(DEVICE)
                    numpy_to_params(tmp, params)
                    ai = accuracy(tmp, self.valloader)
                    if ai > best_acc:
                        best_acc, best_c = ai, cid
                if best_c is not None:
                    valid_ids = [best_c]

            print(f"[Agg {self.agg_id}] valid={sorted(valid_ids)} failed={sorted(failed_ids)}")

            # 5) Aggregate valid updates
            aggregated = fedavg_weighted(client_params, client_sizes, valid_ids, current_params)

            # 6) Evaluate candidate and submit proposal
            tmp = create_model().to(DEVICE)
            numpy_to_params(tmp, aggregated)
            val_acc = accuracy(tmp, self.valloader)
            prop_hash = hash_params_rounded(aggregated)
            self.chain.submit_proposal(rid, self.agg_id, prop_hash)
            print(f"[Agg {self.agg_id}] proposal hash={prop_hash[:12]}.. A={val_acc:.4f}")

            # 7) Finalize & publish winner pointer (demo: single-aggregator or self-finalize)
            self._finalize_and_publish(rid, aggregated, prop_hash)

            # 8) Update local global model to winner; check convergence
            numpy_to_params(self.global_model, aggregated)
            current_params = params_to_numpy(self.global_model)
            delta = abs(val_acc - self.prev_val_acc)
            print(f"[Agg {self.agg_id}] Δ={delta:.6f}, prev={self.prev_val_acc:.4f}, new={val_acc:.4f}")
            self.prev_val_acc = val_acc

            if delta < self.epsilon:
                print(f"[Agg {self.agg_id}] Converged (Δ < ε={self.epsilon}). Stopping.")
                break

            # 9) Next round
            rid += 1
