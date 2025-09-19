# src/node_agent.py
from __future__ import annotations
import os, time, json, hashlib
from typing import Dict, List, Optional

import numpy as np
import torch
from dotenv import load_dotenv

from .model import create_model
from .utils import (
    DEVICE, load_validation_loader, accuracy,
    params_to_numpy, numpy_to_params,
    load_partition_for_client
)
from .storage_chain import FLStorageChain, pack_params_float32, unpack_params_float32
from .onchain_v2 import FLChainV2


# ---- Global HALT marker ----
HALT_NS_ID = 9_999_999
HALT_BYTES = b"HALT"


def hash_params_rounded(params: List[np.ndarray], decimals: int = 6) -> str:
    h = hashlib.sha256()
    for a in params:
        arr = np.asarray(a, dtype=np.float32)
        arr = np.round(arr, decimals=decimals)
        h.update(arr.tobytes(order="C"))
    return h.hexdigest()


def deterministic_aggregators(round_id: int, num_clients: int, k: int) -> List[int]:
    """Fallback aggregator set: stable random choice based on round_id."""
    rng = np.random.default_rng(seed=int(round_id))
    all_ids = np.arange(num_clients, dtype=int)
    k = max(1, min(int(k), int(num_clients)))
    if k >= num_clients:
        return all_ids.tolist()
    sel = rng.choice(all_ids, size=k, replace=False)
    return sorted(int(x) for x in sel)


class NodeAgent:
    def __init__(self, cid: int):
        load_dotenv(dotenv_path=".env")

        # identity & env
        self.cid = int(cid)
        self.rpc_url      = os.getenv("RPC_URL", "http://127.0.0.1:8545")
        self.coord_addr   = os.getenv("CONTRACT_ADDRESS_V2") or os.getenv("COORDINATOR_V2_ADDRESS")
        self.storage_addr = os.getenv("FLSTORAGE_ADDRESS")
        self.priv         = os.getenv(f"PRIVKEY_{self.cid}") or os.getenv("PRIVKEY")
        if not (self.coord_addr and self.storage_addr and self.priv):
            raise RuntimeError("Missing CONTRACT_ADDRESS_V2/COORDINATOR_V2_ADDRESS, FLSTORAGE_ADDRESS or PRIVKEY(_i)")

        # timing
        self.commit_window_s   = int(float(os.getenv("ROUND_COMMIT_WINDOW_SEC", "60")))
        self.propose_window_s  = int(float(os.getenv("PROPOSE_WINDOW_SEC", "20")))
        self.no_commit_grace_s = float(os.getenv("GRACE_NO_COMMIT_SEC", "10"))
        self.finalize_buffer_s = float(os.getenv("FINALIZE_BUFFER_SEC", "3"))
        self.finalize_wait_s   = int(os.getenv("FINALIZE_WAIT_SEC", "120"))

        # FL hyperparams
        self.local_epochs = int(os.getenv("LOCAL_EPOCHS", "3"))
        self.lr           = float(os.getenv("LR", "0.03"))
        self.batch_size   = int(os.getenv("BATCH_SIZE", "32"))
        self.epsilon      = float(os.getenv("EPSILON", "0.001"))
        self.tau          = float(os.getenv("TAU", "0.0"))
        self.weight_decay = float(os.getenv("WEIGHT_DECAY", "0.0005"))
        self.weight_cap   = float(os.getenv("WEIGHT_CAP", "1.0"))

        # rounds / chunk sizes
        self.max_rounds       = int(os.getenv("MAX_ROUNDS", "50"))
        self.client_chunk_sz  = int(os.getenv("CLIENT_CHUNK", "4096"))
        self.global_chunk_sz  = int(os.getenv("GLOBAL_CHUNK", "4096"))

        # topology (for fallback aggregator selection)
        self.num_clients  = int(os.getenv("NUM_CLIENTS", "6"))
        self.k_aggs = int(os.getenv("K_PROPOSERS", os.getenv("K_AGGREGATORS", "2")))

        # on-chain & storage
        self.chain = FLChainV2(self.rpc_url, self.coord_addr, self.priv)
        self.store = FLStorageChain(self.rpc_url, self.storage_addr, self.priv)

        # data loaders
        self.trainloader = load_partition_for_client(
            cid=self.cid,
            num_clients=self.num_clients,
            batch_size=self.batch_size,
            non_iid=bool(int(os.getenv("NON_IID", "0"))),
            labels_per_client=int(os.getenv("LABELS_PER_CLIENT", "2")),
        )
        self.valloader = load_validation_loader()

        # models
        self.model = create_model().to(DEVICE)
        self.global_model = create_model().to(DEVICE)
        self.template = params_to_numpy(self.global_model)

        # round begin timestamps
        self._round_begin_ts: Dict[int, float] = {}

        # baseline bootstrap (must exist on-chain)
        set_, _, rid, wid, _ = self.chain.get_baseline()
        if not set_:
            raise RuntimeError("Baseline not assigned on-chain.")
        blob = self.store.download_blob(rid, wid, chunk_size=self.global_chunk_sz)
        base_params = unpack_params_float32(blob, self.template)
        numpy_to_params(self.global_model, base_params)
        numpy_to_params(self.model, base_params)
        self.template = params_to_numpy(self.global_model)

        # Aggregators compute convergence; initialized per-round after sync
        self.prev_val_acc: Optional[float] = None

        print(f"[Node {self.cid}] bootstrapped baseline", flush=True)

        # expected client sizes
        sizes_env = os.getenv("CLIENT_SIZES_JSON", "")
        self.client_sizes = {i: 1000 for i in range(self.num_clients)}
        if sizes_env:
            self.client_sizes.update({int(k): int(v) for k, v in json.loads(sizes_env).items()})

    # ----------------- utils -----------------
    def _is_halted(self) -> bool:
        try:
            blob = self.store.download_blob(0, HALT_NS_ID, chunk_size=64)
            return bool(blob == HALT_BYTES)
        except Exception:
            return False

    def _get_round(self, rid: int):
        try:
            return self.chain.get_round(rid)
        except Exception:
            return False, False, 0, 0, 0, 0

    def _note_round_begun(self, rid: int):
        if rid not in self._round_begin_ts:
            self._round_begin_ts[rid] = time.time()

    def _first_open_round(self, start_hint: int = 1) -> int:
        r = max(1, int(start_hint))
        while True:
            begun, finalized, *_ = self._get_round(r)
            if not begun and not finalized:
                return r
            if begun and not finalized:
                return r
            r += 1

    def _sleep_until_propose_close(self, rid: int):
        begin_ts = self._round_begin_ts.get(rid, time.time())
        deadline = begin_ts + self.commit_window_s + self.propose_window_s
        now = time.time()
        if now < deadline:
            time.sleep(deadline - now + self.finalize_buffer_s)

    def _is_finalized(self, rid: int) -> bool:
        _, finalized, *_ = self._get_round(rid)
        return bool(finalized)

    def _is_leader(self, rid: int, agg_set: List[int]) -> bool:
        leader = min(agg_set) if agg_set else 0
        return self.cid == leader

    def _pull_finalized_blocking(self, rid: int, timeout_s: int = 300) -> bool:
        """
        Block until round rid is finalized and pull its global. Returns True on success.
        This guarantees every node starts next round from the same global.
        """
        if rid <= 0:
            return True
        deadline = time.time() + timeout_s
        while time.time() < deadline and not self._is_halted():
            begun, finalized, _h, writer_id, _n, _ = self._get_round(rid)
            if finalized and writer_id:
                try:
                    blob = self.store.download_blob(rid, writer_id, chunk_size=self.global_chunk_sz)
                    if blob:
                        new_params = unpack_params_float32(blob, self.template)
                        numpy_to_params(self.global_model, new_params)
                        numpy_to_params(self.model, new_params)
                        self.template = params_to_numpy(self.global_model)
                        return True
                except Exception as e:
                    print(f"[Node {self.cid}] pull r={rid} failed: {e}", flush=True)
            time.sleep(1.0)
        print(f"[Node {self.cid}] pull r={rid} timed out.", flush=True)
        return False

    # ----------------- training -----------------
    def _train_local(self, rid: int, start_hash: str):
        optimizer = torch.optim.SGD(self.model.parameters(), lr=self.lr, momentum=0.9, weight_decay=self.weight_decay)
        criterion = torch.nn.CrossEntropyLoss()
        self.model.train()
        for _ in range(self.local_epochs):
            for (x, y) in self.trainloader:
                x, y = x.to(DEVICE), y.to(DEVICE)
                optimizer.zero_grad()
                loss = criterion(self.model(x), y)
                loss.backward()
                optimizer.step()
        after_hash = hash_params_rounded(params_to_numpy(self.model))[:12]
        print(f"[Node {self.cid}] r={rid} trained hash {start_hash} -> {after_hash}", flush=True)

    def _upload_and_commit(self, rid: int):
        if self._is_finalized(rid) or self._is_halted():
            return
        params = params_to_numpy(self.model)
        blob = pack_params_float32(params)
        self.store.upload_blob(rid, self.cid, blob, chunk_size=self.client_chunk_sz)
        h = hash_params_rounded(params)
        try:
            self.chain.post_commit(rid, self.cid, h)
            print(f"[Node {self.cid}] r={rid} committed {h[:12]}..", flush=True)
        except Exception as e:
            print(f"[Node {self.cid}] commit failed: {e}", flush=True)

    # ----------------- aggregation -----------------
    def _download_client_params(self, rid: int, cids: List[int]):
        out = {}
        for k in cids:
            try:
                blob = self.store.download_blob(rid, k, chunk_size=self.client_chunk_sz)
                if blob:
                    out[k] = unpack_params_float32(blob, self.template)
            except Exception:
                pass
        return out

    def _wait_for_commits(self, rid: int, expected_clients: List[int]) -> List[int]:
        deadline = time.time() + self.commit_window_s
        committed = []
        while time.time() < deadline:
            if self._is_halted():
                return []
            for k in expected_clients:
                if k in committed:
                    continue
                hh = self.chain.get_client_commit(rid, k)
                if hh and int(hh, 16) != 0:
                    committed.append(k)
            if len(committed) == len(expected_clients):
                break
            time.sleep(0.5)
        return committed

    def _finalize_as_leader(self, rid: int, params: List[np.ndarray], h: str, used_agg: bool):
        writer_id = 1_000_000 + self.cid
        blob = pack_params_float32(params)
        self.store.upload_blob(rid, writer_id, blob, chunk_size=self.global_chunk_sz)
        self._sleep_until_propose_close(rid)
        if self._is_finalized(rid) or self._is_halted():
            return
        self.chain.finalize(rid, self.cid, h, writer_id, 0)
        print(f"[Node {self.cid}] r={rid} FINALIZE leader=True used={'AGGREGATED' if used_agg else 'KEEP'} writer={writer_id}", flush=True)

    # ----------------- main loop -----------------
    def run_loop(self, start_round_id: int):
        rid_hint = int(start_round_id)
        rounds_done = 0

        while rounds_done < self.max_rounds:
            if self._is_halted():
                print(f"[Node {self.cid}] HALT detected — exiting.", flush=True)
                return

            # --- STRONG SYNC: ensure we pulled the last finalized global (r-1) before starting r
            r_prev = (rid_hint - 1) if rid_hint > 1 else 0
            if r_prev > 0:
                ok = self._pull_finalized_blocking(r_prev, timeout_s=self.finalize_wait_s)
                if not ok:
                    # If previous round never finalized within window, still proceed but warn
                    print(f"[Node {self.cid}] WARN: proceeding without r={r_prev} pull.", flush=True)

            rid = self._first_open_round(rid_hint)
            rounds_done += 1

            # --- aggregator set: chain if available, else deterministic fallback
            try:
                agg_set = self.chain.get_aggregators(rid)  # may not exist
                if not isinstance(agg_set, list):
                    agg_set = []
            except AttributeError:
                agg_set = deterministic_aggregators(rid, self.num_clients, self.k_aggs)
            except Exception:
                agg_set = deterministic_aggregators(rid, self.num_clients, self.k_aggs)

            is_agg = self.cid in agg_set
            is_leader = self._is_leader(rid, agg_set)

            # ensure round begun (leader only)
            begun, finalized, *_ = self._get_round(rid)
            if not begun and not finalized and is_leader:
                now = int(time.time())
                try:
                    self.chain.begin_round(
                        rid,
                        now + self.commit_window_s,
                        now + self.commit_window_s + self.propose_window_s,
                        agg_set
                    )
                    self._note_round_begun(rid)
                    print(f"[Node {self.cid}] began round {rid} (agg_set={agg_set})", flush=True)
                except Exception as e:
                    if "Round already begun" not in str(e):
                        print(f"[Node {self.cid}] begin_round skipped: {e}", flush=True)
            else:
                if begun:
                    self._note_round_begun(rid)

            role_str = f"{'AGG' if is_agg else 'CLIENT'} leader={is_leader}"
            print(f"\n=== Round {rid} (Node {self.cid}) role={role_str} ===", flush=True)

            # ---- unify A_prev for aggregators at round start (after sync/pull of r-1) ----
            if is_agg:
                self.prev_val_acc = accuracy(self.global_model, self.valloader)
                print(f"[Node {self.cid}] (agg) round-start acc={self.prev_val_acc:.4f}", flush=True)

            # client training & commit (all nodes)
            start_hash = hash_params_rounded(params_to_numpy(self.model))[:12]
            self._train_local(rid, start_hash)
            self._upload_and_commit(rid)

            # ---------- AGGREGATOR-ONLY path ----------
            if is_agg:
                committed = self._wait_for_commits(rid, list(self.client_sizes.keys()))
                print(f"[Node {self.cid}] r={rid} commits={committed}", flush=True)

                if committed:
                    cparams = self._download_client_params(rid, committed)
                    if cparams:
                        # Weighted FedAvg by |D_i|
                        aggregated = [np.zeros_like(p, dtype=np.float32) for p in self.template]
                        total = float(sum(self.client_sizes.get(i, 1) for i in committed)) or 1.0
                        for cid in committed:
                            w = self.client_sizes.get(cid, 1) / total
                            for j in range(len(aggregated)):
                                aggregated[j] += cparams[cid][j].astype(np.float32) * w

                        # Evaluate on shared V (AGGREGATORS ONLY)
                        tmp = create_model().to(DEVICE)
                        numpy_to_params(tmp, aggregated)
                        new_acc = accuracy(tmp, self.valloader)

                        hh = hash_params_rounded(aggregated)
                        try:
                            self.chain.submit_proposal(rid, self.cid, hh)
                        except Exception as e:
                            print(f"[Node {self.cid}] submit_proposal skipped: {e}", flush=True)

                        # Convergence against same A_prev for all aggs
                        base = self.prev_val_acc if self.prev_val_acc is not None else new_acc
                        delta = abs(new_acc - base)
                        status = "CONVERGED" if delta < self.epsilon else "NOT_CONVERGED"
                        print(f"[Node {self.cid}] (agg) Δ={delta:.6f} ε={self.epsilon} → {status} | A_new={new_acc:.4f}", flush=True)

                        # Leader uploads & finalizes (single writer)
                        if is_leader:
                            self._finalize_as_leader(rid, aggregated, hh, used_agg=True)

                        # If converged, leader writes HALT
                        if delta < self.epsilon:
                            if is_leader:
                                try:
                                    self.store.upload_blob(0, HALT_NS_ID, HALT_BYTES, chunk_size=64)
                                    print(f"[Node {self.cid}] Converged — wrote HALT marker.", flush=True)
                                except Exception as e:
                                    print(f"[Node {self.cid}] HALT write failed: {e}", flush=True)
                            return
                elif is_leader:
                    # No commits: keep current global, but still single-writer finalize
                    keep = params_to_numpy(self.global_model)
                    hh = hash_params_rounded(keep)
                    self._finalize_as_leader(rid, keep, hh, used_agg=False)

            # ---------- ALL NODES: strictly pull the finalized global for r ----------
            ok = self._pull_finalized_blocking(rid, timeout_s=self.finalize_wait_s)
            if not ok:
                print(f"[Node {self.cid}] WARN: no finalized global pulled for r={rid}.", flush=True)
            else:
                if is_agg:
                    # Aggregators refresh A_prev to finalized model for next round
                    new_acc = accuracy(self.global_model, self.valloader)
                    base = self.prev_val_acc if self.prev_val_acc is not None else new_acc
                    delta = abs(new_acc - base)
                    print(f"[Node {self.cid}] (agg) pulled r={rid} | Δ={delta:.6f} | acc={new_acc:.4f}", flush=True)
                    self.prev_val_acc = new_acc
                else:
                    print(f"[Node {self.cid}] pulled r={rid} (updated local models).", flush=True)

            rid_hint = rid + 1

        print(f"[Node {self.cid}] Max rounds reached — exiting.", flush=True)


if __name__ == "__main__":
    load_dotenv(dotenv_path=".env")
    cid = int(os.getenv("CID", "0"))
    start_r = int(os.getenv("START_ROUND_ID", "1"))
    NodeAgent(cid).run_loop(start_round_id=start_r)
