# src/aggregator_agent.py
from __future__ import annotations
import os, json, time, hashlib, sys
import numpy as np
import torch
from typing import Dict, List
from dotenv import load_dotenv

from .model import create_model
from .utils import DEVICE, load_validation_loader, accuracy, params_to_numpy, numpy_to_params
from .storage_chain import FLStorageChain, unpack_params_float32, pack_params_float32
from .onchain_v2 import FLChainV2


# ---- Global HALT marker (storage-based) ----
HALT_NS_ID = 9_999_999
HALT_BYTES = b"HALT"


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
    Multi-aggregators off-chain, leader finalizes after commit+propose window.
    - Syncs round from chain each loop.
    - Grace wait/recheck when no commits.
    - Non-leaders skip finalize.
    - NO monotonic safeguard: always accept the new aggregated model.
    - Global HALT: when any aggregator converges, write a HALT marker so all stop.
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
        self.commit_window_s   = int(float(os.getenv("ROUND_COMMIT_WINDOW_SEC", "60")))
        self.propose_window_s  = int(float(os.getenv("PROPOSE_WINDOW_SEC", "20")))
        self.no_commit_grace_s = float(os.getenv("GRACE_NO_COMMIT_SEC", "5"))
        self.finalize_buffer_s = float(os.getenv("FINALIZE_BUFFER_SEC", "3"))
        self.client_chunk_sz   = int(os.getenv("CLIENT_CHUNK", "4096"))
        self.global_chunk_sz   = int(os.getenv("GLOBAL_CHUNK", "4096"))
        self.epsilon           = float(os.getenv("EPSILON", "0.001"))
        self.tau               = float(os.getenv("TAU", "1.0"))
        self.max_rounds        = int(os.getenv("MAX_ROUNDS", "20"))

        # Aggregator set config
        self.use_onchain_agglist = int(os.getenv("USE_ONCHAIN_AGG_LIST", "1")) == 1
        self.agg_count_fallback  = int(os.getenv("AGG_COUNT", "3"))
        self.agg_set_json        = os.getenv("AGG_SET_JSON", "")

        # Chain + storage
        self.chain = FLChainV2(self.rpc_url, self.coord_addr, self.priv)
        self.store = FLStorageChain(self.rpc_url, self.storage_addr, self.priv)

        # Validation dataset + model
        self.valloader = load_validation_loader()
        self.global_model = create_model().to(DEVICE)
        self.template = params_to_numpy(self.global_model)

        # Track when rounds begin (local timestamps)
        self._round_begin_ts: Dict[int, float] = {}

        # Bootstrap from baseline
        try:
            set_, h_hex, rid, wid, _ = self.chain.get_baseline()
        except Exception as e:
            print(f"[Agg {self.agg_id}] RPC unavailable during get_baseline: {e}", flush=True)
            sys.exit(1)
        if not set_:
            raise RuntimeError("Baseline not assigned on-chain.")
        blob = self.store.download_blob(rid, wid, chunk_size=self.global_chunk_sz)
        base_params = unpack_params_float32(blob, self.template)
        numpy_to_params(self.global_model, base_params)
        self.template = base_params
        self.prev_val_acc = accuracy(self.global_model, self.valloader)
        print(f"[Agg {self.agg_id}] bootstrapped baseline, prev_val_acc={self.prev_val_acc:.4f}", flush=True)

    # ---------- sync helpers ----------

    def _first_open_round(self, start_hint: int = 1) -> int:
        r = max(1, int(start_hint))
        while True:
            try:
                begun, finalized, *_ = self.chain.get_round(r)
            except Exception as e:
                print(f"[Agg {self.agg_id}] RPC unavailable during get_round({r}): {e}", flush=True)
                sys.exit(1)
            if not begun and not finalized:
                return r
            if begun and not finalized:
                return r
            r += 1

    def _note_round_begun(self, rid: int):
        if rid not in self._round_begin_ts:
            self._round_begin_ts[rid] = time.time()

    def _sleep_until_propose_close(self, rid: int):
        begin_ts = self._round_begin_ts.get(rid, time.time())
        deadline = begin_ts + self.commit_window_s + self.propose_window_s
        now = time.time()
        if now < deadline:
            time.sleep(deadline - now + self.finalize_buffer_s)

    def _get_aggregator_set(self, rid: int) -> List[int]:
        if self.use_onchain_agglist:
            try:
                lst = self.chain.get_aggregators(rid)
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

    def _is_leader(self, rid: int) -> bool:
        A_t = self._get_aggregator_set(rid)
        if not A_t:
            A_t = list(range(self.agg_count_fallback))
        leader_id = min(A_t)
        is_leader = (self.agg_id == leader_id)
        print(f"[Agg {self.agg_id}] round {rid} aggregators={A_t} leader={leader_id} -> leader? {is_leader}", flush=True)
        return is_leader

    def _is_finalized(self, rid: int) -> bool:
        try:
            begun, finalized, *_ = self.chain.get_round(rid)
            return bool(finalized)
        except Exception:
            return False

    def _is_halted(self) -> bool:
        try:
            blob = self.store.download_blob(0, HALT_NS_ID, chunk_size=64)
            return bool(blob == HALT_BYTES)
        except Exception:
            return False

    # ---------- orchestration ----------

    def _begin_round_if_needed(self, rid: int, is_leader: bool):
        if not is_leader:
            try:
                begun, finalized, *_ = self.chain.get_round(rid)
                if begun:
                    self._note_round_begun(rid)
            except Exception:
                pass
            return

        try:
            begun, finalized, *_ = self.chain.get_round(rid)
        except Exception as e:
            print(f"[Agg {self.agg_id}] RPC unavailable during get_round: {e}", flush=True)
            sys.exit(1)

        if begun:
            self._note_round_begun(rid)
            return

        now = int(time.time())
        try:
            self.chain.begin_round(
                rid,
                now + self.commit_window_s,
                now + self.commit_window_s + self.propose_window_s,
                self._get_aggregator_set(rid),
            )
            print(f"[Agg {self.agg_id}] began round {rid}", flush=True)
            self._note_round_begun(rid)
            time.sleep(0.5)
        except Exception as e:
            if "Round already begun" not in str(e):
                print(f"[Agg {self.agg_id}] begin_round skipped: {e}", flush=True)

    def _wait_for_commits_or_timeout(self, rid: int, expected_clients: List[int]) -> List[int]:
        deadline = int(time.time()) + self.commit_window_s
        committed, seen = [], set()
        while True:
            if self._is_halted():
                return []
            try:
                for cid in expected_clients:
                    if cid in seen: continue
                    hh = self.chain.get_client_commit(rid, cid)
                    if hh and int(hh, 16) != 0:
                        seen.add(cid); committed.append(cid)
            except Exception as e:
                print(f"[Agg {self.agg_id}] RPC unavailable during get_client_commit: {e}", flush=True)
                sys.exit(1)
            if len(seen) == len(expected_clients): return committed
            if time.time() >= deadline: return committed
            time.sleep(0.5)

    def _download_client_params(self, rid: int, cids: List[int]):
        out = {}
        for cid in cids:
            try:
                blob = self.store.download_blob(rid, cid, chunk_size=self.client_chunk_sz)
                if not blob: continue
                params = unpack_params_float32(blob, self.template)
                out[cid] = params
            except Exception as e:
                print(f"[Agg {self.agg_id}] download error for cid={cid}: {e}", flush=True)
        return out

    def _upload_and_maybe_finalize(self, rid: int, params, prop_hash: str, is_leader: bool):
        if not self._is_finalized(rid) and not self._is_halted():
            try:
                self.chain.submit_proposal(rid, self.agg_id, prop_hash)
            except Exception as e:
                if "Finalized" not in str(e):
                    print(f"[Agg {self.agg_id}] submit_proposal skipped: {e}", flush=True)

        if not is_leader:
            print(f"[Agg {self.agg_id}] non-leader -> skip upload/finalize", flush=True)
            return

        if self._is_halted():
            print(f"[Agg {self.agg_id}] HALT detected before upload — skipping finalize.", flush=True)
            return

        writer_id = 1_000_000 + self.agg_id
        blob = pack_params_float32(params)
        n_chunks, _ = self.store.upload_blob(rid, writer_id, blob, chunk_size=self.global_chunk_sz)

        self._sleep_until_propose_close(rid)
        if self._is_finalized(rid) or self._is_halted():
            print(f"[Agg {self.agg_id}] already finalized or HALT — skipping finalize.", flush=True)
            return

        try:
            self.chain.finalize(rid, self.agg_id, prop_hash, writer_id, n_chunks)
            print(f"[Agg {self.agg_id}] finalized r={rid} writer={writer_id} chunks={n_chunks}", flush=True)
        except Exception as e:
            print(f"[Agg {self.agg_id}] finalize skipped: {e}", flush=True)

    # ---------- main loop ----------

    def run_until_convergence(self, start_round_id: int, client_sizes: Dict[int, int]):
        current_params = self.template
        rid_hint = int(start_round_id)
        rounds_done = 0

        while rounds_done < self.max_rounds:
            if self._is_halted():
                print(f"[Agg {self.agg_id}] HALT detected — exiting.", flush=True)
                return

            rid = self._first_open_round(rid_hint)
            rounds_done += 1
            print(f"\n=== Round {rid} (Agg {self.agg_id}) ===", flush=True)

            try:
                b, f, *_ = self.chain.get_round(rid)
                if b:
                    self._note_round_begun(rid)
            except Exception:
                pass

            is_leader = self._is_leader(rid)
            self._begin_round_if_needed(rid, is_leader=is_leader)

            expected_clients = list(client_sizes.keys())
            committed_cids = self._wait_for_commits_or_timeout(rid, expected_clients)

            if not committed_cids and not self._is_halted():
                if is_leader:
                    print(f"[Agg {self.agg_id}] no commits; grace wait {self.no_commit_grace_s}s then recheck...", flush=True)
                    time.sleep(self.no_commit_grace_s)
                    old_window = self.commit_window_s
                    self.commit_window_s = int(max(1, self.no_commit_grace_s))
                    committed_cids = self._wait_for_commits_or_timeout(rid, expected_clients)
                    self.commit_window_s = old_window
                else:
                    time.sleep(self.no_commit_grace_s)

                if not committed_cids or self._is_halted():
                    print(f"[Agg {self.agg_id}] still no commits; publishing current global.", flush=True)
                    keep_hash = hash_params_rounded(current_params)
                    self._upload_and_maybe_finalize(rid, current_params, keep_hash, is_leader)
                    if self._is_halted():
                        print(f"[Agg {self.agg_id}] HALT detected after no-commit path — exiting.", flush=True)
                        return
                    rid_hint = rid
                    continue

            client_params = self._download_client_params(rid, committed_cids)
            if not client_params or self._is_halted():
                print(f"[Agg {self.agg_id}] no params downloadable or HALT — publishing current global.", flush=True)
                keep_hash = hash_params_rounded(current_params)
                self._upload_and_maybe_finalize(rid, current_params, keep_hash, is_leader)
                if self._is_halted():
                    print(f"[Agg {self.agg_id}] HALT detected after download path — exiting.", flush=True)
                    return
                rid_hint = rid
                continue

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

            aggregated = fedavg_weighted(client_params, client_sizes, valid_ids, current_params)

            tmp = create_model().to(DEVICE)
            numpy_to_params(tmp, aggregated)
            val_acc = accuracy(tmp, self.valloader)
            prop_hash = hash_params_rounded(aggregated)
            print(f"[Agg {self.agg_id}] proposal hash={prop_hash[:12]}.. A={val_acc:.4f}", flush=True)

            self._upload_and_maybe_finalize(rid, aggregated, prop_hash, is_leader=is_leader)

            numpy_to_params(self.global_model, aggregated)
            current_params = params_to_numpy(self.global_model)
            delta = abs(val_acc - self.prev_val_acc)
            print(f"[Agg {self.agg_id}] Δ={delta:.6f}, prev={self.prev_val_acc:.4f}, new={val_acc:.4f}", flush=True)
            self.prev_val_acc = val_acc

            if delta < self.epsilon:
                print(f"[Agg {self.agg_id}] Converged (Δ < ε={self.epsilon}). Stopping.", flush=True)
                try:
                    self.store.upload_blob(0, HALT_NS_ID, HALT_BYTES, chunk_size=64)
                    print(f"[Agg {self.agg_id}] wrote HALT marker.", flush=True)
                except Exception as e:
                    print(f"[Agg {self.agg_id}] failed to write HALT marker: {e}", flush=True)
                break

            rid_hint = rid


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
