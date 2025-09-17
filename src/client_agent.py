# src/client_agent.py
from __future__ import annotations
import os, time, hashlib, sys
import numpy as np
import torch
from typing import List, Dict
from dotenv import load_dotenv

from .model import create_model
from .utils import DEVICE, train_one_epoch, params_to_numpy, numpy_to_params, load_partition_for_client
from .storage_chain import FLStorageChain, pack_params_float32
from .onchain_v2 import FLChainV2


def hash_params_rounded(params: List[np.ndarray], decimals: int = 6) -> str:
    h = hashlib.sha256()
    for a in params:
        arr = np.asarray(a, dtype=np.float32)
        arr = np.round(arr, decimals=decimals)
        h.update(arr.tobytes(order="C"))
    return h.hexdigest()


class ClientAgent:
    def __init__(self, cid: int):
        load_dotenv(dotenv_path=".env")
        self.cid = int(cid)

        self.rpc_url      = os.getenv("RPC_URL", "http://127.0.0.1:8545")
        self.coord_addr   = os.getenv("CONTRACT_ADDRESS_V2")
        self.storage_addr = os.getenv("FLSTORAGE_ADDRESS")
        self.priv         = os.getenv(f"PRIVKEY_{self.cid}") or os.getenv("PRIVKEY")
        if not (self.coord_addr and self.storage_addr and self.priv):
            raise RuntimeError("Missing CONTRACT_ADDRESS_V2 / FLSTORAGE_ADDRESS / PRIVKEY_*")

        self.local_epochs    = int(os.getenv("LOCAL_EPOCHS", "1"))
        self.lr              = float(os.getenv("LR", "0.01"))
        self.batch_size      = int(os.getenv("BATCH_SIZE", "32"))
        self.client_chunk_sz = int(os.getenv("CLIENT_CHUNK", "4096"))
        self.finalize_wait_s = int(os.getenv("FINALIZE_WAIT_SEC", "90"))

        self.chain = FLChainV2(self.rpc_url, self.coord_addr, self.priv)
        self.store = FLStorageChain(self.rpc_url, self.storage_addr, self.priv)

        # Local dataset
        self.trainloader = load_partition_for_client(
            cid=self.cid,
            num_clients=int(os.getenv("NUM_CLIENTS", "6")),
            batch_size=self.batch_size,
            non_iid=bool(int(os.getenv("NON_IID", "0"))),
            labels_per_client=int(os.getenv("LABELS_PER_CLIENT", "2")),
        )

        # Local model
        self.model = create_model().to(DEVICE)

        # Bootstrap baseline from chain
        set_, h_hex, rid, wid, n_chunks = self.chain.get_baseline()
        if not set_:
            raise RuntimeError("Baseline not assigned on-chain.")
        blob = self.store.download_blob(rid, wid, chunk_size=self.client_chunk_sz)
        params = pack_params_float32(params_to_numpy(self.model))  # just for shape template
        from .storage_chain import unpack_params_float32
        base_params = unpack_params_float32(blob, params_to_numpy(self.model))
        numpy_to_params(self.model, base_params)
        print(f"[Client {self.cid}] bootstrapped baseline", flush=True)

    # ---------- Helpers ----------

    def _is_finalized(self, round_id: int) -> bool:
        try:
            begun, finalized, *_ = self.chain.get_round(round_id)
            return bool(finalized)
        except Exception as e:
            print(f"[Client {self.cid}] get_round failed while checking finalized: {e}", flush=True)
            return False

    def _train_local(self):
        optimizer = torch.optim.SGD(self.model.parameters(), lr=self.lr, momentum=0.9)
        criterion = torch.nn.CrossEntropyLoss()
        for _ in range(self.local_epochs):
            train_one_epoch(self.model, self.trainloader, criterion, optimizer)

    def _upload_and_commit(self, round_id: int):
        # Skip if already finalized
        if self._is_finalized(round_id):
            print(f"[Client {self.cid}] r={round_id} already finalized — skipping commit.", flush=True)
            return

        params = params_to_numpy(self.model)
        blob = pack_params_float32(params)

        try:
            n_chunks, _ = self.store.upload_blob(round_id, self.cid, blob, chunk_size=self.client_chunk_sz)
        except Exception as e:
            print(f"[Client {self.cid}] upload_blob failed: {e}", flush=True)
            return

        upd_hash = hash_params_rounded(params)
        tries = 6
        for t in range(tries):
            if self._is_finalized(round_id):
                print(f"[Client {self.cid}] r={round_id} finalized before commit attempt {t+1} — aborting.", flush=True)
                return
            try:
                self.chain.post_commit(round_id, self.cid, upd_hash)
                print(f"[Client {self.cid}] r={round_id} uploaded {n_chunks} chunk(s), commit={upd_hash[:12]}..", flush=True)
                return
            except Exception as e:
                msg = str(e)
                if "Finalized" in msg or "finalized" in msg:
                    print(f"[Client {self.cid}] r={round_id} commit rejected (finalized) — aborting retries.", flush=True)
                    return
                print(f"[Client {self.cid}] post_commit attempt {t+1}/{tries} failed: {e}", flush=True)
                time.sleep(0.5 + 0.2 * t)

        print(f"[Client {self.cid}] post_commit failed after {tries} attempts — giving up for r={round_id}.", flush=True)

    def _wait_for_finalized_and_pull(self, round_id: int, timeout_s: int = None):
        if timeout_s is None:
            timeout_s = self.finalize_wait_s
        deadline = time.time() + timeout_s
        while time.time() < deadline:
            begun, finalized, _, writer_id, n_chunks, _ = self.chain.get_round(round_id)
            if finalized:
                blob = self.store.download_blob(round_id, writer_id, chunk_size=self.client_chunk_sz)
                from .storage_chain import unpack_params_float32
                new_params = unpack_params_float32(blob, params_to_numpy(self.model))
                numpy_to_params(self.model, new_params)
                print(f"[Client {self.cid}] pulled global from r={round_id}", flush=True)
                return
            time.sleep(2.0)
        raise RuntimeError(f"Round {round_id} not finalized after {timeout_s}s")

    # ---------- Main loop ----------

    def run_loop(self, start_round_id: int):
        rid_hint = start_round_id
        while True:
            rid = rid_hint
            # 1) Skip if round already finalized
            if self._is_finalized(rid):
                print(f"[Client {self.cid}] r={rid} already finalized — skipping train/commit.", flush=True)
                rid_hint = rid + 1
                continue

            # 2) Train locally
            self._train_local()

            # 3) Upload + commit
            self._upload_and_commit(rid)

            # 4) Wait for finalize & pull new global
            self._wait_for_finalized_and_pull(rid)

            # 5) Move to next round
            rid_hint = rid + 1


if __name__ == "__main__":
    load_dotenv(dotenv_path=".env")
    cid = int(os.getenv("CID", "0"))
    rid = int(os.getenv("START_ROUND_ID", "1"))

    agent = ClientAgent(cid)
    agent.run_loop(start_round_id=rid)
