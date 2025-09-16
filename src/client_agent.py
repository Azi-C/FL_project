# src/client_agent.py
from __future__ import annotations
import os, time, hashlib
import numpy as np
import torch
from dotenv import load_dotenv
from web3 import Web3

from .model import create_model
from .utils import DEVICE, load_partition_for_client, train_one_epoch, params_to_numpy, numpy_to_params
from .storage_chain import FLStorageChain, pack_params_float32, unpack_params_float32
from .onchain_v2 import FLChainV2


def hash_params_rounded(params, decimals=6) -> str:
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
        self.num_clients  = int(os.getenv("NUM_CLIENTS", "6"))
        self.lr           = float(os.getenv("LR", "0.01"))
        self.local_epochs = int(os.getenv("LOCAL_EPOCHS", "1"))

        self.rpc_url      = os.getenv("RPC_URL", "http://127.0.0.1:8545")
        self.coord_addr   = os.getenv("CONTRACT_ADDRESS_V2")
        self.storage_addr = os.getenv("FLSTORAGE_ADDRESS")
        self.priv         = os.getenv(f"PRIVKEY_{self.cid}") or os.getenv("PRIVKEY")
        if not (self.coord_addr and self.storage_addr and self.priv):
            raise RuntimeError("Missing CONTRACT_ADDRESS_V2 / FLSTORAGE_ADDRESS / PRIVKEY_*")

        self.client_chunk_sz = int(os.getenv("CLIENT_CHUNK", "131072"))   # 128 KiB default
        self.global_chunk_sz = int(os.getenv("GLOBAL_CHUNK", "131072"))   # 128 KiB default
        self.max_rounds      = int(os.getenv("MAX_ROUNDS", "20"))
        self.round_wait_s    = float(os.getenv("ROUND_WAIT_SEC", "0.2"))
        self.finalize_wait_s = float(os.getenv("FINALIZE_WAIT_SEC", "60"))

        self.chain = FLChainV2(self.rpc_url, self.coord_addr, self.priv)
        self.store = FLStorageChain(self.rpc_url, self.storage_addr, self.priv)

        # Log which address and contracts this client uses (helps detect duplicate keys / wrong env)
        print(
            f"[Client {self.cid}] addr={Web3.to_checksum_address(self.store.sender)} "
            f"contract={self.coord_addr} storage={self.storage_addr}",
            flush=True
        )

        # Model + data partition
        self.model = create_model().to(DEVICE)
        self.trainloader = load_partition_for_client(
            cid=self.cid, num_clients=self.num_clients, batch_size=32, non_iid=False
        )

        # ----- Bootstrap from on-chain baseline (with size guard) -----
        set_, h_hex, rid, wid, _ = self.chain.get_baseline()
        if not set_:
            raise RuntimeError("Baseline not assigned on-chain.")

        tmpl = params_to_numpy(self.model)
        expected_bytes = sum(int(np.prod(p.shape)) * 4 for p in tmpl)
        blob = self.store.download_blob(rid, wid, chunk_size=self.global_chunk_sz)

        if len(blob) != expected_bytes:
            raise RuntimeError(
                f"[Client {self.cid}] Baseline blob size mismatch: expected {expected_bytes} bytes "
                f"for current model, got {len(blob)}. Pointer round={rid}, writer={wid}. "
                f"Redeploy coordinator and re-run assign_baseline to fix."
            )

        params = unpack_params_float32(blob, tmpl)
        numpy_to_params(self.model, params)
        print(f"[Client {self.cid}] bootstrapped baseline (bytes={len(blob)})", flush=True)

    def _wait_until_round_begun(self, round_id: int, timeout_s: float = 30.0):
        t0 = time.time()
        while True:
            begun, finalized, *_ = self.chain.get_round(round_id)
            if begun:
                return True
            if time.time() - t0 > timeout_s:
                print(f"[Client {self.cid}] round {round_id} not begun after {timeout_s}s — skipping.", flush=True)
                return False
            time.sleep(self.round_wait_s)

    def _wait_for_finalized_and_pull(self, round_id: int, timeout_s: float):
        t0 = time.time()
        while True:
            begun, finalized, winner_hash, win_agg, writer_id, n_chunks = self.chain.get_round(round_id)
            if begun and finalized and writer_id != 0 and n_chunks != 0:
                blob = self.store.download_blob(round_id, writer_id, chunk_size=self.global_chunk_sz)
                if blob:
                    tmpl = params_to_numpy(self.model)
                    # Optional size check to guard corrupted uploads
                    exp_bytes = sum(int(np.prod(p.shape)) * 4 for p in tmpl)
                    if len(blob) != exp_bytes:
                        print(f"[Client {self.cid}] warning: finalized blob size {len(blob)} != expected {exp_bytes}", flush=True)
                    params = unpack_params_float32(blob, tmpl)
                    numpy_to_params(self.model, params)
                    print(f"[Client {self.cid}] pulled new global for r={round_id}", flush=True)
                    return True
            if time.time() - t0 > timeout_s:
                print(f"[Client {self.cid}] round {round_id} not finalized after {timeout_s}s — moving on.", flush=True)
                return False
            time.sleep(self.round_wait_s)

    def _local_train_once(self):
        opt = torch.optim.SGD(self.model.parameters(), lr=self.lr, momentum=0.9)
        crit = torch.nn.CrossEntropyLoss()
        for _ in range(self.local_epochs):
            train_one_epoch(self.model, self.trainloader, crit, opt)

    def _upload_and_commit(self, round_id: int):
        params = params_to_numpy(self.model)
        blob = pack_params_float32(params)
        n_chunks, _ = self.store.upload_blob(round_id, self.cid, blob, chunk_size=self.client_chunk_sz)

        upd_hash = hash_params_rounded(params)
        # Retry committing to tolerate transient fee/nonce issues
        tries = 6
        for t in range(tries):
            try:
                self.chain.post_commit(round_id, self.cid, upd_hash)
                print(f"[Client {self.cid}] r={round_id} uploaded {n_chunks} chunk(s), commit={upd_hash[:12]}..", flush=True)
                return
            except Exception as e:
                print(f"[Client {self.cid}] post_commit attempt {t+1}/{tries} failed: {e}", flush=True)
                time.sleep(0.5 + 0.2 * t)
        raise RuntimeError(f"[Client {self.cid}] post_commit failed after {tries} attempts")

    def run_loop(self, start_round_id: int):
        rid = int(start_round_id)
        rounds_done = 0
        while rounds_done < self.max_rounds:
            rounds_done += 1

            # 1) Wait for round to be open; if not, skip to next
            if not self._wait_until_round_begun(rid):
                rid += 1
                continue

            # 2) Train locally for this round
            self._local_train_once()

            # 3) Upload & commit (with retries)
            self._upload_and_commit(rid)

            # 4) Wait for finalize (don’t crash if it never happens)
            self._wait_for_finalized_and_pull(rid, timeout_s=self.finalize_wait_s)

            # 5) Next round
            rid += 1


if __name__ == "__main__":
    load_dotenv(dotenv_path=".env")
    rid = int(os.getenv("START_ROUND_ID", "1"))
    cid = int(os.getenv("CID", "0"))
    agent = ClientAgent(cid)
    agent.run_loop(start_round_id=rid)
