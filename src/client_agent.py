# src/client_agent.py
from __future__ import annotations
import os, time, hashlib
import numpy as np
import torch
from dotenv import load_dotenv

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
        load_dotenv()
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

        self.client_chunk_sz = int(os.getenv("CLIENT_CHUNK", "4096"))
        self.global_chunk_sz = int(os.getenv("GLOBAL_CHUNK", "4096"))
        self.max_rounds      = int(os.getenv("MAX_ROUNDS", "20"))
        self.round_wait_s    = float(os.getenv("ROUND_WAIT_SEC", "0.2"))

        self.chain = FLChainV2(self.rpc_url, self.coord_addr, self.priv)
        self.store = FLStorageChain(self.rpc_url, self.storage_addr, self.priv)

        # Model + partition
        self.model = create_model().to(DEVICE)
        self.trainloader = load_partition_for_client(
            cid=self.cid, num_clients=self.num_clients, batch_size=32, non_iid=False
        )

        # Bootstrap baseline
        set_, h_hex, rid, wid, _ = self.chain.get_baseline()
        if not set_:
            raise RuntimeError("Baseline not assigned on-chain.")
        blob = self.store.download_blob(rid, wid, chunk_size=self.global_chunk_sz)
        params = unpack_params_float32(blob, params_to_numpy(self.model))
        numpy_to_params(self.model, params)

    def _wait_until_round_begun(self, round_id: int, timeout_s: float = 30.0) -> None:
        t0 = time.time()
        while True:
            begun, finalized, *_ = self.chain.get_round(round_id)
            if begun:
                return
            if time.time() - t0 > timeout_s:
                raise RuntimeError(f"Round {round_id} not begun after {timeout_s}s")
            time.sleep(self.round_wait_s)

    def _wait_for_finalized_and_pull(self, round_id: int, timeout_s: float = 60.0) -> None:
        """After submitting, wait for finalize to pull the new global model."""
        t0 = time.time()
        while True:
            begun, finalized, winner_hash, win_agg, writer_id, n_chunks = self.chain.get_round(round_id)
            if begun and finalized and writer_id != 0 and n_chunks != 0:
                blob = self.store.download_blob(round_id, writer_id, chunk_size=self.global_chunk_sz)
                if blob:
                    params = unpack_params_float32(blob, params_to_numpy(self.model))
                    numpy_to_params(self.model, params)
                    return
            if time.time() - t0 > timeout_s:
                raise RuntimeError(f"Round {round_id} not finalized after {timeout_s}s")
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
        self.chain.post_commit(round_id, self.cid, upd_hash)
        print(f"[Client {self.cid}] r={round_id} uploaded {n_chunks} chunk(s), commit={upd_hash[:12]}..")

    def run_loop(self, start_round_id: int):
        rid = int(start_round_id)
        rounds_done = 0
        while rounds_done < self.max_rounds:
            rounds_done += 1
            # 1) Wait for round begin
            self._wait_until_round_begun(rid)
            # 2) Train locally for this round
            self._local_train_once()
            # 3) Upload & commit
            self._upload_and_commit(rid)
            # 4) Wait for finalize → pull global model for next round
            self._wait_for_finalized_and_pull(rid)
            # 5) Next round id
            rid += 1


if __name__ == "__main__":
    import sys
    rid = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    cid = int(os.getenv("CID", "0"))
    agent = ClientAgent(cid)
    agent.run_loop(start_round_id=rid)
