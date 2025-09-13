# storage_chain.py
from __future__ import annotations
import os
import json
import time
import math
import threading
from dataclasses import dataclass
from typing import Tuple, List

import numpy as np
from web3 import Web3
from web3.exceptions import TransactionNotFound, Web3RPCError
from eth_account import Account

_TX_LOCK = threading.Lock()  # serialize nonces for a single account

def pack_params_float32(params: List[np.ndarray]) -> bytes:
    # Concatenate as float32 with simple header (shape/lengths are not stored; use template for unpack)
    parts = [np.asarray(p, dtype=np.float32).tobytes(order="C") for p in params]
    return b"".join(parts)

def unpack_params_float32(blob: bytes, template: List[np.ndarray]) -> List[np.ndarray]:
    out: List[np.ndarray] = []
    offset = 0
    for t in template:
        nbytes = np.asarray(t, dtype=np.float32).nbytes
        chunk = blob[offset:offset + nbytes]
        arr = np.frombuffer(chunk, dtype=np.float32).reshape(t.shape)
        out.append(arr.copy())  # copy to detach from buffer
        offset += nbytes
    if offset != len(blob):
        # We allow extra padding, but warn
        pass
    return out

@dataclass
class FLStorageChain:
    rpc_url: str
    contract_address: str
    privkey: str

    def __post_init__(self):
        self.w3 = Web3(Web3.HTTPProvider(self.rpc_url))
        with open("artifacts/contracts/FLStorage.sol/FLStorage.json") as f:
            j = json.load(f)
        self.abi = j["abi"]
        self.address = Web3.to_checksum_address(self.contract_address)
        self.contract = self.w3.eth.contract(address=self.address, abi=self.abi)

        self.account = Account.from_key(self.privkey)
        self.sender = self.account.address
        self.chain_id = self.w3.eth.chain_id

    def _base_fee_tip(self):
        base = self.w3.eth.gas_price
        prio = int(base * 0.1) or 1
        return base + prio, prio

    def _send_tx(self, fn, *args, gas: int | None = None, value: int = 0, retries: int = 3):
        for attempt in range(retries):
            with _TX_LOCK:
                nonce = self.w3.eth.get_transaction_count(self.sender, "pending")
                max_fee, max_prio = self._base_fee_tip()
                tx = fn(*args).build_transaction({
                    "from": self.sender,
                    "nonce": nonce,
                    "chainId": self.chain_id,
                    "value": value,
                    "maxFeePerGas": max_fee,
                    "maxPriorityFeePerGas": max_prio,
                    "gas": gas or 7_000_000,  # storage writes can be heavier
                })
                signed = self.w3.eth.account.sign_transaction(tx, private_key=self.privkey)
                try:
                    h = self.w3.eth.send_raw_transaction(signed.raw_transaction)
                except Web3RPCError as e:
                    msg = (e.args[0].get("message") if e.args and isinstance(e.args[0], dict) else str(e))
                    if "Nonce too low" in msg or "replacement transaction underpriced" in msg:
                        time.sleep(0.2)
                        continue
                    raise
            receipt = self._wait_receipt(h)
            if receipt.status != 1:
                raise RuntimeError(f"Tx failed: {h.hex()}")
            return h
        raise RuntimeError("Failed to send tx after nonce retries")

    def _wait_receipt(self, tx_hash, timeout=120.0, poll=0.2):
        start = time.time()
        while True:
            try:
                r = self.w3.eth.get_transaction_receipt(tx_hash)
                return r
            except TransactionNotFound:
                if time.time() - start > timeout:
                    raise TimeoutError("Timed out waiting for transaction receipt")
                time.sleep(poll)

    # --- Public API ---

    def put_chunk(self, round_id: int, agg_or_client_id: int, idx: int, data: bytes):
        return self._send_tx(self.contract.functions.putChunk, round_id, agg_or_client_id, idx, data)

    def get_chunk(self, round_id: int, agg_or_client_id: int, idx: int) -> bytes:
        return self.contract.functions.getChunk(round_id, agg_or_client_id, idx).call()

    def upload_blob(self, round_id: int, agg_or_client_id: int, blob: bytes, chunk_size: int = 4096) -> Tuple[int, List[str]]:
        n_chunks = math.ceil(len(blob) / max(1, chunk_size))
        tx_hashes: List[str] = []
        for i in range(n_chunks):
            part = blob[i*chunk_size : (i+1)*chunk_size]
            h = self.put_chunk(round_id, agg_or_client_id, i, part)
            tx_hashes.append(h.hex())
        return n_chunks, tx_hashes

    def download_blob(self, round_id: int, agg_or_client_id: int, chunk_size: int | None = None) -> bytes:
        # If contract doesn’t store total chunks, call sequentially until empty chunk is returned.
        # Assuming FLStorage stores contiguous chunks from 0..N-1 and returns empty bytes for missing.
        parts: List[bytes] = []
        idx = 0
        while True:
            data = self.get_chunk(round_id, agg_or_client_id, idx)
            if not data:
                break
            parts.append(data)
            idx += 1
        return b"".join(parts)
