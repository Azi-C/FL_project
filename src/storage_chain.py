# src/storage_chain.py
from __future__ import annotations
import json, time, threading
from dataclasses import dataclass
from typing import Tuple, Optional, List
from web3 import Web3
from web3.exceptions import TransactionNotFound, Web3RPCError
from eth_account import Account

_TX_LOCK = threading.Lock()

def pack_params_float32(params: List) -> bytes:
    import numpy as np
    bufs = []
    for p in params:
        arr = np.asarray(p, dtype=np.float32, order="C")
        bufs.append(arr.tobytes(order="C"))
    return b"".join(bufs)

def unpack_params_float32(blob: bytes, template: List) -> List:
    import numpy as np
    out = []
    off = 0
    for p in template:
        shape = np.asarray(p).shape
        n_elems = int(np.prod(shape))
        n_bytes = n_elems * 4  # float32
        chunk = blob[off:off+n_bytes]
        arr = np.frombuffer(chunk, dtype=np.float32).reshape(shape)
        out.append(arr.copy())
        off += n_bytes
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

    # -------- gas helpers (EIP-1559 strict) --------
    def _estimate_gas_eip1559(self, fn, *args) -> int:
        call = {"from": self.sender}
        tx_for_est = fn(*args).build_transaction(call)
        est = self.w3.eth.estimate_gas(tx_for_est)
        gas_limit_block = self.w3.eth.get_block("latest").get("gasLimit", 30_000_000)
        gas = min(int(est * 1.2) + 10_000, int(gas_limit_block * 0.9))
        return max(gas, 50_000)

    def _wait_receipt(self, tx_hash, timeout=60.0, poll=0.2):
        start = time.time()
        while True:
            try:
                return self.w3.eth.get_transaction_receipt(tx_hash)
            except TransactionNotFound:
                if time.time() - start > timeout:
                    raise TimeoutError("Timed out waiting for transaction receipt")
                time.sleep(poll)

    def _send_tx(self, fn, *args, value: int = 0, retries: int = 3):
        """Strict EIP-1559 (type=2) tx with estimated gas and fee caps."""
        for _ in range(retries):
            with _TX_LOCK:
                nonce = self.w3.eth.get_transaction_count(self.sender, "pending")
                gas = self._estimate_gas_eip1559(fn, *args)

                base_fee = self.w3.eth.get_block("latest")["baseFeePerGas"]
                try:
                    tip = self.w3.eth.max_priority_fee
                except Exception:
                    tip = self.w3.to_wei(1, "gwei")

                tx_args = {
                    "from": self.sender,
                    "nonce": nonce,
                    "chainId": self.chain_id,
                    "value": value,
                    "gas": gas,
                    "type": 2,
                    "maxPriorityFeePerGas": int(tip),
                    "maxFeePerGas": int(base_fee + 2 * tip),
                }

                tx = fn(*args).build_transaction(tx_args)
                signed = self.w3.eth.account.sign_transaction(tx, private_key=self.privkey)
                try:
                    h = self.w3.eth.send_raw_transaction(signed.raw_transaction)
                except Web3RPCError as e:
                    msg = (e.args[0].get("message") if e.args and isinstance(e.args[0], dict) else str(e))
                    if ("Nonce too low" in msg) or ("replacement transaction underpriced" in msg):
                        time.sleep(0.2)
                        continue
                    raise
            rcpt = self._wait_receipt(h)
            if rcpt.status != 1:
                raise RuntimeError(f"Tx failed: {h.hex()}")
            return h
        raise RuntimeError("Failed to send tx after nonce retries")

    # -------- public --------
    def put_chunk(self, round_id: int, writer_id: int, idx: int, data: bytes):
        return self._send_tx(self.contract.functions.putChunk, round_id, writer_id, idx, data)

    def get_chunk(self, round_id: int, writer_id: int, idx: int) -> bytes:
        return self.contract.functions.getChunk(round_id, writer_id, idx).call()

    def upload_blob(self, round_id: int, writer_id: int, blob: bytes, chunk_size: int = 4096) -> Tuple[int, List[str]]:
        tx_hashes = []
        n = (len(blob) + chunk_size - 1) // chunk_size
        for i in range(n):
            part = blob[i*chunk_size:(i+1)*chunk_size]
            h = self.put_chunk(round_id, writer_id, i, part)
            tx_hashes.append(h.hex())
        return n, tx_hashes

    def download_blob(self, round_id: int, writer_id: int, chunk_size: int = 4096, max_chunks: int = 10_000) -> bytes:
        parts = []
        for i in range(max_chunks):
            try:
                chunk = self.get_chunk(round_id, writer_id, i)
            except Exception:
                break
            if not chunk:
                break
            parts.append(chunk)
            if len(chunk) < chunk_size:
                break
        return b"".join(parts)
