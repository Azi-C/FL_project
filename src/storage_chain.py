# src/storage_chain.py
from __future__ import annotations
import json
import time
import threading
from dataclasses import dataclass
from typing import Tuple, Optional, List

from web3 import Web3
from web3.exceptions import TransactionNotFound
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
        chunk = blob[off:off + n_bytes]
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

    # ---------- helpers ----------

    def _latest_block(self):
        return self.w3.eth.get_block("latest")

    def _block_gas_limit(self) -> int:
        return int(self._latest_block()["gasLimit"])

    def _base_fee(self) -> Optional[int]:
        b = self._latest_block()
        return int(b["baseFeePerGas"]) if "baseFeePerGas" in b and b["baseFeePerGas"] is not None else None

    def _supports_eip1559(self) -> bool:
        return self._base_fee() is not None

    def _next_pending_nonce(self) -> int:
        return self.w3.eth.get_transaction_count(self.sender, "pending")

    def _wait_receipt(self, tx_hash, timeout: float = 180.0, poll: float = 0.2):
        start = time.time()
        while True:
            try:
                return self.w3.eth.get_transaction_receipt(tx_hash)
            except TransactionNotFound:
                if time.time() - start > timeout:
                    raise TimeoutError(f"Timed out waiting for transaction receipt: {tx_hash.hex()}")
                time.sleep(poll)

    def _build_fee_fields(self, attempt: int):
        import random
        base = self._base_fee()
        if base is not None:
            # EIP-1559
            tip = 1_000_000_000 + random.randint(0, 200_000_000)  # 1.0–1.2 gwei
            max_fee = base * 2 + tip  # ample headroom for local chains
            return {"maxFeePerGas": int(max_fee), "maxPriorityFeePerGas": int(tip)}
        else:
            # legacy fallback with bump + jitter
            node_suggest = int(self.w3.eth.gas_price) or 1_000_000_000
            bumped = int(node_suggest * (1.2 ** attempt))
            jitter = random.randint(0, 50_000_000)
            return {"gasPrice": int(max(node_suggest, bumped + jitter))}

    def _send_tx(
        self,
        fn,
        *args,
        gas: Optional[int] = None,
        value: int = 0,
        retries: int = 12,
        base_backoff_s: float = 0.25,
        safety_buffer: float = 1.15,  # add 15% over estimate
        reserve_gas: int = 100_000,   # keep some headroom vs block gas limit
    ):
        """
        Adaptive, robust sender:
          • estimates gas via eth_estimateGas
          • caps to (blockGasLimit - reserve_gas); errors with clear message if exceeded
          • uses EIP-1559 fees when supported; legacy otherwise
          • pending nonce + lock + retries
        """
        last_err = None
        for attempt in range(retries):
            try:
                with _TX_LOCK:
                    nonce = self._next_pending_nonce()
                    fee_fields = self._build_fee_fields(attempt)

                    # First, estimate gas
                    est_kwargs = {
                        "from": self.sender,
                        "value": value,
                    }
                    # Some clients require explicit gas for estimate; try without first
                    try:
                        est = fn(*args).estimate_gas(est_kwargs)
                    except Exception:
                        # Retry estimate with fee fields included (some nodes check affordability)
                        est = fn(*args).estimate_gas({**est_kwargs, **fee_fields})

                    est = int(est * safety_buffer)
                    block_limit = self._block_gas_limit()
                    hard_cap = max(21_000, block_limit - reserve_gas)

                    if est > hard_cap:
                        raise RuntimeError(
                            f"Chunk too large for current block gas limit: "
                            f"estimated_gas={est} > cap={hard_cap} (blockGasLimit={block_limit}). "
                            f"Reduce CLIENT_CHUNK / GLOBAL_CHUNK size."
                        )

                    tx_kwargs = {
                        "from": self.sender,
                        "nonce": nonce,
                        "chainId": self.chain_id,
                        "value": value,
                        "gas": est if gas is None else min(gas, hard_cap),
                        **fee_fields,
                    }

                    tx = fn(*args).build_transaction(tx_kwargs)
                    signed = self.w3.eth.account.sign_transaction(tx, private_key=self.privkey)
                    tx_hash = self.w3.eth.send_raw_transaction(signed.raw_transaction)

                rcpt = self._wait_receipt(tx_hash)
                if rcpt.status != 1:
                    raise RuntimeError(f"Tx failed: {tx_hash.hex()}")
                return tx_hash
            except Exception as e:
                last_err = e
                time.sleep(base_backoff_s + 0.1 * attempt)
                continue

        raise RuntimeError(f"Failed to send tx after retries: {last_err}")

    # ---------- public API ----------

    def put_chunk(self, round_id: int, writer_id: int, idx: int, data: bytes):
        return self._send_tx(self.contract.functions.putChunk, round_id, writer_id, idx, data)

    def get_chunk(self, round_id: int, writer_id: int, idx: int) -> bytes:
        return self.contract.functions.getChunk(round_id, writer_id, idx).call()

    def upload_blob(
        self,
        round_id: int,
        writer_id: int,
        blob: bytes,
        chunk_size: int = 4096,  # default conservative; override via env in agents if needed
    ) -> Tuple[int, List[str]]:
        tx_hashes: List[str] = []
        n = (len(blob) + chunk_size - 1) // chunk_size
        for i in range(n):
            part = blob[i * chunk_size : (i + 1) * chunk_size]
            h = self.put_chunk(round_id, writer_id, i, part)
            tx_hashes.append(h.hex())
        return n, tx_hashes

    def download_blob(
        self,
        round_id: int,
        writer_id: int,
        chunk_size: int = 4096,
        max_chunks: int = 10_000,
    ) -> bytes:
        """
        Do NOT stop on len(chunk) < chunk_size; only an empty/missing chunk means 'end'.
        """
        parts: List[bytes] = []
        for i in range(max_chunks):
            try:
                chunk = self.get_chunk(round_id, writer_id, i)
            except Exception:
                break
            if not chunk:
                break
            parts.append(chunk)
        return b"".join(parts)
