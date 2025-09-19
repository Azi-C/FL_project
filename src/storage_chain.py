# src/storage_chain.py
from __future__ import annotations

import os
import json
import time
import random
import threading
from dataclasses import dataclass
from typing import Tuple, Optional, List

from web3 import Web3
from web3.exceptions import TransactionNotFound, Web3RPCError
from eth_account import Account


# ---------- Numpy pack/unpack ----------
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
        n = int(np.prod(shape))
        n_bytes = n * 4
        chunk = blob[off:off + n_bytes]
        if len(chunk) != n_bytes:
            raise ValueError(
                f"unpack_params_float32: slice size {len(chunk)} != expected {n_bytes} for shape {shape}"
            )
        arr = np.frombuffer(chunk, dtype=np.float32).reshape(shape)
        out.append(arr.copy())
        off += n_bytes
    return out


# ---------- Per-process TX lock ----------
_TX_LOCK = threading.Lock()


@dataclass
class FLStorageChain:
    rpc_url: str
    contract_address: str
    privkey: str

    def __post_init__(self):
        self.w3 = Web3(Web3.HTTPProvider(self.rpc_url))
        if not self.w3.is_connected():
            raise RuntimeError(f"Cannot connect to RPC at {self.rpc_url}")

        with open("artifacts/contracts/FLStorage.sol/FLStorage.json", "r") as f:
            j = json.load(f)
        self.abi = j["abi"]
        self.address = Web3.to_checksum_address(self.contract_address)
        self.contract = self.w3.eth.contract(address=self.address, abi=self.abi)

        self.account = Account.from_key(self.privkey)
        self.sender = self.account.address
        self.chain_id = self.w3.eth.chain_id

        # verbosity / tuning
        self.quiet_nonce = bool(int(os.getenv("QUIET_NONCE_ERRORS", "1")))
        self.debug_gas   = bool(int(os.getenv("DEBUG_GAS_LOGS", "0")))

        # default fee knobs
        self.tip_default   = int(os.getenv("MAX_PRIORITY_FEE_WEI", str(self.w3.to_wei("1", "gwei"))))
        self.maxfee_cap    = int(os.getenv("MAX_FEE_WEI", str(self.w3.to_wei("200", "gwei"))))
        self.gas_limit_def = int(os.getenv("STORAGE_TX_GAS_LIMIT", "4000000"))

        # how much to bump on each retry (in basis points)
        # 1250 = +12.5% (meets/ exceeds most nodes' min bump)
        self.replace_bump_bps = int(os.getenv("GAS_REPLACEMENT_BUMP_BPS", "1250"))

    # ---------- internals ----------
    def _wait_receipt(self, tx_hash, timeout=120.0, poll=0.3):
        start = time.time()
        while True:
            try:
                return self.w3.eth.get_transaction_receipt(tx_hash)
            except TransactionNotFound:
                if time.time() - start > timeout:
                    raise TimeoutError("Timed out waiting for transaction receipt")
                time.sleep(poll)

    def _latest_base_fee(self) -> int:
        try:
            blk = self.w3.eth.get_block("latest")
            b = blk.get("baseFeePerGas")
            if b is None:
                return int(self.w3.eth.gas_price)
            return int(b)
        except Exception:
            return int(self.w3.eth.gas_price)

    def _gas_params_initial(self) -> tuple[int, int]:
        base = self._latest_base_fee()
        tip = self.tip_default
        # headroom for spikes; will be capped by maxfee_cap
        maxfee = min(self.maxfee_cap, base * 2 + tip)
        if self.debug_gas:
            print(f"[storage] gas init base={base} tip={tip} maxFee={maxfee}", flush=True)
        return tip, maxfee

    def _bump_fees(self, tip: int, maxfee: int, attempt: int) -> tuple[int, int]:
        # multiplicative bump per attempt
        mult = (10_000 + self.replace_bump_bps * attempt) / 10_000.0
        base = self._latest_base_fee()
        tip2 = max(int(tip * mult), self.tip_default + int(1e6))  # keep nudging tip upward
        maxfee2 = max(int(maxfee * mult), base * 2 + tip2)
        maxfee2 = min(maxfee2, self.maxfee_cap)
        if self.debug_gas:
            print(f"[storage] bump attempt={attempt} -> tip={tip2} maxFee={maxfee2} (base={base})", flush=True)
        return tip2, maxfee2

    def _send_tx(
        self,
        fn,
        *args,
        value: int = 0,
        gas: Optional[int] = None,
        retries: int = 8,
        base_backoff: float = 0.15,
        backoff_jitter: float = 0.10,
    ):
        last_err = None
        tip, maxfee = self._gas_params_initial()

        for attempt in range(retries):
            try:
                with _TX_LOCK:
                    nonce = self.w3.eth.get_transaction_count(self.sender, "pending")
                    tx = fn(*args).build_transaction({
                        "from": self.sender,
                        "nonce": nonce,
                        "chainId": self.chain_id,
                        "value": int(value),
                        "gas": int(gas or self.gas_limit_def),
                        "maxPriorityFeePerGas": tip,
                        "maxFeePerGas": maxfee,
                    })
                    signed = self.w3.eth.account.sign_transaction(tx, private_key=self.privkey)
                    txh = self.w3.eth.send_raw_transaction(signed.raw_transaction)

                rcpt = self._wait_receipt(txh)
                if int(rcpt.get("status", 0)) == 1:
                    return txh

                last_err = {"message": "Receipt status != 1", "txHash": txh.hex()}

            except Web3RPCError as e:
                msg = (e.args[0].get("message") if e.args and isinstance(e.args[0], dict) else str(e))
                last_err = {"message": msg}

                # nonce race / replacement rules → bump fees and retry
                if ("Nonce too low" in msg) or ("replacement transaction underpriced" in msg):
                    if not self.quiet_nonce:
                        print(f"[storage] {_short(self.sender)}: {msg} (attempt {attempt+1}/{retries})", flush=True)
                    # bump fees for *next* attempt
                    tip, maxfee = self._bump_fees(tip, maxfee, attempt + 1)
                else:
                    if self.debug_gas:
                        print(f"[storage] RPC error: {msg}", flush=True)

            except Exception as e:
                last_err = {"message": str(e)}

            # backoff with jitter
            time.sleep(min(base_backoff * (1.6 ** attempt) + random.uniform(0.0, backoff_jitter), 2.5))

        raise RuntimeError(f"Failed to send tx after retries: {last_err}")

    # ---------- public ABI wrappers ----------
    def put_chunk(self, round_id: int, writer_id: int, idx: int, data: bytes):
        return self._send_tx(self.contract.functions.putChunk, int(round_id), int(writer_id), int(idx), data)

    def get_chunk(self, round_id: int, writer_id: int, idx: int) -> bytes:
        return self.contract.functions.getChunk(int(round_id), int(writer_id), int(idx)).call()

    def upload_blob(self, round_id: int, writer_id: int, blob: bytes, chunk_size: int = 4096) -> Tuple[int, List[str]]:
        if chunk_size <= 0:
            raise ValueError("chunk_size must be > 0")
        n = (len(blob) + chunk_size - 1) // chunk_size
        tx_hashes: List[str] = []
        for i in range(n):
            part = blob[i * chunk_size:(i + 1) * chunk_size]
            txh = self.put_chunk(round_id, writer_id, i, part)
            tx_hashes.append(txh.hex() if txh is not None else "")
        return n, tx_hashes

    def download_blob(self, round_id: int, writer_id: int, chunk_size: int = 4096, max_chunks: int = 10_000) -> bytes:
        parts: List[bytes] = []
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


# ---------- util ----------
def _short(addr: str) -> str:
    try:
        s = Web3.to_checksum_address(addr)
    except Exception:
        s = addr
    return s[:8] + "…" + s[-6:]
