# storage_chain.py
from web3 import Web3
import json
import math
import numpy as np
from typing import List

def to_checksum(addr: str) -> str:
    return Web3.to_checksum_address(addr)

class FLStorageChain:
    def __init__(self, rpc_url: str, contract_address: str, privkey: str):
        self.w3 = Web3(Web3.HTTPProvider(rpc_url))
        assert self.w3.is_connected(), "Web3 not connected (is node running?)"

        with open("artifacts/contracts/FLStorage.sol/FLStorage.json") as f:
            abi = json.load(f)["abi"]

        self.contract = self.w3.eth.contract(
            address=to_checksum(contract_address),
            abi=abi
        )
        self.account = self.w3.eth.account.from_key(privkey)

    def _tx(self, fn, *args):
        """Build, estimate gas, sign, and send a state-changing tx."""
        tx = fn(*args).build_transaction({
            "from": self.account.address,
            "nonce": self.w3.eth.get_transaction_count(self.account.address),
            "gasPrice": self.w3.to_wei("1", "gwei"),
        })
        gas_est = self.w3.eth.estimate_gas(tx)
        tx["gas"] = int(gas_est * 1.25)  # headroom
        signed = self.account.sign_transaction(tx)
        h = self.w3.eth.send_raw_transaction(signed.raw_transaction)
        return self.w3.eth.wait_for_transaction_receipt(h)

    # --- Rolling hash: keccak(prev || idx || chunk) ---
    @staticmethod
    def rolling_hash(chunks: list[bytes]) -> bytes:
        h = b"\x00" * 32
        for i, ch in enumerate(chunks):
            h = Web3.keccak(h + i.to_bytes(4, "big") + ch)
        return h

    def begin_blob(self, round_id: int, agg_id: int, size: int, total_chunks: int, expected_hash: bytes):
        return self._tx(self.contract.functions.beginBlob, round_id, agg_id, size, total_chunks, expected_hash)

    def put_chunk(self, round_id: int, agg_id: int, idx: int, data: bytes):
        return self._tx(self.contract.functions.putChunk, round_id, agg_id, idx, data)

    def finalize_blob(self, round_id: int, agg_id: int):
        return self._tx(self.contract.functions.finalizeBlob, round_id, agg_id)

    def get_meta(self, round_id: int, agg_id: int):
        return self.contract.functions.blobMeta(round_id, agg_id).call()

    def get_chunk(self, round_id: int, agg_id: int, idx: int) -> bytes:
        return self.contract.functions.getChunk(round_id, agg_id, idx).call()

    # --- Convenience helpers ---
    def upload_blob(self, round_id: int, agg_id: int, blob: bytes, chunk_size: int = 4 * 1024):
        """Upload a blob in small chunks (default 4KB) to reduce per-tx gas."""
        n = math.ceil(len(blob) / chunk_size)
        parts = [blob[i * chunk_size:(i + 1) * chunk_size] for i in range(n)]
        expected = self.rolling_hash(parts)

        self.begin_blob(round_id, agg_id, len(blob), n, expected)

        for idx, part in enumerate(parts):
            self.put_chunk(round_id, agg_id, idx, part)

        self.finalize_blob(round_id, agg_id)
        meta = self.get_meta(round_id, agg_id)
        assert bool(meta[5]) is True, "blob not finalized"
        return n, expected.hex()

    def download_blob(self, round_id: int, agg_id: int) -> bytes:
        meta = self.get_meta(round_id, agg_id)
        total_chunks = int(meta[1])
        parts = []
        for i in range(total_chunks):
            parts.append(self.get_chunk(round_id, agg_id, i))
        # verify rolling hash
        h = b"\x00" * 32
        for i, ch in enumerate(parts):
            h = Web3.keccak(h + i.to_bytes(4, "big") + ch)
        assert h == meta[3], "rolling hash mismatch"
        return b"".join(parts)


# --- Parameter packing/unpacking helpers (float32) ---

def pack_params_float32(params: list[np.ndarray]) -> bytes:
    """Flatten and pack a list of float32 arrays into bytes."""
    return b"".join([np.asarray(p, dtype=np.float32).tobytes(order="C") for p in params])

def unpack_params_float32(blob: bytes, template_params: List[np.ndarray]) -> List[np.ndarray]:
    """Unpack bytes into a list of arrays matching template shapes/dtypes (float32)."""
    params: List[np.ndarray] = []
    offset = 0
    for t in template_params:
        arr = np.asarray(t, dtype=np.float32)
        nbytes = arr.size * 4  # float32
        chunk = blob[offset: offset + nbytes]
        assert len(chunk) == nbytes, "blob too short for template shape"
        p = np.frombuffer(chunk, dtype=np.float32).reshape(arr.shape).copy()
        params.append(p)
        offset += nbytes
    assert offset == len(blob), "blob has extra bytes"
    return params
