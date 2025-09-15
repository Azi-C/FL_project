import json, os
from web3 import Web3
from eth_account import Account

def _load_artifact(path: str):
    with open(path, "r") as f:
        art = json.load(f)
    return art["abi"], art.get("bytecode")

class FLStorageChain:
    def __init__(self, rpc_url: str, contract_address: str, privkey: str):
        self.w3 = Web3(Web3.HTTPProvider(rpc_url))
        self.acct = Account.from_key(privkey)
        self.addr = Web3.to_checksum_address(contract_address)
        abi, _ = _load_artifact(os.path.join("artifacts","contracts","FLStorage.sol","FLStorage.json"))
        self.contract = self.w3.eth.contract(address=self.addr, abi=abi)

    def _tx(self, fn, *args):
        tx = fn(*args).build_transaction({
            "from": self.acct.address,
            "nonce": self.w3.eth.get_transaction_count(self.acct.address),
            "gasPrice": self.w3.eth.gas_price,
        })
        signed = self.acct.sign_transaction(tx)
        h = self.w3.eth.send_raw_transaction(signed.raw_transaction)
        return self.w3.eth.wait_for_transaction_receipt(h)

    def put_chunk(self, round_id, writer_id, idx, data: bytes):
        return self._tx(self.contract.functions.putChunk, round_id, writer_id, idx, data)

    def get_chunk(self, round_id, writer_id, idx):
        return self.contract.functions.getChunk(round_id, writer_id, idx).call()

    def upload_blob(self, round_id, writer_id, blob: bytes, chunk_size=4096):
        for i in range(0, len(blob), chunk_size):
            self.put_chunk(round_id, writer_id, i//chunk_size, blob[i:i+chunk_size])

    def download_blob(self, round_id, writer_id, chunk_size=4096):
        out = bytearray()
        idx = 0
        while True:
            chunk = self.get_chunk(round_id, writer_id, idx)
            if not chunk:
                break
            out.extend(chunk)
            idx += 1
        return bytes(out)

# Utilities
import numpy as np
def pack_params_float32(params):
    return b"".join(np.asarray(p, dtype=np.float32).tobytes(order="C") for p in params)

def unpack_params_float32(blob, template):
    arr = np.frombuffer(blob, dtype=np.float32)
    out, offset = [], 0
    for t in template:
        size = int(np.prod(t.shape))
        vals = arr[offset:offset+size].reshape(t.shape)
        out.append(vals.copy())
        offset += size
    return out
