import json, os
from web3 import Web3
from eth_account import Account
from eth_account.signers.local import LocalAccount

ART_DIR = os.path.join("artifacts", "contracts")

def _load(path: str):
    with open(path, "r") as f:
        j = json.load(f)
    return j["abi"], j.get("bytecode")

class FLChainV2:
    """Python wrapper for FLCoordinatorV2 (registration, eval updates, rounds, anchors)."""
    def __init__(self, rpc_url: str, contract_address: str, privkey: str):
        self.w3 = Web3(Web3.HTTPProvider(rpc_url))
        self.acct: LocalAccount = Account.from_key(privkey)
        self.addr = Web3.to_checksum_address(contract_address)
        abi, _ = _load(os.path.join(ART_DIR, "FLCoordinatorV2.sol", "FLCoordinatorV2.json"))
        self.contract = self.w3.eth.contract(address=self.addr, abi=abi)

    # --- tx helper ---
    def _send(self, fn, *args):
        tx = fn(*args).build_transaction({
            "from": self.acct.address,
            "nonce": self.w3.eth.get_transaction_count(self.acct.address),
            "gasPrice": self.w3.eth.gas_price,
        })
        signed = self.acct.sign_transaction(tx)
        h = self.w3.eth.send_raw_transaction(signed.raw_transaction)
        return self.w3.eth.wait_for_transaction_receipt(h)

    # --- registration / init ---
    def register_client(self, data_size: int):
        return self._send(self.contract.functions.registerClient, int(data_size))

    def close_registration_and_init(self):
        return self._send(self.contract.functions.closeRegistrationAndInit)

    # --- anchors ---
    def set_baseline_hash(self, h_hex: str):
        h = bytes.fromhex(h_hex[2:] if h_hex.startswith("0x") else h_hex)
        return self._send(self.contract.functions.setBaselineHash, Web3.to_bytes(h))

    def set_validation_hash(self, h_hex: str):
        h = bytes.fromhex(h_hex[2:] if h_hex.startswith("0x") else h_hex)
        return self._send(self.contract.functions.setValidationHash, Web3.to_bytes(h))

    # --- rounds ---
    def begin_round(self, round_id: int):
        return self._send(self.contract.functions.beginRound, int(round_id))

    def finalize(self, round_id: int, total_selected: int):
        return self._send(self.contract.functions.finalize, int(round_id), int(total_selected))

    def get_round(self, round_id: int):
        begun, finalized, h = self.contract.functions.getRound(int(round_id)).call()
        return begun, finalized, "0x" + h.hex()

    def mark_converged(self, round_id: int, h_hex: str):
        h = bytes.fromhex(h_hex[2:] if h_hex.startswith("0x") else h_hex)
        return self._send(self.contract.functions.markConverged, int(round_id), Web3.to_bytes(h))

    # --- eval / reputation ---
    def submit_eval(self, round_id: int, who: str, a_i_fp: int, s_i_part: int):
        return self._send(self.contract.functions.submitEval, int(round_id), Web3.to_checksum_address(who), int(a_i_fp), int(s_i_part))

    def get_participants(self):
        return self.contract.functions.getParticipants().call()

    def get_client(self, who: str):
        d, r = self.contract.functions.getClient(Web3.to_checksum_address(who)).call()
        return int(d), int(r)

class FLStorageChain:
    def __init__(self, rpc_url: str, contract_address: str, privkey: str):
        self.w3 = Web3(Web3.HTTPProvider(rpc_url))
        self.acct = Account.from_key(privkey)
        self.addr = Web3.to_checksum_address(contract_address)
        abi, _ = _load(os.path.join(ART_DIR, "FLStorage.sol", "FLStorage.json"))
        self.contract = self.w3.eth.contract(address=self.addr, abi=abi)

    def _send(self, fn, *args):
        tx = fn(*args).build_transaction({
            "from": self.acct.address,
            "nonce": self.w3.eth.get_transaction_count(self.acct.address),
            "gasPrice": self.w3.eth.gas_price,
        })
        signed = self.acct.sign_transaction(tx)
        h = self.w3.eth.send_raw_transaction(signed.raw_transaction)
        return self.w3.eth.wait_for_transaction_receipt(h)

    def put_chunk(self, round_id, writer_id, idx, data: bytes):
        return self._send(self.contract.functions.putChunk, int(round_id), int(writer_id), int(idx), data)

    def get_chunk(self, round_id, writer_id, idx):
        return self.contract.functions.getChunk(int(round_id), int(writer_id), int(idx)).call()

    def upload_blob(self, round_id, writer_id, blob: bytes, chunk_size=4096):
        for i in range(0, len(blob), chunk_size):
            self.put_chunk(round_id, writer_id, i // chunk_size, blob[i:i+chunk_size])

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
