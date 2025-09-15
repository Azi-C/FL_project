from __future__ import annotations
import json
import os
from typing import Tuple, List, Optional

from web3 import Web3
from eth_account import Account
from eth_account.signers.local import LocalAccount


# ---------- helpers ----------

def _load_artifact(path: str):
    with open(path, "r") as f:
        art = json.load(f)
    return art["abi"], art.get("bytecode")


# ======================================================================
# V1 COORDINATOR WRAPPER (your existing contract with propose/finalize)
# Expects artifacts at:
#   artifacts/contracts/FLCoordinator.sol/FLCoordinator.json
# ======================================================================
class FLChain:
    def __init__(self, rpc_url: str, contract_address: str, privkey: str,
                 abi_path: Optional[str] = None):
        self.w3 = Web3(Web3.HTTPProvider(rpc_url))
        self.acct: LocalAccount = Account.from_key(privkey)
        self.addr = Web3.to_checksum_address(contract_address)

        if abi_path is None:
            abi_path = os.path.join(
                "artifacts", "contracts", "FLCoordinator.sol", "FLCoordinator.json"
            )
        abi, _ = _load_artifact(abi_path)
        self.contract = self.w3.eth.contract(address=self.addr, abi=abi)

    def _send(self, fn, *args):
        tx = fn(*args).build_transaction({
            "from": self.acct.address,
            "nonce": self.w3.eth.get_transaction_count(self.acct.address),
            "gasPrice": self.w3.eth.gas_price,
        })
        signed = self.acct.sign_transaction(tx)
        h = self.w3.eth.send_raw_transaction(signed.raw_transaction)
        rcpt = self.w3.eth.wait_for_transaction_receipt(h)
        return rcpt

    # --- V1 functions you already rely on ---
    def submit_proposal(self, round_id: int, agg_id: int, hash_hex: str):
        # hash_hex should be "0x" + 64 hex chars
        return self._send(self.contract.functions.submitProposal, int(round_id), int(agg_id), hash_hex)

    def finalize(self, round_id: int, total_selected: int):
        return self._send(self.contract.functions.finalize, int(round_id), int(total_selected))

    def get_round(self, round_id: int) -> Tuple[bool, str]:
        # returns (finalized: bool, winnerHash: bytes32)
        finalized, h = self.contract.functions.getRound(int(round_id)).call()
        # convert bytes32 to 0x + hex
        if isinstance(h, (bytes, bytearray)):
            return finalized, "0x" + h.hex()
        return finalized, str(h)


# ======================================================================
# V2 COORDINATOR WRAPPER (registration, reputation, rewards, convergence)
# Expects artifacts at:
#   artifacts/contracts/FLCoordinatorV2.sol/FLCoordinatorV2.json
# ======================================================================
class FLChainV2:
    def __init__(self, rpc_url: str, contract_address: str, privkey: str,
                 abi_path: Optional[str] = None):
        self.w3 = Web3(Web3.HTTPProvider(rpc_url))
        self.acct: LocalAccount = Account.from_key(privkey)
        self.addr = Web3.to_checksum_address(contract_address)

        if abi_path is None:
            abi_path = os.path.join(
                "artifacts", "contracts", "FLCoordinatorV2.sol", "FLCoordinatorV2.json"
            )
        abi, _ = _load_artifact(abi_path)
        self.contract = self.w3.eth.contract(address=self.addr, abi=abi)

    # -------- tx helper --------
    def _send(self, fn, *args):
        tx = fn(*args).build_transaction({
            "from": self.acct.address,
            "nonce": self.w3.eth.get_transaction_count(self.acct.address),
            "gasPrice": self.w3.eth.gas_price,
        })
        signed = self.acct.sign_transaction(tx)
        h = self.w3.eth.send_raw_transaction(signed.raw_transaction)
        rcpt = self.w3.eth.wait_for_transaction_receipt(h)
        return rcpt

    # -------- admin/setup --------
    def set_baseline_hash(self, h: str):
        # h must be 0x + 64 hex (bytes32)
        return self._send(self.contract.functions.setBaselineHash, h)

    def set_validation_hash(self, h: str):
        return self._send(self.contract.functions.setValidationHash, h)

    def close_registration_and_init(self):
        return self._send(self.contract.functions.closeRegistrationAndInit)

    # -------- registration --------
    def register_client(self, data_size: int):
        return self._send(self.contract.functions.registerClient, int(data_size))

    # -------- views --------
    def get_participants(self) -> List[str]:
        return list(self.contract.functions.getParticipants().call())

    def get_client(self, addr: str):
        # returns (registered: bool, dataSize: int, repFp: int)
        return self.contract.functions.getClient(Web3.to_checksum_address(addr)).call()

    # -------- reputation / rewards / convergence --------
    def submit_eval(self, round_id: int, who: str, a_i_fp: int, s_i_part: int):
        return self._send(
            self.contract.functions.submitEval,
            int(round_id),
            Web3.to_checksum_address(who),
            int(a_i_fp),
            int(s_i_part),
        )

    def record_reward(self, round_id: int, who: str, amount_fp: int):
        return self._send(
            self.contract.functions.recordReward,
            int(round_id),
            Web3.to_checksum_address(who),
            int(amount_fp),
        )

    def mark_converged(self, round_id: int, notes_bytes32: bytes | str):
        # Accept bytes (32) or hex string "0x.."
        if isinstance(notes_bytes32, str) and notes_bytes32.startswith("0x"):
            return self._send(self.contract.functions.markConverged, int(round_id), notes_bytes32)
        elif isinstance(notes_bytes32, (bytes, bytearray)):
            return self._send(self.contract.functions.markConverged, int(round_id), "0x" + notes_bytes32.hex())
        else:
            raise ValueError("notes must be bytes32 or 0x-hex string")
