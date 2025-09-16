# src/onchain.py
from __future__ import annotations
from typing import Tuple
import json

from web3 import Web3
from web3.exceptions import ABIFunctionNotFound, ContractLogicError

ART_COORD_PATH = "artifacts/contracts/FLCoordinator.sol/FLCoordinator.json"


class FLChain:
    """
    Wrapper around the FLCoordinator contract.
    Handles baseline assignment, initial reputations, proposals/finalize.
    """

    def __init__(self, rpc_url: str, contract_address: str, privkey: str) -> None:
        self.w3 = Web3(Web3.HTTPProvider(rpc_url))
        if not self.w3.is_connected():
            raise RuntimeError(f"Cannot connect to RPC at {rpc_url}")

        with open(ART_COORD_PATH, "r") as f:
            coord_art = json.load(f)

        self.abi = coord_art["abi"]
        self.contract = self.w3.eth.contract(
            address=Web3.to_checksum_address(contract_address),
            abi=self.abi,
        )

        self.privkey = privkey
        self.acct = self.w3.eth.account.from_key(privkey)
        self.chain_id = self.w3.eth.chain_id

    # ---------------- internals (EIP-1559 strict) ----------------

    def _estimate_gas_eip1559(self, fn, *args) -> int:
        call = {"from": self.acct.address}
        tx_for_est = fn(*args).build_transaction(call)
        est = self.w3.eth.estimate_gas(tx_for_est)
        gas_limit_block = self.w3.eth.get_block("latest").get("gasLimit", 30_000_000)
        gas = min(int(est * 1.2) + 10_000, int(gas_limit_block * 0.9))
        return max(gas, 50_000)

    def _build_tx(self, fn, *args) -> dict:
        nonce = self.w3.eth.get_transaction_count(self.acct.address, "pending")
        gas = self._estimate_gas_eip1559(fn, *args)

        base_fee = self.w3.eth.get_block("latest")["baseFeePerGas"]
        try:
            tip = self.w3.eth.max_priority_fee
        except Exception:
            tip = self.w3.to_wei(1, "gwei")

        tx_args = {
            "from": self.acct.address,
            "nonce": nonce,
            "gas": gas,
            "chainId": self.chain_id,
            "type": 2,
            "maxPriorityFeePerGas": int(tip),
            "maxFeePerGas": int(base_fee + 2 * tip),
        }
        return fn(*args).build_transaction(tx_args)

    def _send(self, tx: dict) -> str:
        signed = self.w3.eth.account.sign_transaction(tx, private_key=self.privkey)
        tx_hash = self.w3.eth.send_raw_transaction(signed.raw_transaction)
        rcpt = self.w3.eth.wait_for_transaction_receipt(tx_hash)
        if rcpt["status"] != 1:
            raise RuntimeError("Transaction failed")
        return tx_hash.hex()

    def _has_function(self, name: str) -> bool:
        for entry in self.abi:
            if entry.get("type") == "function" and entry.get("name") == name:
                return True
        return False

    # ---------------- baseline ----------------

    def assign_baseline(self, hash_hex: str, round_id: int, writer_id: int, num_chunks: int) -> str:
        if hash_hex.startswith(("0x", "0X")):
            hash_hex = hash_hex[2:]
        if len(hash_hex) != 64:
            raise ValueError("baseline hash must be 32 bytes (64 hex chars)")
        b32 = bytes.fromhex(hash_hex)
        fn = self.contract.functions.assignBaseline
        tx = self._build_tx(fn, b32, int(round_id), int(writer_id), int(num_chunks))
        return self._send(tx)

    def get_baseline(self) -> Tuple[bool, str, int, int, int]:
        set_, h, rid, wid, n = self.contract.functions.getBaseline().call()
        return bool(set_), h.hex(), int(rid), int(wid), int(n)

    # ---------------- initial reputations ----------------

    def assign_initial_reputations(self, client_ids: list[int], scaled_reps: list[int]) -> str:
        if len(client_ids) != len(scaled_reps):
            raise ValueError("client_ids and scaled_reps must have same length")
        fn = self.contract.functions.assignInitialReputations
        tx = self._build_tx(fn, client_ids, scaled_reps)
        return self._send(tx)

    def get_initial_reputation(self, client_id: int) -> int:
        return int(self.contract.functions.getInitialReputation(int(client_id)).call())

    # ---------------- proposals/finalization ----------------

    def submit_proposal(self, round_id: int, agg_id: int, hash_hex: str) -> str:
        if hash_hex.startswith(("0x", "0X")):
            hash_hex = hash_hex[2:]
        if len(hash_hex) != 64:
            raise ValueError("hash_hex must be 32 bytes (64 hex chars)")
        b32 = bytes.fromhex(hash_hex)
        fn = self.contract.functions.submitProposal
        tx = self._build_tx(fn, int(round_id), int(agg_id), b32)
        return self._send(tx)

    def finalize(self, round_id: int, total_selected: int) -> str:
        fn = self.contract.functions.finalize
        tx = self._build_tx(fn, int(round_id), int(total_selected))
        return self._send(tx)

    def get_round(self, round_id: int) -> Tuple[bool, str]:
        finalized, h = self.contract.functions.getRound(int(round_id)).call()
        return bool(finalized), h.hex()

    def get_votes(self, round_id: int, hash_hex: str) -> int:
        try:
            if not self._has_function("getVotes"):
                return 0
            if hash_hex.startswith(("0x", "0X")):
                hash_hex = hash_hex[2:]
            if len(hash_hex) != 64:
                return 0
            b32 = bytes.fromhex(hash_hex)
            return int(self.contract.functions.getVotes(int(round_id), b32).call())
        except (ABIFunctionNotFound, ContractLogicError, ValueError):
            return 0
