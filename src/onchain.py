# onchain.py
from __future__ import annotations
import os
import json
import time
import threading
from dataclasses import dataclass
from typing import Tuple

from web3 import Web3
from web3.exceptions import TransactionNotFound, Web3RPCError
from eth_account import Account

_TX_LOCK = threading.Lock()  # serialize nonces for a single account

@dataclass
class FLChain:
    rpc_url: str
    contract_address: str
    privkey: str

    def __post_init__(self):
        self.w3 = Web3(Web3.HTTPProvider(self.rpc_url))
        with open("artifacts/contracts/FLCoordinator.sol/FLCoordinator.json") as f:
            j = json.load(f)
        self.abi = j["abi"]
        self.address = Web3.to_checksum_address(self.contract_address)
        self.contract = self.w3.eth.contract(address=self.address, abi=self.abi)

        self.account = Account.from_key(self.privkey)
        self.sender = self.account.address
        self.chain_id = self.w3.eth.chain_id

    def _base_fee_tip(self):
        # Reasonable defaults for Hardhat
        base = self.w3.eth.gas_price  # works fine on Hardhat
        prio = int(base * 0.1) or 1
        return base + prio, prio

    def _send_tx(self, fn, *args, gas: int | None = None, value: int = 0, retries: int = 3):
        for attempt in range(retries):
            with _TX_LOCK:
                # Always get pending nonce (automining means no queue)
                nonce = self.w3.eth.get_transaction_count(self.sender, "pending")
                max_fee, max_prio = self._base_fee_tip()
                tx = fn(*args).build_transaction({
                    "from": self.sender,
                    "nonce": nonce,
                    "chainId": self.chain_id,
                    "value": value,
                    # Legacy fields are fine on Hardhat; we’ll keep EIP-1559-ish gas too
                    "maxFeePerGas": max_fee,
                    "maxPriorityFeePerGas": max_prio,
                    "gas": gas or 4_000_000,
                })
                signed = self.w3.eth.account.sign_transaction(tx, private_key=self.privkey)
                try:
                    h = self.w3.eth.send_raw_transaction(signed.raw_transaction)
                except Web3RPCError as e:
                    msg = (e.args[0].get("message") if e.args and isinstance(e.args[0], dict) else str(e))
                    if "Nonce too low" in msg or "replacement transaction underpriced" in msg:
                        # Refresh then retry
                        time.sleep(0.2)
                        continue
                    raise

            # Wait outside the lock so others can build
            receipt = self._wait_receipt(h)
            if receipt.status != 1:
                raise RuntimeError(f"Tx failed: {h.hex()}")
            return h
        raise RuntimeError("Failed to send tx after nonce retries")

    def _wait_receipt(self, tx_hash, timeout=60.0, poll=0.2):
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

    def submit_proposal(self, round_id: int, agg_id: int, hash_hex: str):
        # hash_hex should be '0x' + 64 hex chars
        return self._send_tx(self.contract.functions.submitProposal, round_id, agg_id, hash_hex)

    def finalize(self, round_id: int, total_selected: int):
        return self._send_tx(self.contract.functions.finalizeRound, round_id, total_selected)

    def get_round(self, round_id: int) -> Tuple[bool, str]:
        # returns (finalized, winningHash)
        finalized, h = self.contract.functions.getRound(round_id).call()
        return bool(finalized), h

    def get_votes(self, round_id: int, hash_hex: str) -> int:
        return int(self.contract.functions.getVotes(round_id, hash_hex).call())

    # onchain.py (add inside FLChain)
    def begin_round(self, round_id: int):
        # Solidity fn likely named beginRound(uint256)
        return self._send_tx(self.contract.functions.beginRound, round_id)
