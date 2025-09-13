from __future__ import annotations
import json
import time
import threading
from dataclasses import dataclass
from typing import Tuple

from web3 import Web3
from web3.exceptions import TransactionNotFound, Web3RPCError
from eth_account import Account

_TX_LOCK = threading.Lock()

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

    def _base_fee_tip(self) -> tuple[int, int]:
        base = int(self.w3.eth.gas_price)
        prio = max(1, int(base * 0.1))
        return base + prio, prio

    def _wait_receipt(self, tx_hash, timeout=60.0, poll=0.2):
        start = time.time()
        while True:
            try:
                return self.w3.eth.get_transaction_receipt(tx_hash)
            except TransactionNotFound:
                if time.time() - start > timeout:
                    raise TimeoutError("Timed out waiting for transaction receipt")
                time.sleep(poll)

    def _send_tx(self, fn, *args, gas: int | None = None, value: int = 0, retries: int = 3):
        for _ in range(retries):
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
                    "gas": gas or 4_000_000,
                })
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

    # -------- public API (matches your coordinator ABI) --------
    def submit_proposal(self, round_id: int, agg_id: int, hash_hex: str):
        return self._send_tx(self.contract.functions.submitProposal, round_id, agg_id, hash_hex)

    def finalize(self, round_id: int, total_selected: int):
        return self._send_tx(self.contract.functions.finalize, round_id, total_selected)

    def get_round(self, round_id: int) -> Tuple[bool, str]:
        finalized, h = self.contract.functions.getRound(round_id).call()
        return bool(finalized), h

    def get_votes(self, round_id: int, hash_hex: str) -> int:
        return int(self.contract.functions.getVotes(round_id, hash_hex).call())
