from __future__ import annotations
import json, os
from typing import List, Tuple
from web3 import Web3
# (no relative imports needed here)


ART_COORD_V2 = "artifacts/contracts/FLCoordinatorV2.sol/FLCoordinatorV2.json"

class FLChainV2:
    def __init__(self, rpc_url: str, contract_address: str, privkey: str):
        self.w3 = Web3(Web3.HTTPProvider(rpc_url))
        if not self.w3.is_connected():
            raise RuntimeError(f"Cannot connect to {rpc_url}")
        with open(ART_COORD_V2, "r") as f:
            j = json.load(f)
        self.abi = j["abi"]
        self.contract = self.w3.eth.contract(address=Web3.to_checksum_address(contract_address), abi=self.abi)
        self.acct = self.w3.eth.account.from_key(privkey)
        self.chain_id = self.w3.eth.chain_id

    def _tx(self, fn, *args, gas: int = 3_000_000):
        nonce = self.w3.eth.get_transaction_count(self.acct.address, "pending")
        base = int(self.w3.eth.gas_price)
        tip  = max(1, int(base * 0.1))
        tx = fn(*args).build_transaction({
            "from": self.acct.address,
            "nonce": nonce,
            "chainId": self.chain_id,
            "maxFeePerGas": base + tip,
            "maxPriorityFeePerGas": tip,
            "gas": gas,
        })
        signed = self.w3.eth.account.sign_transaction(tx, private_key=self.acct.key)
        h = self.w3.eth.send_raw_transaction(signed.raw_transaction)
        rcpt = self.w3.eth.wait_for_transaction_receipt(h)
        if rcpt.status != 1:
            raise RuntimeError("Tx failed")
        return h.hex()

    # ---- baseline / reps ----
    def assign_baseline(self, hash_hex: str, round_id: int, writer_id: int, num_chunks: int):
        hh = bytes.fromhex(hash_hex[2:] if hash_hex.startswith("0x") else hash_hex)
        return self._tx(self.contract.functions.assignBaseline, hh, int(round_id), int(writer_id), int(num_chunks))

    def get_baseline(self):
        return self.contract.functions.getBaseline().call()

    def assign_initial_reputations(self, client_ids: List[int], scaled_reps: List[int]):
        return self._tx(self.contract.functions.assignInitialReputations, [int(x) for x in client_ids], [int(x) for x in scaled_reps])

    def get_initial_reputation(self, cid: int) -> int:
        return int(self.contract.functions.getInitialReputation(int(cid)).call())

    # ---- rounds ----
    def begin_round(self, round_id: int, commit_deadline_ts: int, propose_deadline_ts: int, aggregators: List[int]):
        return self._tx(self.contract.functions.beginRound, int(round_id), int(commit_deadline_ts), int(propose_deadline_ts), [int(a) for a in aggregators])

    def post_commit(self, round_id: int, client_id: int, update_hash_hex: str):
        hh = bytes.fromhex(update_hash_hex[2:] if update_hash_hex.startswith("0x") else update_hash_hex)
        return self._tx(self.contract.functions.postCommit, int(round_id), int(client_id), hh)

    def submit_proposal(self, round_id: int, agg_id: int, proposal_hash_hex: str):
        hh = bytes.fromhex(proposal_hash_hex[2:] if proposal_hash_hex.startswith("0x") else proposal_hash_hex)
        return self._tx(self.contract.functions.submitProposal, int(round_id), int(agg_id), hh)

    def finalize(self, round_id: int, winner_agg_id: int, winner_hash_hex: str, writer_id: int, num_chunks: int):
        hh = bytes.fromhex(winner_hash_hex[2:] if winner_hash_hex.startswith("0x") else winner_hash_hex)
        return self._tx(self.contract.functions.finalize, int(round_id), int(winner_agg_id), hh, int(writer_id), int(num_chunks))

    def get_round(self, round_id: int):
        return self.contract.functions.getRound(int(round_id)).call()

    def get_client_commit(self, round_id: int, client_id: int) -> str:
        b = self.contract.functions.getClientCommit(int(round_id), int(client_id)).call()
        return b.hex()
