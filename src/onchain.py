from web3 import Web3
import json
import os

class FLChain:
    def __init__(self, rpc_url="http://127.0.0.1:8545", contract_address=None, privkey=None):
        self.w3 = Web3(Web3.HTTPProvider(rpc_url))
        assert self.w3.is_connected(), "Web3 not connected (is Hardhat node running?)"

        # Load ABI from Hardhat artifacts
        with open("artifacts/contracts/FLCoordinator.sol/FLCoordinator.json") as f:
            artifact = json.load(f)
        abi = artifact["abi"]

        if not contract_address:
            raise ValueError("Provide deployed contract address")
        self.contract = self.w3.eth.contract(
            address=Web3.to_checksum_address(contract_address),
            abi=abi
        )

        if not privkey:
            raise ValueError("Provide a private key from Hardhat node output")
        self.account = self.w3.eth.account.from_key(privkey)

    # ---- internal helper to send a state-changing tx ----
    def _tx(self, fn, *args):
        tx = fn(*args).build_transaction({
            "from": self.account.address,
            "nonce": self.w3.eth.get_transaction_count(self.account.address),
            "gas": 5_000_000,
            "gasPrice": self.w3.to_wei("1", "gwei"),
        })
        signed = self.account.sign_transaction(tx)
        h = self.w3.eth.send_raw_transaction(signed.raw_transaction)
        return self.w3.eth.wait_for_transaction_receipt(h)

    # ---- contract wrappers ----
    def submit_proposal(self, round_id: int, agg_id: int, hash_hex: str):
        # accept with or without 0x
        if not hash_hex.startswith(("0x", "0X")):
            hash_hex = "0x" + hash_hex
        b = Web3.to_bytes(hexstr=hash_hex)
        return self._tx(self.contract.functions.submitProposal, round_id, agg_id, b)

    def finalize(self, round_id: int, total_selected: int):
        return self._tx(self.contract.functions.finalize, round_id, total_selected)

    def get_round(self, round_id: int):
        finalized, h = self.contract.functions.getRound(round_id).call()
        # return hex string without altering case/length
        return finalized, h.hex()

    def get_votes(self, round_id: int, hash_hex: str) -> int:
        """Read-only: return vote count for a given hash in a round."""
        if not hash_hex.startswith(("0x", "0X")):
            hash_hex = "0x" + hash_hex
        b = Web3.to_bytes(hexstr=hash_hex)
        return int(self.contract.functions.getVotes(round_id, b).call())
