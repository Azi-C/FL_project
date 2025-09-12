from web3 import Web3
import json, os

class FLChain:
    def __init__(self, rpc_url="http://127.0.0.1:8545", contract_address=None, privkey=None):
        self.w3 = Web3(Web3.HTTPProvider(rpc_url))
        assert self.w3.is_connected(), "Web3 not connected (is hardhat node running?)"

        with open("artifacts/contracts/FLCoordinator.sol/FLCoordinator.json") as f:
            artifact = json.load(f)
        abi = artifact["abi"]

        if not contract_address:
            raise ValueError("Missing contract_address")
        self.contract = self.w3.eth.contract(address=Web3.to_checksum_address(contract_address), abi=abi)

        if not privkey:
            raise ValueError("Missing privkey (use one from hardhat node)")
        self.acct = self.w3.eth.account.from_key(privkey)

    def _tx(self, fn, *args):
        tx = fn(*args).build_transaction({
            "from": self.acct.address,
            "nonce": self.w3.eth.get_transaction_count(self.acct.address),
            "gas": 5_000_000,
            "gasPrice": self.w3.to_wei("1", "gwei"),
        })
        signed = self.acct.sign_transaction(tx)
        h = self.w3.eth.send_raw_transaction(signed.raw_transaction)
        return self.w3.eth.wait_for_transaction_receipt(h)

    def submit_proposal(self, round_id: int, agg_id: int, hash_hex: str):
        b = Web3.to_bytes(hexstr=hash_hex if hash_hex.startswith("0x") else "0x"+hash_hex)
        return self._tx(self.contract.functions.submitProposal, round_id, agg_id, b)

    def finalize(self, round_id: int, total_selected: int):
        return self._tx(self.contract.functions.finalize, round_id, total_selected)

    def get_round(self, round_id: int):
        finalized, h = self.contract.functions.getRound(round_id).call()
        return finalized, h.hex()
