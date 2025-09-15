# quick_check.py
from web3 import Web3
import os
from dotenv import load_dotenv
load_dotenv()

w3 = Web3(Web3.HTTPProvider(os.getenv("RPC_URL")))
print("Connected:", w3.is_connected())
print("Client:", w3.client_version)
blk = w3.eth.get_block("latest")
print("baseFeePerGas in block:", "baseFeePerGas" in blk)
print("block gasLimit:", blk.get("gasLimit"))
