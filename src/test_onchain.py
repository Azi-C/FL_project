# test_storage.py
from web3 import Web3
import json, os
from dotenv import load_dotenv

load_dotenv()
w3 = Web3(Web3.HTTPProvider(os.getenv("RPC_URL")))
addr = Web3.to_checksum_address(os.getenv("FLSTORAGE_ADDRESS"))

print("Code @ FLSTORAGE:", w3.eth.get_code(addr).hex()[:18])  # should not be 0x

with open("artifacts/contracts/FLStorage.sol/FLStorage.json") as f:
    abi = json.load(f)["abi"]
c = w3.eth.contract(address=addr, abi=abi)

# read a view to confirm ABI is correct
print("Meta(1,1):", c.functions.blobMeta(1,1).call())  # returns tuple of zeros if not begun yet
