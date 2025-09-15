# src/test_onchain.py
from web3 import Web3
import json, os
from dotenv import load_dotenv

def main():
    load_dotenv()
    rpc = os.getenv("RPC_URL")
    storage_addr_env = os.getenv("FLSTORAGE_ADDRESS")

    if not (rpc and storage_addr_env):
        raise RuntimeError("Set RPC_URL and FLSTORAGE_ADDRESS in .env")

    w3 = Web3(Web3.HTTPProvider(rpc))
    if not w3.is_connected():
        raise RuntimeError(f"Cannot connect to RPC at {rpc}")

    storage_addr = Web3.to_checksum_address(storage_addr_env)
    code = w3.eth.get_code(storage_addr).hex()
    print("Bytecode @ FLSTORAGE:", (code[:20] + ("..." if len(code) > 20 else "")))
    if code == "0x":
        raise RuntimeError(f"No contract code at {storage_addr} (did you deploy FLStorage?)")

    with open("artifacts/contracts/FLStorage.sol/FLStorage.json") as f:
        abi = json.load(f)["abi"]
    c = w3.eth.contract(address=storage_addr, abi=abi)

    # Lecture d'un chunk existant/inexistant (API réelle du contrat)
    # Ici on lit l'index 0 du writer 1 au round 1 : renverra "" si rien n'a été écrit.
    data = c.functions.getChunk(1, 1, 0).call()
    print("getChunk(1,1,0) length:", len(data))

if __name__ == "__main__":
    main()
