# src/debug_post_commit.py
from __future__ import annotations
import os, json, hashlib, numpy as np
from dotenv import load_dotenv
from web3 import Web3

def hparams() -> bytes:
    # just 32 bytes; contract stores commit hash, not weights
    arr = np.arange(16, dtype=np.uint8)
    return hashlib.sha256(arr.tobytes()).digest()

def main():
    load_dotenv(".env")
    rpc = os.getenv("RPC_URL", "http://127.0.0.1:8545")
    coord = os.getenv("CONTRACT_ADDRESS_V2")
    pk    = os.getenv("PRIVKEY_0") or os.getenv("PRIVKEY")
    rid   = int(os.getenv("START_ROUND_ID", "1"))
    cid   = 0

    w3 = Web3(Web3.HTTPProvider(rpc))
    acct = w3.eth.account.from_key(pk)

    # load ABI (same JSON your onchain_v2 uses)
    with open("artifacts/contracts/FLCoordinatorV2.sol/FLCoordinatorV2.json", "r") as f:
        abi = json.load(f)["abi"]
    c = w3.eth.contract(address=Web3.to_checksum_address(coord), abi=abi)

    # sanity: list function names and check postCommit exists
    fnames = [e.get("name") for e in abi if e.get("type") == "function"]
    print("ABI fns:", fnames)
    if "postCommit" not in fnames:
        print("❌ ABI has no postCommit. You are likely pointing to the wrong contract or wrong artifacts path.")
        return

    # read round status so we know it's begun
    try:
        begun, finalized, *_ = c.functions.getRound(rid).call()
        print(f"Round {rid} begun={begun} finalized={finalized}")
        if not begun:
            print("❌ Round not begun. Start your aggregator (it calls begin_round) or call it manually.")
            return
    except Exception as e:
        print("getRound() failed:", e)
        return

    # send commit
    commit_hash = hparams()
    tx = c.functions.postCommit(rid, cid, commit_hash).build_transaction({
        "from": acct.address,
        "nonce": w3.eth.get_transaction_count(acct.address, "pending"),
        "gas": 500_000,
        # adaptive EIP-1559
        "maxPriorityFeePerGas": w3.to_wei(1, "gwei"),
        "maxFeePerGas": int(w3.eth.get_block("latest")["baseFeePerGas"]) + w3.to_wei(2, "gwei"),
        "chainId": w3.eth.chain_id,
    })
    signed = w3.eth.account.sign_transaction(tx, acct.key)
    h = w3.eth.send_raw_transaction(signed.raw_transaction)
    rcpt = w3.eth.wait_for_transaction_receipt(h)
    print("postCommit rcpt status:", rcpt.status)

    # read back
    got = c.functions.getClientCommit(rid, cid).call()
    print("getClientCommit:", got.hex())

if __name__ == "__main__":
    main()
