# src/round_starter.py
from __future__ import annotations
import os, time
from dotenv import load_dotenv
from .onchain_v2 import FLChainV2

def main():
    load_dotenv()
    rid = int(os.getenv("RID", "1"))
    rpc = os.getenv("RPC_URL", "http://127.0.0.1:8545")
    addr = os.getenv("CONTRACT_ADDRESS_V2")
    pk   = os.getenv("PRIVKEY")
    c = FLChainV2(rpc, addr, pk)

    begun, finalized, *_ = c.get_round(rid)
    if begun:
        print(f"[round] {rid} already begun")
        return
    now = int(time.time())
    tx = c.begin_round(rid, now + 120, now + 240, [])
    print(f"[round] begun rid={rid} tx={tx}")

if __name__ == "__main__":
    main()
