from __future__ import annotations
import os, hashlib, numpy as np, torch
from dotenv import load_dotenv

from src.model import create_model
from src.utils import DEVICE, params_to_numpy, numpy_to_params
from src.storage_chain import FLStorageChain, pack_params_float32, unpack_params_float32
from src.onchain_v2 import FLChainV2

def hash_params_rounded(params, decimals=6) -> str:
    h = hashlib.sha256()
    for a in params:
        arr = np.asarray(a, dtype=np.float32)
        arr = np.round(arr, decimals=decimals)
        h.update(arr.tobytes(order="C"))
    return h.hexdigest()

def main():
    load_dotenv()
    rpc_url   = os.getenv("RPC_URL", "http://127.0.0.1:8545")
    coord_v2  = os.getenv("CONTRACT_ADDRESS_V2")
    storage   = os.getenv("FLSTORAGE_ADDRESS")
    priv      = os.getenv("PRIVKEY")
    if not (coord_v2 and storage and priv):
        raise RuntimeError("Missing CONTRACT_ADDRESS_V2 / FLSTORAGE_ADDRESS / PRIVKEY in .env")

    chain = FLChainV2(rpc_url, coord_v2, priv)
    store = FLStorageChain(rpc_url, storage, priv)

    # If already set, just show it and exit (idempotent)
    set_, h_hex, rid, wid, n = chain.get_baseline()
    if set_:
        print(f"[baseline] already set: hash={h_hex[:12]}.. round={rid} writer={wid} chunks={n}")
        return

    # Build initial (randomly initialized) model params
    model = create_model().to(DEVICE)
    params = params_to_numpy(model)

    # Upload to FLStorage at round 0, "global" writer id 1_000_000
    ROUND0 = 0
    GLOBAL_WRITER_ID = 1_000_000
    blob = pack_params_float32(params)
    n_chunks, _ = store.upload_blob(ROUND0, GLOBAL_WRITER_ID, blob, chunk_size=4096)

    # Hash (rounded) must match what aggregators/clients will verify
    h = hash_params_rounded(params)

    # Assign on-chain
    chain.assign_baseline(h, ROUND0, GLOBAL_WRITER_ID, n_chunks)
    print(f"[baseline] assigned: hash={h[:12]}.. round={ROUND0} writer={GLOBAL_WRITER_ID} chunks={n_chunks}")

if __name__ == "__main__":
    main()

