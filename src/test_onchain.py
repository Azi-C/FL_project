import os
from dotenv import load_dotenv
from onchain import FLChain

load_dotenv()
chain = FLChain(
    rpc_url=os.getenv("RPC_URL"),
    contract_address=os.getenv("CONTRACT_ADDRESS"),
    privkey=os.getenv("PRIVKEY"),
)

rnd = 1
# two aggregators propose the same hash (majority), one proposes different
chain.submit_proposal(rnd, agg_id=0, hash_hex="0x" + "11"*32)
chain.submit_proposal(rnd, agg_id=1, hash_hex="0x" + "11"*32)
chain.submit_proposal(rnd, agg_id=2, hash_hex="0x" + "22"*32)

chain.finalize(rnd, total_selected=3)
print("Round state:", chain.get_round(rnd))  # should show the majority hash
