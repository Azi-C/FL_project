from web3 import Web3
import json, os
from dotenv import load_dotenv
import csv

load_dotenv()
RPC_URL = os.getenv("RPC_URL", "http://127.0.0.1:8545")
ADDR    = os.getenv("CONTRACT_ADDRESS")

w3 = Web3(Web3.HTTPProvider(RPC_URL))
assert w3.is_connected(), "Web3 not connected"

with open("artifacts/contracts/FLCoordinator.sol/FLCoordinator.json") as f:
    abi = json.load(f)["abi"]
c = w3.eth.contract(address=Web3.to_checksum_address(ADDR), abi=abi)

# Scan from block 0; for longer runs you can store the last scanned block
from_block = 0
to_block = "latest"

props = c.events.ProposalSubmitted().get_logs(from_block=from_block, to_block=to_block)
reached = c.events.ConsensusReached().get_logs(from_block=from_block, to_block=to_block)
failed  = c.events.ConsensusFailed().get_logs(from_block=from_block, to_block=to_block)

print(f"Proposals: {len(props)}, Reached: {len(reached)}, Failed: {len(failed)}")

# Write a CSV for your appendix
os.makedirs("artifacts", exist_ok=True)
with open("artifacts/onchain_events.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["type","round","aggId/hash","hash_or_votes","block"])
    for e in props:
        w.writerow(["ProposalSubmitted", e["args"]["round"], e["args"]["aggId"], e["args"]["hash"].hex(), e["blockNumber"]])
    for e in reached:
        w.writerow(["ConsensusReached", e["args"]["round"], e["args"]["hash"].hex(), int(e["args"]["votes"]), e["blockNumber"]])
    for e in failed:
        w.writerow(["ConsensusFailed", e["args"]["round"], "", int(e["args"]["topVotes"]), e["blockNumber"]])

print("Wrote artifacts/onchain_events.csv")
