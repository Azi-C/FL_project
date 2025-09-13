import json

with open("artifacts/contracts/FLCoordinator.sol/FLCoordinator.json") as f:
    abi = json.load(f)["abi"]

fns = [i["name"] for i in abi if i.get("type") == "function"]
print("Available functions in FLCoordinator ABI:")
for fn in fns:
    print(" -", fn)
