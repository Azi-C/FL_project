const hre = require("hardhat");
require("dotenv").config();

function pickCoordinatorFromEnv() {
  const c =
    process.env.COORDINATOR_V2_ADDRESS ||
    process.env.CONTRACT_ADDRESS_V2 ||   // your current .env key
    process.env.CONTRACT_ADDRESS;        // legacy fallback
  return c;
}

function isAddress(a) {
  return typeof a === "string" && /^0x[0-9a-fA-F]{40}$/.test(a);
}

async function main() {
  const [deployer] = await hre.ethers.getSigners();
  console.log("Deploying FLStorage with:", deployer.address);

  const coordinator = pickCoordinatorFromEnv();
  console.log("Coordinator address =", coordinator);

  if (!isAddress(coordinator)) {
    throw new Error(
      "Missing or invalid coordinator address. " +
      "Set COORDINATOR_V2_ADDRESS (or CONTRACT_ADDRESS_V2) in .env to a 0x... address."
    );
  }

  const FLStorage = await hre.ethers.getContractFactory("FLStorage");
  const storage = await FLStorage.deploy(coordinator); // constructor(address)
  await storage.waitForDeployment();

  console.log("FLStorage deployed to:", await storage.getAddress());
  console.log("Coordinator set to:", coordinator);
}

main().catch((e) => {
  console.error(e);
  process.exitCode = 1;
});
