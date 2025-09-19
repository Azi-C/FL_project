// scripts/deploy_storage.js
// Simple deployer for the open FLStorage above.

const hre = require("hardhat");
const { config } = require("dotenv");

// Load .env if present (optional; not strictly needed here)
config({ path: ".env" });

async function main() {
  const [deployer] = await hre.ethers.getSigners();
  console.log("Deploying FLStorage with:", deployer.address);

  const FLStorage = await hre.ethers.getContractFactory("FLStorage");
  const storage = await FLStorage.deploy();
  await storage.waitForDeployment();

  const addr = await storage.getAddress();
  console.log("FLStorage deployed to:", addr);
}

main().catch((e) => {
  console.error(e);
  process.exitCode = 1;
});
