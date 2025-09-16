// scripts/deploy_coordinator_v2.js
const hre = require("hardhat");

async function main() {
  const [deployer] = await hre.ethers.getSigners();
  console.log("Deploying FLCoordinatorV2 with:", deployer.address);

  const FLCoordinatorV2 = await hre.ethers.getContractFactory("FLCoordinatorV2");
  const coord = await FLCoordinatorV2.deploy();
  await coord.waitForDeployment();

  const addr = await coord.getAddress();
  console.log("FLCoordinatorV2 deployed to:", addr);
}

main().catch((e) => {
  console.error(e);
  process.exitCode = 1;
});
