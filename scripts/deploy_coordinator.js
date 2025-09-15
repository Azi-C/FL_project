const hre = require("hardhat");

async function main() {
  const [deployer] = await hre.ethers.getSigners();
  console.log("Deploying FLCoordinator with:", deployer.address);

  const FLCoordinator = await hre.ethers.getContractFactory("FLCoordinator");
  const coordinator = await FLCoordinator.deploy();
  await coordinator.waitForDeployment();

  const addr = await coordinator.getAddress();
  console.log("FLCoordinator deployed to:", addr);
}

main().catch((e) => {
  console.error(e);
  process.exitCode = 1;
});