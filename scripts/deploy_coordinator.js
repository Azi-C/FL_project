const hre = require("hardhat");

async function main() {
  const [deployer] = await hre.ethers.getSigners();
  console.log("Deploying FLCoordinator with:", deployer.address);

  const C = await hre.ethers.getContractFactory("FLCoordinator");
  const c = await C.deploy();
  await c.waitForDeployment();

  const addr = await c.getAddress();
  console.log("FLCoordinator deployed to:", addr);
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});
