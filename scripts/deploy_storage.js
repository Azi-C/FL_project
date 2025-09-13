const hre = require("hardhat");

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
