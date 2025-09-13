const hre = require("hardhat");

async function main() {
  const [deployer] = await hre.ethers.getSigners();
  console.log("Deploying with:", deployer.address);

  const Storage = await hre.ethers.getContractFactory("FLStorage");
  const storage = await Storage.deploy();
  await storage.waitForDeployment();

  console.log("FLStorage deployed to:", await storage.getAddress());
}

main().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
