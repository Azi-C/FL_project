async function main() {
  const [deployer] = await ethers.getSigners();
  console.log("Deploying with:", deployer.address);

  const C = await ethers.getContractFactory("FLCoordinator");
  const c = await C.deploy();
  await c.waitForDeployment();
  const addr = await c.getAddress();

  console.log("FLCoordinator deployed to:", addr);
}
main().catch((e) => { console.error(e); process.exitCode = 1; });
