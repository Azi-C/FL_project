async function main() {
  const [deployer] = await ethers.getSigners();
  console.log("Deploying with:", deployer.address);

  const Minimal = await ethers.getContractFactory("Minimal");
  const minimal = await Minimal.deploy();

  // ethers v6 style:
  await minimal.waitForDeployment();
  const addr = await minimal.getAddress();

  console.log("Minimal deployed to:", addr);
}

main().catch((e) => {
  console.error(e);
  process.exitCode = 1;
});
