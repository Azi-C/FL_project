// npx hardhat run --network localhost scripts/deploy_v2.js
async function main() {
  const [deployer] = await ethers.getSigners();
  console.log("Deployer:", deployer.address);

  // alpha+beta must equal 1e6
  const FP = 1_000_000;
  const alpha = Math.floor(0.6 * FP); // 0.6
  const beta  = FP - alpha;           // 0.4

  const V2 = await ethers.getContractFactory("FLCoordinatorV2");
  const v2 = await V2.deploy(alpha, beta);
  await v2.waitForDeployment();
  const v2addr = await v2.getAddress();
  console.log("FLCoordinatorV2:", v2addr);

  const ST = await ethers.getContractFactory("FLStorage");
  const st = await ST.deploy();
  await st.waitForDeployment();
  const staddr = await st.getAddress();
  console.log("FLStorage:", staddr);

  // Optionally set baseline/validation anchors (dummy)
  await (await v2.setBaselineHash("0x" + "11".repeat(32))).wait();
  await (await v2.setValidationHash("0x" + "22".repeat(32))).wait();

  console.log("Done.");
}

main().catch((e) => { console.error(e); process.exit(1); });
