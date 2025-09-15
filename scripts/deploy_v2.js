// npx hardhat run --network localhost scripts/deploy_v2.js
// α=0.7, β=0.3 at 1e6 fixed-point scale
const ALPHA_FP = 700000;
const BETA_FP  = 300000;

async function main() {
  const [deployer] = await ethers.getSigners();
  console.log("Deploying with:", deployer.address);

  const FLC = await ethers.getContractFactory("FLCoordinatorV2");
  const coord = await FLC.deploy(ALPHA_FP, BETA_FP);
  await coord.waitForDeployment();

  console.log("FLCoordinatorV2 deployed at:", await coord.getAddress());
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});
