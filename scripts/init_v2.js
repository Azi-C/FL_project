// npx hardhat run --network localhost scripts/init_v2.js
async function main() {
  const addr = process.env.COORDINATOR_V2_ADDRESS;
  if (!addr) throw new Error("Set COORDINATOR_V2_ADDRESS in .env");

  const v2 = await ethers.getContractAt("FLCoordinatorV2", addr);

  console.log("registrationOpen:", await v2.registrationOpen());
  console.log("initialized:", await v2.initialized());
  console.log("participants:", (await v2.getParticipants()).length);
  console.log("totalData:", (await v2.totalData()).toString());

  if (await v2.registrationOpen()) {
    const tx = await v2.closeRegistrationAndInit();
    await tx.wait();
    console.log("✅ closeRegistrationAndInit");
  } else {
    console.log("Registration already closed.");
  }

  console.log("initialized:", await v2.initialized());
  console.log("totalRep:", (await v2.totalRep()).toString());
}

main().catch((e) => { console.error(e); process.exit(1); });
