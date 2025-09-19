/** @type import('hardhat/config').HardhatUserConfig */
require("@nomicfoundation/hardhat-ethers");

module.exports = {
  solidity: "0.8.20",
  networks: {
    hardhat: {
      blockGasLimit: 120000000, // 120M
      gasPrice: 1_000_000_000,  // 1 gwei
      mining: { auto: false, interval: 1000 }
    },
    localhost: { url: "http://127.0.0.1:8545" }
  }
};
