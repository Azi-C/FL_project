// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

// Minimal storage used in your last working flow: just put/get chunk.
// No owner checks, no "begin round" gate.
contract FLStorage {
    // roundId => writerId => chunkIdx => data
    mapping(uint256 => mapping(uint256 => mapping(uint256 => bytes))) private chunks;

    function putChunk(
        uint256 roundId,
        uint256 writerId,
        uint256 idx,
        bytes calldata data
    ) external {
        chunks[roundId][writerId][idx] = data;
    }

    function getChunk(
        uint256 roundId,
        uint256 writerId,
        uint256 idx
    ) external view returns (bytes memory) {
        return chunks[roundId][writerId][idx];
    }
}
