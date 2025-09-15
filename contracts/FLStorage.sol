// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

contract FLStorage {
    struct Chunk { bytes data; }
    mapping(uint256 => mapping(uint256 => mapping(uint256 => Chunk))) public blobs;

    event ChunkStored(uint256 indexed roundId, uint256 indexed writerId, uint256 idx, bytes data);

    function putChunk(uint256 roundId, uint256 writerId, uint256 idx, bytes calldata data) external {
        blobs[roundId][writerId][idx] = Chunk({data: data});
        emit ChunkStored(roundId, writerId, idx, data);
    }

    function getChunk(uint256 roundId, uint256 writerId, uint256 idx) external view returns (bytes memory) {
        return blobs[roundId][writerId][idx].data;
    }
}
