// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

/// @title FLStorage - chunked on-chain storage for model blobs
/// @notice Stores client updates and global models as blobs of bytes, in chunks
contract FLStorage {
    struct BlobMeta {
        uint256 size;         // total bytes
        uint32 totalChunks;   // expected number of chunks
        bytes32 expectedHash; // keccak256 rolling hash expected
        bytes32 rollingHash;  // updated as chunks arrive
        uint32 received;      // how many chunks received
        bool complete;        // true after finalize
    }

    // roundId => aggId => meta
    mapping(uint256 => mapping(uint256 => BlobMeta)) public blobMeta;

    // roundId => aggId => chunkIndex => bytes
    mapping(uint256 => mapping(uint256 => mapping(uint32 => bytes))) public chunks;

    event BeginBlob(uint256 indexed roundId, uint256 indexed aggId, uint256 size, uint32 totalChunks, bytes32 expectedHash);
    event PutChunk(uint256 indexed roundId, uint256 indexed aggId, uint32 indexed idx, uint256 len);
    event FinalizeBlob(uint256 indexed roundId, uint256 indexed aggId, bytes32 rollingHash);

    error AlreadyBegun();
    error NotBegun();
    error AlreadyComplete();
    error BadIndex();
    error BadLengths();
    error BadHash();

    /// @notice Start uploading a blob for (roundId, aggId)
    function beginBlob(
        uint256 roundId,
        uint256 aggId,
        uint256 size,
        uint32 totalChunks,
        bytes32 expectedHash
    ) external {
        BlobMeta storage m = blobMeta[roundId][aggId];
        if (m.totalChunks != 0) revert AlreadyBegun();
        require(size > 0 && totalChunks > 0, "bad params");
        m.size = size;
        m.totalChunks = totalChunks;
        m.expectedHash = expectedHash;
        m.rollingHash = bytes32(0);
        m.received = 0;
        m.complete = false;
        emit BeginBlob(roundId, aggId, size, totalChunks, expectedHash);
    }

    /// @notice Upload one chunk
    /// @dev rollingHash = keccak256(rollingHash || idx || data)
    function putChunk(
        uint256 roundId,
        uint256 aggId,
        uint32 idx,
        bytes calldata data
    ) external {
        BlobMeta storage m = blobMeta[roundId][aggId];
        if (m.totalChunks == 0) revert NotBegun();
        if (m.complete) revert AlreadyComplete();
        if (idx >= m.totalChunks) revert BadIndex();
        if (chunks[roundId][aggId][idx].length != 0) revert BadIndex(); // already set

        chunks[roundId][aggId][idx] = data;

        m.rollingHash = keccak256(abi.encodePacked(m.rollingHash, idx, data));
        m.received += 1;

        emit PutChunk(roundId, aggId, idx, data.length);
    }

    /// @notice Finalize and check integrity
    function finalizeBlob(uint256 roundId, uint256 aggId) external {
        BlobMeta storage m = blobMeta[roundId][aggId];
        if (m.totalChunks == 0) revert NotBegun();
        if (m.complete) revert AlreadyComplete();
        if (m.received != m.totalChunks) revert BadLengths();
        if (m.rollingHash != m.expectedHash) revert BadHash();
        m.complete = true;
        emit FinalizeBlob(roundId, aggId, m.rollingHash);
    }

    /// @notice Read one chunk (off-chain readers can iterate)
    function getChunk(uint256 roundId, uint256 aggId, uint32 idx) external view returns (bytes memory) {
        return chunks[roundId][aggId][idx];
    }
}
