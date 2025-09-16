// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

/**
 * Minimal coordinator (V1+) used to:
 *  - assign a baseline model once (hash + pointer into FLStorage)
 *  - anchor proposals and finalize a round (as before)
 */
contract FLCoordinator {
    struct Round {
        bool finalized;
        bytes32 consensusHash;
    }
    mapping(uint256 => Round) public rounds;

    // -------- Baseline assignment (one-shot) --------
    bool    public baselineSet;
    bytes32 public baselineHash;       // hash of baseline params (e.g., sha256 over packed float32)
    uint256 public baselineRoundId;    // pointer into FLStorage
    uint256 public baselineWriterId;   // pointer into FLStorage
    uint256 public baselineNumChunks;  // how many chunks were uploaded

    event Proposal(uint256 indexed roundId, uint256 aggId, bytes32 hash);
    event Finalized(uint256 indexed roundId, bytes32 consensusHash);
    event BaselineAssigned(bytes32 baselineHash, uint256 roundId, uint256 writerId, uint256 numChunks);

    /**
     * Assign the baseline exactly once.
     * You can add onlyOwner or role gating if desired.
     */
    function assignBaseline(
        bytes32 _hash,
        uint256 _roundId,
        uint256 _writerId,
        uint256 _numChunks
    ) external {
        require(!baselineSet, "Baseline already set");
        require(_hash != bytes32(0), "Invalid baseline hash");
        baselineSet = true;
        baselineHash = _hash;
        baselineRoundId = _roundId;
        baselineWriterId = _writerId;
        baselineNumChunks = _numChunks;
        emit BaselineAssigned(_hash, _roundId, _writerId, _numChunks);
    }

    function getBaseline()
        external
        view
        returns (bool set_, bytes32 h, uint256 roundId, uint256 writerId, uint256 numChunks)
    {
        return (baselineSet, baselineHash, baselineRoundId, baselineWriterId, baselineNumChunks);
    }

    // -------- Existing V1 proposal/finalize stubs --------
    function submitProposal(uint256 roundId, uint256 aggId, bytes32 h) external {
        require(!rounds[roundId].finalized, "Already finalized");
        emit Proposal(roundId, aggId, h);
    }

    function finalize(uint256 roundId, uint256 /*totalSelected*/) external {
        require(!rounds[roundId].finalized, "Already finalized");
        rounds[roundId].finalized = true;
        rounds[roundId].consensusHash = bytes32(0);
        emit Finalized(roundId, rounds[roundId].consensusHash);
    }

    function getRound(uint256 roundId) external view returns (bool, bytes32) {
        Round storage r = rounds[roundId];
        return (r.finalized, r.consensusHash);
    }
}
