// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

/**
 * Coordinator contract:
 *  - Assigns the baseline model once (hash + pointer into FLStorage)
 *  - Assigns initial reputations for all clients once
 *  - Anchors proposals and finalizes rounds
 */
contract FLCoordinator {
    struct Round {
        bool finalized;
        bytes32 consensusHash;
    }
    mapping(uint256 => Round) public rounds;

    // -------- Baseline assignment --------
    bool    public baselineSet;
    bytes32 public baselineHash;
    uint256 public baselineRoundId;
    uint256 public baselineWriterId;
    uint256 public baselineNumChunks;

    event Proposal(uint256 indexed roundId, uint256 aggId, bytes32 hash);
    event Finalized(uint256 indexed roundId, bytes32 consensusHash);
    event BaselineAssigned(bytes32 baselineHash, uint256 roundId, uint256 writerId, uint256 numChunks);

    function assignBaseline(bytes32 _hash, uint256 _roundId, uint256 _writerId, uint256 _numChunks) external {
        require(!baselineSet, "Baseline already set");
        require(_hash != bytes32(0), "Invalid baseline hash");
        baselineSet = true;
        baselineHash = _hash;
        baselineRoundId = _roundId;
        baselineWriterId = _writerId;
        baselineNumChunks = _numChunks;
        emit BaselineAssigned(_hash, _roundId, _writerId, _numChunks);
    }

    function getBaseline() external view returns (bool, bytes32, uint256, uint256, uint256) {
        return (baselineSet, baselineHash, baselineRoundId, baselineWriterId, baselineNumChunks);
    }

    // -------- Initial reputations --------
    uint256 public constant REP_SCALE = 1_000_000; // fixed-point scale (1.0 = 1_000_000)
    bool public initialReputationsSet;
    mapping(uint256 => uint32) private _rep; // clientId => scaled reputation

    event InitialReputationsAssigned(uint256 count);

    function assignInitialReputations(uint256[] calldata clientIds, uint32[] calldata scaledReps) external {
        require(!initialReputationsSet, "Initial reputations already set");
        require(clientIds.length == scaledReps.length, "Length mismatch");
        uint256 n = clientIds.length;
        for (uint256 i = 0; i < n; i++) {
            _rep[clientIds[i]] = scaledReps[i];
        }
        initialReputationsSet = true;
        emit InitialReputationsAssigned(n);
    }

    function getInitialReputation(uint256 clientId) external view returns (uint32) {
        return _rep[clientId];
    }

    // -------- Round proposals/finalization --------
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
