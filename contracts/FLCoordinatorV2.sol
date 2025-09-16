// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

/**
 * FLCoordinatorV2 — serverless-friendly coordinator
 * - Round lifecycle (begin -> commit -> propose -> finalize)
 * - Clients commit update hashes on-chain; raw updates live in FLStorage
 * - Aggregators are selected off-chain and passed to beginRound
 * - Proposals are hashed aggregated models; winner is finalized and its pointer recorded
 *
 * NOTE: No access control here (for simplicity). Add Ownable/roles later if needed.
 */
contract FLCoordinatorV2 {
    struct RoundMeta {
        bool begun;
        bool finalized;
        uint256 roundId;
        // phase windows are informational (off-chain timing); enforcement is optional/minimal
        uint64 commitDeadline;    // last block timestamp to accept client commits
        uint64 proposeDeadline;   // last ts to accept aggregator proposals
        // elected aggregators for this round (by off-chain election)
        uint256[] aggregators;
        // consensus outcome
        bytes32 winnerHash;
        uint256 winnerAggId;
        // where to fetch the winning global model
        uint256 winnerWriterId;
        uint32  winnerNumChunks;
    }

    // roundId => RoundMeta
    mapping(uint256 => RoundMeta) public rounds;

    // roundId => clientId => update commit hash (bytes32)
    mapping(uint256 => mapping(uint256 => bytes32)) public clientCommit;

    // roundId => proposalHash => votes (optional; you can ignore for now)
    mapping(uint256 => mapping(bytes32 => uint256)) public votes;

    // Baseline pointer (set once)
    bool public baselineSet;
    bytes32 public baselineHash;
    uint256 public baselineRoundId;
    uint256 public baselineWriterId;
    uint32  public baselineNumChunks;

    // Initial reputations (scaled ints) — optional
    bool public initialReputationsSet;
    mapping(uint256 => uint32) private _rep;

    // -------- Events --------
    event BaselineAssigned(bytes32 hash, uint256 roundId, uint256 writerId, uint32 numChunks);
    event InitialReputationsAssigned(uint256 count);

    event RoundBegun(uint256 indexed roundId, uint64 commitDeadline, uint64 proposeDeadline, uint256[] aggregators);
    event ClientCommitted(uint256 indexed roundId, uint256 indexed clientId, bytes32 updateHash);
    event ProposalSubmitted(uint256 indexed roundId, uint256 indexed aggId, bytes32 proposalHash);
    event Finalized(uint256 indexed roundId, uint256 winnerAggId, bytes32 winnerHash, uint256 writerId, uint32 numChunks);

    // -------- Baseline --------
    function assignBaseline(bytes32 _hash, uint256 _roundId, uint256 _writerId, uint32 _numChunks) external {
        require(!baselineSet, "Baseline already set");
        require(_hash != bytes32(0), "Invalid hash");
        baselineSet = true;
        baselineHash = _hash;
        baselineRoundId = _roundId;
        baselineWriterId = _writerId;
        baselineNumChunks = _numChunks;
        emit BaselineAssigned(_hash, _roundId, _writerId, _numChunks);
    }

    function getBaseline() external view returns (bool, bytes32, uint256, uint256, uint32) {
        return (baselineSet, baselineHash, baselineRoundId, baselineWriterId, baselineNumChunks);
    }

    // -------- Initial reputations (optional) --------
    function assignInitialReputations(uint256[] calldata clientIds, uint32[] calldata scaledReps) external {
        require(!initialReputationsSet, "Initial reputations already set");
        require(clientIds.length == scaledReps.length, "Len mismatch");
        for (uint256 i = 0; i < clientIds.length; i++) {
            _rep[clientIds[i]] = scaledReps[i];
        }
        initialReputationsSet = true;
        emit InitialReputationsAssigned(clientIds.length);
    }

    function getInitialReputation(uint256 clientId) external view returns (uint32) {
        return _rep[clientId];
    }

    // -------- Round lifecycle --------
    function beginRound(
        uint256 roundId,
        uint64 commitDeadline,
        uint64 proposeDeadline,
        uint256[] calldata aggregators
    ) external {
        RoundMeta storage r = rounds[roundId];
        require(!r.begun, "Round already begun");
        r.begun = true;
        r.roundId = roundId;
        r.commitDeadline = commitDeadline;
        r.proposeDeadline = proposeDeadline;
        for (uint256 i = 0; i < aggregators.length; i++) {
            r.aggregators.push(aggregators[i]);
        }
        emit RoundBegun(roundId, commitDeadline, proposeDeadline, aggregators);
    }

    function postCommit(uint256 roundId, uint256 clientId, bytes32 updateHash) external {
        RoundMeta storage r = rounds[roundId];
        require(r.begun, "Round not begun");
        require(!r.finalized, "Finalized");
        // Optional: enforce time window — here we skip strict ts checks to keep dev simple
        clientCommit[roundId][clientId] = updateHash;
        emit ClientCommitted(roundId, clientId, updateHash);
    }

    function submitProposal(uint256 roundId, uint256 aggId, bytes32 proposalHash) external {
        RoundMeta storage r = rounds[roundId];
        require(r.begun, "Round not begun");
        require(!r.finalized, "Finalized");
        emit ProposalSubmitted(roundId, aggId, proposalHash);
        // votes[roundId][proposalHash]++ ; // enable if you implement on-chain voting
    }

    function finalize(
        uint256 roundId,
        uint256 winnerAggId,
        bytes32 winnerHash,
        uint256 winnerWriterId,
        uint32  winnerNumChunks
    ) external {
        RoundMeta storage r = rounds[roundId];
        require(r.begun, "Round not begun");
        require(!r.finalized, "Already finalized");
        r.finalized = true;
        r.winnerAggId = winnerAggId;
        r.winnerHash = winnerHash;
        r.winnerWriterId = winnerWriterId;
        r.winnerNumChunks = winnerNumChunks;
        emit Finalized(roundId, winnerAggId, winnerHash, winnerWriterId, winnerNumChunks);
    }

    function getRound(uint256 roundId)
        external
        view
        returns (
            bool begun,
            bool finalized,
            bytes32 winnerHash,
            uint256 winnerAggId,
            uint256 winnerWriterId,
            uint32  winnerNumChunks
        )
    {
        RoundMeta storage r = rounds[roundId];
        return (r.begun, r.finalized, r.winnerHash, r.winnerAggId, r.winnerWriterId, r.winnerNumChunks);
    }

    function getAggregators(uint256 roundId) external view returns (uint256[] memory) {
        return rounds[roundId].aggregators;
    }

    function getClientCommit(uint256 roundId, uint256 clientId) external view returns (bytes32) {
        return clientCommit[roundId][clientId];
    }
}
