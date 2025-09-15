// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

/**
 * Minimal coordinator (V1) used only to anchor proposals and finalize a round.
 * No gating: you can submitProposal/finalize any round id.
 */
contract FLCoordinator {
    struct Round {
        bool finalized;
        bytes32 consensusHash;
    }
    mapping(uint256 => Round) public rounds;

    event Proposal(uint256 indexed roundId, uint256 aggId, bytes32 hash);
    event Finalized(uint256 indexed roundId, bytes32 consensusHash);

    function submitProposal(uint256 roundId, uint256 aggId, bytes32 h) external {
        require(!rounds[roundId].finalized, "Already finalized");
        emit Proposal(roundId, aggId, h);
    }

    function finalize(uint256 roundId, uint256 totalSelected) external {
        require(!rounds[roundId].finalized, "Already finalized");
        rounds[roundId].finalized = true;
        // In this V1 stub we just store zero (your Python decides the hash)
        rounds[roundId].consensusHash = bytes32(0);
        emit Finalized(roundId, rounds[roundId].consensusHash);
    }

    function getRound(uint256 roundId) external view returns (bool, bytes32) {
        Round storage r = rounds[roundId];
        return (r.finalized, r.consensusHash);
    }
}
