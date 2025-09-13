// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

/// @notice Minimal coordinator for hash-based consensus per round.
/// Methods expected by your Python onchain bridge:
/// - submitProposal(uint256 roundId, uint256 aggId, bytes32 hash)
/// - finalize(uint256 roundId, uint256 totalSelected)
/// - getRound(uint256 roundId) -> (bool finalized, bytes32 consensusHash)
/// - getVotes(uint256 roundId, bytes32 hash) -> (uint256)
contract FLCoordinator {
    struct Round {
        bool finalized;
        bytes32 consensusHash;
        uint256 totalSelected; // number of selected aggregators in this round
        bytes32[] hashesSeen;  // list of unique proposal hashes
        mapping(bytes32 => uint256) votes; // hash => votes
        mapping(bytes32 => bool) seen;     // track unique hashes
    }

    mapping(uint256 => Round) private rounds;

    event ProposalSubmitted(uint256 indexed roundId, uint256 indexed aggId, bytes32 hash);
    event Finalized(uint256 indexed roundId, bytes32 consensusHash, uint256 maxVotes);

    function submitProposal(uint256 roundId, uint256 /*aggId*/, bytes32 hash) external {
        Round storage r = rounds[roundId];
        require(!r.finalized, "Round finalized");
        if (!r.seen[hash]) {
            r.seen[hash] = true;
            r.hashesSeen.push(hash);
        }
        r.votes[hash] += 1;
        emit ProposalSubmitted(roundId, 0, hash);
    }

    /// @param totalSelected number of aggregators selected this round (k)
    function finalize(uint256 roundId, uint256 totalSelected) external {
        Round storage r = rounds[roundId];
        require(!r.finalized, "Round already finalized");
        r.finalized = true;
        r.totalSelected = totalSelected;

        // choose hash with max votes; require strict majority (>= floor(k/2)+1)
        uint256 maxVotes = 0;
        bytes32 winner = bytes32(0);
        for (uint256 i = 0; i < r.hashesSeen.length; i++) {
            bytes32 h = r.hashesSeen[i];
            uint256 v = r.votes[h];
            if (v > maxVotes) {
                maxVotes = v;
                winner = h;
            }
        }

        // strict majority threshold
        uint256 threshold = (totalSelected / 2) + 1;
        if (maxVotes >= threshold) {
            r.consensusHash = winner;
        } else {
            r.consensusHash = bytes32(0);
        }

        emit Finalized(roundId, r.consensusHash, maxVotes);
    }

    function getRound(uint256 roundId) external view returns (bool finalized, bytes32 consensusHash) {
        Round storage r = rounds[roundId];
        return (r.finalized, r.consensusHash);
    }

    function getVotes(uint256 roundId, bytes32 hash) external view returns (uint256) {
        Round storage r = rounds[roundId];
        return r.votes[hash];
    }
}
