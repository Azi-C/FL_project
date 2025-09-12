// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

contract FLCoordinator {
    struct RoundState { bool finalized; bytes32 consensusHash; }

    // round => round state
    mapping(uint256 => RoundState) public rounds;

    // round => (aggId => hasSubmitted)
    mapping(uint256 => mapping(uint256 => bool)) public hasSubmitted;

    // round => (aggId => bytes32 hash)
    mapping(uint256 => mapping(uint256 => bytes32)) public proposalOf;

    // round => (hash => votes)
    mapping(uint256 => mapping(bytes32 => uint256)) public votes;

    event ProposalSubmitted(uint256 indexed round, uint256 indexed aggId, bytes32 hash);
    event ConsensusReached(uint256 indexed round, bytes32 hash, uint256 votes, uint256 totalSelected);
    event ConsensusFailed(uint256 indexed round, uint256 topVotes, uint256 totalSelected);

    modifier notFinalized(uint256 round) {
        require(!rounds[round].finalized, "Round finalized");
        _;
    }

    /// Aggregator posts exactly one proposal hash for a round
    function submitProposal(uint256 round, uint256 aggId, bytes32 h) external notFinalized(round) {
        require(!hasSubmitted[round][aggId], "Already submitted");
        hasSubmitted[round][aggId] = true;
        proposalOf[round][aggId] = h;
        votes[round][h] += 1;
        emit ProposalSubmitted(round, aggId, h);
    }

    /// Finalize when some hash has strict majority (> totalSelected/2)
    /// totalSelected is the number of aggregators chosen OFF-CHAIN for this round
    function finalize(uint256 round, uint256 totalSelected) external notFinalized(round) {
        require(totalSelected > 0, "totalSelected=0");

        bytes32 winner;
        uint256 maxVotes;

        // Simple scan over submitted aggIds (0..1023 is plenty for this project)
        for (uint256 aggId = 0; aggId < 1024; aggId++) {
            if (hasSubmitted[round][aggId]) {
                bytes32 h = proposalOf[round][aggId];
                uint256 v = votes[round][h];
                if (v > maxVotes) {
                    maxVotes = v;
                    winner = h;
                }
            }
        }

        if (maxVotes > totalSelected / 2) {
            rounds[round] = RoundState(true, winner);
            emit ConsensusReached(round, winner, maxVotes, totalSelected);
        } else {
            rounds[round] = RoundState(true, bytes32(0));
            emit ConsensusFailed(round, maxVotes, totalSelected);
        }
    }

    function getVotes(uint256 round, bytes32 h) external view returns (uint256) {
        return votes[round][h];
    }

    function getRound(uint256 round) external view returns (bool finalized, bytes32 consensusHash) {
        RoundState memory rs = rounds[round];
        return (rs.finalized, rs.consensusHash);
    }
}
