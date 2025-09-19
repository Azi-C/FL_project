// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

/**
 * FLStorage (Aggregator-Read Gated)
 *
 * - putChunk(): still open — any client can upload its local update.
 * - getChunk():
 *     * Client updates (writerId < GLOBAL_NS_OFFSET): ONLY aggregators for that round may read.
 *     * Global models (writerId >= GLOBAL_NS_OFFSET): public — all clients must be able to pull.
 *
 * Notes:
 * - We reference the coordinator to know who the aggregators are for a given round.
 * - We support either:
 *     (A) coordinator exposing `isAggregator(roundId, who) -> bool`, or
 *     (B) coordinator exposing `getAggregators(roundId) -> address[]`
 *   via try/catch to maximize compatibility with your deployed coordinator.
 */

interface IFLCoordinatorIsAgg {
    function isAggregator(uint256 roundId, address who) external view returns (bool);
}

interface IFLCoordinatorGetAggs {
    function getAggregators(uint256 roundId) external view returns (address[] memory);
}

contract FLStorage {
    // roundId => writerId => chunkIdx => data
    mapping(uint256 => mapping(uint256 => mapping(uint256 => bytes))) private chunks;

    // Public namespace threshold: global models are stored under writerId >= GLOBAL_NS_OFFSET
    uint256 public constant GLOBAL_NS_OFFSET = 1_000_000;

    // Coordinator address to check aggregator sets
    address public coordinator;
    address public owner;

    event CoordinatorSet(address indexed prev, address indexed curr);
    event OwnerTransferred(address indexed prev, address indexed curr);

    modifier onlyOwner() {
        require(msg.sender == owner, "Ownable: caller is not the owner");
        _;
    }

    constructor(address _coordinator) {
        owner = msg.sender;
        coordinator = _coordinator;
        emit CoordinatorSet(address(0), _coordinator);
        emit OwnerTransferred(address(0), msg.sender);
    }

    function setCoordinator(address _coordinator) external onlyOwner {
        address prev = coordinator;
        coordinator = _coordinator;
        emit CoordinatorSet(prev, _coordinator);
    }

    function transferOwnership(address newOwner) external onlyOwner {
        require(newOwner != address(0), "newOwner is zero");
        address prev = owner;
        owner = newOwner;
        emit OwnerTransferred(prev, newOwner);
    }

    // ---------------- Internal helper: aggregator check ----------------

    function _isAggregator(uint256 roundId, address who) internal view returns (bool) {
        // First try the isAggregator(bool) selector
        if (coordinator != address(0)) {
            // try isAggregator
            try IFLCoordinatorIsAgg(coordinator).isAggregator(roundId, who) returns (bool ok) {
                return ok;
            } catch {
                // try getAggregators
                try IFLCoordinatorGetAggs(coordinator).getAggregators(roundId) returns (address[] memory set) {
                    for (uint256 i = 0; i < set.length; i++) {
                        if (set[i] == who) {
                            return true;
                        }
                    }
                    return false;
                } catch {
                    // If neither interface works, deny access (conservative)
                    return false;
                }
            }
        }
        return false;
    }

    // ---------------- Write path (open) ----------------

    function putChunk(
        uint256 roundId,
        uint256 writerId,
        uint256 idx,
        bytes calldata data
    ) external {
        // uploads are open: any client can store its local update
        chunks[roundId][writerId][idx] = data;
    }

    // ---------------- Read path (restricted) ----------------

    function getChunk(
        uint256 roundId,
        uint256 writerId,
        uint256 idx
    ) external view returns (bytes memory) {
        // Global models are public so all clients can pull the finalized model
        if (writerId >= GLOBAL_NS_OFFSET) {
            return chunks[roundId][writerId][idx];
        }

        // Client updates are restricted to current round aggregators only
        require(_isAggregator(roundId, msg.sender), "Forbidden: not an aggregator for this round");
        return chunks[roundId][writerId][idx];
    }
}
