// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

/**
 * FLStorage (Aggregator-Read Gated)
 *
 * Design:
 *  - Client updates (writerId < GLOBAL_NS_OFFSET) can be uploaded by anyone via putChunk(),
 *    but can only be READ by current round's aggregators.
 *  - Finalized global models (writerId >= GLOBAL_NS_OFFSET) are PUBLICLY readable so that all
 *    clients can pull the latest global.
 *
 * Coordinator compatibility:
 *  - Tries coordinator.isAggregator(roundId, who) first.
 *  - Falls back to coordinator.getAggregators(roundId) and scans membership.
 *  - If neither interface works, access is denied (conservative).
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

    // Writer IDs >= GLOBAL_NS_OFFSET are reserved for finalized globals (public read)
    uint256 public constant GLOBAL_NS_OFFSET = 1_000_000;

    // Admin + coordinator
    address public owner;
    address public coordinator;

    event CoordinatorSet(address indexed previous, address indexed current);
    event OwnerTransferred(address indexed previous, address indexed current);

    modifier onlyOwner() {
        require(msg.sender == owner, "FLStorage: caller is not the owner");
        _;
    }

    constructor(address _coordinator) {
        owner = msg.sender;
        coordinator = _coordinator;
        emit OwnerTransferred(address(0), msg.sender);
        emit CoordinatorSet(address(0), _coordinator);
    }

    function transferOwnership(address newOwner) external onlyOwner {
        require(newOwner != address(0), "FLStorage: newOwner is zero");
        emit OwnerTransferred(owner, newOwner);
        owner = newOwner;
    }

    function setCoordinator(address _coordinator) external onlyOwner {
        emit CoordinatorSet(coordinator, _coordinator);
        coordinator = _coordinator;
    }

    // ============== Internal: aggregator check ==============

    function _isAggregator(uint256 roundId, address who) internal view returns (bool) {
        if (coordinator == address(0)) return false;

        // Try isAggregator(roundId, who)
        try IFLCoordinatorIsAgg(coordinator).isAggregator(roundId, who) returns (bool ok) {
            return ok;
        } catch { /* fall through */ }

        // Try getAggregators(roundId)
        try IFLCoordinatorGetAggs(coordinator).getAggregators(roundId) returns (address[] memory set) {
            for (uint256 i = 0; i < set.length; i++) {
                if (set[i] == who) return true;
            }
            return false;
        } catch {
            // No compatible interface -> deny
            return false;
        }
    }

    // ============== Write path (open) ==============

    /**
     * Store one chunk.
     * - Open to all callers (clients upload their local updates; leader/agg uploads finalized global).
     */
    function putChunk(
        uint256 roundId,
        uint256 writerId,
        uint256 idx,
        bytes calldata data
    ) external {
        chunks[roundId][writerId][idx] = data;
    }

    // ============== Read path (gated for client updates) ==============

    /**
     * Read one chunk.
     * - If writerId >= GLOBAL_NS_OFFSET (finalized global): public read.
     * - Else (client updates): only aggregators of that round may read.
     */
    function getChunk(
        uint256 roundId,
        uint256 writerId,
        uint256 idx
    ) external view returns (bytes memory) {
        if (writerId >= GLOBAL_NS_OFFSET) {
            // Publicly readable finalized global
            return chunks[roundId][writerId][idx];
        }
        // Client update: aggregator-only
        require(_isAggregator(roundId, msg.sender), "FLStorage: forbidden (not aggregator)");
        return chunks[roundId][writerId][idx];
    }
}
