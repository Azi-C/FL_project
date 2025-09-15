// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

/**
 * Federated-learning coordinator (V2)
 * - Step 1: registration: clients call registerClient(|D_i|)
 * - Owner calls closeRegistrationAndInit(): computes initial rep r_i^(0) = |D_i| / sum_j |D_j|
 * - Optional: setBaselineHash / setValidationHash (pointers/hashes only)
 * - Per round:
 *    * beginRound(r) (anyone can open the round once)
 *    * off-chain: peers train/aggregate; on-chain: submitEval(r, who, a_i_fp, s_i_part)
 *      ri <- ri + α*ai*β*s  (all fixed-point with FP=1e6, clamp to [0, FP])
 *    * v1-style proposals/finalize still available for simple consensus anchoring
 *    * markConverged(r, hash)
 * - Optional: distributeRewards(k) (owner): credits rewardBalances based on r_i / Σr_j
 */
contract FLCoordinatorV2 {
    // ------- fixed point scale -------
    uint256 public constant FP = 1_000_000; // 1.000000

    address public owner;

    // registration / participants
    bool public registrationOpen = true;
    bool public initialized = false;

    address[] private participants;
    mapping(address => bool) public isParticipant;
    mapping(address => uint256) public dataSize;    // |D_i|
    mapping(address => uint256) public rep;         // reputation in FP
    uint256 public totalData;                       // Σ|D_i|
    uint256 public totalRep;                        // Σrep_i  (in FP)

    // alpha/beta weights (must sum to FP on init)
    uint256 public alpha_fp;  // e.g. 600000
    uint256 public beta_fp;   // e.g. 400000

    // rewards (simple on-chain ledger; not ERC20)
    uint256 public tokenPool;                 // optional global pool to draw from
    mapping(address => uint256) public rewardBalance; // credits accrued

    // baseline / validation anchors (hashes only)
    bytes32 public baselineHash;
    bytes32 public validationHash;

    // round/meta (simple)
    struct Round {
        bool begun;
        bool finalized;
        bytes32 consensusHash;  // from V1-style finalize or external agreement
        bool converged;
        bytes32 convergedHash;
    }
    mapping(uint256 => Round) public rounds;

    // --- V1-like proposal events (for off-chain listeners) ---
    event Proposal(uint256 indexed roundId, uint256 aggId, bytes32 hash);
    event Finalized(uint256 indexed roundId, bytes32 consensusHash);

    // --- V2 events ---
    event Registered(address indexed who, uint256 dataSize);
    event Initialized(uint256 participants, uint256 totalData, uint256 totalRep);
    event RoundBegan(uint256 indexed roundId);
    event EvalSubmitted(uint256 indexed roundId, address indexed who, uint256 a_i_fp, int8 s_i_part, uint256 newRep);
    event Converged(uint256 indexed roundId, bytes32 hash);
    event RewardsDistributed(uint256 k, uint256 totalRep);

    modifier onlyOwner() {
        require(msg.sender == owner, "Only owner");
        _;
    }

    constructor(uint256 _alpha_fp, uint256 _beta_fp) {
        require(_alpha_fp + _beta_fp == FP, "alpha+beta != 1.0");
        owner = msg.sender;
        alpha_fp = _alpha_fp;
        beta_fp = _beta_fp;
    }

    // --------- Step 1: Registration ----------
    function registerClient(uint256 _dataSize) external {
        require(registrationOpen, "Registration closed");
        require(!isParticipant[msg.sender], "Already registered");
        require(_dataSize > 0, "dataSize=0");

        isParticipant[msg.sender] = true;
        participants.push(msg.sender);
        dataSize[msg.sender] = _dataSize;
        totalData += _dataSize;

        emit Registered(msg.sender, _dataSize);
    }

    function closeRegistrationAndInit() external onlyOwner {
        require(registrationOpen, "Already closed");
        require(!initialized, "Already initialized");
        require(participants.length > 0, "No participants");
        require(totalData > 0, "totalData=0");

        registrationOpen = false;

        // r_i^(0) = |D_i| / Σ|D|
        // Ensure Σrep_i = FP
        uint256 sumRep = 0;
        for (uint256 i = 0; i < participants.length; i++) {
            address a = participants[i];
            uint256 r = (dataSize[a] * FP) / totalData;
            rep[a] = r;
            sumRep += r;
        }
        // Fix rounding (assign remainder to first)
        if (sumRep != FP) {
            uint256 diff;
            if (sumRep < FP) {
                diff = FP - sumRep;
                rep[participants[0]] += diff;
                sumRep = FP;
            } else {
                diff = sumRep - FP;
                if (rep[participants[0]] > diff) {
                    rep[participants[0]] -= diff;
                    sumRep = FP;
                }
            }
        }
        totalRep = sumRep;
        initialized = true;
        emit Initialized(participants.length, totalData, totalRep);
    }

    // --------- Anchors for baseline/validation ----------
    function setBaselineHash(bytes32 h) external onlyOwner { baselineHash = h; }
    function setValidationHash(bytes32 h) external onlyOwner { validationHash = h; }

    // --------- Rounds ----------
    function beginRound(uint256 roundId) external {
        Round storage r = rounds[roundId];
        require(!r.finalized, "Already finalized");
        r.begun = true;
        emit RoundBegan(roundId);
    }

    // V1-like proposal/finalize (hash anchoring only)
    function submitProposal(uint256 roundId, uint256 aggId, bytes32 h) external {
        Round storage r = rounds[roundId];
        require(!r.finalized, "Already finalized");
        require(r.begun, "NotBegun()");
        emit Proposal(roundId, aggId, h);
    }

    function finalize(uint256 roundId, uint256 /*totalSelected*/) external {
        Round storage r = rounds[roundId];
        require(!r.finalized, "Already finalized");
        require(r.begun, "NotBegun()");
        r.finalized = true;
        // In a real impl, you’d pass the decided hash; we keep a placeholder.
        r.consensusHash = bytes32(0);
        emit Finalized(roundId, r.consensusHash);
    }

    // --------- Per-client eval update (your formula) ----------
    // ri(t+1) = ri(t) + α*ai * β*s_part
    // where ai, α, β are in FP, s_part in {-1, +1}
    function submitEval(uint256 roundId, address who, uint256 a_i_fp, int8 s_i_part) external {
        require(initialized, "Not initialized");
        Round storage r = rounds[roundId];
        require(r.begun, "NotBegun()");
        require(isParticipant[who], "Not participant");
        require(a_i_fp <= FP, "ai>1.0");
        require(s_i_part == -1 || s_i_part == 1, "s in {-1,1}");

        uint256 oldRep = rep[who];

        // delta = α * ai
        uint256 delta = (alpha_fp * a_i_fp) / FP;
        // delta2 = delta * β
        uint256 delta2 = (delta * beta_fp) / FP;

        uint256 newRep;
        if (s_i_part == 1) {
            // add
            newRep = oldRep + delta2;
            if (newRep > FP) newRep = FP;
        } else {
            // subtract
            if (oldRep > delta2) {
                newRep = oldRep - delta2;
            } else {
                newRep = 0;
            }
        }

        rep[who] = newRep;

        // keep totalRep = Σrep_i (adjust by difference, clamp)
        if (newRep >= oldRep) {
            totalRep += (newRep - oldRep);
            if (totalRep > FP * participants.length) {
                totalRep = FP * participants.length;
            }
        } else {
            totalRep -= (oldRep - newRep);
        }

        emit EvalSubmitted(roundId, who, a_i_fp, s_i_part, newRep);
    }

    // Optional: distribute reward k proportionally to reputation
    function distributeRewards(uint256 k) external onlyOwner {
        require(initialized, "Not initialized");
        require(totalRep > 0, "totalRep=0");
        for (uint256 i = 0; i < participants.length; i++) {
            address a = participants[i];
            // p_i = k * rep_i / Σrep
            uint256 p = (k * rep[a]) / totalRep;
            rewardBalance[a] += p;
        }
        if (tokenPool >= k) tokenPool -= k;
        emit RewardsDistributed(k, totalRep);
    }

    // Convergence marker (hash anchoring)
    function markConverged(uint256 roundId, bytes32 h) external {
        Round storage r = rounds[roundId];
        require(r.begun, "NotBegun()");
        r.converged = true;
        r.convergedHash = h;
        emit Converged(roundId, h);
    }

    // ------ views / helpers ------
    function getParticipants() external view returns (address[] memory) {
        return participants;
    }

    function getClient(address a) external view returns (uint256 dsize, uint256 r_fp) {
        return (dataSize[a], rep[a]);
    }

    function getRound(uint256 roundId) external view returns (bool begun, bool finalized, bytes32 hash) {
        Round storage rr = rounds[roundId];
        return (rr.begun, rr.finalized, rr.consensusHash);
    }
}
