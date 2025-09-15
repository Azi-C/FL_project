// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

/**
 * FLCoordinatorV2
 *
 * What this contract does (lightweight, gas-aware):
 * - Client registration with dataset size |D_i| and participant registry
 * - One-time normalization to initialize r_i^(0) ∝ |D_i| / sum|D|
 * - Store baseline/validation hashes for auditability (blobs stay in FLStorage)
 * - Accept off-chain evaluation results and update on-chain reputation:
 *     r_i(t+1) = r_i(t) + (α * a_i) * (β * s_i_part)   (fixed-point math)
 *   where a_i in [0,1] scaled by 1e6, s_i_part ∈ {+1,-1}
 * - Mark round convergence on-chain (for transparency)
 *
 * Notes:
 * - Heavy work (training/eval) remains off-chain; only the results are recorded here.
 * - This keeps gas low and still matches your protocol’s state machine.
 * - Rewards: we expose a hook to record per-round rewards off-chain; you can extend
 *   to a full ERC20 later (here we only log “virtual rewards”).
 */
contract FLCoordinatorV2 {
    // ---------- Fixed-point scale (1.0 == 1_000_000) ----------
    uint256 public constant FP = 1_000_000;

    // ---------- Configurable weights ----------
    // α + β should = 1.0 (i.e., alpha + beta == FP)
    uint256 public alpha;  // e.g., 700_000  (0.7)
    uint256 public beta;   // e.g., 300_000  (0.3)

    address public owner;
    bool    public registrationOpen = true;
    bool    public initialized      = false;

    // Baseline and validation "references" (hashes/checksums/metadata)
    bytes32 public baselineHash;  // hash of baseline params blob in FLStorage
    bytes32 public valsetHash;    // hash/seed/metadata of shared validation dataset

    // Participants
    address[] public participants;

    struct ClientInfo {
        bool    registered;
        uint64  dataSize;     // |D_i|
        uint256 rep;          // reputation in fixed-point (scaled by FP)
    }

    mapping(address => ClientInfo) public clients;

    uint256 public totalData;     // sum |D_i|
    uint256 public totalRep;      // sum r_i (scaled by FP) — recomputed at init, updated on changes

    // Round status (complementary to your existing FLCoordinator)
    struct RoundMeta {
        bool converged;       // set when |A(w_t+1,V) - A(w_t,V)| < ε
        bytes32 notes;        // optional metadata (e.g., hash of winner’s params, or blank)
    }
    mapping(uint256 => RoundMeta) public rounds;

    // ---------- Events ----------
    event Registered(address indexed who, uint64 dataSize);
    event InitReputation(address indexed who, uint256 repFp);
    event RegistrationClosed(uint256 totalData, uint256 totalRep);
    event BaselineSet(bytes32 baselineHash);
    event ValSetHash(bytes32 valHash);
    event ReputationUpdated(address indexed who, uint256 oldRep, uint256 newRep, int256 deltaSigned);
    event EvalSubmitted(uint256 indexed roundId, address indexed who, uint256 a_i_fp, int8 s_i_part);
    event RewardRecorded(uint256 indexed roundId, address indexed who, uint256 amountFp);
    event Converged(uint256 indexed roundId);

    modifier onlyOwner() {
        require(msg.sender == owner, "Only owner");
        _;
    }

    constructor(uint256 alphaFp, uint256 betaFp) {
        require(alphaFp + betaFp == FP, "alpha+beta!=1.0");
        alpha = alphaFp;
        beta = betaFp;
        owner = msg.sender;
    }

    // ---------- Admin / setup ----------

    function setBaselineHash(bytes32 h) external onlyOwner {
        baselineHash = h;
        emit BaselineSet(h);
    }

    function setValidationHash(bytes32 h) external onlyOwner {
        valsetHash = h;
        emit ValSetHash(h);
    }

    // Client self-registration (while open)
    function registerClient(uint64 dataSize) external {
        require(registrationOpen, "Registration closed");
        require(!clients[msg.sender].registered, "Already registered");
        require(dataSize > 0, "dataSize=0");

        clients[msg.sender] = ClientInfo({
            registered: true,
            dataSize: dataSize,
            rep: 0
        });
        participants.push(msg.sender);
        totalData += dataSize;

        emit Registered(msg.sender, dataSize);
    }

    // One-time initialization: set r_i^(0) ∝ |D_i|/sum|D|
    // We pick repFp = round( FP * dataSize / totalData ), and also set totalRep.
    function closeRegistrationAndInit() external onlyOwner {
        require(registrationOpen, "Already closed");
        registrationOpen = false;

        require(!initialized, "Already initialized");
        require(totalData > 0, "No data");

        uint256 _totalRep = 0;
        for (uint256 i = 0; i < participants.length; i++) {
            address p = participants[i];
            ClientInfo storage ci = clients[p];
            uint256 repFp = (uint256(ci.dataSize) * FP) / totalData;
            ci.rep = repFp;
            _totalRep += repFp;
            emit InitReputation(p, repFp);
        }
        totalRep = _totalRep;
        initialized = true;

        emit RegistrationClosed(totalData, totalRep);
    }

    // ---------- Views ----------

    function getParticipants() external view returns (address[] memory) {
        return participants;
    }

    function getClient(address who) external view returns (bool registered, uint64 dataSize, uint256 repFp) {
        ClientInfo storage ci = clients[who];
        return (ci.registered, ci.dataSize, ci.rep);
    }

    // ---------- Eval / reputation / rewards ----------

    // Submit off-chain eval result for a given round:
    // a_i_fp is accuracy [0..FP], s_i_part = +1 (valid) or -1 (invalid)
    // Updates reputation: r_new = r_old + (alpha * a_i_fp) * (beta * s_i_part) / (FP*FP)  (with fixed-point)
    function submitEval(uint256 roundId, address who, uint256 a_i_fp, int8 s_i_part) external {
        require(initialized, "Not initialized");
        require(clients[who].registered, "Not registered");
        require(s_i_part == 1 || s_i_part == -1, "spart must be +/-1");
        require(a_i_fp <= FP, "a_i out of range");

        emit EvalSubmitted(roundId, who, a_i_fp, s_i_part);

        ClientInfo storage ci = clients[who];
        uint256 rOld = ci.rep;

        // delta = (alpha * a_i_fp / FP) * (beta * s / FP) in fixed-point
        // Combine as (alpha * a_i_fp * beta) / FP^2, then apply sign.
        uint256 mag = (alpha * a_i_fp) / FP;           // α * a_i
        mag = (mag * beta) / FP;                       // α * a_i * β
        int256 deltaSigned = s_i_part == 1 ? int256(mag) : -int256(mag);

        // apply
        int256 rNewSigned = int256(rOld) + deltaSigned;
        if (rNewSigned < 0) rNewSigned = 0;
        uint256 rNew = uint256(rNewSigned);

        // update totals
        if (rNew >= rOld) {
            totalRep += (rNew - rOld);
        } else {
            totalRep -= (rOld - rNew);
        }
        ci.rep = rNew;

        emit ReputationUpdated(who, rOld, rNew, deltaSigned);
    }

    // Record "virtual" reward (scaled FP). Off-chain decides the amount per round,
    // typically: p_i = K_round * r_i / sum_j r_j. This function only emits/logs it.
    function recordReward(uint256 roundId, address who, uint256 amountFp) external {
        require(clients[who].registered, "Not registered");
        emit RewardRecorded(roundId, who, amountFp);
    }

    // Mark round as converged (for auditability)
    function markConverged(uint256 roundId, bytes32 notes) external {
        rounds[roundId].converged = true;
        rounds[roundId].notes = notes; // e.g., hash of winner global (optional)
        emit Converged(roundId);
    }
}
