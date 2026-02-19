"""
Matching Pennies Task - Pure Python + Numba Implementation

This module provides a Numba-accelerated implementation of the matching pennies
algorithm, matching the interface of `multiEnvironmentBanditsMP_julia`.

Author: Auto-ported from matching_pennies.jl
"""

import numpy as np
from numba import njit, types, prange
from numba.typed import Dict, List
from typing import Sequence, Tuple, Optional
import pandas as pd
import math

from .abstract import RL_multiEnvironment


# ==============================================================================
# Numba-compatible Binomial Test
# ==============================================================================

@njit(cache=True)
def _log_beta(a: float, b: float) -> float:
    """Log of the beta function using lgamma."""
    return math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b)


@njit(cache=True)
def _betainc_cf(a: float, b: float, x: float) -> float:
    """
    Regularized incomplete beta function using continued fraction expansion.
    Uses Lentz's algorithm for numerical stability.
    """
    if x == 0.0:
        return 0.0
    if x == 1.0:
        return 1.0
    
    # Use symmetry relation for numerical stability
    if x > (a + 1.0) / (a + b + 2.0):
        return 1.0 - _betainc_cf(b, a, 1.0 - x)

    
    # Continued fraction using Lentz's algorithm
    TINY = 1e-30
    EPS = 1e-14
    MAX_ITER = 200
    
    # Prefactor
    lbeta_ab = _log_beta(a, b)
    front = math.exp(a * math.log(x) + b * math.log(1.0 - x) - lbeta_ab) / a
    
    # Lentz's algorithm
    f = 1.0
    c = 1.0
    d = 0.0
    
    for m in range(1, MAX_ITER + 1):
        # Even step
        m2 = 2 * m
        
        # a_{2m}
        num = m * (b - m) * x / ((a + m2 - 1.0) * (a + m2))
        d = 1.0 + num * d
        if abs(d) < TINY:
            d = TINY
        c = 1.0 + num / c
        if abs(c) < TINY:
            c = TINY
        d = 1.0 / d
        f *= c * d
        
        # a_{2m+1}
        num = -(a + m) * (a + b + m) * x / ((a + m2) * (a + m2 + 1.0))
        d = 1.0 + num * d
        if abs(d) < TINY:
            d = TINY
        c = 1.0 + num / c
        if abs(c) < TINY:
            c = TINY
        d = 1.0 / d
        delta = c * d
        f *= delta
        
        if abs(delta - 1.0) < EPS:
            break
    
    return front * f


@njit(cache=True)
def binomial_cdf(k: int, n: int, p: float) -> float:
    """
    Cumulative distribution function P(X <= k) for Binomial(n, p).
    Uses regularized incomplete beta function.
    """
    if k < 0:
        return 0.0
    if k >= n:
        return 1.0
    if p == 0.0:
        return 1.0
    if p == 1.0:
        return 0.0 if k < n else 1.0
    
    # P(X <= k) = I_{1-p}(n-k, k+1) where I is regularized incomplete beta
    # Explicit float conversion to avoid Numba type issues
    a = np.float64(n - k)
    b = np.float64(k + 1)
    x = 1.0 - p
    return _betainc_cf(a, b, x)


@njit(cache=True)
def binomial_pmf(k: int, n: int, p: float) -> float:
    """
    Probability mass function P(X = k) for Binomial(n, p).
    """
    if k < 0 or k > n:
        return 0.0
    if n == 0:
        return 1.0 if k == 0 else 0.0
    
    # Use log to avoid overflow
    # log(C(n,k)) = lgamma(n+1) - lgamma(k+1) - lgamma(n-k+1)
    log_comb = math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1)
    
    if p == 0.0:
        return 1.0 if k == 0 else 0.0
    if p == 1.0:
        return 1.0 if k == n else 0.0
    
    log_pmf = log_comb + k * math.log(p) + (n - k) * math.log(1.0 - p)
    return math.exp(log_pmf)


@njit(cache=True)
def binomial_pvalue_two_tailed(k: int, n: int, p: float = 0.5) -> float:
    """
    Two-tailed p-value for binomial test.
    Tests H0: probability = p vs H1: probability != p
    
    Uses the minimum likelihood method: sum probabilities of all outcomes
    that have probability <= P(X=k).
    
    This matches scipy.stats.binomtest and Julia's HypothesisTests.BinomialTest
    """
    if n == 0:
        return 1.0
    
    # Probability of observed value
    prob_k = binomial_pmf(k, n, p)
    
    # Sum probabilities of all outcomes with probability <= prob_k
    # Add small tolerance to handle floating point comparisons
    tolerance = prob_k * 1e-10
    
    p_val = 0.0
    for i in range(n + 1):
        prob_i = binomial_pmf(i, n, p)
        if prob_i <= prob_k + tolerance:
            p_val += prob_i
    
    # Clamp to [0, 1]
    if p_val > 1.0:
        p_val = 1.0
    if p_val < 0.0:
        p_val = 0.0
    
    return p_val


# ==============================================================================
# History Sequence Encoding
# ==============================================================================

@njit(cache=True)
def encode_histseq(choices: np.ndarray, rewards: np.ndarray, 
                   end_pos: int, depth: int, if_outcome: bool) -> np.int64:
    """
    Encode a history sequence as a unique integer key.
    
    For depth=3, if_outcome=True, sequence "0+1-0-" encodes as:
    Each trial uses 2 bits if no outcome (choice only: 0 or 1)
    Each trial uses 3 bits if outcome included (choice: 0/1, reward: +/-)
    
    Args:
        choices: array of choices (0 or 1)
        rewards: array of rewards (0 or 1)  
        end_pos: position up to which to include (exclusive, 0-indexed)
        depth: number of trials to include
        if_outcome: whether to include outcome in encoding
    
    Returns:
        Unique integer key, or -1 for depth=0 empty sequence
    """
    if depth <= 0:
        return np.int64(-1)  # Special key for empty sequence (alg-0)
    
    start_pos = end_pos - depth
    if start_pos < 0:
        return np.int64(-2)  # Invalid - not enough history
    
    key = np.int64(0)
    
    if if_outcome:
        # 2 bits per trial: choice (0/1) + reward (0/1)
        # Encode as: (choice << 1) | reward, then shift left by 2 for each trial
        for i in range(depth):
            idx = start_pos + i
            trial_code = (np.int64(choices[idx]) << 1) | np.int64(rewards[idx])
            key = (key << 2) | trial_code
        # Add a marker bit to distinguish from no-outcome sequences
        key = (key << 1) | np.int64(1)
    else:
        # 1 bit per trial: just choice
        for i in range(depth):
            idx = start_pos + i
            key = (key << 1) | np.int64(choices[idx])
        # Add a marker bit (0) to distinguish from outcome sequences
        key = key << 1
    
    # Also encode depth to make keys unique across depths
    key = (key << 4) | np.int64(depth)
    
    return key


# Note: This function is NOT JIT-compiled because Numba doesn't support
# string operations well. It's only used for output/debugging.
def decode_histseq_to_string(key: int) -> str:
    """
    Decode an integer key back to a human-readable string (for debugging/output).
    """
    if key == -1:
        return ""
    if key == -2:
        return "<invalid>"
    
    depth = int(key & 0xF)
    key = key >> 4
    
    if_outcome = bool(key & 1)
    key = key >> 1
    
    result = ""
    
    if if_outcome:
        # Extract 2 bits per trial, in reverse order
        trials = []
        for _ in range(depth):
            trial_code = key & 0x3
            choice = (trial_code >> 1) & 1
            reward = trial_code & 1
            trials.append((choice, reward))
            key = key >> 2
        
        for choice, reward in reversed(trials):
            result += str(choice)
            result += "+" if reward else "-"
    else:
        # Extract 1 bit per trial, in reverse order
        choices = []
        for _ in range(depth):
            choices.append(key & 1)
            key = key >> 1
        
        for choice in reversed(choices):
            result += str(choice)
    
    return result


# ==============================================================================
# Core Algorithm Data Structures
# ==============================================================================

# BiasCount entry indices (for structured array access)
BC_HISTSEQ_KEY = 0
BC_RIGHTCH = 1
BC_TOTAL = 2
BC_PVAL = 3
BC_ALG_TYPE = 4
BC_DEPTH = 5
BC_N_FIELDS = 6

# TrialBias entry indices
TB_DETECTED = 0
TB_DEPTH = 1
TB_MAGNITUDE = 2
TB_WHICH_ALG = 3
TB_N_FIELDS = 4


@njit(cache=True)
def init_bias_count_arrays(max_entries: int) -> Tuple[np.ndarray, np.ndarray, int]:
    """
    Initialize bias count storage arrays.
    
    Returns:
        bc_int: int array for histseq_key, rightCh, total, alg_type, depth
        bc_float: float array for p_val
        n_entries: current number of entries (starts at 1 for alg-0 entry)
    """
    bc_int = np.zeros((max_entries, 5), dtype=np.int64)
    bc_float = np.zeros((max_entries, 1), dtype=np.float64)
    
    # Initialize first entry for algorithm 0 (empty history, no outcome)
    bc_int[0, 0] = -1  # histseq_key for empty
    bc_int[0, 1] = 0   # rightCh
    bc_int[0, 2] = 0   # total
    bc_int[0, 3] = 0   # alg_type
    bc_int[0, 4] = 0   # depth
    bc_float[0, 0] = np.nan  # p_val
    
    return bc_int, bc_float, 1


@njit(cache=True)
def lookup_or_create_bias(bc_int: np.ndarray, bc_float: np.ndarray, 
                          n_entries: int, histseq_key: np.int64,
                          depth: int, if_outcome: bool) -> Tuple[int, int]:
    """
    Find or create a bias count entry for the given history sequence.
    
    Returns:
        (index, new_n_entries)
    """
    # Search for existing entry
    for i in range(n_entries):
        if bc_int[i, 0] == histseq_key:
            return i, n_entries
    
    # Create new entry
    idx = n_entries
    bc_int[idx, 0] = histseq_key
    bc_int[idx, 1] = 0  # rightCh
    bc_int[idx, 2] = 0  # total
    bc_int[idx, 3] = 1 if if_outcome else 0  # alg_type (0=no outcome, 1=with outcome)
    bc_int[idx, 4] = depth
    bc_float[idx, 0] = np.nan  # p_val
    
    return idx, n_entries + 1


# ==============================================================================
# Core Algorithm Functions
# ==============================================================================

@njit(cache=True)
def detect_and_update_biases(
    choices: np.ndarray, rewards: np.ndarray, n_trials: int,
    bc_int: np.ndarray, bc_float: np.ndarray, n_bc_entries: int,
    choice: int, max_depth: int, alpha: float, right_choice: int
) -> Tuple[np.ndarray, int, int]:
    """
    Detect biases based on history and update bias counts.
    
    Args:
        choices, rewards: history arrays (only first n_trials entries are valid)
        n_trials: number of trials in history
        bc_int, bc_float: bias count storage
        n_bc_entries: current number of bias entries
        choice: current trial's choice
        max_depth: maximum depth for algorithm
        alpha: significance level
        right_choice: which choice counts as "right" (typically 1)
    
    Returns:
        significant_biases: array of (freq, p_val, if_outcome, depth) for detected biases
        n_significant: number of significant biases detected
        new_n_bc_entries: updated number of bias entries
    """
    # Max possible biases: (max_depth+1) depths × 2 outcome modes - 1 (no depth=0 with outcome)
    max_biases = (max_depth + 1) * 2
    significant_biases = np.zeros((max_biases, 4), dtype=np.float64)
    n_significant = 0
    
    curr_tri = n_trials + 1  # 1-based trial number
    
    for dep in range(max_depth + 1):
        for if_out_int in range(2):
            if_out = bool(if_out_int)
            
            # Skip depth=0 with outcome (same as depth=0 without outcome)
            if if_out and dep == 0:
                continue
            
            # Need enough history
            if n_trials < dep:
                continue
            
            # Encode history sequence
            histseq_key = encode_histseq(choices, rewards, n_trials, dep, if_out)
            
            # Lookup or create bias entry
            idx, n_bc_entries = lookup_or_create_bias(
                bc_int, bc_float, n_bc_entries, histseq_key, dep, if_out)
            
            # Check for existing significant bias (before updating with current choice)
            if bc_int[idx, 2] > 0:  # total > 0
                p_val = bc_float[idx, 0]
                if p_val < alpha:
                    freq = bc_int[idx, 1] / bc_int[idx, 2]  # rightCh / total
                    significant_biases[n_significant, 0] = freq
                    significant_biases[n_significant, 1] = p_val
                    significant_biases[n_significant, 2] = 1.0 if if_out else 0.0
                    significant_biases[n_significant, 3] = np.float64(dep)
                    n_significant += 1
            
            # Update bias count with current choice
            if choice == right_choice:
                bc_int[idx, 1] += 1  # rightCh
            bc_int[idx, 2] += 1  # total
            
            # Recalculate p-value
            bc_float[idx, 0] = binomial_pvalue_two_tailed(
                int(bc_int[idx, 1]), int(bc_int[idx, 2]), 0.5)
    
    return significant_biases, n_significant, n_bc_entries


@njit(cache=True)
def select_bias(significant_biases: np.ndarray, n_significant: int) -> Tuple[float, int, int, float, int]:
    """
    Select the most extreme bias to punish.
    
    Returns:
        (p_comp_chx, detected, depth, magnitude, which_alg)
    """
    if n_significant == 0:
        return 0.5, 0, -1, 0.0, 0
    
    # Find bias with largest deviation from 0.5
    max_dev = 0.0
    max_idx = 0
    
    for i in range(n_significant):
        freq = significant_biases[i, 0]
        dev = abs(freq - 0.5)
        if dev > max_dev:
            max_dev = dev
            max_idx = i
    
    freq = significant_biases[max_idx, 0]
    if_out = significant_biases[max_idx, 2] > 0.5
    dep = int(significant_biases[max_idx, 3])
    
    p_comp_chx = 1.0 - freq  # Computer chooses opposite
    detected = 2 if freq > 0.5 else 1
    magnitude = freq - 0.5
    which_alg = 2 if if_out else 1
    
    return p_comp_chx, detected, dep, magnitude, which_alg


@njit(cache=True)
def run_step_numba(
    choices: np.ndarray, rewards: np.ndarray, com_pRs: np.ndarray, n_trials: int,
    bc_int: np.ndarray, bc_float: np.ndarray, n_bc_entries: int,
    tb_int: np.ndarray, tb_float: np.ndarray,
    choice: int, max_depth: int, alpha: float, 
    right_choice: int, left_choice: int, rand_val: float
) -> Tuple[int, int, float, int, int, float, int]:
    """
    Execute one step of the matching pennies task.
    
    Args:
        choices, rewards, com_pRs: history arrays
        n_trials: current number of trials
        bc_int, bc_float: bias count storage
        n_bc_entries: number of bias entries
        tb_int, tb_float: trial bias storage
        choice: agent's choice
        max_depth, alpha: algorithm parameters
        right_choice, left_choice: choice encoding
        rand_val: random value for computer choice
    
    Returns:
        (reward, new_n_trials, new_n_bc_entries, detected, depth, magnitude, which_alg)
    """
    # Detect and update biases
    sig_biases, n_sig, n_bc_entries = detect_and_update_biases(
        choices, rewards, n_trials,
        bc_int, bc_float, n_bc_entries,
        choice, max_depth, alpha, right_choice
    )
    
    # Select bias to punish
    p_comp_chx, detected, depth, magnitude, which_alg = select_bias(sig_biases, n_sig)
    
    # Computer makes choice
    if rand_val < p_comp_chx:
        comp_choice = right_choice
    else:
        comp_choice = left_choice
    
    # Determine reward
    reward = 1 if comp_choice == choice else 0
    
    # Update history
    choices[n_trials] = choice
    rewards[n_trials] = reward
    com_pRs[n_trials] = p_comp_chx
    
    # Update trial bias
    tb_int[n_trials, 0] = detected
    tb_int[n_trials, 1] = depth
    tb_float[n_trials, 0] = magnitude
    tb_int[n_trials, 2] = which_alg
    
    return reward, n_trials + 1, n_bc_entries, detected, depth, magnitude, which_alg


@njit(cache=True)
def cal_mp_algs_numba(
    choices: np.ndarray, rewards: np.ndarray, com_pRs: np.ndarray, n_trials: int,
    bc_int: np.ndarray, bc_float: np.ndarray, n_bc_entries: int,
    tb_int: np.ndarray, tb_float: np.ndarray,
    max_depth: int, alpha: float, right_choice: int
) -> int:
    """
    Calculate MP algorithm outputs for an existing dataset.
    This is equivalent to Julia's cal_MP_algs function.
    
    Returns:
        new_n_bc_entries
    """
    # First trial: only algorithm 0 (no history yet)
    if n_trials > 0:
        histseq_key = np.int64(-1)  # Empty sequence
        idx, n_bc_entries = lookup_or_create_bias(
            bc_int, bc_float, n_bc_entries, histseq_key, 0, False)
        
        # Update with first trial choice
        if choices[0] == right_choice:
            bc_int[idx, 1] += 1
        bc_int[idx, 2] += 1
        bc_float[idx, 0] = binomial_pvalue_two_tailed(
            int(bc_int[idx, 1]), int(bc_int[idx, 2]), 0.5)
        
        com_pRs[0] = 0.5
        tb_int[0, 0] = 0  # detected
        tb_int[0, 1] = -1  # depth
        tb_float[0, 0] = 0.0  # magnitude
        tb_int[0, 2] = 0  # which_alg
    
    # Process remaining trials
    for tri in range(1, n_trials):
        # Detect biases using history up to tri-1
        sig_biases, n_sig, n_bc_entries = detect_and_update_biases(
            choices, rewards, tri,  # History up to (not including) current trial
            bc_int, bc_float, n_bc_entries,
            choices[tri],  # Current trial's choice for updating
            max_depth, alpha, right_choice
        )
        
        # Select bias
        p_comp_chx, detected, depth, magnitude, which_alg = select_bias(sig_biases, n_sig)
        
        # Store results
        com_pRs[tri] = p_comp_chx
        tb_int[tri, 0] = detected
        tb_int[tri, 1] = depth
        tb_float[tri, 0] = magnitude
        tb_int[tri, 2] = which_alg
    
    return n_bc_entries


# ==============================================================================
# Parallel Environment Support
# ==============================================================================

@njit(parallel=True, cache=True)
def step_parallel_numba(
    all_choices: np.ndarray,    # (n_sess, max_trials)
    all_rewards: np.ndarray,    # (n_sess, max_trials)
    all_com_pRs: np.ndarray,    # (n_sess, max_trials)
    n_trials: np.ndarray,       # (n_sess,)
    all_bc_int: np.ndarray,     # (n_sess, max_bc, 5)
    all_bc_float: np.ndarray,   # (n_sess, max_bc, 1)
    n_bc_entries: np.ndarray,   # (n_sess,)
    all_tb_int: np.ndarray,     # (n_sess, max_trials, 3)
    all_tb_float: np.ndarray,   # (n_sess, max_trials, 1)
    agent_choices: np.ndarray,  # (n_sess,) choices from agent
    max_depth: int,
    alpha: float,
    right_choice: int,
    left_choice: int,
    rand_vals: np.ndarray       # (n_sess,) pre-generated random values
) -> np.ndarray:
    """
    Step all environments in parallel using Numba prange.
    
    Each environment is processed independently in parallel threads.
    Random values must be pre-generated before calling this function
    to ensure thread safety.
    
    Returns:
        rewards: (n_sess,) array of rewards for each environment
    """
    n_sess = len(agent_choices)
    rewards = np.zeros(n_sess, dtype=np.int32)
    
    for i in prange(n_sess):
        reward, new_n_trials, new_n_bc, _, _, _, _ = run_step_numba(
            all_choices[i], all_rewards[i], all_com_pRs[i], n_trials[i],
            all_bc_int[i], all_bc_float[i], n_bc_entries[i],
            all_tb_int[i], all_tb_float[i],
            agent_choices[i], max_depth, alpha, right_choice, left_choice,
            rand_vals[i]
        )
        rewards[i] = reward
        n_trials[i] = new_n_trials
        n_bc_entries[i] = new_n_bc
    
    return rewards


@njit(parallel=True, cache=True)
def cal_mp_algs_parallel_numba(
    all_choices: np.ndarray,    # (n_sess, max_trials)
    all_rewards: np.ndarray,    # (n_sess, max_trials)
    all_com_pRs: np.ndarray,    # (n_sess, max_trials)
    all_n_trials: np.ndarray,   # (n_sess,) number of trials per env
    all_bc_int: np.ndarray,     # (n_sess, max_bc, 5)
    all_bc_float: np.ndarray,   # (n_sess, max_bc, 1)
    n_bc_entries: np.ndarray,   # (n_sess,)
    all_tb_int: np.ndarray,     # (n_sess, max_trials, 3)
    all_tb_float: np.ndarray,   # (n_sess, max_trials, 1)
    max_depth: int,
    alpha: float,
    right_choice: int
) -> None:
    """
    Calculate MP algorithm outputs for existing datasets in parallel.
    This is the parallel version of cal_mp_algs_numba.
    """
    n_sess = len(all_n_trials)
    
    for i in prange(n_sess):
        n_bc_entries[i] = cal_mp_algs_numba(
            all_choices[i], all_rewards[i], all_com_pRs[i], all_n_trials[i],
            all_bc_int[i], all_bc_float[i], n_bc_entries[i],
            all_tb_int[i], all_tb_float[i],
            max_depth, alpha, right_choice
        )


class ParallelMPEnvironments(RL_multiEnvironment):
    """
    Vectorized storage for multiple MP environments with parallel stepping.
    
    This class stores all environment state in contiguous 2D/3D arrays
    for efficient parallel access using Numba prange.
    
    Arguments:
        n_sess: number of parallel environments
        max_depth: maximum depth for bias detection algorithm
        alpha: significance level for statistical testing
        choices: tuple of (left_choice, right_choice)
        max_trials: maximum trials per environment
        max_bc_entries: maximum bias count entries per environment
    """
    
    def __init__(
        self, 
        n_sess: int,
        max_depth: int = 4,
        alpha: float = 0.05,
        choices: Tuple[int, int] = (0, 1),
        max_trials: int = 2000,
        max_bc_entries: int = 500
    ):
        self.n_sess = n_sess
        self.max_depth = max_depth
        self.alpha = alpha
        self.choices = choices
        self.max_trials = max_trials
        self.max_bc_entries = max_bc_entries
        
        self._rng = np.random.default_rng()
        self._sess_ids: Optional[list] = None
        
        self._allocate_arrays()
    
    def _allocate_arrays(self):
        """Allocate contiguous arrays for all environments."""
        n = self.n_sess
        mt = self.max_trials
        mb = self.max_bc_entries
        
        # History arrays: (n_sess, max_trials)
        self.all_choices = np.zeros((n, mt), dtype=np.int32)
        self.all_rewards = np.zeros((n, mt), dtype=np.int32)
        self.all_com_pRs = np.zeros((n, mt), dtype=np.float64)
        self.n_trials = np.zeros(n, dtype=np.int32)
        
        # Bias count arrays: (n_sess, max_bc, fields)
        self.all_bc_int = np.zeros((n, mb, 5), dtype=np.int64)
        self.all_bc_float = np.zeros((n, mb, 1), dtype=np.float64)
        self.n_bc_entries = np.ones(n, dtype=np.int32)  # Start at 1 for alg-0
        
        # Initialize first entry for algorithm 0 in each environment
        for i in range(n):
            self.all_bc_int[i, 0, 0] = -1  # histseq_key for empty
            self.all_bc_float[i, 0, 0] = np.nan  # p_val
        
        # Trial bias arrays: (n_sess, max_trials, fields)
        self.all_tb_int = np.zeros((n, mt, 3), dtype=np.int32)
        self.all_tb_float = np.zeros((n, mt, 1), dtype=np.float64)
    
    def set_random_seed(self, seed: int):
        """Set random seed for reproducibility."""
        self._rng = np.random.default_rng(seed)
        return self
    
    def new_session(self) -> None:
        """Reset all environments to initial state."""
        self._allocate_arrays()
    
    @property
    def sess_ids(self) -> list:
        """Session IDs for each environment."""
        if self._sess_ids is None:
            return list(range(self.n_sess))
        return self._sess_ids
    
    @sess_ids.setter
    def sess_ids(self, val):
        if val is not None:
            assert len(val) == self.n_sess
        self._sess_ids = val
    
    def step(self, choice: np.ndarray) -> np.ndarray:
        """
        Step all environments in parallel.
        
        Args:
            choices: (n_sess,) array of agent choices
            
        Returns:
            rewards: (n_sess,) array of rewards
        """
        choice = np.asarray(choice, dtype=np.int32)
        assert len(choice) == self.n_sess
        
        # Pre-generate random values for thread safety
        rand_vals = self._rng.random(self.n_sess)
        
        return step_parallel_numba(
            self.all_choices, self.all_rewards, self.all_com_pRs, self.n_trials,
            self.all_bc_int, self.all_bc_float, self.n_bc_entries,
            self.all_tb_int, self.all_tb_float,
            choice, self.max_depth, self.alpha,
            self.choices[1], self.choices[0], rand_vals
        )
    
    @property
    def history(self) -> pd.DataFrame:
        """Obtain history of all environments as DataFrame."""
        dfs = []
        for i, sid in enumerate(self.sess_ids):
            n = self.n_trials[i]
            if n > 0:
                df = pd.DataFrame({
                    'choice': self.all_choices[i, :n].astype(int),
                    'reward': self.all_rewards[i, :n].astype(int),
                    'com_pR': self.all_com_pRs[i, :n].astype(float),
                    'sess_id': sid,
                    'tri_id': np.arange(n)
                })
                dfs.append(df)
        
        if dfs:
            return pd.concat(dfs, ignore_index=True)
        return pd.DataFrame(columns=['choice', 'reward', 'com_pR', 'sess_id', 'tri_id'])
    
    @property
    def biasCount(self) -> pd.DataFrame:
        """Get bias count from all environments."""
        dfs = []
        for i, sid in enumerate(self.sess_ids):
            n = self.n_bc_entries[i]
            histseq_strs = [decode_histseq_to_string(int(k)) 
                           for k in self.all_bc_int[i, :n, 0]]
            df = pd.DataFrame({
                'histseq': histseq_strs,
                'rightCh': self.all_bc_int[i, :n, 1].astype(int),
                'total': self.all_bc_int[i, :n, 2].astype(int),
                'p_val': self.all_bc_float[i, :n, 0].astype(float),
                'alg_type': self.all_bc_int[i, :n, 3].astype(int),
                'depth': self.all_bc_int[i, :n, 4].astype(int),
                'sess_id': sid
            })
            dfs.append(df)
        
        if dfs:
            return pd.concat(dfs, ignore_index=True)
        return pd.DataFrame(columns=['histseq', 'rightCh', 'total', 'p_val',
                                      'alg_type', 'depth', 'sess_id'])
    
    @property
    def trialBias(self) -> pd.DataFrame:
        """Get trial bias info from all environments."""
        dfs = []
        for i, sid in enumerate(self.sess_ids):
            n = self.n_trials[i]
            if n > 0:
                # Generate histseq strings for detected biases
                histseq_strs = []
                for j in range(n):
                    if self.all_tb_int[i, j, 0] > 0 and j > 0:
                        if_out = self.all_tb_int[i, j, 2] == 2
                        depth = self.all_tb_int[i, j, 1]
                        key = encode_histseq(
                            self.all_choices[i], self.all_rewards[i],
                            j, depth, if_out
                        )
                        histseq_strs.append(decode_histseq_to_string(key))
                    else:
                        histseq_strs.append("")
                
                df = pd.DataFrame({
                    'detected': self.all_tb_int[i, :n, 0].astype(int),
                    'depth': self.all_tb_int[i, :n, 1].astype(int),
                    'magnitude': self.all_tb_float[i, :n, 0].astype(float),
                    'which_alg': self.all_tb_int[i, :n, 2].astype(int),
                    'histseq': histseq_strs,
                    'sess_id': sid,
                    'tri_id': np.arange(n)
                })
                dfs.append(df)
        
        if dfs:
            return pd.concat(dfs, ignore_index=True)
        return pd.DataFrame(columns=['detected', 'depth', 'magnitude',
                                      'which_alg', 'histseq', 'sess_id', 'tri_id'])
    
    @property
    def n_actions(self) -> int:
        return len(self.choices)
    
    @property
    def reward_probs_all(self) -> np.ndarray:
        """Get reward probabilities for all trials."""
        hist = self.history
        pR = hist.pivot(columns='sess_id', values='com_pR').to_numpy()
        return np.stack((1 - pR, pR), axis=-1)
    
    def eval_on_dataset(self, df: pd.DataFrame, ch_col: str = 'choice',
                        rew_col: str = 'reward', tri_id_col: Optional[str] = 'tri_id',
                        sess_id_col: Optional[str] = 'sess_id'):
        """
        Calculate MP task variables on an existing behavioral dataset.
        Uses parallel processing for multiple sessions.
        """
        df = df.rename(columns={
            ch_col: 'choice',
            rew_col: 'reward',
        }).astype({
            'choice': np.int64,
            'reward': np.int64,
        })
        
        if tri_id_col is not None:
            df = df.sort_values(tri_id_col).drop(columns=tri_id_col)
        
        if sess_id_col is None:
            # Single session
            self.n_sess = 1
            self._allocate_arrays()
            
            choices = df['choice'].to_numpy()
            rewards = df['reward'].to_numpy()
            n_trials = len(choices)
            
            self.all_choices[0, :n_trials] = choices
            self.all_rewards[0, :n_trials] = rewards
            self.n_trials[0] = n_trials
            
            self.n_bc_entries[0] = cal_mp_algs_numba(
                self.all_choices[0], self.all_rewards[0], self.all_com_pRs[0], n_trials,
                self.all_bc_int[0], self.all_bc_float[0], self.n_bc_entries[0],
                self.all_tb_int[0], self.all_tb_float[0],
                self.max_depth, self.alpha, self.choices[1]
            )
        else:
            # Multiple sessions - process in parallel
            sess_ids = sorted(df[sess_id_col].unique().tolist())
            self.n_sess = len(sess_ids)
            self._sess_ids = sess_ids
            self._allocate_arrays()
            
            # Copy data into arrays
            all_n_trials = np.zeros(self.n_sess, dtype=np.int32)
            for idx, (sid, group) in enumerate(df.groupby(sess_id_col)):
                choices = group['choice'].to_numpy()
                rewards = group['reward'].to_numpy()
                n_trials = len(choices)
                
                self.all_choices[idx, :n_trials] = choices
                self.all_rewards[idx, :n_trials] = rewards
                all_n_trials[idx] = n_trials
            
            self.n_trials[:] = all_n_trials
            
            # Run parallel calculation
            cal_mp_algs_parallel_numba(
                self.all_choices, self.all_rewards, self.all_com_pRs, self.n_trials,
                self.all_bc_int, self.all_bc_float, self.n_bc_entries,
                self.all_tb_int, self.all_tb_float,
                self.max_depth, self.alpha, self.choices[1]
            )


# ==============================================================================
# Python Wrapper Class
# ==============================================================================

class MatchingPenniesTaskNumba:
    """
    Single session matching pennies task using Numba.
    Internal class used by multiEnvironmentBanditsMP_numba.
    """
    
    def __init__(self, max_depth: int = 4, alpha: float = 0.05, 
                 choices: Tuple[int, int] = (0, 1), max_trials: int = 50000):
        assert max_depth >= 0, "maxdepth has to be non-negative"
        assert 0 < alpha < 1, "alpha has to be in the range of (0,1)"
        assert len(choices) == 2, "only 2-armed bandit task is supported"
        
        self.max_depth = max_depth
        self.alpha = alpha
        self.choices = choices
        self.max_trials = max_trials
        
        # Max bias entries: for each (depth, if_outcome) pair
        # depth 0-4, if_outcome True/False = 5*2 - 1 = 9 alg types
        # Each can have 2^(depth * bits_per_trial) history sequences
        # For max_depth=4 with outcome: 2^8 = 256 per type
        # Conservative estimate: 10000 entries should be plenty
        self.max_bc_entries = 10000
        
        self.reset()
    
    def reset(self):
        """Reset all state for a new session."""
        # History arrays
        self._choices = np.zeros(self.max_trials, dtype=np.int32)
        self._rewards = np.zeros(self.max_trials, dtype=np.int32)
        self._com_pRs = np.zeros(self.max_trials, dtype=np.float64)
        self._n_trials = 0
        
        # Bias count arrays
        self._bc_int, self._bc_float, self._n_bc_entries = init_bias_count_arrays(self.max_bc_entries)
        
        # Trial bias arrays
        self._tb_int = np.zeros((self.max_trials, 3), dtype=np.int32)  # detected, depth, which_alg
        self._tb_float = np.zeros((self.max_trials, 1), dtype=np.float64)  # magnitude
        
        # Random state
        self._rng = np.random.default_rng()
    
    def set_random_seed(self, seed: int):
        """Set random seed for reproducibility."""
        self._rng = np.random.default_rng(seed)
    
    def step(self, choice: int) -> int:
        """Execute one trial and return reward."""
        assert choice in self.choices, f"choice must be in {self.choices}"
        
        rand_val = self._rng.random()
        
        reward, self._n_trials, self._n_bc_entries, _, _, _, _ = run_step_numba(
            self._choices, self._rewards, self._com_pRs, self._n_trials,
            self._bc_int, self._bc_float, self._n_bc_entries,
            self._tb_int, self._tb_float,
            choice, self.max_depth, self.alpha,
            self.choices[1], self.choices[0], rand_val
        )
        
        return reward
    
    def get_history(self) -> dict:
        """Return history as dict of arrays."""
        n = self._n_trials
        return {
            'choice': self._choices[:n].copy(),
            'reward': self._rewards[:n].copy(),
            'com_pR': self._com_pRs[:n].copy()
        }
    
    def get_trial_bias(self) -> dict:
        """Return trial bias info as dict of arrays."""
        n = self._n_trials
        return {
            'detected': self._tb_int[:n, 0].copy(),
            'depth': self._tb_int[:n, 1].copy(),
            'magnitude': self._tb_float[:n, 0].copy(),
            'which_alg': self._tb_int[:n, 2].copy()
        }
    
    def get_bias_count(self) -> dict:
        """Return bias count as dict of arrays."""
        n = self._n_bc_entries
        return {
            'histseq_key': self._bc_int[:n, 0].copy(),
            'rightCh': self._bc_int[:n, 1].copy(),
            'total': self._bc_int[:n, 2].copy(),
            'p_val': self._bc_float[:n, 0].copy(),
            'alg_type': self._bc_int[:n, 3].copy(),
            'depth': self._bc_int[:n, 4].copy()
        }

# TODO deprecate this class in the future, since it's superseded by `ParallelMPEnvironments`
class multiEnvironmentBanditsMP_numba(RL_multiEnvironment):
    """
    Multi-session matching pennies environment using Numba.
    
    This class provides the same interface as multiEnvironmentBanditsMP_julia
    but uses pure Python+Numba for performance without Julia dependency.
    
    Arguments:
        n_sess: number of sessions to run simultaneously
        max_depth: the maximum depth of computer algorithm
        alpha: the p-value significance level for statistically testing bias
        ch_code: how to code the L and R choice
    """
    
    def __init__(self, n_sess: int = 1, max_depth: int = 4, 
                 alpha: float = 0.05, ch_code: Sequence = (0, 1)):
        self.n_sess = n_sess
        self._max_depth = max_depth
        self._alpha = alpha
        self._choices = tuple(ch_code)
        
        self._sess_ids = None
        self._rng = np.random.default_rng()
        
        self.new_session()
    
    def set_random_seed(self, seed: int):
        """Set random seed for reproducibility."""
        self._rng = np.random.default_rng(seed)
        # Propagate to each task with derived seeds
        for i, mpt in enumerate(self._MPTs):
            mpt.set_random_seed(seed + i)
        return self
    
    def new_session(self) -> None:
        """Reset all sessions."""
        self._MPTs = [
            MatchingPenniesTaskNumba(
                self._max_depth, self._alpha, self._choices
            ) for _ in range(self.n_sess)
        ]
        # Set random seeds
        base_seed = self._rng.integers(0, 2**31)
        for i, mpt in enumerate(self._MPTs):
            mpt.set_random_seed(base_seed + i)
    
    @property
    def sess_ids(self) -> list:
        """Session IDs for each session."""
        if self._sess_ids is None:
            return list(range(self.n_sess))
        return self._sess_ids
    
    @sess_ids.setter
    def sess_ids(self, val):
        if val is not None:
            assert len(val) == self.n_sess, \
                f"length of the value {len(val)} does not match attribute `n_sess` {self.n_sess}"
        self._sess_ids = val
    
    @property
    def history(self) -> pd.DataFrame:
        """Obtain history of all sessions as DataFrame."""
        dfs = []
        for i, (sid, mpt) in enumerate(zip(self.sess_ids, self._MPTs)):
            hist = mpt.get_history()
            n = len(hist['choice'])
            df = pd.DataFrame({
                'choice': hist['choice'].astype(int),
                'reward': hist['reward'].astype(int),
                'com_pR': hist['com_pR'].astype(float),
                'sess_id': sid,
                'tri_id': np.arange(n)
            })
            dfs.append(df)
        
        if dfs:
            result = pd.concat(dfs, ignore_index=True)
        else:
            result = pd.DataFrame(columns=['choice', 'reward', 'com_pR', 'sess_id', 'tri_id'])
        
        return result
    
    @property
    def biasCount(self) -> pd.DataFrame:
        """Get bias count from all sessions."""
        dfs = []
        for i, (sid, mpt) in enumerate(zip(self.sess_ids, self._MPTs)):
            bc = mpt.get_bias_count()
            n = len(bc['histseq_key'])
            
            # Decode histseq keys to strings for compatibility
            histseq_strs = [decode_histseq_to_string(int(k)) for k in bc['histseq_key']]
            
            df = pd.DataFrame({
                'histseq': histseq_strs,
                'rightCh': bc['rightCh'].astype(int),
                'total': bc['total'].astype(int),
                'p_val': bc['p_val'].astype(float),
                'alg_type': bc['alg_type'].astype(int),
                'depth': bc['depth'].astype(int),
                'sess_id': sid
            })
            dfs.append(df)
        
        if dfs:
            result = pd.concat(dfs, ignore_index=True)
        else:
            result = pd.DataFrame(columns=['histseq', 'rightCh', 'total', 'p_val', 
                                           'alg_type', 'depth', 'sess_id'])
        
        return result
    
    @property
    def trialBias(self) -> pd.DataFrame:
        """Get trial bias info from all sessions."""
        dfs = []
        for i, (sid, mpt) in enumerate(zip(self.sess_ids, self._MPTs)):
            tb = mpt.get_trial_bias()
            hist = mpt.get_history()
            n = len(tb['detected'])
            
            # Generate histseq strings for detected biases
            histseq_strs = []
            for j in range(n):
                if tb['detected'][j] > 0 and j > 0:
                    if_out = tb['which_alg'][j] == 2
                    depth = tb['depth'][j]
                    key = encode_histseq(hist['choice'], hist['reward'], j, depth, if_out)
                    histseq_strs.append(decode_histseq_to_string(key))
                else:
                    histseq_strs.append("")
            
            df = pd.DataFrame({
                'detected': tb['detected'].astype(int),
                'depth': tb['depth'].astype(int),
                'magnitude': tb['magnitude'].astype(float),
                'which_alg': tb['which_alg'].astype(int),
                'histseq': histseq_strs,
                'sess_id': sid,
                'tri_id': np.arange(n)
            })
            dfs.append(df)
        
        if dfs:
            result = pd.concat(dfs, ignore_index=True)
        else:
            result = pd.DataFrame(columns=['detected', 'depth', 'magnitude', 
                                           'which_alg', 'histseq', 'sess_id', 'tri_id'])
        
        return result
    
    @property
    def n_actions(self) -> int:
        return len(self._choices)
    
    @property
    def reward_probs_all(self) -> np.ndarray:
        """Get reward probabilities for all trials."""
        hist = self.history
        pR = hist.pivot(columns='sess_id', values='com_pR').to_numpy()
        return np.stack((1 - pR, pR), axis=-1)
    
    def step(self, choice: np.ndarray) -> np.ndarray:
        """Step all sessions forward with given choices."""
        choice = np.asarray(choice)
        assert len(choice) == self.n_sess, \
            f"choice array length {len(choice)} != n_sess {self.n_sess}"
        
        rewards = np.zeros(self.n_sess, dtype=np.float64)
        for i, (c, mpt) in enumerate(zip(choice, self._MPTs)):
            rewards[i] = mpt.step(int(c))
        
        return rewards
    
    def eval_on_dataset(self, df: pd.DataFrame, ch_col: str = 'choice', 
                        rew_col: str = 'reward', tri_id_col: Optional[str] = 'tri_id',
                        sess_id_col: Optional[str] = 'sess_id'):
        """
        Calculate MP task variables on an existing behavioral dataset.
        
        Args:
            df: DataFrame with behavioral data
            ch_col: column name for choices
            rew_col: column name for rewards
            tri_id_col: column for trial index (for sorting)
            sess_id_col: column for session ID
        """
        df = df.rename(columns={
            ch_col: 'choice',
            rew_col: 'reward',
        }).astype({
            'choice': np.int64,
            'reward': np.int64,
        })
        
        if tri_id_col is not None:
            df = df.sort_values(tri_id_col).drop(columns=tri_id_col)
        
        if sess_id_col is None:
            # Single session
            self.n_sess = 1
            self.new_session()
            
            choices = df['choice'].to_numpy()
            rewards = df['reward'].to_numpy()
            n_trials = len(choices)
            
            mpt = self._MPTs[0]
            mpt._choices[:n_trials] = choices
            mpt._rewards[:n_trials] = rewards
            mpt._n_trials = n_trials
            
            mpt._n_bc_entries = cal_mp_algs_numba(
                mpt._choices, mpt._rewards, mpt._com_pRs, n_trials,
                mpt._bc_int, mpt._bc_float, mpt._n_bc_entries,
                mpt._tb_int, mpt._tb_float,
                self._max_depth, self._alpha, self._choices[1]
            )
        else:
            # Multiple sessions
            sess_ids = sorted(df[sess_id_col].unique().tolist())
            self.n_sess = len(sess_ids)
            self.sess_ids = sess_ids
            self.new_session()
            
            for sid, group in df.groupby(sess_id_col):
                idx = list(self.sess_ids).index(sid)
                mpt = self._MPTs[idx]
                
                choices = group['choice'].to_numpy()
                rewards = group['reward'].to_numpy()
                n_trials = len(choices)
                
                mpt._choices[:n_trials] = choices
                mpt._rewards[:n_trials] = rewards
                mpt._n_trials = n_trials
                
                mpt._n_bc_entries = cal_mp_algs_numba(
                    mpt._choices, mpt._rewards, mpt._com_pRs, n_trials,
                    mpt._bc_int, mpt._bc_float, mpt._n_bc_entries,
                    mpt._tb_int, mpt._tb_float,
                    self._max_depth, self._alpha, self._choices[1]
                )
