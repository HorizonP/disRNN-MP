# Parallel Environment Simulation - Walkthrough

## Summary

Added `ParallelMPEnvironments` class to `MP_numba.py` for multi-threaded parallel simulation of matching pennies environments using Numba's `prange`.

## Changes Made

### [MP_numba.py](file:///home/peiyu/Documents/codes/2023Oct05_RNN_behavioral_modeling_test/disRNN_MatchingPennies/disRNN_MP/agent/MP_numba.py)

1. **New parallel step function** (`step_parallel_numba()`) using `prange` for multi-threaded execution
2. **New parallel eval function** (`cal_mp_algs_parallel_numba()`) for evaluating datasets in parallel
3. **New `ParallelMPEnvironments` class** with:
   - Contiguous 2D/3D array storage for all environments
   - Same DataFrame interface as `multiEnvironmentBanditsMP_numba`
   - Pre-generated random values for thread-safe execution

Key additions (lines 574-920):
- `step_parallel_numba()` - Numba function with `@njit(parallel=True)` using `prange`
- `cal_mp_algs_parallel_numba()` - Parallel dataset evaluation
- `ParallelMPEnvironments` - Python wrapper class

### [test_MP_numba.py](file:///home/peiyu/Documents/codes/2023Oct05_RNN_behavioral_modeling_test/disRNN_MatchingPennies/tests/test_MP_numba.py)

Added 7 new tests for `ParallelMPEnvironments`:
- `test_initialization` - Basic array shape verification
- `test_step` - Parallel stepping works correctly
- `test_history_format` - DataFrame interface matches serial
- `test_correctness_vs_serial` - Algorithm produces identical results
- `test_deterministic_with_seed` - Same seed = same results
- `test_eval_on_dataset` - Evaluation produces same results as serial
- `test_parallel_performance` - Performance benchmark

## Usage Example

```python
from disRNN_MP.agent.MP_numba import ParallelMPEnvironments

# Create 100 parallel environments
env = ParallelMPEnvironments(n_envs=100, max_trials=1000, max_bc_entries=500)
env.set_random_seed(42)

# Step all environments in parallel
for _ in range(500):
    choices = np.random.randint(0, 2, size=100)
    rewards = env.step(choices)  # Uses prange internally

# Get history as DataFrame (same interface as before)
history_df = env.history
bias_df = env.biasCount
```

## Memory Layout

| Array | Shape | Size (100 envs × 1000 trials) |
|-------|-------|-------------------------------|
| `all_choices` | (n_envs, max_trials) | 0.4 MB |
| `all_rewards` | (n_envs, max_trials) | 0.4 MB |
| `all_com_pRs` | (n_envs, max_trials) | 0.8 MB |
| `all_bc_int` | (n_envs, max_bc, 5) | 2.0 MB |
| **Total** | | **~5 MB** |

## Test Results

```
22 passed, 2 skipped
```

All existing tests continue to pass. New parallel tests verify correctness against serial implementation.
