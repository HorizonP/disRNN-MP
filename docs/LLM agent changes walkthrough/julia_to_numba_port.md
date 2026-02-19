# Matching Pennies: Julia → Python+Numba Port

## Summary

Successfully ported the matching pennies algorithm from Julia to Python+Numba with **100% numerical equivalence** verified.

## Files Created/Modified

| File | Description |
|------|-------------|
| `disRNN_MP/agent/MP_numba.py` | Pure Python+Numba implementation |
| `tests/test_MP_numba.py` | Test suite with Julia comparison |

---

## Test Results

```
tests/test_MP_numba.py::TestBinomialPValue::test_binomial_pvalue_matches_scipy PASSED
tests/test_MP_numba.py::TestHistseqEncoding::test_encode_decode_roundtrip PASSED
tests/test_MP_numba.py::TestHistseqEncoding::test_empty_sequence PASSED
tests/test_MP_numba.py::TestMatchingPenniesTaskNumba::test_basic_step PASSED
tests/test_MP_numba.py::TestMatchingPenniesTaskNumba::test_deterministic_with_seed PASSED
tests/test_MP_numba.py::TestMultiEnvironmentBanditsMP_numba::test_initialization PASSED
tests/test_MP_numba.py::TestMultiEnvironmentBanditsMP_numba::test_step PASSED
tests/test_MP_numba.py::TestMultiEnvironmentBanditsMP_numba::test_history_format PASSED
tests/test_MP_numba.py::TestMultiEnvironmentBanditsMP_numba::test_eval_on_dataset PASSED
tests/test_MP_numba.py::TestNumericalEquivalenceWithJulia::test_single_session_equivalence PASSED
tests/test_MP_numba.py::TestNumericalEquivalenceWithJulia::test_multi_session_equivalence PASSED
tests/test_MP_numba.py::TestNumericalEquivalenceWithJulia::test_eval_on_dataset_equivalence PASSED
tests/test_MP_numba.py::TestPerformance::test_numba_warmup PASSED
tests/test_MP_numba.py::TestPerformance::test_performance_benchmark PASSED

================== 15 passed, 2 skipped in 35.67s ==================
```

---

## Implementation Highlights

### Numba JIT Functions
- `binomial_pmf()` / `binomial_pvalue_two_tailed()` - Statistical tests matching scipy
- `encode_histseq()` / `decode_histseq_to_string()` - History encoding as integers
- `detect_and_update_biases()` - Core bias detection
- `run_step_numba()` / `cal_mp_algs_numba()` - Step and batch evaluation
- `step_parallel_numba()` / `cal_mp_algs_parallel_numba()` - Parallel versions using `prange`

### Classes
- `MatchingPenniesTaskNumba` - Single session wrapper
- `multiEnvironmentBanditsMP_numba` - Multi-session serial wrapper (matches Julia interface)
- `ParallelMPEnvironments` - Multi-session parallel wrapper using Numba `prange`

### Key Design Decisions
1. **History encoding**: String keys → integer keys for Numba compatibility
2. **DataFrames → NumPy arrays**: Pre-allocated arrays for performance
3. **P-value**: Minimum likelihood method matches scipy/Julia exactly
4. **Parallel support**: `prange` for multi-environment parallel stepping

---

## Usage

### Serial (matches Julia interface)
```python
from disRNN_MP.agent.MP_numba import multiEnvironmentBanditsMP_numba
import numpy as np

env = multiEnvironmentBanditsMP_numba(n_sess=5, max_depth=4, alpha=0.05)
env.set_random_seed(42)

for _ in range(1000):
    rewards = env.step(np.random.randint(0, 2, size=5))

history = env.history  # Same format as Julia version
```

### Parallel
```python
from disRNN_MP.agent.MP_numba import ParallelMPEnvironments

env = ParallelMPEnvironments(n_envs=32, max_depth=4, alpha=0.05)
env.set_random_seed(42)

for _ in range(1000):
    rewards = env.step(np.random.randint(0, 2, size=32))

history = env.history
```

## Run Tests
```bash
pixi run -e default pytest tests/test_MP_numba.py -v
```
