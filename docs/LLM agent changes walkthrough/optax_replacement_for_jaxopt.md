# Replacement of jaxopt with optax in classic_RL

## Summary

Deprecated `jaxopt` as the default optimizer in `RLmodelWrapper` and replaced it with `optax.lbfgs`. This removes `jaxopt` from mandatory dependencies while preserving backward compatibility for legacy code.

## Files Created/Modified

| File | Description |
|------|-------------|
| `disRNN_MP/classic_RL/classic_RL.py` | Added `optax` support, deprecated `jaxopt`, added `optimizer` arg |
| `scripts/archive_scripts/legacy_classicRL.py` | New home for legacy `_classicRL` and `_forgetQ` classes |
| `tests/test_classicRL.py` | Converted to proper pytest suite, updated expected values |
| `usage_example/classic_RL/classic_RL.md` | Updated docs with optimizer options |
| `pyproject.toml` | Moved `jaxopt` to optional `legacy` extra |
| `requirements.txt` | Commented out `jaxopt` |

---

## Test Results

```
tests/test_classicRL.py::test_forgetQ_fitting PASSED
tests/test_classicRL.py::test_optimizer_deprecation_warning PASSED

================== 2 passed in 12.34s ==================
```

---

## Implementation Highlights

### 1. Optax L-BFGS Implementation
- Added `_run_opt_bounded` helper function to handle box constraints with `optax`
- Implemented `_fit_optax` method in `RLmodelWrapper`
- Default behavior now uses `optax.lbfgs`
- Key difference: `optax` implementation uses unnormalized log-likelihood (`norm=False`) which scales the loss differently, resulting in slightly different but equally valid parameter solutions.

### 2. Legacy Support
- `optimizer='jaxopt'` argument triggers legacy behavior
- Lazy import of `jaxopt` ensures it's only required when explicitly requested
- Raises `DeprecationWarning` when used
- Suggests installation of `disRNN-MP[legacy]` if missing

### 3. Cleanup
- Moved unused `_classicRL` and `_forgetQ` classes to archive script
- Removed top-level `jaxopt` import to prevent dependency errors

---

## Usage

### Default (Optax)
```python
import jax
from disRNN_MP.classic_RL import forgetQ, RLmodelWrapper

rlmd = forgetQ()
# Uses optax L-BFGS by default
mdw = RLmodelWrapper(rlmd, dataset=trainD, rng=jax.random.key(0), run_fitting=True)
```

### Legacy (Jaxopt)
Requires `pip install .[legacy]` or `pip install jaxopt`

```python
# Explicitly use legacy optimizer
mdw.fit(jax.random.key(0), optimizer='jaxopt')
```

## Run Tests
```bash
pixi run pytest tests/test_classicRL.py -v
```
