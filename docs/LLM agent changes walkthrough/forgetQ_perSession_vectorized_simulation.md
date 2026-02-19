# forgetQ_perSession: Vectorized Per-Session Parameter Simulation

## Summary

Added `forgetQ_perSession`, a variant of the `forgetQ` model that supports per-session parameters for simulating heterogeneous agents. This enables sampling parameters from a multivariate Gaussian (fitted to session-level behavioral fits) and running vectorized simulations where each session has unique parameter values.

## Files Created/Modified

| File | Description |
|------|-------------|
| `disRNN_MP/classic_RL/forgetQ_hetero.py` | New class implementing vectorized per-session forgetQ |
| `disRNN_MP/classic_RL/__init__.py` | Added export for `forgetQ_perSession` |
| `tests/test_forgetQ_perSession.py` | Comprehensive test suite (7 tests) |

---

## Test Results

```
tests/test_forgetQ_perSession.py::TestForgetQPerSession::test_init_values_shape PASSED
tests/test_forgetQ_perSession.py::TestForgetQPerSession::test_init_values_n_sess_mismatch_raises PASSED
tests/test_forgetQ_perSession.py::TestForgetQPerSession::test_value_update_shape PASSED
tests/test_forgetQ_perSession.py::TestForgetQPerSession::test_value_update_per_session_params PASSED
tests/test_forgetQ_perSession.py::TestForgetQPerSession::test_choice_selection_shape PASSED
tests/test_forgetQ_perSession.py::TestForgetQPerSession::test_forward_pass PASSED
tests/test_forgetQ_perSession.py::TestForgetQPerSession::test_integration_with_RLmodel_multiAgent PASSED

======================== 7 passed in 6.87s ========================
```

---

## Motivation

The standard `forgetQ` model uses scalar parameters shared across all sessions:
- `decay_rate`: scalar
- `positive_evi`: scalar  
- `negative_evi`: scalar
- `v0`: shape `(2,)`

This works well for fitting a single model to pooled data, but doesn't support simulating **heterogeneous agents** where each session has different parameter values (e.g., sampled from a distribution fitted to session-level behavioral fits).

## Implementation Highlights

### 1. Per-Session Parameter Shapes

`forgetQ_perSession` accepts `n_sess` at construction and defines parameters with vectorized shapes:

```python
@property
def _default_paramSpecs(self) -> List[boundedParam]:
    return [
        boundedParam('decay_rate', 0, 1, shape=(self.n_sess,)),
        boundedParam('positive_evi', -10, 10, shape=(self.n_sess,)),
        boundedParam('negative_evi', -10, 10, shape=(self.n_sess,)),
        boundedParam('v0', -10, 10, shape=(self.n_sess, 2)),
    ]
```

### 2. Vectorized Value Update

The `value_update` method handles per-session parameters correctly:

```python
def value_update(self, obs: jax.Array, values: jax.Array) -> jax.Array:
    # decay_rate is (n_sess,), reshape for broadcasting
    values = self.decay_rate.reshape(-1, 1) * values + evid
    return values
```

### 3. Integration with Existing Agent System

Works seamlessly with `RLmodel_multiAgent`:

```python
model = forgetQ_perSession(n_sess=500)
params = {
    'params': {
        'decay_rate': jnp.array(samples[:, 0]),
        'positive_evi': jnp.array(samples[:, 1]),
        'negative_evi': jnp.array(samples[:, 2]),
        'v0': jnp.stack([samples[:, 3], samples[:, 4]], axis=1),
    }
}
agent = RLmodel_multiAgent(model, params=params, n_sess=500, seed=42)
```

---

## Usage Example

### Fit Session-Level Parameters, Sample, and Simulate

```python
from scipy.stats import multivariate_normal
from disRNN_MP.classic_RL import forgetQ_perSession
from disRNN_MP.agent.agents import RLmodel_multiAgent

# 1. Fit multivariate Gaussian to session-level fitted parameters
param_matrix = session_fits_df.select(['decay_rate', 'positive_evi', 'negative_evi', 'v0_0', 'v0_1']).to_numpy()
mean = np.mean(param_matrix, axis=0)
cov = np.cov(param_matrix, rowvar=False)

# 2. Sample new parameter sets
n_sess_sim = 500
samples = multivariate_normal.rvs(mean=mean, cov=cov, size=n_sess_sim, random_state=42)

# 3. Clip to parameter bounds
samples[:, 0] = np.clip(samples[:, 0], 0, 1)  # decay_rate

# 4. Create vectorized model and agent
model = forgetQ_perSession(n_sess=n_sess_sim)
params = {
    'params': {
        'decay_rate': jnp.array(samples[:, 0]),
        'positive_evi': jnp.array(samples[:, 1]),
        'negative_evi': jnp.array(samples[:, 2]),
        'v0': jnp.stack([samples[:, 3], samples[:, 4]], axis=1),
    }
}
agent = RLmodel_multiAgent(model, params=params, n_sess=n_sess_sim, seed=42)

# 5. Simulate
simtri, _ = simul_MP(agent, n_sess=n_sess_sim, envCls=ParallelMPEnvironments)
```

---

## Run Tests

```bash
eval "$(pixi shell-hook -e default)" && pytest disRNN_MatchingPennies/tests/test_forgetQ_perSession.py -v
```

---

## Design Decisions

1. **Separate File**: `forgetQ_perSession` lives in its own file rather than `classic_RL.py` to keep the main module focused on the core fitting workflow.

2. **Not for Fitting**: This class is designed for simulation only. The parameter shapes are fixed at construction time based on `n_sess`, making it unsuitable for the standard fitting workflow.

3. **Inherits from RLmodel**: Full compatibility with the existing `RLmodel` interface, including `init_values`, `value_update`, `choice_selection`, and `__call__`.
