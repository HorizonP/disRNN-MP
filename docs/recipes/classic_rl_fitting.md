# Fit Classic RL Models with Grid Search

## When to use
Use when fitting explicit RL models (e.g., `forgetQ`) to behavioral data. Multi-start optimization avoids local minima, ensuring more robust parameter estimation.

## Expected inputs
- `trainingDataset` (for fitting) and `testDataset` (for selection)
- An `RLmodel` subclass (e.g., `forgetQ`)
- Grid of initial parameter values

## Expected outputs
- Best-fitting model wrapper (`RLmodelWrapper`)
- Full optimization results for diagnostic purposes

## Minimal example (Single Fit)

A single fit is straightforward using `RLmodelWrapper` with `run_fitting=True`.

```python
import jax
from disRNN_MP.classic_RL import forgetQ, RLmodelWrapper
from disRNN_MP.dataset import train_test_datasets_from_df

# Load data
trainD, testD = train_test_datasets_from_df(
    df='path/to/behavior.csv',
    x_vars=['choice', 'reward'],
    y_vars=['next_choice'],
    seed=0
)

# Initialize and fit
model = forgetQ()
mdw = RLmodelWrapper(model, dataset=trainD, rng=jax.random.key(0), run_fitting=True)

print(f"Fitted params: {mdw.params}")
print(f"Test metric: {mdw.metric(testD)}")
```

## Grid search with parallelization

For research results, it is recommended to run multi-start optimization.

### Environment Setup
L-BFGS performs better with higher precision (`f64`). On many systems, CPU handles double precision more efficiently than GPU for these small models.

```python
import os
# Simulate multiple devices for pmap
os.environ["XLA_FLAGS"] = '--xla_force_host_platform_device_count=30'

import jax
import jax.numpy as jnp
import numpy as np
from tqdm import tqdm

# Force CPU usage
jax.config.update('jax_platform_name', 'cpu')
```

### Define Grid and Fitting Function

```python
from disRNN_MP.classic_RL import forgetQ, boundedParam, RLmodelWrapper

def fit_with_init(dr, pe, ne, seed):
    # Override default param specs with specific initial values
    model = forgetQ(param_specs=[
        boundedParam('decay_rate', 0, 1, init=dr),
        boundedParam('positive_evi', -10, 10, init=pe),
        boundedParam('negative_evi', -10, 10, init=ne),
    ])
    
    # Create wrapper (run_fitting=False) and call .fit()
    mdw = RLmodelWrapper(model, dataset=trainD, rng=jax.random.key(seed))
    params, state = mdw.fit(jax.random.key(seed))
    return params, state

# Create grid
init_grids = np.meshgrid(
    np.linspace(0, 1, 5),      # decay_rate
    np.linspace(-10, 10, 4),   # positive_evi
    np.linspace(-10, 10, 4),   # negative_evi
)
# Flatten and add seeds
flat_grids = [g.flatten() for g in init_grids] + [np.arange(len(init_grids[0].flatten()))]
```

### Run Parallelized Fits

```python
fit_pmap = jax.pmap(fit_with_init)
N_jobs = len(flat_grids[0])
N_batches = int(np.ceil(N_jobs / jax.device_count()))

results = []
for i_batch in tqdm(range(N_batches)):
    # Split grid into batches for devices
    batch = jax.tree_util.tree_map(
        lambda arr: jnp.array_split(arr, N_batches)[i_batch], 
        flat_grids
    )
    results.append(fit_pmap(*batch))

# Concatenate across batches
all_results = jax.tree_util.tree_map(lambda *args: jnp.concat(args, axis=0), *results)
```

### Select Best Model

Evaluate all non-failing fits on the test set.

```python
from disRNN_MP.rnn.utils import nan_in_dict

# Filter valid results and wrap into mdw
valid_params = [r['params'] for r in all_results if not nan_in_dict(r)]
mdws = [RLmodelWrapper(forgetQ(), params=p) for p in valid_params]

# Select best by test set metric (exp of normalized LL)
test_metrics = [mdw.metric(testD) for mdw in mdws]
best_idx = np.argmax(test_metrics)
best_mdw = mdws[best_idx]

print(f"Best test metric: {test_metrics[best_idx]}")
print(f"Best params: {best_mdw.params}")
```

## Notes
- **Precision:** L-BFGS requires high gradient precision; `f64` on CPU is preferred.
- **Diagnostics:** It is good practice to save `all_results` before selection to check for convergence issues across the grid.
- **NaN Handling:** Use `nan_in_dict` to detect optimizations that failed to produce valid parameters.

## Related recipes
- [Load datasets](data_loading.md)
- [Compute evaluation metrics](evaluation_metrics.md)
