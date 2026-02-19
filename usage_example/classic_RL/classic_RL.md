# RLmodelWrapper Usage Examples

`RLmodelWrapper` is a utility class in `disRNN_MP.classic_RL` that wraps `RLmodel` definitions. It provides a consistent interface for fitting models to behavioral data, running forward passes to extract internal variables, and evaluating model performance.

## 1. Model Fitting and Batch Optimization

The most common use of `RLmodelWrapper` is to fit a reinforcement learning model to a dataset. Often, multiple initializations (seeds) are used to ensure the optimizer finds a global minimum.

### Single Model Fitting
```python
import jax
from disRNN_MP.classic_RL import forgetQ, boundedParam, RLmodelWrapper

# Define model with specific parameter constraints
rlmd = forgetQ(param_specs=[
    boundedParam('decay_rate', 0, 1),
    boundedParam('positive_evi', -10, 10),
])

# Initialize and fit (uses optax L-BFGS by default)
mdw = RLmodelWrapper(rlmd, dataset=trainD, rng=jax.random.key(97), run_fitting=True)
print(f"Minimized Loss: {mdw.minimum}")
```

### Optimizer Selection

The `fit()` method supports two optimizers via the `optimizer` parameter:

- **`'optax'` (default)**: Uses `optax.lbfgs` with box constraints. This is the recommended optimizer.
- **`'jaxopt'` (deprecated)**: Uses `jaxopt.LBFGSB`. Requires installing the legacy extra: `pip install disRNN-MP[legacy]`

```python
# Explicit optimizer selection (optax is default)
mdw.fit(rng=jax.random.key(97), optimizer='optax')

# Legacy jaxopt optimizer (deprecated, requires [legacy] extra)
mdw.fit(rng=jax.random.key(97), optimizer='jaxopt')
```

### Batch Fitting and Model Selection
```python
# Fit multiple models with different seeds
mdws = [RLmodelWrapper(rlmd, dataset=trainD, rng=jax.random.key(seed), run_fitting=True) 
        for seed in [23, 1, 183, 2, 9]]

# Select the best model based on performance on a test dataset
best_mdw = mdws[np.argmax([m.metric(testD) for m in mdws])]
```

## 2. Inference and Performance Evaluation

Once a model is fitted (or initialized with parameters), you can use it to generate predictions and evaluate its accuracy against observed behavior.

### Forward Pass (Extracting Latents)
```python
# Get internal values (latents) and choice probabilities for a dataset
all_values, all_ch_probs = mdw.forward(testD)

# all_values shape: (N_trials, N_sessions, N_values)
# all_ch_probs shape: (N_trials, N_sessions, N_actions)
```

### Evaluation Metrics
```python
# Calculate the exponential of the normalized log-likelihood
# This represents the average probability the model assigns to the observed choices
perf = mdw.metric(testD)
print(f"Test Metric (exp-nLL): {perf:.4f}")
```

## 3. Using Pre-trained Parameters

`RLmodelWrapper` allows initializing a model with parameters obtained from a previous training session or loaded from a file.

```python
from disRNN_MP.utils import unpickle

# Load saved parameters (e.g., from a msgpack file)
fQ_par = unpickle('results/m211_fQ_best_par.msgpack')

# Initialize wrapper with fixed parameters (skipping fitting)
fQ = RLmodelWrapper(forgetQ(), params=fQ_par)

# Use the pre-trained model for analysis
values, _ = fQ.forward(mon_td['m211'])
```

## 4. Behavioral Simulation

Fitted models can be wrapped into agents to simulate behavior in various environments.

```python
from disRNN_MP.agent import RLmodel_agent, run_experiment, EnvironmentBanditsMP_julia

# Create an agent using the fitted model wrapper
age = RLmodel_agent(mdw, seed=123)

# Run an experiment in a Bandit environment
env = EnvironmentBanditsMP_julia()
results = run_experiment(age, env, n_trials=500, n_sessions=30)

# The results contain the simulated choices and rewards
sim_df = results[0].df
```

## 5. Advanced: Parallel Fitting with Grid Initialization

For large-scale fitting tasks, you can leverage JAX's parallelization capabilities. The `fit()` method uses optax L-BFGS internally, which is compatible with `jax.pmap` for parallel fitting across multiple parameter initializations.

```python
import jax
import optax
from disRNN_MP.classic_RL import forgetQ, boundedParam, RLmodelWrapper

def wrap_fq(trainD, dr=0.1, pe=1, ne=-1, seed=0):
    """Create a model wrapper with specific initial parameter values."""
    rlmd = forgetQ(param_specs=[
        boundedParam('decay_rate', 0, 1, init=dr),
        boundedParam('positive_evi', -10, 10, init=pe),
        boundedParam('negative_evi', -10, 10, init=ne),
    ])
    return RLmodelWrapper(rlmd, dataset=trainD, rng=jax.random.key(seed), run_fitting=False)

def fit_on_trainD(dr=0.1, pe=1, ne=-1, seed=0):
    """Fit a single model - can be pmapped for parallelization."""
    mdw = wrap_fq(trainD, dr=dr, pe=pe, ne=ne, seed=seed)
    return mdw.fit(jax.random.key(seed))

# Parallel fitting across multiple initializations
# (See scripts/2025Jan26_fit_fQ_grid_init_params.py for full implementation details)
fun = jax.pmap(fit_on_trainD)
```
