# Reproducible runs

## When to use
Use this to keep results repeatable across training and evaluation.

## Expected inputs
- Dataset creation parameters
- A training routine that accepts random keys

## Expected outputs
- Stable metrics when rerunning the workflow

## Minimal example

```python
import jax
from disRNN_MP.dataset import train_test_datasets

train_dataset, test_dataset = train_test_datasets(
    "path/to/dataset.npy",
    seed=42,
)

rng_key = jax.random.PRNGKey(42)
```

## Notes
- Avoid mixing `jax.random.PRNGKey` and `jax.random.key` styles.
- Keep seeds explicit in dataset creation and training steps.

## Related recipes
- [Load datasets](data_loading.md)
- [Train a disRNN model](model_training.md)
