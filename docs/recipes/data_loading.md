# Load datasets

## When to use
Use this when converting raw arrays or dataframes into `trainingDataset` objects.

## Expected inputs
- `*.npy` array shaped `(n_trials, n_sessions, n_features)`
- Or a long-form dataframe with trial/session columns

## Expected outputs
- `trainingDataset` instances for train/test

## Minimal example

```python
from disRNN_MP.dataset import train_test_datasets

train_dataset, test_dataset = train_test_datasets(
    "path/to/dataset.npy",
    n_sess_sample=0.8,
    seed=3,
)
```

## Notes
- Use `trainingDataset.from_df(...)` when starting from a dataframe.
- Keep input and output feature indices consistent.

## Related recipes
- [Train a disRNN model](model_training.md)
- [Reproducible runs](reproducibility.md)
