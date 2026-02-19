# Common pitfalls

## When to use
Use this as a checklist when results look off or errors appear.

## Expected inputs
- A dataset or training run that behaves unexpectedly

## Expected outputs
- A narrowed-down set of issues to investigate

## Minimal example

```python
from disRNN_MP.dataset import trainingDataset

assert isinstance(train_dataset, trainingDataset)
```

## Notes
- Ensure input/output arrays are shaped `(trials, sessions, features)`.
- Avoid mixing NumPy arrays with JAX arrays in one computation chain.
- Keep device placement consistent when using torch or JAX.

## Related recipes
- [Reproducible runs](reproducibility.md)
- [Load datasets](data_loading.md)
