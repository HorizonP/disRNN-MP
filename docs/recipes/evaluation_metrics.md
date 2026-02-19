# Compute evaluation metrics

## When to use
Use this to compare train/test log-likelihood or model fit.

## Expected inputs
- `trainingDataset` instances
- A trained model function and parameters

## Expected outputs
- Scalar metrics (log-likelihood, BIC, etc.)

## Minimal example

```python
from disRNN_MP.rnn.utils import compute_log_likelihood

train_ll = compute_log_likelihood(train_dataset, trainer.eval_model, trainer.params)
test_ll = compute_log_likelihood(test_dataset, trainer.eval_model, trainer.params)
```

## Notes
- Use analysis modules for task-specific metrics (e.g., bandit analyzers).

## Related recipes
- [Train a disRNN model](model_training.md)
- [Plot training diagnostics](visualization.md)
