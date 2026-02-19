# Plot training diagnostics

## When to use
Use this after training to summarize model progress and fit.

## Expected inputs
- A trained `RNNtraining` instance
- Optional output directory

## Expected outputs
- Plotly figures or saved plots

## Minimal example

```python
from disRNN_MP.rnn.plots import training_progress_plot

fig = training_progress_plot(trainer)
```

## Notes
- Some plotting functions can write HTML outputs for sharing.

## Related recipes
- [Train a disRNN model](model_training.md)
- [Compute evaluation metrics](evaluation_metrics.md)
