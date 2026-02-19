# Train a disRNN model

## When to use
Use this after preparing `trainingDataset` instances.

## Expected inputs
- `(train_dataset, test_dataset)` tuple
- A disRNN model function from `make_disrnn_funcs`

## Expected outputs
- A trained `RNNtraining` object with metrics and params

## Minimal example

```python
import optax
from disRNN_MP.rnn import RNNtraining, make_disrnn_funcs
from disRNN_MP.dataset import train_test_datasets

train_dataset, test_dataset = train_test_datasets("path/to/dataset.npy", seed=3)

make_disrnn, make_disrnn_eval = make_disrnn_funcs(
    latent_size=5,
    update_mlp_shape=[3, 3],
    choice_mlp_shape=[2],
    sample_dataset=train_dataset,
)

trainer = RNNtraining(
    model=make_disrnn,
    eval_model=make_disrnn_eval,
    datasets=(train_dataset, test_dataset),
    optimizer=optax.adam(1e-3),
)

trainer.train(
    "baseline",
    n_block=2,
    steps_per_block=10,
    loss_type="penalized_categorical",
)
```

## Notes
- Keep random keys explicit when adding custom training loops.

## Related recipes
- [Compute evaluation metrics](evaluation_metrics.md)
- [Plot training diagnostics](visualization.md)
