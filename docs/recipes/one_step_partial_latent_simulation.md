# One-step simulation from partial latent state

Use this pattern when you only know a subset of latent dimensions (for example, 3 plotted axes) and want one-step transitions from `stepFun`.

## Inputs
- `mda[mon]`: analyzer object for the target monkey/model.
- `lat_info`: latent metadata table with `name` and `ind`.
- `sampled_starts`: table with start-point columns like `current_STD`, `current_RRM`, `current_STCM`.
- `curr_lats`: names of known latent columns and corresponding latent names.

## Recipe

```python
import numpy as np
import jax.numpy as jnp
import polars as pl

mid = 18
mon = f"m{mid}"
known_lats = ["STD", "RRM", "STCM"]
curr_cols = [f"current_{lat}" for lat in known_lats]

lat_map = lat_info.filter(monkey=mid).select("name", "ind")
lat_name_to_ind = {name: ind for name, ind in lat_map.iter_rows()}
n_lat = mda[mon].n_latents

# 1) Build full latent state with zeros for unknown dimensions
start_states = np.zeros((sampled_starts.height, n_lat), dtype=np.float32)
for lat_name, col in zip(known_lats, curr_cols):
    start_states[:, lat_name_to_ind[lat_name]] = sampled_starts[col].to_numpy()

# 2) Choose next observations for your dataset/task
# NOTE: the coding below is an example for matching-pennies style data only.
next_obs_df = pl.DataFrame({"monCh": [0, 0, 1, 1], "rew": [0, 1, 0, 1]})

obs_batch = np.tile(next_obs_df.select("monCh", "rew").to_numpy(), (sampled_starts.height, 1))
state_batch = np.repeat(start_states, next_obs_df.height, axis=0)

# 3) Simulate one step
_, next_states = mda[mon].stepFun(jnp.array(obs_batch), jnp.array(state_batch))
next_states = np.asarray(next_states)

# 4) Select next-state dimensions you care about
result = pl.DataFrame({
    "next_STD": next_states[:, lat_name_to_ind["STD"]],
    "next_RRM": next_states[:, lat_name_to_ind["RRM"]],
    "next_STCM": next_states[:, lat_name_to_ind["STCM"]],
})
```

## Notes
- `stepFun` supports batched inputs; use one vectorized call rather than nested loops.
- Confirm your observation coding before simulation. Different tasks or preprocessing pipelines may use different encodings and labels.
- Keep a `start_id` in output tables to preserve one-to-many mapping from start state to simulated branches.

## See also
For a ready-made utility that wraps this pattern, see `partial_state_evolve` in `functions/simul_analy.py`. It handles observation meshgrid construction, batching, and result assembly automatically:

```python
from functions.simul_analy import partial_state_evolve

colmap = {"STD": 0, "RRM": 1, "STCM": 2}  # latent name -> index
obs_grid = {"monCh": [0, 1], "rew": [0, 1]}

result = partial_state_evolve(mda[mon], sampled_starts, colmap, obs_grid)
# result has columns: ...all sampled_starts cols..., next_monCh, next_rew, next_STD, next_RRM, next_STCM
```
