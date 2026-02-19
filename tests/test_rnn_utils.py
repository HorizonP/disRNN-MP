# %%
import os
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "False"

import haiku as hk
import jax
import jax.numpy as jnp

from disRNN_MP.rnn import utils

# %% test make_RNNtransformed, transform_hkRNN
md = utils.make_RNNtransformed('disRNN_MP.rnn.disrnn4.hkDisRNN', latent_size = 7, eval_mode = True)

init = jax.jit(md.init)
apply = jax.jit(md.apply)

pars = init(jax.random.PRNGKey(5), jnp.zeros((100,10,2)))

xs = jax.random.uniform(jax.random.key(0), (200,10,2))
fo1 = apply(pars, jax.random.PRNGKey(10), xs)

# %% test patched_forward: when no latent is patched, this function should behave exactly same as the apply function of RNNtransformed
patch_state_ids = []
exo_states = jnp.array([])
fo2 = utils.patched_forward(md.model_haiku, pars, patch_state_ids, xs, exo_states)

assert jnp.all(fo1[1] == fo2[1])
assert jnp.all(fo1[0]['prediction'] == fo2[0]['prediction'])
# %% test patched_forward: updated latent states should reflect patched state values
patch_state_ids = [0,1]
exo_states = jnp.array([123, 101])
fo3 = utils.patched_forward(md.model_haiku, pars, patch_state_ids, xs, exo_states)

assert jnp.all(fo3[1][...,patch_state_ids] == exo_states)

exo_states = jax.random.uniform(jax.random.key(101), xs.shape) * 100
fo4 = utils.patched_forward(md.model_haiku, pars, patch_state_ids, xs, exo_states)

assert jnp.all(fo4[1][...,patch_state_ids] == exo_states)

# %%
