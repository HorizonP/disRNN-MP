# %%
import os
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "False"

import haiku as hk
import jax
import jax.numpy as jnp

from disRNN_MP.utils import isequal_pytree
from disRNN_MP.rnn.utils import transform_hkRNN
from disRNN_MP.rnn.disrnn4 import (
    hkDisRNN, 
    make_loss_fun as disrnn_make_loss_fun, 
    make_param_metric_expLL as disrnn_make_param_metric_expLL)
from disRNN_MP.experimental.deep_disRNN2 import (
    deepDisRNN,
    make_loss_fun as deepDisRNN_make_loss_fun,
    make_param_metric_expLL as deepDisRNN_make_param_metric_expLL)
# %%

def make_disrnn():
    disrnn = hkDisRNN(
        latent_size=4,
        update_mlp_shape=(3,2),
        choice_mlp_shape=(4,2)
    )
    return disrnn

def make_deep_disrnn_1lyr():
    disrnn = deepDisRNN(
        latent_size=4,
        update_mlp_shape=(3,2),
        choice_mlp_shape=(4,2),
        latent_shape=None
    )
    return disrnn

def make_deep_disrnn_2lyr():
    disrnn = deepDisRNN(
        latent_size=6,
        update_mlp_shape=(3,2),
        choice_mlp_shape=(4,2),
        latent_shape=(4,2)
    )
    return disrnn
# %%

disrnn = transform_hkRNN(make_disrnn)
deep_disrnn_1lyr = transform_hkRNN(make_deep_disrnn_1lyr)
deep_disrnn_2lyr = transform_hkRNN(make_deep_disrnn_2lyr)

params = jax.jit(disrnn.init)(jax.random.PRNGKey(1234), jnp.zeros((10, 2, 4)))
params2 = jax.jit(deep_disrnn_1lyr.init)(jax.random.PRNGKey(1234), jnp.zeros((10, 2, 4)))

pseudo_input = jax.random.choice(jax.random.key(34), 2, (10, 2, 4))

out1 = disrnn.apply(params, jax.random.PRNGKey(10), pseudo_input)
out2 = deep_disrnn_1lyr.apply(params2, jax.random.PRNGKey(10), pseudo_input)

assert isequal_pytree(out1, out2)