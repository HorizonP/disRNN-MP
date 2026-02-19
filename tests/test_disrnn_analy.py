# %%
#!%load_ext autoreload
#!%autoreload 3
# %%
import os
from pathlib import Path
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import polars as pl
import flax.linen as nn
import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt

import disRNN_MP.rnn.plots as rnnplt
import disRNN_MP.rnn.disrnn_analy as dana
from disRNN_MP import rnn
from disRNN_MP.rnn.utils import evo_state, get_initial_state

pwd = Path(__file__).parent

# %%
train_data = rnn.unpack(pwd / "../data/disRNN_RNNtraining.cloudpickle")
df = dana.make_MP_phase_df_fromRNNtraining(train_data)
dana.MP_phase_dynamic_figure(df, [1, 3], 'tst.html')
dana.MP_phase_dynamic_figure(df, [1, 3, 5], 'tst2.html')

# %%

param = train_data.params
make_network = train_data.eval_model

step_fun = evo_state(make_network, param)

initial_state = get_initial_state(make_network, param)
reference_state = jnp.zeros_like(initial_state)

observations = ([0, 0], [0, 1], [1, 0], [1, 1])
fig1 = dana.plot_update_1d(step_fun, reference_state, 5, observations, np.repeat(["tst"], 4))
fig2 = dana.plot_update_2d(step_fun, reference_state, 1, 5, observations, np.repeat(["tst"], 4))
# %%
from plotly.tools import mpl_to_plotly

mpl_to_plotly(fig1)

