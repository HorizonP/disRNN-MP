# %%
#!%load_ext autoreload
#!%autoreload 3
# %%
import os
from pathlib import Path
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import jax
import jax.numpy as jnp

from disRNN_MP import rnn
from disRNN_MP.rnn.plots import training_progress_plot, disrnn_dashboard_figure, disrnn2_dashboard_html
pwd = Path(__file__).parent
# %%
train_data1 = rnn.unpack(pwd / "../data/disRNN_RNNtraining.cloudpickle")

train_data2 = rnn.unpack(pwd / "../data/GRU_RNNtraining.cloudpickle")
# %%
# fig1 = disrnn_dashboard_figure(train_data1)
disrnn2_dashboard_html(train_data1, outdir=pwd)
# %%
training_progress_plot(train_data2, outdir=pwd)
# %%
