# %%

#!%load_ext snakeviz
#!%snakeviz_config -h localhost -p 8901

# %%
#!%load_ext autoreload
#!%autoreload 3
# %%
import os
from pathlib import Path
from functools import partial
import logging
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import optax
import jax
import jax.numpy as jnp

from disRNN_MP import rnn
from disRNN_MP.rnn import make_disrnn_funcs, RNNtraining, train_session, train_model, RNNtraining_interactive, compute_log_likelihood
from disRNN_MP.rnn.utils import transform_hkRNN, make_loss_fun, make_train_step, train_with_step_fun, _train_model
from disRNN_MP.dataset import train_test_datasets
import disRNN_MP.agent
from disRNN_MP.agent import hkNetwork_agent, EnvironmentBanditsMP_julia, run_experiment
# %%

(train_dataset, test_dataset) = train_test_datasets(Path("../data/mp_beh_m18_500completed.npy"), 40, seed=3)

# define neural network
make_disrnn, make_disrnn_eval = make_disrnn_funcs(
    latent_size= 5,
    update_mlp_shape= [3,3],
    choice_mlp_shape= [2],
    sample_dataset=train_dataset
)

optimizer = optax.adam(1e-3)
init, apply = transform_hkRNN(make_disrnn)
params = jax.jit(init)(jax.random.PRNGKey(0), next(train_dataset)[0])
opt_state = optimizer.init(params)
train_step = jax.jit(make_train_step(apply, optimizer=optimizer, loss_type='penalized_categorical', penalty_scale=0))
train_step2 = jax.jit(make_train_step(apply, optimizer=optimizer, loss_type='penalized_categorical', penalty_scale=1e-3))
# %%
lossf = make_loss_fun(apply)
loss = lossf(params, jax.random.PRNGKey(123), next(train_dataset)[0], next(train_dataset)[1])
float(loss) == 879097.125 # type: ignore

# %%

params, opt_state, losses = train_with_step_fun(train_step, train_dataset, params, opt_state, jax.random.PRNGKey(50), n_steps=1000)

print(compute_log_likelihood(train_dataset, make_disrnn, params))
print(compute_log_likelihood(test_dataset, make_disrnn, params))


# %% test RNNtraining disRNN
with jax.log_compiles(True):
    # put model, dataset, optimizer together
    disRNN0 = RNNtraining(model=make_disrnn, eval_model=make_disrnn_eval, 
        datasets=(train_dataset, test_dataset),
        optimizer= optax.adam(1e-3))

    disRNN0.train("test",
        n_block=2, 
        steps_per_block=10,
        loss_type='penalized_categorical',
        )


# %% test compute_log_likelihood
compute_log_likelihood(train_dataset, disRNN0.eval_model, disRNN0.params)
# %% test RNNtraining_interactive disRNN

disRNN = RNNtraining_interactive(model=make_disrnn, eval_model=make_disrnn_eval, 
    datasets=(train_dataset, test_dataset),
    optimizer= optax.adam(1e-3))

# this is the plot that will be updated while the model is being trained
disRNN.likelihood_plot

# %%

disRNN.train("no_penalty",
        n_block=10, 
        steps_per_block=100,
        loss_type='penalized_categorical',
        )


disRNN.train(
        n_block=10, 
        steps_per_block=100,  
        loss_type='penalized_categorical', penalty_scale=1e-3,
        beta_scale=1
        )



# %%
env = EnvironmentBanditsMP_julia()
age = hkNetwork_agent(disRNN0.eval_model, disRNN0.params)

res = run_experiment(age, env, 100)