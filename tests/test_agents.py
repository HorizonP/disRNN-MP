# %%
import os
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "False")

import numpy as np
import jax.numpy as jnp

from disRNN_MP.agent.agents import RLmodel_multiAgent
from disRNN_MP.classic_RL import forgetQ, boundedParam


def _make_agent(n_sess: int = 10, seed: int = 32) -> RLmodel_multiAgent:
    rlmd = forgetQ(param_specs=[
        boundedParam("decay_rate", 0, 1, init=0.5),
        boundedParam("positive_evi", -10, 10, init=1),
        boundedParam("negative_evi", -10, 10, init=-1),
    ])
    return RLmodel_multiAgent(rlmd, n_sess=n_sess, seed=seed)


def test_get_choice_repeatable_within_trial():
    n_sess = 10
    agent = _make_agent(n_sess=n_sess)

    # repeated calls of `get_choice` on same trial should return random but repeatable samples
    ch1 = agent.get_choice()
    ch2 = agent.get_choice()
    np.testing.assert_array_equal(ch1, ch2)

    reward = jnp.zeros_like(jnp.asarray(ch1))
    agent.update(jnp.asarray(ch1), reward)

    ch3 = agent.get_choice()
    ch4 = agent.get_choice()
    np.testing.assert_array_equal(ch3, ch4)
    assert ch3.shape == (n_sess,)

# %%
