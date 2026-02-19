"""forgetQ variant with per-session parameters for heterogeneous agent simulation."""

from dataclasses import field
from typing import List

import jax
from jax import lax
import jax.numpy as jnp
import flax.linen as nn

from .classic_RL import RLmodel, boundedParam


class forgetQ_perSession(RLmodel):
    """forgetQ with per-session parameters for simulation.
    
    Unlike standard forgetQ where params are scalars shared across sessions,
    this variant expects params with shape (n_sess,) or (n_sess, 2) for v0,
    allowing different parameter values per session.
    
    This is designed for simulation with heterogeneous agents, where each
    session has its own parameter set (e.g., sampled from a fitted 
    multivariate Gaussian distribution).
    
    Note: This is for simulation only, not fitting.
    
    Attributes:
        n_sess: Number of sessions (determines param array shapes)
        N_actions: Number of possible actions (default 2)
    
    Example:
        # Create model for 100 sessions
        model = forgetQ_perSession(n_sess=100)
        
        # Build params from sampled values
        params = {
            'decay_rate': jnp.array(samples[:, 0]),      # (100,)
            'positive_evi': jnp.array(samples[:, 1]),    # (100,)
            'negative_evi': jnp.array(samples[:, 2]),    # (100,)
            'v0': jnp.array(samples[:, 3:5]),            # (100, 2)
        }
        
        # Use with RLmodel_multiAgent
        agent = RLmodel_multiAgent(model, params=params, n_sess=100)
    """
    n_sess: int = 1
    N_actions: int = 2
    N_values: int = field(init=False)
    N_obs_feat: int = field(default=2, init=False)
    
    def __post_init__(self):
        self.N_values = self.N_actions
        super().__post_init__()

    @property
    @nn.nowrap
    def _default_paramSpecs(self) -> List[boundedParam]:
        return [
            boundedParam('decay_rate', 0, 1, shape=(self.n_sess,)),
            boundedParam('positive_evi', -10, 10, shape=(self.n_sess,)),
            boundedParam('negative_evi', -10, 10, shape=(self.n_sess,)),
            boundedParam('v0', -10, 10, shape=(self.n_sess, 2)),
        ]
    
    def init_values(self, n_sess: int) -> jax.Array:
        """Return v0 directly - already (n_sess, 2)."""
        if n_sess != self.n_sess:
            raise ValueError(f"n_sess mismatch: got {n_sess}, expected {self.n_sess}")
        return self.v0
    
    def value_update(self, obs: jax.Array, values: jax.Array) -> jax.Array:
        """Update Q-values with per-session parameters.
        
        Args:
            obs: (n_sess, 2) - [choice, reward] for each session
            values: (n_sess, 2) - current Q-values
            
        Returns:
            Updated Q-values (n_sess, 2)
        """
        N_actions = self.N_values
        
        # choice one-hot: (n_sess, 2)
        chOH = jax.nn.one_hot(obs[:, 0].astype(jnp.int32), N_actions)
        
        # evidence: select positive_evi or negative_evi based on reward
        evid_scalar = lax.select(
            obs[:, 1].astype('bool'),
            self.positive_evi,  # (n_sess,)
            self.negative_evi   # (n_sess,)
        ).reshape(-1, 1)  # (n_sess, 1)
        
        evid = chOH * evid_scalar  # (n_sess, 2)
        
        # value update with per-session decay_rate
        values = self.decay_rate.reshape(-1, 1) * values + evid  # (n_sess, 2)
        
        return values
    
    def choice_selection(self, values: jax.Array) -> jax.Array:
        """Softmax over Q-values to get choice probabilities."""
        return jax.nn.softmax(values, axis=-1)
