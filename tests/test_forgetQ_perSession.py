"""Tests for forgetQ_perSession class."""
import os
import pytest
import numpy as np
import jax
import jax.numpy as jnp

from disRNN_MP.classic_RL import forgetQ_perSession

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "False"
jax.config.update("jax_platform_name", "cpu")

# %%
class TestForgetQPerSession:
    """Tests for forgetQ_perSession class."""
    
    def test_init_values_shape(self):
        """Test that init_values returns correct shape."""
        n_sess = 10
        model = forgetQ_perSession(n_sess=n_sess)
        
        rng = jax.random.key(42)
        variables = model.init(rng, model._get_dummy_inputs(n_sess))
        
        init_vals = model.apply(variables, n_sess=n_sess, method='init_values')
        
        assert init_vals.shape == (n_sess, 2)
    
    def test_init_values_n_sess_mismatch_raises(self):
        """Test that init_values raises error on n_sess mismatch."""
        model = forgetQ_perSession(n_sess=10)
        rng = jax.random.key(42)
        variables = model.init(rng, model._get_dummy_inputs(10))
        
        with pytest.raises(ValueError, match="n_sess mismatch"):
            model.apply(variables, n_sess=5, method='init_values')
    
    def test_value_update_shape(self):
        """Test that value_update returns correct shape."""
        n_sess = 10
        model = forgetQ_perSession(n_sess=n_sess)
        rng = jax.random.key(42)
        variables = model.init(rng, model._get_dummy_inputs(n_sess))
        
        obs = jnp.zeros((n_sess, 2))
        values = jnp.zeros((n_sess, 2))
        
        updated = model.apply(variables, obs=obs, values=values, method='value_update')
        
        assert updated.shape == (n_sess, 2)
    
    def test_value_update_per_session_params(self):
        """Test that value_update uses per-session parameters correctly."""
        n_sess = 3
        model = forgetQ_perSession(n_sess=n_sess)
        
        params = {
            'decay_rate': jnp.array([0.9, 0.5, 0.1]),
            'positive_evi': jnp.array([1.0, 2.0, 3.0]),
            'negative_evi': jnp.array([-1.0, -2.0, -3.0]),
            'v0': jnp.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]]),
        }
        variables = {'params': params}
        
        # All sessions choose action 0, get reward
        obs = jnp.array([[0, 1], [0, 1], [0, 1]])
        values = jnp.zeros((n_sess, 2))
        
        updated = model.apply(variables, obs=obs, values=values, method='value_update')
        
        # Each session should have different updates based on their positive_evi
        # Session 0: 0.9 * 0 + 1.0 = 1.0 for action 0
        # Session 1: 0.5 * 0 + 2.0 = 2.0 for action 0
        # Session 2: 0.1 * 0 + 3.0 = 3.0 for action 0
        expected = jnp.array([[1.0, 0.0], [2.0, 0.0], [3.0, 0.0]])
        
        assert jnp.allclose(updated, expected)
    
    def test_choice_selection_shape(self):
        """Test that choice_selection returns valid probabilities."""
        n_sess = 10
        model = forgetQ_perSession(n_sess=n_sess)
        rng = jax.random.key(42)
        variables = model.init(rng, model._get_dummy_inputs(n_sess))
        
        values = jax.random.normal(rng, (n_sess, 2))
        probs = model.apply(variables, values, method='choice_selection')
        
        assert probs.shape == (n_sess, 2)
        assert jnp.allclose(probs.sum(axis=1), 1.0)
        assert jnp.all(probs >= 0)
    
    def test_forward_pass(self):
        """Test full forward pass through the model."""
        n_sess = 5
        n_trials = 20
        model = forgetQ_perSession(n_sess=n_sess)
        rng = jax.random.key(42)
        
        # Create dummy inputs: (n_trials, n_sess, 2)
        inputs = jax.random.randint(rng, (n_trials, n_sess, 2), 0, 2).astype(jnp.float32)
        
        variables = model.init(rng, inputs)
        all_values, all_probs = model.apply(variables, inputs)
        
        assert all_values.shape == (n_trials, n_sess, 2)
        assert all_probs.shape == (n_trials, n_sess, 2)
    
    def test_integration_with_RLmodel_multiAgent(self):
        """Test that forgetQ_perSession works with RLmodel_multiAgent."""
        from disRNN_MP.agent.agents import RLmodel_multiAgent
        
        n_sess = 5
        model = forgetQ_perSession(n_sess=n_sess)
        
        # Create explicit params
        params = {
            'params': {
                'decay_rate': jnp.ones(n_sess) * 0.9,
                'positive_evi': jnp.ones(n_sess) * 1.0,
                'negative_evi': jnp.ones(n_sess) * -1.0,
                'v0': jnp.zeros((n_sess, 2)),
            }
        }
        
        # Create agent
        agent = RLmodel_multiAgent(model, params=params, n_sess=n_sess, seed=42)
        
        # Check agent has correct properties
        assert agent.n_sess == n_sess
        assert agent.choice_probs.shape == (n_sess, 2)
        
        # Simulate one step
        choices = jnp.array([0, 1, 0, 1, 0])
        rewards = jnp.array([1, 0, 1, 0, 1])
        agent.update(choices, rewards)
        
        assert agent.choice_probs.shape == (n_sess, 2)
