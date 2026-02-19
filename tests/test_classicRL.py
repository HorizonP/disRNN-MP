import os
import pytest
import numpy as np
import jax
import jax.numpy as jnp
from pathlib import Path

from disRNN_MP.classic_RL import forgetQ, boundedParam, RLmodelWrapper
from disRNN_MP.dataset import makeDataset_nparr
from disRNN_MP.utils import isequal_pytree

# Ensure deterministic JAX behavior
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "False"
jax.config.update("jax_platform_name", "cpu")

@pytest.fixture
def data_path():
    # Resolves to: disRNN_MatchingPennies/../data/mp_beh_m18_500completed.npy
    current_dir = Path(__file__).parent
    return current_dir.parent.parent / "data" / "mp_beh_m18_500completed.npy"

def test_forgetQ_fitting(data_path):
    if not data_path.exists():
        pytest.skip(f"Data file not found at {data_path}")
        
    dat = np.load(data_path)
    
    trainD = makeDataset_nparr(dat[:,::2,:])
    testD = makeDataset_nparr(dat[:,1::2,:])
    
    rlmd = forgetQ(param_specs=[
        boundedParam('decay_rate', 0, 1),
        boundedParam('positive_evi', -10, 10),
    ])
    
    # Fit using optax (default)
    mdw = RLmodelWrapper(rlmd, dataset=trainD, rng=jax.random.key(97), run_fitting=True)
    
    # Expected values from optax L-BFGS fitting
    expected_params = {
        'decay_rate': jnp.array(0.8813225),
        'negative_evi': jnp.array(-0.142954),
        'positive_evi': jnp.array(0.17330836),
        'v0': jnp.array([1.944603, 2.1570194]),
    }
    
    # Use np.allclose for robustness against tiny floating point differences
    for k, v in expected_params.items():
        assert np.allclose(np.array(mdw.params[k]), np.array(v), rtol=1e-5, atol=1e-6)
    
    # Check metrics
    train_metric = float(mdw.metric(trainD))
    test_metric = float(mdw.metric(testD))
    
    assert np.isclose(train_metric, 0.5079038, rtol=1e-5)
    assert np.isclose(test_metric, 0.50979125, rtol=1e-5)
    
    # Ensure forward pass works
    mdw.forward(makeDataset_nparr(dat[:450,1::2,:]))

