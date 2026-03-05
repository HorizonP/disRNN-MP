import os
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "False"

import pytest
import haiku as hk
import jax
import jax.numpy as jnp

from disRNN_MP.rnn import utils


# ── shared fixture ──────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def model_setup():
    """Create a disRNN model, params, and dummy inputs once per module."""
    md = utils.make_RNNtransformed(
        'disRNN_MP.rnn.disrnn4.hkDisRNN', latent_size=7, eval_mode=True
    )
    init = jax.jit(md.init)
    apply = jax.jit(md.apply)

    pars = init(jax.random.PRNGKey(5), jnp.zeros((100, 10, 2)))
    xs = jax.random.uniform(jax.random.key(0), (200, 10, 2))
    fo = apply(pars, jax.random.PRNGKey(10), xs)

    return md, pars, xs, fo


# ── patched_forward tests ──────────────────────────────────────────────────

def test_patched_forward_no_patch(model_setup):
    """Empty patch list should produce identical output to normal forward."""
    md, pars, xs, fo = model_setup
    fo2 = utils.patched_forward(md.model_haiku, pars, [], xs, jnp.array([]))

    assert jnp.all(fo[1] == fo2[1])
    assert jnp.all(fo[0]['prediction'] == fo2[0]['prediction'])


def test_patched_forward_scalar_exo(model_setup):
    """Scalar exo states should be reflected in the patched latent columns."""
    md, pars, xs, _ = model_setup
    patch_state_ids = [0, 1]
    exo_states = jnp.array([123, 101])
    fo3 = utils.patched_forward(md.model_haiku, pars, patch_state_ids, xs, exo_states)

    assert jnp.all(fo3[1][..., patch_state_ids] == exo_states)


def test_patched_forward_array_exo(model_setup):
    """Per-trial array exo states should be reflected in the patched latent columns."""
    md, pars, xs, _ = model_setup
    patch_state_ids = [0, 1]
    exo_states = jax.random.uniform(jax.random.key(101), xs.shape) * 100
    fo4 = utils.patched_forward(md.model_haiku, pars, patch_state_ids, xs, exo_states)

    assert jnp.all(fo4[1][..., patch_state_ids] == exo_states)


# ── eval_model_from_state tests ────────────────────────────────────────────

def test_eval_from_default_state(model_setup):
    """Starting from the default initial state should match eval_model output."""
    md, pars, xs, _ = model_setup
    make_network = md.model_haiku

    y1, s1 = utils.eval_model(make_network, pars, xs)

    init_state = utils.get_initial_state(make_network, pars)
    init_state = jnp.broadcast_to(init_state, (xs.shape[1], init_state.shape[-1]))
    y2, s2 = utils.eval_model_from_state(make_network, pars, xs, init_state)

    assert jnp.allclose(s1, s2, atol=1e-5)


def test_eval_from_mid_session_state(model_setup):
    """Starting from state at trial T should reproduce states[T+1:]."""
    md, pars, xs, _ = model_setup
    make_network = md.model_haiku

    _, s1 = utils.eval_model(make_network, pars, xs)

    T = 5
    mid_state = s1[T]  # state after processing xs[T]; shape (N_sessions, N_latents)
    _, s3 = utils.eval_model_from_state(make_network, pars, xs[T + 1:], mid_state)

    assert jnp.allclose(s1[T + 1:], s3, atol=1e-5)
