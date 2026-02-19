import os
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "False")

import jax
import jax.numpy as jnp
import pytest
from optax import softmax_cross_entropy_with_integer_labels

from disRNN_MP.metrics import BerLL_logit, BerLL_prob


def test_berll_logit_matches_optax_cross_entropy():
    key = jax.random.key(12)
    label_key, logits_key = jax.random.split(key, 2)

    # Labels can be (...,) or (..., 1); both should map to the same log-likelihood.
    labels = jax.random.randint(label_key, shape=(3, 2, 2), minval=0, maxval=2)
    labels_expanded = labels[..., None]
    logits = jax.random.uniform(logits_key, shape=(3, 2, 2, 2), minval=-5.0, maxval=5.0)

    # BerLL_logit returns log-likelihood; negate to compare to optax cross-entropy loss.
    expected = jnp.sum(softmax_cross_entropy_with_integer_labels(logits, labels))
    actual = -BerLL_logit(labels, logits)
    actual_expanded = -BerLL_logit(labels_expanded, logits)

    assert jnp.allclose(actual, expected)
    assert jnp.allclose(actual_expanded, expected)


def test_berll_prob_matches_logit():
    key = jax.random.key(33)
    labels_key, logits_key = jax.random.split(key, 2)

    # Probabilities from softmax(logits) should agree with the logit-based path.
    labels = jax.random.randint(labels_key, shape=(4, 3), minval=0, maxval=2)
    logits = jax.random.normal(logits_key, shape=(4, 3, 2))
    probs = jax.nn.softmax(logits, axis=-1)

    assert jnp.allclose(BerLL_prob(labels, probs), BerLL_logit(labels, logits))


def test_berll_logit_raises_on_incompatible_shape():
    # Shape mismatch should raise early to prevent silent broadcasting errors.
    labels = jax.random.randint(jax.random.key(12), shape=(3, 4, 2), minval=0, maxval=2)
    logits = jax.random.normal(jax.random.key(50), shape=(3, 2, 2, 2))

    with pytest.raises(ValueError):
        BerLL_logit(labels, logits)
    
