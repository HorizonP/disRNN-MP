from typing import Callable

import jax
import jax.numpy as jnp
from jax import lax

def BerLL(
        labels: jax.Array, log_probs: jax.Array, norm: bool = False
    ) -> jax.Array:
    """calculate Bernoulli log likelihood with labels and log(p)

    assume labels and output_logits sharing same dimensions except last one
    labels: (...) or (..., 1)
    log_probs: (..., n)

    value of labels corresponds to the number of column of last dim of output_logits
    Masks: value of labels < 0 will be ignored in the calculation

    Returns:
        jax.Array: the sum
    """

    if labels.shape == log_probs.shape[:-1]:
        _labels = labels
    elif labels.shape[:-1] == log_probs.shape[:-1] and labels.shape[-1] == 1:
        _labels = labels[...,0]
    else:
        raise ValueError(f"label shape {labels.shape} is not compatible with log_probs {log_probs.shape}")

    one_hot_labels = jax.nn.one_hot(
        _labels.astype(jnp.int32), num_classes = log_probs.shape[-1]
    ) # (...) -> (..., log_probs.shape[-1])

    mask = jnp.expand_dims(jnp.logical_not(_labels < 0), axis = -1)

    sum_log_liks = mask * one_hot_labels * log_probs

    return lax.cond(
        norm,
        lambda logLik, mask: jnp.nansum(logLik) / jnp.sum(mask),
        lambda logLik, mask: jnp.nansum(logLik),
        sum_log_liks, mask    
    )


@jax.jit
def BerLL_logit(labels: jax.Array, logits: jax.Array, norm: bool = False) -> jax.Array:
    """calculate Bernoulli log likelihood with logits input
    see BerLL
    """
    log_probs = jax.nn.log_softmax(logits, axis = -1)
    return BerLL(labels, log_probs, norm)

@jax.jit
def BerLL_prob(labels: jax.Array, probs: jax.Array, norm: bool = False) -> jax.Array:
    """calculate Bernoulli log likelihood with probabilities input
    see BerLL
    """
    log_probs = jnp.log(probs)
    return BerLL(labels, log_probs, norm)


@jax.jit
def bic(N_params, N_data, ll):
    """Bayesian information criterion
    
    """
    return N_params * jnp.log(N_data) - 2 * ll

