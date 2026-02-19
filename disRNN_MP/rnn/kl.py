import jax
import jax.numpy as jnp

from ..typing import Array

def kl_gaussian_mu_sigma(mu: Array, sigma: Array) -> jax.Array:
    r"""Calculate KL divergence between a diagonal-covariance gaussian and standard gaussian using mus and sigmas.

    KL(p, q) = H(p, q) - H(p) = -\int p(x)log(q(x))dx - -\int p(x)log(p(x))dx
            = 0.5 * [log(|s2|/|s1|) - 1 + tr(s1/s2) + (m1-m2)^2/s2]
            = 0.5 * [-log(|s1|) - 1 + tr(s1) + m1^2] (if m2 = 0, s2 = 1)
    Args:
      mean: mean vector of the first distribution
      sigma: vector of sigmas of the gaussian, whose square is the diagonal of covariance matrix of the given distribution

    Returns:
      A scalar representing KL divergence of the two Gaussian distributions.
    """

    var = jnp.square(sigma)

    return 0.5 * jnp.sum(-jnp.log(var) - 1.0 + var + jnp.square(mu), axis=-1)