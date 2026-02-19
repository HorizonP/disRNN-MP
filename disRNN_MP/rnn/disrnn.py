from typing import Iterable, Callable, Any

import haiku as hk
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import warnings
from flax import traverse_util

from ..typing import ListLike, Array
from disRNN_MP.rnn.utils import Params

warnings.filterwarnings("ignore")


def kl_gaussian(mean: Array, var: Array) -> jax.Array:
    r"""Calculate KL divergence between given and standard gaussian distributions.

    KL(p, q) = H(p, q) - H(p) = -\int p(x)log(q(x))dx - -\int p(x)log(p(x))dx
            = 0.5 * [log(|s2|/|s1|) - 1 + tr(s1/s2) + (m1-m2)^2/s2]
            = 0.5 * [-log(|s1|) - 1 + tr(s1) + m1^2] (if m2 = 0, s2 = 1)
    Args:
      mean: mean vector of the first distribution
      var: diagonal vector of covariance matrix of the first distribution

    Returns:
      A scalar representing KL divergence of the two Gaussian distributions.
    """

    return 0.5 * jnp.sum(-jnp.log(var) - 1.0 + var + jnp.square(mean), axis=-1)

def _sigma_squashed(sigma_unsquashed):
    return 2 * jax.nn.sigmoid(sigma_unsquashed)

class HkDisRNN(hk.RNNCore):
    """
    Disentangled RNN.
    separate update rule bottlenecks penalty from global bottlenecks
    """

    def __init__(
        self,
        # vector size of observations on each trial, usually 2 for choice and reward
        obs_size: int = 2,
        # vector size of target (prediction), usually 2 for 2-arm bandit -- the log-p for each action
        target_size: int = 2,
        latent_size: int = 10,
        update_mlp_shape: Iterable[int] = (10, 10, 10),
        choice_mlp_shape: Iterable[int] = (10, 10, 10),
        eval_mode: float = 0, # eval_mode will turn off randomness
        activation: Callable[[Any], Any] = jax.nn.relu,
    ):
        super().__init__()

        self._target_size = target_size
        self._latent_size = latent_size
        self._update_mlp_shape = update_mlp_shape
        self._choice_mlp_shape = choice_mlp_shape
        self._eval_mode = eval_mode
        self._activation = activation

        # Each update MLP gets input from both the latents and the observations.
        # It has a sigma and a multiplier associated with each.
        mlp_input_size = latent_size + obs_size
        # At init the bottlenecks should all be open: sigmas small and multipliers 1
        update_mlp_sigmas_unsquashed = hk.get_parameter(
            'update_mlp_sigmas_unsquashed',
            (mlp_input_size, latent_size),
            init=hk.initializers.RandomUniform(minval=-3, maxval=-2),
        )
        # Training encourages sigmas to be ~1 or smaller. Bound them between 0 and 2
        self._update_mlp_sigmas = (
            _sigma_squashed(update_mlp_sigmas_unsquashed) * (1 - eval_mode)
        )
        self._update_mlp_multipliers = hk.get_parameter(
            'update_mlp_gates',
            (mlp_input_size, latent_size),
            init=hk.initializers.Constant(constant=1),
        )

        # Latents will also go through a bottleneck
        self.latent_sigmas_unsquashed = hk.get_parameter(
            'latent_sigmas_unsquashed',
            (latent_size,),
            init=hk.initializers.RandomUniform(minval=-3, maxval=-2),
        )
        self._latent_sigmas = (
            _sigma_squashed(self.latent_sigmas_unsquashed) * (1 - eval_mode)
        )

        # Latent initial values are also free parameters
        self._latent_inits = hk.get_parameter(
            'latent_inits',
            (latent_size,),
            init=hk.initializers.RandomUniform(minval=-0.01, maxval=0.01),
        )

    def __call__(
            self, observations: jnp.ndarray, prev_latents: jnp.ndarray):
        """run a single step (one trial) for all batches"""
        # observations is of shape (batch_size, obs_size)

        # Accumulator for KL costs
        penalty = jnp.zeros((observations.shape[0], 2))
        # size: (batch_size, 2)
        # penalty[:,0] is for update rule bottlenecks, penalty[:,1] is for global bottlenecks

        ################
        #  UPDATE MLPs #
        ################
        # Each update MLP updates one latent
        # It sees previous latents and current observation
        # It outputs a weight and an update to apply to its latent

        # update_mlp_mus_unscaled: (batch_size, obs_size + latent_size)
        update_mlp_mus_unscaled = jnp.concatenate(
            (observations, prev_latents), axis=1
        )
        # update_mlp_mus: (batch_size, obs_size + latent_size, latent_size)
        update_mlp_mus = (
            jnp.expand_dims(update_mlp_mus_unscaled, 2)
            * self._update_mlp_multipliers
        )
        # update_mlp_sigmas: (obs_size + latent_size, latent_size)
        update_mlp_sigmas = self._update_mlp_sigmas * (1 - self._eval_mode)
        # update_mlp_inputs: (batch_size, obs_size + latent_size, latent_size)
        update_mlp_inputs = update_mlp_mus + update_mlp_sigmas * jax.random.normal(
            hk.next_rng_key(), update_mlp_mus.shape
        )
        # new_latents: (batch_size, latent_size)
        new_latents = jnp.zeros(shape=(prev_latents.shape))

        # Loop over latents. Update each usings its own MLP
        for mlp_i in np.arange(self._latent_size):
            penalty = penalty.at[:, 0].add(kl_gaussian(
                update_mlp_mus[:, :, mlp_i], update_mlp_sigmas[:, mlp_i]
            ))
            update_mlp_output = hk.nets.MLP(
                self._update_mlp_shape,
                activation=self._activation,
                name=f"latent{mlp_i+1}_update_MLP"
            )(update_mlp_inputs[:, :, mlp_i])  # type: ignore
            # update, w, new_latent: (batch_size,)
            update = hk.Linear(1, name=f"latent{mlp_i+1}_update_target")(  # type: ignore
                update_mlp_output
            )[:, 0]
            w = jax.nn.sigmoid(hk.Linear(1, name=f"latent{mlp_i+1}_update_lr")(  # type: ignore
                update_mlp_output)
            )[:, 0]
            new_latent = w * update + (1 - w) * prev_latents[:, mlp_i]
            # new_latent = prev_latents[:, mlp_i] + update
            new_latents = new_latents.at[:, mlp_i].set(new_latent)

        #####################
        # Global Bottleneck #
        #####################
        # noised_up_latents: (batch_size, latent_size)
        noised_up_latents = new_latents + self._latent_sigmas * jax.random.normal(
            hk.next_rng_key(), new_latents.shape
        )
        penalty = penalty.at[:, 1].set(
            kl_gaussian(new_latents, self._latent_sigmas))

        ###############
        #  CHOICE MLP #
        ###############
        # Predict targets for current time step
        # This sees previous state but does _not_ see current observation
        choice_mlp_output = hk.nets.MLP(
            self._choice_mlp_shape, activation=self._activation,
            name="choice_MLP"
        )(noised_up_latents)  # type: ignore
        # (batch_size, target_size)
        y_hat = hk.Linear(self._target_size, name="choice_logits")( # type: ignore
            choice_mlp_output)  # type: ignore

        # Append the penalty, so that rnn_utils can apply it as part of the loss

        # If we are in eval mode, there should be no penalty
        penalty = penalty * (1 - self._eval_mode)

        # output: (batch_size, target_size + 2)
        output = jnp.concatenate((y_hat, penalty), axis=1)

        return output, noised_up_latents

    def initial_state(self, batch_size):
        # (batch_size, latent_size)
        latents = jnp.ones([batch_size, self._latent_size]
                           ) * self._latent_inits
        return latents

def _get_disrnn_prefix(params: Params) -> str:
    """get the model prefix used in the params
    can be either 'hk_dis_rnn' or 'hk_dis_rnn2'
    otherwise, raise ValueError
    """
    if 'hk_dis_rnn2' in params.keys():
        return 'hk_dis_rnn2'
    elif 'hk_dis_rnn' in params.keys():
        return 'hk_dis_rnn'
    elif 'deep_dis_rnn' in params.keys():
        return 'deep_dis_rnn'
    else:
        raise ValueError("unrecognized params")
    
def sort_bottlenecks(params, sort_latents=True):

    params_disrnn = params[_get_disrnn_prefix(params)]

    latent_dim = params_disrnn['latent_sigmas_unsquashed'].shape[0]
    obs_dim = params_disrnn['update_mlp_sigmas_unsquashed'].shape[0] - latent_dim

    latent_sigmas = 2 * jax.nn.sigmoid(
        jnp.array(params_disrnn['latent_sigmas_unsquashed'])
    )

    update_sigmas = 2 * jax.nn.sigmoid(
        np.transpose(
            params_disrnn['update_mlp_sigmas_unsquashed']
        )
    )
    if sort_latents:
        latent_sigma_order = np.argsort(
            params_disrnn['latent_sigmas_unsquashed']
        )
        latent_sigmas = latent_sigmas[latent_sigma_order]

        update_sigma_order = np.concatenate(
            (np.arange(0, obs_dim, 1), obs_dim + latent_sigma_order), axis=0
        )
        update_sigmas = update_sigmas[latent_sigma_order, :]
        update_sigmas = update_sigmas[:, update_sigma_order]
    else:
        latent_sigma_order = np.arange(latent_dim)

    return latent_sigmas, update_sigmas, latent_sigma_order

def order_bottlenecks(params):
    """order latent and update bottlenecks by latent bottleneck value"""
    params_disrnn = params[_get_disrnn_prefix(params)]
    
    latent_dim = params_disrnn['latent_sigmas_unsquashed'].shape[0]
    obs_dim = params_disrnn['update_mlp_sigmas_unsquashed'].shape[0] - latent_dim

    latent_sigma_order = np.argsort(
        params_disrnn['latent_sigmas_unsquashed']
    )

    update_sigma_order = np.concatenate(
        (np.arange(0, obs_dim, 1), obs_dim + latent_sigma_order), axis=0
    )

    return latent_sigma_order, update_sigma_order

def get_bottlenecks_update(params: ListLike, prefix = None):
    """extract update bottleneck for a list of parameters
    value close to 1 means open bottleneck
    """
    if prefix is None:
        prefix = _get_disrnn_prefix(params[0]) # type: ignore

    if 'update_mlp_sigmas_param' in params[0][prefix]:
        tmp = traverse_util.t_identity.each()[prefix]['update_mlp_sigmas_param'].iterate(params)
        update_bns = 1 - np.abs(np.dstack(list(tmp)))
    else:
        tmp = traverse_util.t_identity.each()[prefix]['update_mlp_sigmas_unsquashed'].iterate(params)
        update_bns = 1 - _sigma_squashed(np.dstack(list(tmp)))
    return np.moveaxis(np.swapaxes(update_bns, 0, 1),-1, 0)

def get_bottlenecks_latent(params: ListLike, prefix = None):
    """extract latent bottleneck for a list of parameters
    value close to 1 means open bottleneck
    """
    if prefix is None:
        prefix = _get_disrnn_prefix(params[0]) # type: ignore
    
    if 'latent_sigmas_param' in params[0][prefix]:
        tmp = traverse_util.t_identity.each()[prefix]['latent_sigmas_param'].iterate(params)
        latent_bns = 1 - np.abs(np.dstack(list(tmp)))
    else:
        tmp = traverse_util.t_identity.each()[prefix]['latent_sigmas_unsquashed'].iterate(params)
        latent_bns = 1 - _sigma_squashed(np.dstack(list(tmp)))
    return np.moveaxis(np.swapaxes(latent_bns, 0, 1), -1, 0)