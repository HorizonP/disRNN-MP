from typing import Iterable, Callable, Any

import haiku as hk
import jax
import jax.numpy as jnp

from disRNN_MP.rnn.disrnn import kl_gaussian, _sigma_squashed
from disRNN_MP.metrics import BerLL_logit
from disRNN_MP.rnn.utils import RNNtransformed

class hkDisRNN(hk.RNNCore):
    """disRNN implemented with haiku

    this one generate output in dictionary

    init Args:
        target_size (int, optional): network output size. Defaults to 2.
        latent_size (int, optional): latent size. Defaults to 10.
        update_mlp_shape (Iterable[int], optional): shape of update MLP. Defaults to (10, 10, 10).
        choice_mlp_shape (Iterable[int], optional): shape of choice MLP. Defaults to (10, 10, 10).
        eval_mode (bool, optional): eval_mode will turn off randomness. Defaults to False.
        activation (Callable[[Any], Any], optional): the activation function used by the network. Defaults to jax.nn.relu
    """

    def __init__(
            self,
            # vector size of target (prediction), usually 2 for 2-arm bandit -- the log-p for each action
            target_size: int = 2,
            latent_size: int = 10,
            update_mlp_shape: Iterable[int] = (10, 10, 10),
            choice_mlp_shape: Iterable[int] = (10, 10, 10),
            eval_mode: bool = False, # eval_mode will turn off randomness
            activation: Callable[[Any], Any] = jax.nn.relu,
            choice_bottleneck: bool = False, # TODO to be implemented
        ):
        """_summary_

        Args:
            target_size (int, optional): network output size. Defaults to 2.
            latent_size (int, optional): latent size. Defaults to 10.
            update_mlp_shape (Iterable[int], optional): shape of update MLP. Defaults to (10, 10, 10).
            choice_mlp_shape (Iterable[int], optional): shape of choice MLP. Defaults to (10, 10, 10).
            eval_mode (bool, optional): eval_mode will turn off randomness. Defaults to False.
            activation (Callable[[Any], Any], optional): the activation function used by the network. Defaults to jax.nn.relu
        """
        super().__init__()
        self.name = 'hk_dis_rnn'
        self._target_size = target_size
        self._latent_size = latent_size
        self._update_mlp_shape = update_mlp_shape
        self._choice_mlp_shape = choice_mlp_shape
        self._eval_mode = float(bool(eval_mode))
        self._activation = activation

    def __call__(
            self, observations: jnp.ndarray, prev_latents: jnp.ndarray):
        """run a single step (one trial) for all batches"""
        # observations is of shape (batch_size, obs_size)

        #################
        # define params #
        #################

        obs_size = observations.shape[1]
        # Each update MLP gets input from both the latents and the observations.
        # It has a sigma and a multiplier associated with each.
        mlp_input_size = self._latent_size + obs_size
        # At init the bottlenecks should all be open: sigmas small and multipliers 1
        update_mlp_sigmas_unsquashed = hk.get_parameter(
            'update_mlp_sigmas_unsquashed',
            (mlp_input_size, self._latent_size),
            init=hk.initializers.RandomUniform(minval=-3, maxval=-2),
        )
        # Training encourages sigmas to be ~1 or smaller. Bound them between 0 and 2
        _update_mlp_sigmas = (
            _sigma_squashed(update_mlp_sigmas_unsquashed) * (1 - self._eval_mode)
        )
        _update_mlp_multipliers = hk.get_parameter(
            'update_mlp_gates',
            (mlp_input_size, self._latent_size),
            init=hk.initializers.Constant(constant=1),
        )

        # Latents will also go through a bottleneck
        latent_sigmas_unsquashed = hk.get_parameter(
            'latent_sigmas_unsquashed',
            (self._latent_size,),
            init=hk.initializers.RandomUniform(minval=-3, maxval=-2),
        )
        _latent_sigmas = (
            _sigma_squashed(latent_sigmas_unsquashed) * (1 - self._eval_mode)
        )

        

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
            * _update_mlp_multipliers
        )
        # update_mlp_sigmas: (obs_size + latent_size, latent_size)
        update_mlp_sigmas = _update_mlp_sigmas * (1 - self._eval_mode)
        # update_mlp_inputs: (batch_size, obs_size + latent_size, latent_size)
        update_mlp_inputs = update_mlp_mus + update_mlp_sigmas * jax.random.normal(
            hk.next_rng_key(), update_mlp_mus.shape
        )
        # new_latents: (batch_size, latent_size)
        new_latents = jnp.zeros(shape=(prev_latents.shape))

        # Loop over latents. Update each usings its own MLP
        for mlp_i in range(self._latent_size):
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
        noised_up_latents = new_latents + _latent_sigmas * jax.random.normal(
            hk.next_rng_key(), new_latents.shape
        )
        penalty = penalty.at[:, 1].set(
            kl_gaussian(new_latents, _latent_sigmas))

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
        # output = jnp.concatenate((y_hat, penalty), axis=1)
        output = {'prediction': y_hat, 'penalty': penalty}

        return output, noised_up_latents
    
    def initial_state(self, batch_size):

        # Latent initial values are also free parameters
        _latent_inits = hk.get_parameter(
            'latent_inits',
            (self._latent_size,),
            init=hk.initializers.RandomUniform(minval=-0.01, maxval=0.01),
        )

        # (batch_size, latent_size)
        latents = jnp.ones([batch_size, self._latent_size]
                           ) * _latent_inits
        return latents
    
# example loss function
def make_loss_fun(
        model_apply, #: Callable[[Params, RandomKey, Inputs], Tuple[Outputs, States]],
        penalty_scale: float = 0,
        beta_scale: float = 1,
        
    ): # -> Callable[[Params, RandomKey, Inputs, Outputs], Loss]:
    """_summary_

    Args:
        model_apply (Callable): _description_

    Returns:
        loss_fun: (params, random_key, xs, ys) -> loss
    """

    def categorical_log_likelihood(
        labels: jax.Array, output_logits: jax.Array
    ) -> jax.Array:
        masked_log_liks = BerLL_logit(labels = labels, logits = output_logits, norm = False)
        loss = -jnp.nansum(masked_log_liks)
        return loss

    def penalized_categorical_loss(
        params, random_key, xs, targets
    ) -> jax.Array:
        """Treats the last two elements of the model outputs as penalty."""
        # (n_steps, n_episodes, n_targets)
        model_output, _ = model_apply(params, random_key, xs)
        output_logits = model_output['prediction']
        penalties = model_output['penalty']     
        penalties = penalties.at[:, :, 0].multiply(beta_scale)

        penalty = jnp.sum(penalties)  # ()
        loss = (
            categorical_log_likelihood(targets, output_logits)
            + penalty_scale * penalty
        )
        return loss

    return penalized_categorical_loss


# example parameter metric function
def make_param_metric_expLL(model: RNNtransformed, ys_feat_ind: int = 0, prediction_slice: slice = slice(None)):
    key = jax.random.PRNGKey(0)
    def _metric(params, xs, ys):
        model_outputs, _ = model.apply(params, key, xs)
        normLL = BerLL_logit(ys[..., ys_feat_ind], model_outputs['prediction'][:,:,prediction_slice], norm = True)
        exp_normLL = jnp.exp(normLL)
        return exp_normLL
    
    return _metric