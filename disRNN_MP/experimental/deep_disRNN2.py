from typing import Iterable, Callable, Any, List, Literal, Optional, Tuple, get_args

import haiku as hk
import jax
import jax.numpy as jnp

from disRNN_MP.typing import Array, patchable_hkRNNCore
from disRNN_MP.rnn.disrnn import kl_gaussian
from disRNN_MP.metrics import BerLL_logit
from disRNN_MP.rnn.utils import RNNtransformed

_TYPE_sigma_parameterization = Literal['abs', 'sigmoid']

class deepDisRNN(patchable_hkRNNCore):
    """disRNN implemented with haiku

    include bottleneck for choiceMLP input, and use a ReLU-like 

    - model outputs a dictionary {'prediction': *, 'penalty': *}

    Implementation
    - eval mode will zero all the sigmas associated with any information bottleneck, which is handled in `_get_sigma_abs` and `_get_sigma_sigmoid`
    - eval mode will also zero the accumulated penalty terms


    Args:
        target_size (int, optional): network output size. Defaults to 2.
        latent_size (int, optional): latent size. Defaults to 10.
        update_mlp_shape (Iterable[int], optional): shape of update MLP. Defaults to (10, 10, 10).
        choice_mlp_shape (Iterable[int], optional): shape of choice MLP. Defaults to (10, 10, 10).
        eval_mode (bool, optional): eval_mode will turn off randomness. Defaults to False.
        activation (Callable[[Any], Any], optional): the activation function used by the network. Defaults to jax.nn.relu
        epsilon (float, optional): sigma = epsilon + abs(sigma_param) the minimum of sigma for bottleneck, just to make sigma never goes to exactly 0
    """

    ## ============= type annotations and docstrings
        
    def apply_latent_bn(self, latent: Array) -> Tuple[jax.Array, jax.Array]:
        """apply latent bottleneck

        add gaussian noise to the latent, calculate latent information bottleneck penalty

        the exact logic of the function is set by `_configure_latent_bn` method with `latent_bn_gate` and `sigma_parameterization` argument when initialized

        Args:
            latent (Array): updated latent without noise

        Returns:
            Tuple[Array, Array]: _description_
        """
        # this is a placeholder definition, the real function is determined by `_configure_latent_bn` in `__init__` method
        ...

    def get_update_sigma(self, obs_size: int) -> jax.Array:
        """get the update sigma as haiku parameter

        how update sigma is parameterized is determined by `sigma_parameterization` argument during initialization

        the update sigma will be zeros when `eval_mode` is True

        Args:
            obs_size (int): the size of observation features

        Returns:
            jax.Array: update sigma
        """
        ...

    def get_choice_sigma(self) -> jax.Array:
        """get the choice sigma as haiku parameter

        how choice sigma is parameterized is determined by `sigma_parameterization` argument during initialization

        the choice sigma will be zeros when `eval_mode` is True

        Args:
            obs_size (int): the size of observation features

        Returns:
            jax.Array: choice sigma
        """
        ...
    
    def get_latent_sigma(self) -> jax.Array:
        """get the latent sigma as haiku parameter

        how latent sigma is parameterized is determined by `sigma_parameterization` argument during initialization

        the latent sigma will be zeros when `eval_mode` is True

        Args:
            obs_size (int): the size of observation features

        Returns:
            jax.Array: latent sigma
        """
        ...

    ## ============= functions for configuring the model

    def _get_sigma_abs(self, name, sz, init = hk.initializers.Constant(constant=0)):
        sigma_param = hk.get_parameter(
            name=name, shape=sz, init=init,
        )
        # Training encourages sigmas to be ~1 or smaller. Bound them between 0 and 2
        sigmas = (
            (self._epsilon + jnp.abs(sigma_param)) * (1 - self._eval_mode)
        )
        return sigmas
    
    def _get_sigma_sigmoid(self, name, sz, init = hk.initializers.RandomUniform(minval=-3, maxval=-2)):
        sigma_param = hk.get_parameter(
            name=name, shape=sz, init=init,
        )
        # Training encourages sigmas to be ~1 or smaller. Bound them between 0 and 2
        sigmas = (
            2 * jax.nn.sigmoid(sigma_param) * (1 - self._eval_mode)
        )
        return sigmas
    
    def _configure_latent_bn(self):
        """with or without a mean multiplier (gate)
        
        this function use `self._latent_bn_gate` to conditionally generate a function to be run inside __call__

        """

        if self._latent_bn_gate:
            def latent_bn(latent):
                _latent_sigmas = self.get_latent_sigma()

                _latent_multipliers = hk.get_parameter(
                    'latent_gates',
                    (self._latent_size,),
                    init=hk.initializers.Constant(constant=1),
                )

                # noised_up_latents: (batch_size, latent_size)
                noisy_latent = latent * _latent_multipliers + _latent_sigmas * jax.random.normal(
                    hk.next_rng_key(), latent.shape
                )

                penalty = kl_gaussian(latent * _latent_multipliers, _latent_sigmas)

                return noisy_latent, penalty
            
        else:
            def latent_bn(latent):
                _latent_sigmas = self.get_latent_sigma()

                # noised_up_latents: (batch_size, latent_size)
                noisy_latent = latent + _latent_sigmas * jax.random.normal(
                    hk.next_rng_key(), latent.shape
                )

                penalty = kl_gaussian(latent, _latent_sigmas)

                return noisy_latent, penalty
        
        return latent_bn

    ## ============= initialization
    def __init__(
            self,
            # vector size of target (prediction), usually 2 for 2-arm bandit -- the log-p for each action
            target_size: int = 2,
            latent_size: int = 10,
            update_mlp_shape: Iterable[int] = (10, 10, 10),
            choice_mlp_shape: Iterable[int] = (10, 10, 10),
            eval_mode: bool = False, # eval_mode will turn off randomness
            activation: Callable[[Any], Any] = jax.nn.relu,
            sigma_parameterization: Optional[_TYPE_sigma_parameterization] = "abs",
            epsilon: float = 1e-4, 
            latent_bn_gate: Optional[bool] = False,
            latent_shape: None | List[int] | Tuple[int, ...] = None
        ):
        """_summary_

        Args:
            target_size (int, optional): network output size. Defaults to 2.
            latent_size (int, optional): latent size. Defaults to 10.
            update_mlp_shape (Iterable[int], optional): shape of update MLP. Defaults to (10, 10, 10).
            choice_mlp_shape (Iterable[int], optional): shape of choice MLP. Defaults to (10, 10, 10).
            eval_mode (bool, optional): eval_mode will turn off randomness. Defaults to False.
            activation (Callable[[Any], Any], optional): the activation function used by the network. Defaults to jax.nn.relu
            epsilon (float, optional): sigma = epsilon + abs(sigma_param) the minimum of sigma for bottleneck, just to make sigma never goes to exactly 0
            latent_shape (None | Iterable[int]): to specify a deep disRNN model. this parameter takes an iterable (list, tuple or others) that specify number of latents for each layer. 
            The sum of this should equal to latent_size. If None, this will be set as (latent_size, ) to effectively specify a single layer disRNN. Latents at the first layer are as usual, 
            whose updateMLP receives inputs from current observation and other 1st-layer latent values on last time step. The updateMLP of latents at deeper layer will take current observation and updated value of latents from 1 layer below. 
            Latents from all layers will input to choiceMLP, so that latents at higher layers are not necessary for the model to run, i.e. they can be with all closed bottleneck
        """
        super().__init__()
        self.name = 'hk_dis_rnn'
        self._target_size = target_size
        self._latent_size = latent_size
        self._update_mlp_shape = update_mlp_shape
        self._choice_mlp_shape = choice_mlp_shape
        self._eval_mode = float(bool(eval_mode))
        self._activation = activation
        self._sigma_parameterization = sigma_parameterization
        self._latent_bn_gate = latent_bn_gate
        self._epsilon = epsilon

        if latent_shape is None:
            self._latent_shape = (self._latent_size, )
        else:
            self._latent_shape = latent_shape
            assert sum(self._latent_shape) == self._latent_size, f"latent shape {self._latent_shape} does not sum to latent_size {self._latent_size}"

        self._max_lat_lyr_size = max(self._latent_shape)
        # Since update MLP of latents at different layers may take different numbers of latent inputs, to still keep sigmas of all updateMLPs in one matrix, the update_mlp_sigmas parameter will be declared as (_max_lat_lyr_size + obs_size, _latent_size), thus need this `_max_lat_lyr_size` attribute

        # ========== configure how to parameterize sigmas in the bottlenecks
        options = get_args(_TYPE_sigma_parameterization)
        assert sigma_parameterization in options, f"'{sigma_parameterization}' is not in {options}"

        if self._sigma_parameterization == "abs":
            
            self.get_update_sigma = lambda obs_size: self._get_sigma_abs(
                'update_mlp_sigmas_param',
                (self._max_lat_lyr_size + obs_size, self._latent_size)
            )

            self.get_latent_sigma = lambda : self._get_sigma_abs(
                'latent_sigmas_param',
                (self._latent_size,)
            )

            self.get_choice_sigma = lambda : self._get_sigma_abs(
                'choice_mlp_sigmas_param',
                (self._latent_size,)
            )
            
        else:

            self.get_update_sigma = lambda obs_size: self._get_sigma_sigmoid(
                'update_mlp_sigmas_unsquashed',
                (self._max_lat_lyr_size + obs_size, self._latent_size)
            )

            self.get_latent_sigma = lambda : self._get_sigma_sigmoid(
                'latent_sigmas_unsquashed',
                (self._latent_size,)
            )

            self.get_choice_sigma = lambda : self._get_sigma_sigmoid(
                'choice_mlp_sigmas_unsquashed',
                (self._latent_size,)
            )

        self.apply_latent_bn = self._configure_latent_bn()

    def set_patch_state_ids(self, patch_state_ids: List[int]):
        """set which latents are going to be patched by exogeneous states

        0-based latent index

        this method simply set the `patch_state_ids` attribute

        Args:
            patch_state_ids (List[int]): the 0-based index of latents that will be patched
        """
        self._patch_state_ids = patch_state_ids

    def latent_update(self, observations: Array, prev_latents: Array) -> Tuple[jax.Array, jax.Array]:
        """run a single step (one trial) for all batches

        deepDisRNN is different from DisRNN only in the latent_update part
        
        Returns:
            A tuple of (new_latents, penalty): first is updated latents, Array of shape (batch_size, latent_size).
            Second is update bottleneck penalty, Array of shape (batch_size, )
        
        """
        # observations is of shape (batch_size, obs_size)
        # prev_latents is of shape (batch_size, latent_size)

        obs_size = observations.shape[1]

        # Each update MLP gets input from both the latents and the observations.
        # It has a sigma and a multiplier associated with each.
        mlp_input_size = self._latent_size + obs_size

        _update_mlp_multipliers = hk.get_parameter(
            'update_mlp_gates',
            (obs_size + self._max_lat_lyr_size, self._latent_size),
            init=hk.initializers.Constant(constant=1),
        )

        # _update_mlp_sigmas: (obs_size + _max_lat_lyr_size, latent_size)
        _update_mlp_sigmas = self.get_update_sigma(obs_size)

        # Accumulator for update information bottleneck costs
        penalty = jnp.zeros((observations.shape[0],))
        # (batch_size, )
        
        ################
        #  UPDATE MLPs #
        ################
        # new_latents: (batch_size, latent_size)
        new_latents = jnp.zeros(shape=(prev_latents.shape))

        # the updateMLP input consist of current observation + some kind of previous latents observation
        # for the 1st layer latents, the prev. latents obs. is the 1st layer latent values at last time step
        # for deeper layers, the prev. latents obs. is the 1-layer-lower latent values at current time step

        # lat_input shape: (batch_size, N_lat_inputs)
        # initialized as 1st layer latent input: previous 1st layer latent values
        lat_input = prev_latents[:, :self._latent_shape[0]]
        lat_ind_offset = 0

        for lyr_i in range(len(self._latent_shape)):
            # number of latents in current layer
            lyr_sz = self._latent_shape[lyr_i]

            #================== batch calculating updateMLP inputs for all latents in current layer

            # the index range for current layer latents in all latents: lyr_latents = all_latents[lat_ind_range] 
            lat_ind_range = slice(lat_ind_offset, lat_ind_offset + lyr_sz)
            # the index range for inputs for current layer latents: lyr_updMLP_inputs = max_updMLP_inputs[input_ind_range]
            input_ind_range = slice(obs_size + lat_input.shape[1])

            # lyr_update_mlp_mus_unscaled: (batch_size, obs_size + lat_input_size)
            lyr_update_mlp_mus_unscaled = jnp.concatenate(
                (observations, lat_input), axis=1
            )
            # lyr_update_mlp_mus: (batch_size, obs_size + lat_input_size, lyr_sz)
            lyr_update_mlp_mus = (
                jnp.expand_dims(lyr_update_mlp_mus_unscaled, 2) # (batch_size, obs_size + lat_input_size, 1)
                * _update_mlp_multipliers[input_ind_range, lat_ind_range] # (obs_size + lat_input_size, lyr_sz)
            )
            # lyr_update_mlp_sigmas: (obs_size + lat_input_size, lyr_sz)
            lyr_update_mlp_sigmas = _update_mlp_sigmas[input_ind_range, lat_ind_range] * (1 - self._eval_mode)
            # lyr_update_mlp_inputs: (batch_size, obs_size + lat_input_size, lyr_sz)
            lyr_update_mlp_inputs = lyr_update_mlp_mus + lyr_update_mlp_sigmas * jax.random.normal(
                hk.next_rng_key(), lyr_update_mlp_mus.shape
            )

            for mlp_i in range(lyr_sz):

                #================== run updateMLP
                mlp = hk.nets.MLP(
                    self._update_mlp_shape,
                    activation=self._activation,
                    name=f"latent{lat_ind_offset + mlp_i + 1}_update_MLP"
                )

                update_mlp_output = mlp(lyr_update_mlp_inputs[:, :, mlp_i])

                #================== update latent
                # update, w, new_latent: (batch_size,)
                update = hk.Linear(1, name=f"latent{lat_ind_offset + mlp_i + 1}_update_target")(  # type: ignore
                    update_mlp_output
                )[:, 0]
                w = jax.nn.sigmoid(hk.Linear(1, name=f"latent{lat_ind_offset + mlp_i + 1}_update_lr")(  # type: ignore
                    update_mlp_output)
                )[:, 0]

                new_latent = w * update + (1 - w) * prev_latents[:, lat_ind_offset + mlp_i]

                new_latents = new_latents.at[:, lat_ind_offset + mlp_i].set(new_latent)

                #================== add up penalty
                penalty += kl_gaussian(
                    lyr_update_mlp_mus[:, :, mlp_i], lyr_update_mlp_sigmas[:, mlp_i]
                )

            # for next layer iteration
            lat_input = new_latents[:, lat_ind_range]
            lat_ind_offset += lyr_sz

        return new_latents, penalty
    
    def choice_selection(self, noised_up_latents: Array) -> Tuple[jax.Array, jax.Array]:

        # get parameters for choiceMLP input bottleneck
        _choice_mlp_sigmas = self.get_choice_sigma()

        _choice_mlp_multipliers = hk.get_parameter(
            'choice_mlp_gates',
            (self._latent_size,),
            init=hk.initializers.Constant(constant=1),
        )

        ###############
        #  CHOICE MLP #
        ###############
        # Predict targets for current time step
        # This sees previous state but does _not_ see current observation

        choice_mlp_input = noised_up_latents * _choice_mlp_multipliers + _choice_mlp_sigmas * jax.random.normal(
            hk.next_rng_key(), noised_up_latents.shape
        )

        # here if I use noised_up_latents as the mean for KL divergence calculation, the latent_sigma will be dragged in two directions during training (as latent_sigmas is added to noised_up_latents)
        # this may be why it used to generate a lot of 0.5 latent sigma for closed bottlenecks
        penalty = kl_gaussian(noised_up_latents * _choice_mlp_multipliers, _choice_mlp_sigmas)

        choice_mlp_output = hk.nets.MLP(
            self._choice_mlp_shape, activation=self._activation,
            name="choice_MLP"
        )(choice_mlp_input)  # type: ignore
        # (batch_size, target_size)
        y_hat = hk.Linear(self._target_size, name="choice_logits")( # type: ignore
            choice_mlp_output)  # type: ignore
        
        return y_hat, penalty

    def __call__(self, observations: Array, prev_latents: Array) -> Tuple[dict[str, jax.Array], jax.Array]:
        """run a single step (one trial) for all batches"""
        # observations is of shape (batch_size, obs_size)
        # prev_latents is of shape (batch_size, latent_size)


        new_latents, update_penalty = self.latent_update(observations, prev_latents)
        noised_up_latents, latent_penalty = self.apply_latent_bn(new_latents)
        y_hat, choice_penalty = self.choice_selection(noised_up_latents)
        

        # Accumulator for KL costs
        penalty = jnp.stack([update_penalty, latent_penalty, choice_penalty], axis=1)
        # size: (batch_size, 3)
        # penalty[:,0] is for update rule bottlenecks
        # penalty[:,1] is for global bottlenecks
        # penalty[:,2] is for choiceMLP input bottlenecks

        # Append the penalty, so that rnn_utils can apply it as part of the loss

        # If we are in eval mode, there should be no penalty
        penalty = penalty * (1 - self._eval_mode)

        # output: (batch_size, target_size + 2)
        # output = jnp.concatenate((y_hat, penalty), axis=1)
        output = {'prediction': y_hat, 'penalty': penalty}

        return output, noised_up_latents
    
    def step_with_exo_state(self, observations: Array, prev_latents: Array, exo_states: Array) -> Tuple[dict[str, jax.Array], jax.Array]:
        """step the network with activation patching

        clamping the latent value just after latent value update and before choice selection
        
        before running this function, it is required to run `set_patch_state_ids` method
        """

        new_latents, update_penalty = self.latent_update(observations, prev_latents)

        # activation patching
        new_latents = new_latents.at[:, self._patch_state_ids].set(exo_states)

        noised_up_latents, latent_penalty = self.apply_latent_bn(new_latents)
        y_hat, choice_penalty = self.choice_selection(noised_up_latents)
        
        # Accumulator for KL costs
        penalty = jnp.stack([update_penalty, latent_penalty, choice_penalty], axis=1)
        # size: (batch_size, 3)

        # If we are in eval mode, there should be no penalty
        penalty = penalty * (1 - self._eval_mode)

        # output: (batch_size, target_size + 2)
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
        theta_scale: float = 1,
        
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
        penalties = penalties.at[:, :, 0].multiply(beta_scale) # for update rule bottlenecks
        penalties = penalties.at[:, :, 2].multiply(theta_scale) # for choice MLP input bottlenecks

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