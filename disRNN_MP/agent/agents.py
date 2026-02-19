from typing import Callable, Optional, Tuple, overload
from abc import ABC, abstractmethod

import jax
import jax.numpy as jnp
import haiku as hk
import numpy as np
from flax.core import FrozenDict

from disRNN_MP.typing import Array, Params
from .abstract import RL_agent, RL_multiAgent
from disRNN_MP.classic_RL import RLmodelWrapper, RLmodel
from disRNN_MP.utils import _select_func_by_1st_arg_type
from disRNN_MP.rnn.utils import has_haiku_static_attrs

class RLmodel_agent(RL_agent):
    """an agent defined from a fitted classic RL model
    
    the model provide probability p(L) and p(R), the agent randomly sample a choice from the distribution

    running one session at a time

    modelw: a fitted classic RL model
    seed: the random seed used for sampling choices
    
    """

    def __init_from_modelDef_params(self, rlmodel, params, seed = 0):
        self._RLmodel = rlmodel
        self.random_seed = seed
        self.params = params

        def _choice_probs(latents):
            """choice selection for single session
            latents(N_values,) -> (N_actions,)
            """
            ch_probs = self._RLmodel.apply(
                params, 
                latents, 
                method='choice_selection')
            return ch_probs
        
        def _update(choice, reward, latents):
            """value update for single session
            choice(0-dim), reward(0-dim), latents(N_values,) -> latents(N_values,)
            """
            obs = jnp.array([choice, reward])
            _latents = self._RLmodel.apply(
                params, 
                obs = obs, values = latents, 
                method='value_update')
            return _latents
        
        # since `init_values` function is for multiple sessions, it returns an array of shape (N_sessions, N_values) even only 1 session. The indexing `[0]` is to make the init value shape compatiable with value update and choice selection
        self._new_sess_latents = self._RLmodel.apply(
            params, 
            n_sess = 1,
            method='init_values')[0]
        
        
        self._choice_probs = jax.jit(_choice_probs)
        self._update = jax.jit(_update)

        self.new_session()
        super().__init__(seed)

    def __init_from_ModelWrapper(self, modelw, seed = None):
        self._modelw = modelw
        self.__init_from_modelDef_params(modelw.model, {'params': modelw.params}, seed)

    @overload
    def __init__(self, modelw:RLmodelWrapper, seed: Optional[int] = ...) -> None:
        ...

    @overload
    def __init__(self, rlmodel:RLmodel, params, seed: Optional[int] = ...) -> None:
        ...

    def __init__(self, *args, **kwargs):

        _select_func_by_1st_arg_type(
                {
                    RLmodelWrapper: self.__init_from_ModelWrapper,
                    RLmodel: self.__init_from_modelDef_params,
                },
                args, kwargs
            )
    

    @property
    def choice_probs(self) -> jax.Array:
        return self._choice_probs(self.latents)
    
    def update(self, choice, reward) -> None:        
        self.latents = self._update(choice, reward, self.latents)
        return super().update(choice, reward) # update random seed

    def new_session(self) -> None:
        self.latents = self._new_sess_latents
        

class hkNetwork_agent(RL_agent):
    """A class that allows running a pretrained RNN as an agent.

    Attributes:
        make_network: A Haiku function that returns an RNN architecture
        params: A set of Haiku parameters suitable for that architecture
        latents: last latent variable state, size: (1, N_latent)
    """

    def __init__(self,
                make_network: Callable[[], hk.RNNCore],
                params: hk.Params|Params,
                n_actions: int = 2,
                state_to_numpy: bool = False,
                random_seed: int = 0):
        """Initialize the agent network.

        Args: 
            make_network: function that instantiates a callable haiku network object
            params: parameters for the network
            n_actions: number of permitted actions (default = 2)
        """
        self._state_to_numpy = state_to_numpy
        self._n_actions = n_actions
        self._params = params

        def _step_network(xs: np.ndarray, state: hk.State) -> Tuple[np.ndarray, hk.State]:
            """Apply one step of the network.
            
            Args:
            xs: array containing network inputs
            state: previous state of the hidden units of the RNN model.
            
            Returns:
            y_hat: output of RNN
            new_state: state of the hidden units of the RNN model
            """
            core = make_network()
            y_hat, new_state = core(xs, state)
            return y_hat, new_state

        def _get_initial_state() -> hk.State:
            """Get the initial state of the hidden units of RNN model."""
            core = make_network()
            state = core.initial_state(1)
            return state

        _, self._step = hk.transform(_step_network)
        _, self._get_init = hk.transform(_get_initial_state)

        self._step = jax.jit(self._step)
        self._get_init = jax.jit(self._get_init)

        self.infuse_seed(random_seed)
        self.new_session()

        # here it tries to figure out the data type for output of the network
        try:
            out_shape = jax.eval_shape(self._step, self._params, self._PRNGkey, self._xs, self.latents)
        except ValueError as e:
            raise TypeError("the input data shape mismatches with what the network takes")
            
        if isinstance(out_shape[0], dict) and 'prediction' in out_shape[0]:
            self._logits_indexing = jax.jit(lambda out: out['prediction'][0,:])
            self._output_type = "dict"
        elif hasattr(out_shape[0], 'shape'): # it's an array
            self._logits_indexing = jax.jit(lambda out: out[0,:self._n_actions])
            self._output_type = "array"
        else:
            raise TypeError(f"unknown output shape from the network's step function: {out_shape[0]}")

        super().__init__()

    def infuse_seed(self, random_seed) -> None:
        """finish the agent initialization with a random_seed
        
        Set up all the instance properties and functions that are dependent on a random seed. 

        Can be used to reset the random seed

        """
        super().infuse_seed(random_seed)

        self._PRNGkey = jax.random.PRNGKey(self.random_seed)

        # self._model_fun = jax.jit(
        #     lambda xs, state: self._step(self._params, key, xs, state))
        

    def new_session(self):
        """Reset the network for the beginning of a new session."""
        self._PRNGkey, key1, key2 = jax.random.split(self._PRNGkey, 3)

        _initial_state = self._get_init(self._params, key1)
        self.latents = _initial_state
        self._next_latent = _initial_state

        self._xs = jax.random.choice(key2, 2, (1,2)) # np.zeros((1, 2))

    @property
    def choice_probs(self) -> Array:
        """Predict the choice probabilities as a softmax over output logits."""

        # output, self._next_latent = self._model_fun(self._xs, self.latents)
        output, self._next_latent = self._step(self._params, self._PRNGkey, self._xs, self.latents)

        output_logits = self._logits_indexing(output)
        choice_probs = jax.nn.softmax(output_logits)
        return choice_probs

    def update(self, choice: int, reward: int):

        # update agent's observation
        self._xs = np.array([[choice, reward]])

        # update agent's latent variable
        if self._state_to_numpy:
            # self.latents = np.array(new_state)
            self.latents = np.array(self._next_latent)
        else:
            # self.latents = new_state
            self.latents = self._next_latent
        
        # update random seed for model's step function
        self._PRNGkey, _ = jax.random.split(self._PRNGkey, 2)

        # update random seed for sampling choice
        return super().update(choice, reward)


class null_agent(RL_agent):
    
    def __init__(self, random_seed: int = 0) -> None:
        super().__init__(random_seed)
        self.latents = np.array([0])

    @property
    def choice_probs(self):
        return np.array([0.5, 0.5])
    
    def new_session(self) -> None:
        pass

    def update(self, choice, reward) -> None:
        return super().update(choice, reward)
    


class hkNetwork_multiAgent(RL_multiAgent):
    """A class that allows running a pretrained haiku RNN as an agent.

    Attributes:
        make_network: A Haiku function that returns an RNN architecture
        params: A set of Haiku parameters suitable for that architecture
        latents: last latent variable state, size: (N_sess, N_latent)
    """

    def __init__(self,
                make_network: Callable[[], hk.RNNCore],
                params: hk.Params|Params,
                n_sess: int = 1,
                n_actions: int = 2,
                random_seed: int = 0,
                additional_inputs: None | list | tuple | Array = None,
                init_ch_probs: None | Array = None,
                ):
        """Initialize the agent network.

        Args: 
            make_network: function that instantiates a callable haiku network object
            params: parameters for the network
            n_actions: number of permitted actions (default = 2)
        """
        self.n_sess = n_sess
        self._n_actions = n_actions
        self._params = params
        if additional_inputs is not None:
            self._additional_inputs = jnp.array(additional_inputs)
            self._build_xs = lambda ch, rew: jnp.column_stack((ch, rew, jnp.tile(self._additional_inputs, (ch.shape[0], 1))))
        else:
            self._build_xs = lambda ch, rew: jnp.column_stack((ch, rew))

        if init_ch_probs is None:
            arr = np.ones((n_sess, n_actions))
            self._init_choice_probs = arr / np.sum(arr, axis=1).reshape(-1,1)
        else:
            assert init_ch_probs.shape == (n_sess, n_actions), f"init_ch_probs has shape {init_ch_probs.shape}, which does not match what is required {(n_sess, n_actions)}"
            self._init_choice_probs = init_ch_probs

        def _step_network(xs: np.ndarray, state: hk.State) -> Tuple[np.ndarray, hk.State]:
            """Apply one step of the network.
            
            Args:
            xs: array containing network inputs
            state: previous state of the hidden units of the RNN model.
            
            Returns:
            y_hat: output of RNN
            new_state: state of the hidden units of the RNN model
            """
            core = make_network()
            y_hat, new_state = core(xs, state)
            return y_hat, new_state

        def _get_initial_state(n_sess) -> hk.State:
            """Get the initial state of the hidden units of RNN model."""
            core = make_network()
            state = core.initial_state(n_sess)
            return state

        _, self._step = hk.transform(_step_network)
        _, self._get_init = hk.transform(_get_initial_state)

        self._step = jax.jit(self._step)
        self._get_init = jax.jit(self._get_init, static_argnums=2)

        self.infuse_seed(random_seed)
        self.new_session()

        # here it tries to figure out the data type for output of the network
        try:
            dummy_xs = self._build_xs(np.ones((self.n_sess, )), np.ones((self.n_sess, )))
            out_shape = jax.eval_shape(self._step, self._params, self._PRNGkey, dummy_xs, self.latents)
        except ValueError as e:
            raise TypeError("the input data shape mismatches with what the network takes")
            
        if isinstance(out_shape[0], dict) and 'prediction' in out_shape[0]:
            self._logits_indexing = jax.jit(lambda out: out['prediction'])
            self._output_type = "dict"
        elif hasattr(out_shape[0], 'shape'): # it's an array
            self._logits_indexing = jax.jit(lambda out: out[:,:self._n_actions])
            self._output_type = "array"
        else:
            raise TypeError(f"unknown output shape from the network's step function: {out_shape[0]}")

        super().__init__()

    def infuse_seed(self, random_seed) -> None:
        """finish the agent initialization with a random_seed
        
        Set up all the instance properties and functions that are dependent on a random seed. 

        Can be used to reset the random seed

        """
        super().infuse_seed(random_seed)

        self._PRNGkey = jax.random.PRNGKey(self.random_seed)

        

    def new_session(self):
        """Reset the network for the beginning of a new session."""
        self._PRNGkey, key1 = jax.random.split(self._PRNGkey, 2)

        _initial_state = self._get_init(self._params, key1, self.n_sess)
        self.latents = _initial_state

        assert self._init_choice_probs.shape == (self.n_sess, self._n_actions), f"init_ch_probs has shape {self._init_choice_probs.shape}, which does not match what is required {(self.n_sess, self._n_actions)}. Please make sure the `n_sess` and `_n_actions` attributes are not modified ad-hoc"

        self._choice_probs = self._init_choice_probs


    @property
    def choice_probs(self) -> jax.Array:
        """Predict the choice probabilities as a softmax over output logits."""

        return self._choice_probs

    def update(self, choice: Array, reward: Array):

        # update agent's observation
        self._xs = self._build_xs(choice, reward)

        # choice, reward + current latent (last trial updated latent) -> next choice_prob + updated_latent
        output, self.latents = self._step(self._params, self._PRNGkey, self._xs, self.latents)

        # next choice_prob
        output_logits = self._logits_indexing(output)
        self._choice_probs = jax.nn.softmax(output_logits, axis=-1)
        
        # update random seed for model's step function
        self._PRNGkey, _ = jax.random.split(self._PRNGkey, 2)

        # update random seed for sampling choice
        return super().update(choice, reward)
    


class RLmodel_multiAgent(RL_multiAgent):
    """a multi-agent defined from a fitted classic RL model
    
    the model provide probability p(L) and p(R), the agent randomly sample a choice from the distribution

    running one session at a time

    modelw: a fitted classic RL model
    seed: the random seed used for sampling choices
    
    """

    def __init_from_modelDef_params(self, 
            rlmodel: RLmodel, 
            params: dict | FrozenDict | None = None, 
            n_sess: int = 1, 
            seed = 0,
            init_ch_probs: None | Array = None,
        ):
        self._RLmodel = rlmodel
        self.random_seed = seed
        if params is None:
            params = rlmodel.init(jax.random.key(seed), rlmodel._get_dummy_inputs(n_sess)) 
        self.params = params
        self.n_sess = n_sess
        self._init_choice_probs = init_ch_probs

        def _cal_choice_probs(latents):
            """choice selection for single session
            latents(N_sess, N_values) -> (N_sess, N_actions,)
            """
            ch_probs = self._RLmodel.apply(
                params, 
                latents, 
                method='choice_selection')
            return ch_probs
        
        def _update(choice, reward, latents):
            """update agent's latent variables and next random seed after observe itself's choice and reward
            both choice and reward should be 1D array of shape (N_sess, )
            """
            # obs = jnp.array([choice, reward])
            if len(choice.shape) == 1:
                choice = choice.reshape(-1, 1)
            if len(reward.shape) == 1:
                reward = reward.reshape(-1, 1)
            obs = jnp.concatenate((choice, reward), axis=1)
            _latents = self._RLmodel.apply(
                params, 
                obs = obs, values = latents, 
                method='value_update')
            return _latents
        
        self._cal_choice_probs = jax.jit(_cal_choice_probs)
        self._update = jax.jit(_update)

        self.new_session()
        super().__init__(seed)

    def __init_from_ModelWrapper(self, modelw, n_sess = 1, seed = 0):
        self._modelw = modelw
        self.__init_from_modelDef_params(modelw.model, {'params': modelw.params}, n_sess=n_sess, seed=seed)

    @overload
    def __init__(self, modelw:RLmodelWrapper, n_sess:int = ..., seed: Optional[int] = ...) -> None:
        ...

    @overload
    def __init__(self, rlmodel:RLmodel, params = ..., n_sess:int = ..., seed: Optional[int] = ...) -> None:
        ...

    def __init__(self, *args, **kwargs):

        _select_func_by_1st_arg_type(
                {
                    RLmodelWrapper: self.__init_from_ModelWrapper,
                    RLmodel: self.__init_from_modelDef_params,
                },
                args, kwargs
            )
    
    @property
    def choice_probs(self) -> jax.Array:
        return self._choice_probs
    
    def update(self, choice, reward) -> None:        
        self.latents = self._update(choice, reward, self.latents)
        self._choice_probs = self._cal_choice_probs(self.latents)
        return super().update(choice, reward) # update random seed

    def new_session(self) -> None:

        _init_latents = self._RLmodel.apply(
            self.params, 
            n_sess = self.n_sess,
            method='init_values')
        self.latents = _init_latents

        # set the very first self._choice_probs
        n_actions = self._RLmodel.N_actions
        if self._init_choice_probs is None:
            # when no init choice prob, choice prob is uniform
            arr = np.ones((self.n_sess, n_actions))
            self._choice_probs = arr / np.sum(arr, axis=1).reshape(-1,1)

        elif self._init_choice_probs.shape == (n_actions, ):
            # when specified for each action, duplicate it for each session
            arr = np.tile(self._init_choice_probs, (self.n_sess, 1))
            self._choice_probs = arr / np.sum(arr, axis=1).reshape(-1,1)

        elif self._init_choice_probs.shape == (self.n_sess, n_actions):
            # when specified for each session and each action, use as is
            arr = self._init_choice_probs
            self._choice_probs = arr / np.sum(arr, axis=1).reshape(-1,1)

        else:
            raise ValueError(f"init_ch_probs has shape {self._init_choice_probs.shape}, which does not match what is required (None or {(n_actions,)} or {(self.n_sess, n_actions)})")


        
