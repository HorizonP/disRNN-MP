from typing import Callable, List, Tuple
from disRNN_MP.agent.abstract import RL_agent, RL_multiAgent
from disRNN_MP.typing import Array, Params, patchable_hkRNNCore
from haiku._src.recurrent import RNNCore

import jax
import jax.numpy as jnp

import numpy as np
from .agents import hkNetwork_agent, hkNetwork_multiAgent
import haiku as hk

STEP_FUN_EXO_STATE = Callable[[Array, Array, Array], Tuple[jax.Array | dict[str, jax.Array], jax.Array]]
""" RNN step function that takes exogeneous state (for replacing endogenous state)

(xs, state, exo_state) -> y_hat, new_state
"""


class patched_hkNetwork_agent(hkNetwork_agent):
    """activation patched RL agent of haiku network 

    Args:
        exo_states (array): Exogeneous states to patch the internal state, of shape (N_trials, N_mod_states). Axis 0 is time, axis 1 represent different latent variables to be patched. Each trial the agent played with the environment will use one row of the exogeneous states to patch the internal latent variable's activation
        patch_state_ids (List[int]): index of internal latent variables to be patched with exogeneous states
    """
    def __init__(self, 
            make_network: Callable[[], patchable_hkRNNCore], 
            params: hk.Params|Params, 
            exo_states,
            n_actions: int = 2, 
            state_to_numpy: bool = False, 
            random_seed: int = 0,
        ):
        """_summary_

        Args:
            make_network (Callable[[], hk.Module  |  STEP_FUN_EXO_STATE]): _description_
            params (hk.Params | Params): _description_
            exo_states (_type_): _description_
            n_actions (int, optional): _description_. Defaults to 2.
            state_to_numpy (bool, optional): _description_. Defaults to False.
            random_seed (int, optional): _description_. Defaults to 0.

        Raises:
            TypeError: _description_
            TypeError: _description_

        Returns:
            _type_: _description_
        """
        
        self._state_to_numpy = state_to_numpy
        self._n_actions = n_actions
        self._params = params
        self.exo_states = exo_states

        def _step_network(xs: np.ndarray, state: hk.State, exo_state: hk.State):
            """Apply one step of the network.
            
            Args:
            xs: array containing network inputs
            state: previous state of the hidden units of the RNN model.
            
            Returns:
            y_hat: output of RNN
            new_state: state of the hidden units of the RNN model
            """
            core = make_network()
            y_hat, new_state = core.step_with_exo_state(xs, state, exo_state)  # type: ignore
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
            out_shape = jax.eval_shape(self._step, self._params, self._PRNGkey, self._xs, self.latents, self.exo_states[[0],:])
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

        super(RL_agent).__init__()

        self.triID = 0
        

    def new_session(self):
        self.triID = 0
        return super().new_session()


    def update(self, choice: int, reward: int):
        """update latent variables of the agent (including patch the specified latents)

        Args:
            choice (int): _description_
            reward (int): _description_
        """
        super().update(choice, reward)
        self.triID += 1

    @property
    def choice_probs(self) -> Array:
        """Predict the choice probabilities as a softmax over output logits."""
        tid = self.triID % self.exo_states.shape[0]

        output, self._next_latent = self._step(self._params, self._PRNGkey, self._xs, self.latents, self.exo_states[tid, :])

        output_logits = self._logits_indexing(output)
        choice_probs = jax.nn.softmax(output_logits)
        return choice_probs

    
class patched_hkNetwork_multiAgent(hkNetwork_multiAgent):
    """activation patched RL agent of haiku network 

    possible sequence to setup the instance with modified attributes:

    # these 3 statements can be in any order
    self.infuse_seed(random_seed) # optional, only if change random seed
    self.n_sess = n_sess # optional, only if change number of session
    self.exo_states = exo_states # optional, only if change exogeneous states for patching

    # this is required and has to be after any change of attributes
    self.new_session()



    Args:
        exo_states (array): Exogeneous states to patch the internal state, of shape (N_trials, N_sessions, N_mod_states). Axis 0 is time (trial), axis 1 is session, axis 2 represent different latent variables to be patched. Each trial the agent played with the environment will use one row of the exogeneous states to patch the internal latent variable's activation
        patch_state_ids (List[int]): index of internal latent variables to be patched with exogeneous states
    """
    def __init__(self, 
            make_network: Callable[[], patchable_hkRNNCore], 
            params: hk.Params|Params, 
            patch_state_ids: List[int],
            exo_states: None | Array = None,
            n_sess: int = 1,
            n_actions: int = 2, 
            random_seed: int = 0,
            additional_inputs: None | list | tuple | Array = None,
            init_ch_probs: None | Array = None,
        ):
        """_summary_

        Args:
            make_network (Callable[[], hk.Module  |  STEP_FUN_EXO_STATE]): _description_
            params (hk.Params | Params): _description_
            exo_states (_type_): _description_
            n_actions (int, optional): _description_. Defaults to 2.
            state_to_numpy (bool, optional): _description_. Defaults to False.
            random_seed (int, optional): _description_. Defaults to 0.

        Raises:
            TypeError: _description_
            TypeError: _description_

        Returns:
            _type_: _description_
        """

        # running RL_multiAgent's __init__
        super(hkNetwork_multiAgent, self).__init__(random_seed)

        def patched_module_haiku():
            disrnn = make_network()
            disrnn.set_patch_state_ids(patch_state_ids)
            return disrnn
        
        self.make_network = patched_module_haiku
        self.patch_state_ids = patch_state_ids

        self._n_actions = n_actions
        self._params = params
        self.n_sess = n_sess

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

        if exo_states is None:
            self.exo_states = jnp.zeros((1,n_sess,len(patch_state_ids)))
        else:
            self.exo_states = exo_states

        

        def _step_network(xs: np.ndarray, state: hk.State, exo_state: hk.State):
            """Apply one step of the network.
            
            Args:
            xs: array containing network inputs
            state: previous state of the hidden units of the RNN model.
            
            Returns:
            y_hat: output of RNN
            new_state: state of the hidden units of the RNN model
            """
            core = patched_module_haiku()
            y_hat, new_state = core.step_with_exo_state(xs, state, exo_state)  # type: ignore
            return y_hat, new_state

        def _get_initial_state(n_sess) -> hk.State:
            """Get the initial state of the hidden units of RNN model."""
            core = patched_module_haiku()
            state = core.initial_state(n_sess)
            return state

        _, self._step = hk.transform(_step_network)
        _, self._get_init = hk.transform(_get_initial_state)

        self._step = jax.jit(self._step)
        self._get_init = jax.jit(self._get_init, static_argnums=2)

        # running hkNetwork_multiAgent's infuse_seed
        self.infuse_seed(random_seed)
        self.new_session()

        # here it tries to figure out the data type for output of the network
        try:
            dummy_xs = self._build_xs(np.ones((self.n_sess, )), np.ones((self.n_sess, )))
            out_shape = jax.eval_shape(self._step, self._params, self._PRNGkey, dummy_xs, self.latents, self.exo_states[0,:,:])
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
        

    def new_session(self):
        self.triID = 0
        assert self.exo_states.shape[1:] == (self.n_sess, len(self.patch_state_ids)), f"shape of exo_states (N_tri, N_sess, N_mod_latents) = {self.exo_states.shape} doesn't match with the instance attributes (n_sess, len(self.patch_state_ids))  {(self.n_sess, len(self.patch_state_ids))}"

        return super().new_session()


    def update(self, choice: Array, reward: Array):
        """update latent variables of the agent (including patch the specified latents)

        Args:
            choice (Array): _description_
            reward (Array): _description_
        """
        # update agent's observation
        self._xs = self._build_xs(choice, reward)

        tid = self.triID % self.exo_states.shape[0]

        # choice, reward + current latent (last trial updated latent) -> next choice_prob + updated_latent
        output, self.latents = self._step(self._params, self._PRNGkey, self._xs, self.latents, self.exo_states[tid, :, :])

        # next choice_prob
        output_logits = self._logits_indexing(output)
        self._choice_probs = jax.nn.softmax(output_logits, axis=-1)
        
        # update random seed for model's step function
        self._PRNGkey, _ = jax.random.split(self._PRNGkey, 2)

        # update random seed for sampling choice
        super(hkNetwork_multiAgent, self).update(choice, reward)
        self.triID += 1

    @property
    def choice_probs(self) -> Array:
        """Predict the choice probabilities as a softmax over output logits."""

        return self._choice_probs
