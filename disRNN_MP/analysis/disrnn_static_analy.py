from ast import mod
from copy import deepcopy
from operator import getitem

from chex import PRNGKey
from disRNN_MP.analysis.model_graph import updateM2graph
from disRNN_MP.rnn.pkl_instantiate import _pkl_instantiate
from disRNN_MP.rnn.disrnn import _sigma_squashed
from disRNN_MP.rnn.train_db import ModelTrainee
from disRNN_MP.rnn.utils import get_haiku_static_attrs, transform_hkRNN
from disRNN_MP.typing import Inputs, Outputs, States, Params, Array
from disRNN_MP.utils import update_r


from typing import Any, Callable, Hashable, Iterable, Literal, Sequence, Sized, Tuple
from functools import partial, reduce

import numpy as np
import jax
import jax.numpy as jnp
import haiku as hk

@partial(jax.jit, static_argnames = ['path'])
def _index_nested_dict(params, path: Tuple[str, ...]):
    return reduce(getitem, path, params)

@partial(jax.jit, static_argnames=['prefix', 'par_name'])
def get_sigma_abs(pars, epsilon, prefix, par_name):
    return epsilon + jnp.abs(pars[prefix][par_name])

@partial(jax.jit, static_argnames=['prefix', 'par_name'])
def get_sigma_sigmoid(pars, prefix, par_name):
    return _sigma_squashed(pars[prefix][par_name])

@partial(jax.jit, static_argnames=['dict_path'])
def modify_params(pars: Params, dict_path: Tuple[Hashable], arr_mask: Array, values: Array):
    alt_pars = jax.device_put(deepcopy(pars))
    # arr_mask = jnp.array(arr_mask)

    # make a copy of the array at `dict_path` with arr_mask positions modified to values
    # modified_arr = reduce(getitem, dict_path, alt_pars).at[arr_mask].set(values)
    ori_arr = _index_nested_dict(alt_pars, dict_path)
    modified_arr = jnp.where(arr_mask, jnp.broadcast_to(values, ori_arr.shape), ori_arr)

    return update_r(alt_pars, dict_path, modified_arr)

class haiku_module_properties:
    """expose static haiku module properties that does not necessarily need transform"""
    def __init__(self, make_network: Callable) -> None:
        self.make_network = make_network
        _static = get_haiku_static_attrs(make_network, ["__class__", "__dict__"])
        self._static_attrs = _static["__dict__"]
        self._network_class = _static["__class__"]

class hkRNN_params_analyzer(haiku_module_properties):
    """common attributes and method for analyzing haiku RNN models"""

    @classmethod
    def from_dry_model_def(cls, dry_model):
        return cls(_pkl_instantiate(dry_model))

    @classmethod
    def from_modelTrainee(cls, mt: 'ModelTrainee'):
        mt._instantiates()
        return cls(mt.eval_model.model_haiku)

    def __init__(self, make_network, forward_func: None | Callable = None, step_func: None | Callable = None) -> None:
        # add attributes: make_network, _static_attrs, _network_class
        haiku_module_properties.__init__(self, make_network)

        # the common prefix for keys of the parameter dictionary
        self._prefix = self._static_attrs['module_name']

        self._forward_func = forward_func
        self._step_func = step_func

    @property
    def forward_func(self) -> Callable[[Params, PRNGKey, Inputs], Tuple[Outputs, States]]:
        """running the model with given parameter and random key on an entire dataset (multiple trials)"""
        
        if self._forward_func is None:
            tfd = transform_hkRNN(self.make_network)
            self._forward_func = jax.jit(tfd.apply)
       
        return self._forward_func
            

    @property
    def step_func(self) -> Callable[[Params, PRNGKey, Inputs, States], Tuple[Outputs, States]]:
        """function to evolve to next state for current model (single trial)
            
        (params, random_key, xs: (batch_sz, obs_sz), state: (batch_sz, latent_sz)) -> output: (batch_sz, out_sz), latents: (batch_sz, latent_sz)
        
        if one of the input has batch_sz as 1, while the other > 1, they will be broadcasted to larger one
        
        both input arguments can also be 1D array if there's only one batch
        
        """

        if self._step_func is None:
            def step_sub(xs, state):
                if len(xs.shape) == 1:
                    xs = jnp.reshape(xs, (1, -1))
                
                if len(state.shape) == 1:
                    state = jnp.reshape(state, (1, -1))
                
                batch_sz = jnp.broadcast_shapes((xs.shape[0],), (state.shape[0],))[0]

                # broadcast first dim size to batch_sz, leave the other dims untouched
                xs = jnp.broadcast_to(xs, (batch_sz, *xs.shape[1:])) 
                state = jnp.broadcast_to(state, (batch_sz, *state.shape[1:]))

                core = self.make_network()
                y_hat, new_state = core(xs, state)
                return y_hat, new_state

            model = hk.transform(step_sub)

            self._step_func = jax.jit(model.apply)
        
        return self._step_func

# TODO let `disRNN_params_analyzer` inherit from `hkRNN_params_analyzer`
class disRNN_params_analyzer(haiku_module_properties):
    """represent a network definition, properties and functions that is suitable for analyzing parameter sets

    all the parameters should work for the same type and shape of dataset

    only provide functions that can be used for analyzing a list of parameters

    """

    @classmethod
    def from_dry_model_def(cls, dry_model):
        return cls(_pkl_instantiate(dry_model))

    @classmethod
    def from_modelTrainee(cls, mt: 'ModelTrainee'):
        mt._instantiates()
        return cls(mt.eval_model.model_haiku)

    def __init__(self, make_network, forward_func: None | Callable = None, step_func: None | Callable = None) -> None:
        # add attributes: make_network, _static_attrs, _network_class
        haiku_module_properties.__init__(self, make_network)

        # the common prefix for keys of the parameter dictionary
        self._prefix = self._static_attrs['module_name']

        self.n_latents: int = self._static_attrs['_latent_size']
        """number of latent obtained from static attributes of model definition"""

        self.target_size: int = self._static_attrs['_target_size']
        """target size obtained from static attributes of model definition"""

        self.update_mlp_shape: Iterable[int] = self._static_attrs['_update_mlp_shape']
        self.choice_mlp_shape: Iterable[int] = self._static_attrs['_choice_mlp_shape']

        self.activation_func: Callable = self._static_attrs['_activation']

        self._forward_func = forward_func
        self._step_func = step_func

        # some dirty work to recognize which version of disRNN is and what configuration it has
        if '_epsilon' in self._static_attrs:
            self.has_separate_choice_bottleneck = True
            if '_sigma_parameterization' in self._static_attrs and self._static_attrs['_sigma_parameterization'] == 'sigmoid':
                self.get_latent_sigma = lambda par: get_sigma_sigmoid(par, self._prefix, 'latent_sigmas_unsquashed')
                self.get_update_sigma = lambda par: get_sigma_sigmoid(par, self._prefix, 'update_mlp_sigmas_unsquashed')
                self.get_choice_sigma = lambda par: get_sigma_sigmoid(par, self._prefix, 'choice_mlp_sigmas_unsquashed')
            else:
                # this suggests the sigmas are calculated as sigma = epsilon + abs(sigma_param)
                self.get_latent_sigma = lambda par: get_sigma_abs(par, self._static_attrs['_epsilon'], self._prefix, 'latent_sigmas_param')
                self.get_update_sigma = lambda par: get_sigma_abs(par, self._static_attrs['_epsilon'], self._prefix, 'update_mlp_sigmas_param')
                self.get_choice_sigma = lambda par: get_sigma_abs(par, self._static_attrs['_epsilon'], self._prefix, 'choice_mlp_sigmas_param')
        else:
            self.has_separate_choice_bottleneck = False

            self.get_latent_sigma = lambda par: get_sigma_sigmoid(par, self._prefix, 'latent_sigmas_unsquashed')
            self.get_update_sigma = lambda par: get_sigma_sigmoid(par, self._prefix, 'update_mlp_sigmas_unsquashed')
            # for older version of disRNN models, the choice sigma is equivalent to latent sigma
            self.get_choice_sigma = self.get_latent_sigma
        
    def get_latent_inits(self, params:Params):
        return _index_nested_dict(params, (self._prefix, 'latent_inits'))
    
    def get_update_gate(self, params:Params):
        return _index_nested_dict(params, (self._prefix, 'update_mlp_gates'))
    
    def get_choice_gate(self, params:Params):
        return _index_nested_dict(params, (self._prefix, 'choice_mlp_gates'))

    def latent_update_graph(self, params:Params, thres = 0.5, **kwargs):
        """ get a graph representing the update dependency

        Args:
            thres: threshold for considering as open bottleneck (larger than it means open)
        """

        updm = np.transpose(self.get_update_sigma(params))

        gs = updateM2graph(updm, thres, **kwargs)

        return dict(zip(
            ['full_graph', 'compact_graph', 'upd_openness'],
            (*gs, np.array(1-updm))
        ))

    def __repr__(self) -> str:
        _repr = (super().__repr__() + "\n"
            + f"model prefix: {self._prefix}\n"
            + f"latent capacity: {self.n_latents}\n"
            + f"number of choices (target size): {self.target_size}\n"
            + f"update_mlp_shape: {self.update_mlp_shape}\n"
            + f"choice_mlp_shape: {self.choice_mlp_shape}\n"
        )
        return _repr

    @property
    def forward_func(self) -> Callable[[Params, PRNGKey, Inputs], Tuple[Outputs, States]]:
        """running the model with given parameter and random key on an entire dataset (multiple trials)"""
        
        if self._forward_func is None:
            tfd = transform_hkRNN(self.make_network)
            self._forward_func = jax.jit(tfd.apply)
       
        return self._forward_func
            

    @property
    def step_func(self) -> Callable[[Params, PRNGKey, Inputs, States], Tuple[Outputs, States]]:
        """function to evolve to next state for current model (single trial)
            
        (params, random_key, xs: (batch_sz, obs_sz), state: (batch_sz, latent_sz)) -> output: (batch_sz, out_sz), latents: (batch_sz, latent_sz)
        
        if one of the input has batch_sz as 1, while the other > 1, they will be broadcasted to larger one
        
        both input arguments can also be 1D array if there's only one batch
        
        """

        if self._step_func is None:
            def step_sub(xs, state):
                if len(xs.shape) == 1:
                    xs = jnp.reshape(xs, (1, -1))
                
                if len(state.shape) == 1:
                    state = jnp.reshape(state, (1, -1))
                
                batch_sz = jnp.broadcast_shapes((xs.shape[0],), (state.shape[0],))[0]

                # broadcast first dim size to batch_sz, leave the other dims untouched
                xs = jnp.broadcast_to(xs, (batch_sz, *xs.shape[1:])) 
                state = jnp.broadcast_to(state, (batch_sz, *state.shape[1:]))

                core = self.make_network()
                y_hat, new_state = core(xs, state)
                return y_hat, new_state

            model = hk.transform(step_sub)

            self._step_func = jax.jit(model.apply)
        
        return self._step_func


    # === functions related to model surgery: making modified parameters
    
    def modify_updateMLP_gates(self, params: Params, mask: Any, verb: Literal['mute', 'preserve'] = 'mute'):

        match verb.lower():
            case 'mute':
                bool_mask = jnp.zeros_like(self.get_update_gate(params), dtype = bool).at[mask].set(True)
            case 'preserve':
                bool_mask = jnp.ones_like(self.get_update_gate(params), dtype = bool).at[mask].set(False)
            case _:
                raise ValueError(f"unknown verb '{verb}'. available verbs are: 'mute', 'preserve'")
        
        return modify_params(params, (self._prefix, 'update_mlp_gates',), bool_mask, 0)

    def modify_choiceMLP_gates(self, params: Params, mask: Any, verb: Literal['mute', 'preserve'] = 'mute'):

        match verb.lower():
            case 'mute':
                bool_mask = jnp.zeros_like(self.get_choice_gate(params), dtype = bool).at[mask].set(True)
            case 'preserve':
                bool_mask = jnp.ones_like(self.get_choice_gate(params), dtype = bool).at[mask].set(False)
            case _:
                raise ValueError(f"unknown verb '{verb}'. available verbs are: 'mute', 'preserve'")
        
        return modify_params(params, (self._prefix, 'choice_mlp_gates',), bool_mask, 0)