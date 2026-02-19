""" this module provides serveral interrelated classes for analyzing disRNN models

The assumption of this module is that all disRNN models are defined with the `haiku` library. 
So, the most fundamental class is `haiku_module_properties` which exposes all static properties of a haiku module. By static it means these properties does not depends on specific parameters nor random seed. They are usually defined during initialization of the module. (such as network size, type of activation function, and etc.). This class is designed to be able to represent any other haiku module networks.

The `disRNN_params_analyzer` class inherits from `haiku_module_properties` and is supposed to include properties and functions that is unique to disRNN models. It pertains to a disRNN model with a specific set of hyperparameters. It does not have any properties related to a specific set of trainable parameters. It has code to accommodate different version of disRNN model and expose a unifying public interface for all these mdoels

"""
from typing import Any, Callable, Dict, Iterable, List, Literal, Tuple
import re
from copy import deepcopy
from functools import partial, cached_property

from chex import PRNGKey
from disRNN_MP.agent.agents import hkNetwork_multiAgent
from disRNN_MP.analysis.disrnn_static_analy import disRNN_params_analyzer
from disRNN_MP.dataset import trainingDataset
from disRNN_MP.rnn.train_db import trainingRecord
from disRNN_MP.typing import Params, Array, States, Outputs
from matplotlib.axes import Axes
import numpy as np
from numpy import ndarray
import pandas as pd
import jax
import jax.numpy as jnp
import haiku as hk
import xarray as xr
import igraph as ig
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns

from .tidy_db import get_mt_model_by_step
from disRNN_MP.rnn.utils import has_haiku_static_attrs, patched_forward
from disRNN_MP.rnn.disrnn_analy import disRNN_choiceMLP_func, disRNN_update_par_func, disRNN_update_pars_func
from disRNN_MP.agent.patched_agent import patched_hkNetwork_multiAgent
from disRNN_MP.utils import ds2df, min_max
from disRNN_MP.rnn.pkl_instantiate import _pkl_instantiate
from ._model_graph_plots import plt_graph_with_igraph, plt_graph_with_networkx


def assemble_eval_dataset(
        dataset: trainingDataset | xr.Dataset, 
        output_latents_tuple: Tuple, 
        output_label: None | List[str] = None,
        ) -> xr.Dataset:
    
    """organize a forward call output with input data into an xarray dataset

    Content of the eval dataset:

    - Variables
        - inputs: the inputs contained in the trainingDataset
        - outputs: the outputs contained in the trainingDataset
        - latents: Updated latents for each trial
        - choice_logits: Prediction of choice on next trial as model output
    - Coordinates
        - tri: trial
        - sess: session
        - in_feat: input features
        - out_feat: output features
        - latent: latent variable
        - choice: available choices
        
    :param dataset: the trainingDataset used as argument for the forward call
    :type dataset: trainingDataset
    :param output_latents_tuple: returns from the forward call
    :type output_latents_tuple: Tuple
    :param output_label: Optional labels for each of the possible action/choice
    :type output_label: 
    :return: xarray Dataset
    :rtype: Dataset
    """
    
    if isinstance(dataset, trainingDataset):
        ds = dataset.to_xr_dataset()
    else:
        ds = dataset.copy()

    output, latents = output_latents_tuple 

    data_vars = {
        'latents': (["tri", "sess", "latent"], latents, {'description': "Updated latents for each trial"}),
        'choice_logits': (["tri", "sess", "choice"], output['prediction'], {'description': "Prediction of choice on next trial as model output"}),
    }        

    if output_label is None:
        output_label = [f"choice {i}" for i in range(output['prediction'].shape[2])]
    
    coords = {
        'latent': np.arange(latents.shape[2]),
        'choice': output_label,
    }

    return ds.assign(variables=data_vars).assign_coords(**coords)



@jax.jit
def cal_MLP_N_params(input_sz: int, layer_szs: List[int]):
    # szs = jnp.array([input_sz] + layer_szs)
    szs = jnp.concatenate((jnp.array([input_sz]), jnp.array(layer_szs)), axis=0)

    N_ws = jnp.sum(szs[:-1] * szs[1:])
    N_bs = jnp.sum(szs[1:])

    return N_ws + N_bs

@jax.jit
def cal_N_effective_params(lat_sigma, upd_sigma, ch_sigma, bn_sigma_thre, static_attrs):
    """calculate effective number of parameters given `bn_sigma_thre`

    consider open latent's inits, choice MLP with open choice latent as input, open latent's update MLP with open update bottleneck inputs

    Returns:
        int: effective number of parameters
    """
    open_lat = lat_sigma < bn_sigma_thre
    open_upd = upd_sigma < bn_sigma_thre
    opne_ch = (ch_sigma < bn_sigma_thre) & open_lat

    N_open_lat = jnp.sum(open_lat)
    N_open_ch = jnp.sum(opne_ch)
    

    # open_upd_inputs = jnp.sum(open_upd[:, open_lat], axis=0)
    open_upd_inputs = jnp.sum(open_upd, axis=0) # N_open_update_bn for each latent
    
    updMLP_shape = jnp.concatenate((jnp.array(static_attrs['update_mlp_shape']), jnp.array([2])), axis=0)
    eff_updMLP_N_pars = jnp.where(open_lat, jax.vmap(cal_MLP_N_params, [0, None])(open_upd_inputs, updMLP_shape), 0)
    # 

    chsh = jnp.concatenate((jnp.array(static_attrs['choice_mlp_shape']), jnp.array([static_attrs['target_size']])), axis=0)
    N_chMLP_pars = cal_MLP_N_params(N_open_ch, chsh)
    # jax.debug.print("N_open_ch={N_open_ch}, chMLP_shape={chsh}, N_chMLP_pars={N_chMLP_pars}", N_open_ch = N_open_ch, chsh=chsh, N_chMLP_pars=N_chMLP_pars)

    return (
        N_open_lat + # latent_inits
        N_chMLP_pars + # effective choice MLP
        jnp.sum(eff_updMLP_N_pars) # sum of effective update MLPs
    )

@partial(jax.jit, static_argnames = ['prefix', 'sigma_parameterization'])
def _trimmed_params(params, lat_sigma, upd_sigma, ch_sigma, bn_sigma_thre, prefix, sigma_parameterization):
    """create modified parameters which all closed bottleneck are strictly disabled 
    By setting associated gate to 0, and sigma to 1

    it is recommended to use eval mode of disRNN when evaluate this trimmed parameter (where all sigma are effectively set to 0 as well)

    criteria for open IB:
    - update: less than bn_sigma_thre
    - latent: less than bn_sigma_thre and at least 1 open update IB
    - choice: less than bn_sigma_thre and open latent IB

    Returns:
        Params: altered parameters
    """
    # 

    open_upd = upd_sigma < bn_sigma_thre
    open_lat = (lat_sigma < bn_sigma_thre) & jnp.any(upd_sigma < bn_sigma_thre, axis=0)
    open_ch = (ch_sigma < bn_sigma_thre) & open_lat

    # since this condition depends on the shape of params, and the function will be recompiled when params shape is different, so it is fine to use the if-else
    # this works because the `trimmed` property creates a patched module where `_latent_bn_gate` is set to True
    if 'latent_gates' in params[prefix]:
        params[prefix]['latent_gates'] = jnp.where(open_lat, params[prefix]['latent_gates'], 0,)
    else:
        params[prefix]['latent_gates'] = jnp.where(open_lat, 1, 0)
    
    params[prefix]['choice_mlp_gates'] = jnp.where(open_ch, params[prefix]['choice_mlp_gates'], 0)
    
    params[prefix]['update_mlp_gates'] = jnp.where(open_upd, params[prefix]['update_mlp_gates'], 0)

    if sigma_parameterization == 'abs':
        params[prefix]['latent_sigmas_param'] = jnp.where(open_lat, params[prefix]['latent_sigmas_param'], 1)
        params[prefix]['update_mlp_sigmas_param'] = jnp.where(open_upd, params[prefix]['update_mlp_sigmas_param'], 1)
        params[prefix]['choice_mlp_sigmas_param'] = jnp.where(open_ch, params[prefix]['choice_mlp_sigmas_param'], 1)
    elif sigma_parameterization == 'sigmoid':
        params[prefix]['latent_sigmas_param'] = jnp.where(open_lat, params[prefix]['latent_sigmas_param'], 0)
        params[prefix]['update_mlp_sigmas_param'] = jnp.where(open_upd, params[prefix]['update_mlp_sigmas_param'], 0)
        params[prefix]['choice_mlp_sigmas_param'] = jnp.where(open_ch, params[prefix]['choice_mlp_sigmas_param'], 0)
    else:
        raise ValueError(f"unsupported sigma_parameterization: {sigma_parameterization}\nsupported: ('abs', 'sigmoid')")
    
    return params

@jax.jit
def add_sub_func_name_to_disrnn_params(par: dict[str, Any]):
    """rename the parameters for disRNN model to be compatible with disRNN model defined with `latent_update` and `choice_selection` methods

    add `~latent_update` to updateMLP related parameter, `~choice_selection` to choiceMLP related parameter

    Args:
        par (_type_): _description_

    Returns:
        _type_: _description_
    """
    if np.all(~pd.Series(par.keys()).str.contains("/~latent_update/")):
        def _update_par_names(name):
            name = re.sub(r"/(?=latent\d+_update)", "/~latent_update/", name)
            name = re.sub(r"/(?=choice)", "/~choice_selection/", name)
            return name
        
        return {_update_par_names(k):v for k, v in par.items()}
    else:
        return par

class NPProxy:
    """
    A proxy class that converts JAX arrays into NumPy arrays on attribute or method access.

    This class is intended to be attached as an attribute to a parent class (e.g., an
    `Analyzer`) that stores or produces JAX arrays. When accessing a JAX array through this
    proxy, `jax.device_get` is automatically called to return a NumPy array on the host.
    
    - If an attribute is a JAX array, it is converted to a NumPy array via `jax.device_get`.
    - If an attribute is a method returning a JAX array, the result is also converted.
    
    Attributes:
        __parent__: The instance of the parent class whose attributes/methods will
            be proxied and converted to NumPy arrays.

    Example:
        >>> class Analyzer:
        ...     def __init__(self, x):
        ...         self.x = jnp.array([x, x+1, x+2])
        ...         self.y = jnp.array([10, 20, 30])
        ...         # Method that returns a JAX array
        ...         self.compute_sum = lambda a, b: jnp.array([a + b])
        ...         # Attach the NPProxy
        ...         self.np = NPProxy(self)
        ...
        >>> a = Analyzer(0)
        >>> type(a.x)                # doctest: +SKIP
        <class 'jaxlib.xla_extension.ArrayImpl'>
        >>> a.np.x                   # doctest: +SKIP
        array([0, 1, 2])  # NumPy array
        >>> a.np.compute_sum(2, 3)   # doctest: +SKIP
        array([5])       # NumPy array
    """

    def __init__(self, parent) -> None:
        self.__parent__ = parent
    
    def __getattr__(self, name: str) -> Any:
        attr = getattr(self.__parent__, name)
        
        if callable(attr):
            # If it's a method, wrap it
            def wrapper(*args, **kwargs):
                result = attr(*args, **kwargs)
                # Convert the result to NumPy if it's a JAX array
                return jax.device_get(result)
            return wrapper
        
        # Otherwise, assume it's a JAX array attribute
        return jax.device_get(attr)


class disRNN_model_analyzer(disRNN_params_analyzer):
    """represent a pair of network definition and parameters

    the smallest unit of a functioning disRNN model (that can run against a dataset)

    sigma close to 1 means the bottleneck is closed, close to 0 means open

    Raises:
        ValueError: _description_

    Returns:
        _type_: _description_
    """

    np: NPProxy
    """
    A proxy attribute that converts JAX arrays into NumPy arrays on demand.

    When you access or call attributes through this proxy (e.g. `self.np.x` instead of
    `self.x`), the underlying JAX array or method return value is automatically transferred
    to the host as a NumPy array via `jax.device_get`. This is particularly convenient for
    quick inspections or when you need a NumPy array for further processing outside JAX.

    Examples:
        >>> analyzer = disRNN_model_analyzer(...)
        >>> type(analyzer.x)
        <class 'jaxlib.xla_extension.ArrayImpl'>    # JAX array
        >>> x_host = analyzer.np.x
        >>> type(x_host)
        <class 'numpy.ndarray'>                     # NumPy array

        >>> y_host = analyzer.np.some_method(1, 2)  # If some_method returns a JAX array
        >>> type(y_host)
        <class 'numpy.ndarray'>                     # NumPy array
    """

    input_feature_name: List[str]
    """the name of each feature of model input. 
    If not provided, default to dataset.input_feature_name
    """

    @classmethod
    def from_db(cls, db_sess, mt_id, step, alt_cwd = None, fix_name: bool = False, dataset: Literal['train', 'test'] | trainingDataset = 'train', **kwargs):
        """create disRNN_model_analyzer from a record in a database
        this will use the `eval_model` of the ModelTrainee, since it makes sense to use deterministic model for analysis

        Args:
            db_sess: Sqlalchemy db session
            mt_id: modelTrainee id
            step: training step for indexing record
            alt_cwd: if not None, will use as alternative working directory when materializing the ModelTrainee
            fix_name: whether call `add_sub_func_name_to_disrnn_params` on the parameters to fix the parameter names

        """
        mt, rec = get_mt_model_by_step(db_sess, mt_id, step, alt_cwd = alt_cwd)

        if isinstance(dataset, trainingDataset):
            ds = dataset
        elif dataset == 'train':
            ds = mt.sessions[0].train_dataset
        elif dataset == 'test':
            ds = mt.sessions[0].test_dataset
        else:
            raise ValueError("unknown dataset argument")
        
        inst = cls.from_trainingRecord(rec, fix_name = fix_name, dataset = ds, **kwargs)
        return (inst, mt, rec)
        
    
    @classmethod
    def from_trainingRecord(cls, trec: trainingRecord, fix_name: bool = False, **kwargs):
        """init disRNN_model_analyzer from a trainingRecording instance

        this is a much light weight init function comparing to `from_db`

        Args:
            trec (trainingRecord): _description_
            fix_name (bool, optional): whether to change parameter names to fit new disrnn4 model. Defaults to False.

        Returns:
            _type_: _description_
        """
        if fix_name:
            par = add_sub_func_name_to_disrnn_params(trec.parameter)
        else:
            par = trec.parameter
        
        trec.parent_training._instantiates()
        return cls(trec.parent_training.eval_model.model_haiku, par, **kwargs)
    
    @classmethod
    def from_dry_model(cls, dry_model, parameter, fix_name: bool = False):
        if fix_name:
            parameter = add_sub_func_name_to_disrnn_params(parameter)
        return cls(_pkl_instantiate(dry_model).model_haiku, parameter)
    
    @classmethod
    def init_fix_name(cls, make_network, params, *args, **kwargs):
        params = add_sub_func_name_to_disrnn_params(params)
        return cls(make_network, params, *args, **kwargs)

    def __init__(self, 
            make_network, 
            params, 
            dataset: None | trainingDataset = None, 
            bn_sigma_thre = 0.5, 
            fix_params_name: bool = False,
            output_label: None | List[str] = None,
            **kwargs) -> None:
        """create a disRNN_model_analyzer instance

        Args:
            make_network (_type_): a closure that makes a disRNN definition
            params (_type_): the parameter of the disRNN model
            dataset (None | trainingDataset, optional): allow optionally include a dataset for the instance. some method will be evaluated on the dataset. Defaults to None.
            bn_sigma_thre (float, optional): threshold for bottleneck sigma that to be considered as open. Defaults to 0.5.
            fix_params_name (bool, optional): whether to modify old parameter names to suit new disRNN definition. Defaults to False.
        """
        
        disRNN_params_analyzer.__init__(self, make_network, **kwargs)
        if fix_params_name:
            self.params = add_sub_func_name_to_disrnn_params(params)
        else:
            self.params = params
        
        self.dataset = dataset
        self.bn_sigma_thre = bn_sigma_thre
        self._trimmed = None
        self.np = NPProxy(self)

        if dataset is not None:
            self.input_feature_name = dataset.input_feature_name
        else:
            self.input_feature_name = [f"in_f{i}" for i in range(self.np.obs_dim)]
        
        self.output_label = output_label or (self.dataset.output_label if self.dataset else None) or [f'choice {i}' for i in range(self.target_size)]
        

    def set_bn_sigma_thre(self, bn_sigma_thre):
        """change the threshold for bottleneck considered as closed"""
        self.bn_sigma_thre = bn_sigma_thre
        return self
    
    def __repr__(self) -> str:
        repr = (super().__repr__()
            + f"threshold for open bottlenecks: less than {self.bn_sigma_thre}\n"
            + f"number of open latents: {len(self.open_latent())}\n" 
            + f"number of open choice latents: {len(self.open_choice_latent())}\n"
            + f"total number of open update bottlenecks: {self.N_open_update_bn}\n"
            + f"input features: {self.input_feature_name}\n"
            + f"available choices: {self.output_label}\n"
        )
        return repr

    @property
    def params(self):
        return self._params
    
    @params.setter
    def params(self, par):
        self._params = jax.device_put(par)
    
    # ====== any attribute that depends on the exact value of the parameters will be defined as dynamic properties
    @property
    def update_sigma(self):
        return self.get_update_sigma(self.params)
    
    @property
    def latent_sigma(self):
        return self.get_latent_sigma(self.params)
    
    @property
    def choice_sigma(self):
        return self.get_choice_sigma(self.params)
    
    @property
    def update_gate(self):
        return self.get_update_gate(self.params)
    
    @property
    def choice_gate(self):
        return self.get_choice_gate(self.params)
    
    @property
    def latent_inits(self):
        """initial latent values for a single session (1D array)"""
        return self.get_latent_inits(self.params)
    
    @property
    def obs_dim(self):
        """dimension of observation vector
        for example input (choice, reward) is of length 2
        """
        update_sigma_shape = self.update_sigma.shape
        return update_sigma_shape[0] - update_sigma_shape[1]
    
    @property
    def stepFun(self):
        """function to evolve to next state for current model
        with random key 0
        
        (xs: (batch_sz, obs_sz), state: (batch_sz, latent_sz)) -> output: (batch_sz, out_sz), latents: (batch_sz, latent_sz)
        if one of the input has batch_sz as 1, while the other > 1, 
        they will be broadcasted to larger one
        both input arguments can also be 1D array if there's only one batch
        
        """
        # return evo_state(self.make_network, self.params)
        return lambda xs, states: self.step_func(self.params, jax.random.PRNGKey(0), xs, states)
    
    @property
    def latent_sigma_order(self):
        """sort latents by associated latent sigma"""
        return jnp.argsort(self.latent_sigma)
    
    @property
    def update_sigma_order(self):
        """the order of update sigma axis-0 after incorporating latent sigma order"""
        return jnp.concatenate(
            (jnp.arange(0, self.obs_dim, 1), self.obs_dim + self.latent_sigma_order), axis=0
        )
     
    @property
    def sorted_update_sigma(self):
        """sorted update sigma matrix after incorporating latent sigma order"""
        update_sigmas = self.update_sigma
        update_sigmas = update_sigmas[:, self.latent_sigma_order]
        update_sigmas = update_sigmas[self.update_sigma_order, :]
        return update_sigmas
    
    @property
    def sorted_latent_sigma(self):
        return self.latent_sigma[self.latent_sigma_order]
    
    @property
    def sorted_choice_sigma(self):
        return self.choice_sigma[self.latent_sigma_order]
    
    @property
    def N_open_update_bn(self):
        return jnp.sum(self.open_update_bn())
    
    @property
    def N_max_lat2lat_update_bn(self):
        N_lat2lat_upd_bn = jnp.sum(self.open_update_bn()[self.obs_dim:, :], axis=0)
        return jnp.max(N_lat2lat_upd_bn)
    
    def LV_upd_open_obs(self, lat_ind):
        """the indices of observation feature with open update bottleneck to latent of index `lat_ind`
        
        Arguments:
            lat_ind: 0-based latent index
        
        Returns:
            0-based indices of observation features
        """
        return jnp.nonzero(self.open_update_bn()[:self.obs_dim, lat_ind])[0]
    
    def LV_upd_open_lat(self, lat_ind):
        """the indices of latent variable inputs with open update bottleneck to latent variable of index `lat_ind`
        
        Arguments:
            lat_ind: 0-based latent index
        
        Returns:
            0-based indices of latent variables
        """
        return jnp.nonzero(self.open_update_bn()[self.obs_dim:, lat_ind])[0]

    @property
    def N_effective_params(self):
        """calculate effective number of parameters given `bn_close_thre`

        consider open latent's inits, choice MLP with open choice latent as input, open latent's update MLP with open update bottleneck inputs

        Returns:
            int: effective number of parameters
        """
        return cal_N_effective_params(self.latent_sigma, self.update_sigma, self.choice_sigma, self.bn_sigma_thre, {k:self.__dict__[k] for k in ['update_mlp_shape', 'choice_mlp_shape', 'target_size']})
    
    def _old_N_effective_params(self):
        """calculate effective number of parameters given `bn_close_thre`

        consider open latent's inits, choice MLP with open choice latent as input, open latent's update MLP with open update bottleneck inputs

        Returns:
            int: effective number of parameters
        """
        open_lat = self.open_latent()
        N_open_lat = open_lat.shape[0]
        N_open_ch = self.open_choice_latent().shape[0]

        open_upd = self.open_update_bn()

        open_upd_inputs = jnp.sum(open_upd[:, open_lat], axis=0)
        eff_updMLP_N_pars = jax.vmap(cal_MLP_N_params, [0, None])(open_upd_inputs, self.update_mlp_shape + [2])

        return (
            N_open_lat + # latent_inits
            cal_MLP_N_params(N_open_ch, self.choice_mlp_shape + [self.target_size]) + # effective choice MLP
            jnp.sum(eff_updMLP_N_pars) # sum of effective update MLPs
        )
    
    @property
    def trimmed(self):
        """a trimmed model version of self
        all the closed bottleneck judged by `bn_sigma_thre` are set to fully closed

        Returns:
            self.__class__: a trimmed instance of same class as self
        """
        if self._trimmed is None:
            # create a modified make_network which enable latent bottleneck gates
            def patched_module():
                disrnn = self.make_network()
                disrnn._latent_bn_gate = True
                disrnn.apply_latent_bn = disrnn._configure_latent_bn()
                return disrnn
            
            # create modified parameters which all closed bottleneck are disabled via setting associated gate to 0
            alt_params = self._trimmed_params()
            
            self._trimmed = self.__class__(
                make_network = patched_module,
                params = alt_params,
                dataset = self.dataset,
                bn_sigma_thre = self.bn_sigma_thre,
                output_label = self.output_label) # type: ignore
            
        elif self._trimmed.bn_sigma_thre != self.bn_sigma_thre:
            # re-do trimming parameters since the bn_sigma_thre has changed
            self._trimmed.params = self._trimmed_params()
            self._trimmed.bn_sigma_thre = self.bn_sigma_thre
        
        return self._trimmed
    
    def _trimmed_params(self):

        # modify bottleneck gates to reflect the trimmed model
        alt_params = _trimmed_params(
            self.params, 
            self.latent_sigma, 
            self.update_sigma, 
            self.choice_sigma, 
            self.bn_sigma_thre, 
            self._prefix, 
            self._static_attrs['_sigma_parameterization']
        )
        return alt_params

    
    def _old_trim_bn_params(self):
        """create modified parameters which all closed bottleneck are strictly disabled 
        By setting associated gate to 0, sigma to 1

        Returns:
            Params: altered parameters
        """
        # 
        alt_params = deepcopy(self.params)

        close_lat_bn_mask = jnp.logical_not(self.open_latent(ret_mask=True))
        if 'latent_gates' in alt_params[self._prefix]:
            alt_params[self._prefix]['latent_gates'] = jax.device_put(alt_params[self._prefix]['latent_gates']).at[close_lat_bn_mask].set(0)
        else:
            alt_params[self._prefix]['latent_gates'] = jnp.ones((self.n_latents, )).at[close_lat_bn_mask].set(0)
        
        alt_params[self._prefix]['choice_mlp_gates'] = jax.device_put(
            alt_params[self._prefix]['choice_mlp_gates']).at[
                jnp.logical_not(self.open_choice_latent(ret_mask=True))].set(0)
        
        alt_params[self._prefix]['update_mlp_gates'] = jax.device_put(
            alt_params[self._prefix]['update_mlp_gates']).at[
                jnp.logical_not(self.open_update_bn())].set(0)
        
        return alt_params


    def build_choiceMLP_func(self, convert_pR = False, **kwargs) -> Callable:
        """reconstruct the choice MLP function from model definition and parameters

        this function has already included `choice_mlp_gate` multiplier

        if convert_pR is True, the result function will return p(R) output
        else, the result function will only return logits output
        """
        if has_haiku_static_attrs(self.make_network, 'choice_selection'):
            # then the following version will produce exactly same numbers as disRNN's __call__ method when it's double vmapped
            # when being used along, it produces same numbers as `disRNN_choiceMLP_func` version
            def choice_slct(latents):
                disrnn = self.make_network()
                y_hat, penalty = disrnn.choice_selection(latents)
                return y_hat
            _, tf_choice_slct = hk.transform(choice_slct)
            tf_choice_slct = jax.jit(tf_choice_slct)
            if convert_pR:
                return lambda lat: jax.nn.softmax(tf_choice_slct(self.params, jax.random.PRNGKey(0), lat))[1]
            else:
                return lambda lat: tf_choice_slct(self.params, jax.random.PRNGKey(0), lat)
        else:
            # this version could have error upto 1e-3, but is stable under vmap 
            return disRNN_choiceMLP_func(self.params, self.make_network, self.activation_func, convert_pR = convert_pR, **kwargs)
       
    def build_updateMLP_func(self, lat_id: None | int = None, restrict_input_lats_id: None | List[int] = None, other_input_lats_val: float = 0, include_upd_mlp_gate: bool = True) -> Callable:
        """obtain the updateMLP function for this disRNN model

        Note all latent indices are 1-based

        Args:
            lat_id (None | int): specify which latent's update MLP. 1-based index for latent
            restrict_input_lats_ind (None | List[int]): When this argument is set to a list of indices, this function will build a update MLP function that specifically takes a subset of latent inputs. The value of latents that are not specified here will be fixated as `other_input_lats_val`. The list of indices will be interpreted as 1-based indices for latent. When this argument is a empty list, this function will build a update MLP function that doesn't take latent inputs at all.
            other_input_lats_val (float): only meaningful when `restrict_input_lats_ind` is not None. the values for other latent inputs that are not specified in `restrict_input_lats_ind`

        Returns:
            - if `lat_id` is integer, return a update MLP function: (obs: vector, latents: vector) -> (lr: scalar, target: scalar)
            - if `lat_id` is None or not provided, return a update MLP function: (obs: vector, latents: vector) -> matrix of shape (lat_sz, 2), column 0 represents learning rates for each latent, column 1 represents targets  for each latent
            - if `restrict_input_lats_ind` is a list of integers, return a update MLP function: (obs: vector, lat_i: scalar, lat_j: scalar, ...) -> either (lr: scalar, target: scalar) or matrix of shape (lat_sz, 2)
            - if `restrict_input_lats_ind` is a empty list, return a update MLP function that all latents value are set to a constant value `other_input_lats_val`: (obs: vector) -> either (lr: scalar, target: scalar) or matrix of shape (lat_sz, 2)
        """        
        if lat_id is not None:
            f = disRNN_update_par_func(self.params, self.make_network, lat_id, include_upd_mlp_gate=include_upd_mlp_gate)
        else:
            f = disRNN_update_pars_func(self.params, self.make_network, include_upd_mlp_gate=include_upd_mlp_gate)
        
        if restrict_input_lats_id is not None:

            if len(restrict_input_lats_id) == 0:
                def updMLP_wrt_lats(obs, *args):
                    latents = jnp.full(self.n_latents, fill_value=other_input_lats_val)
                    return f(obs, latents)
            else:
                def updMLP_wrt_lats(obs, *args):
                    latents = jnp.full(self.n_latents, fill_value=other_input_lats_val)
                    latents = latents.at[jnp.array(restrict_input_lats_id) - 1].set(jnp.array([*args]))
                    return f(obs, latents)
            
            return updMLP_wrt_lats
        
        else:
            return f


    def open_update_bn(self, ):
        """boolean array indicating open update bottleneck
        """
        return self.update_sigma < self.bn_sigma_thre
    
    def open_latent(self, ret_mask: bool = False, consider_update_bn: bool = False):
        """get the (0-based) indices of latents with open latent bottleneck (and any open update bottlenecks if considering update bottleneck)"""

        if consider_update_bn:
            crit = jnp.any(self.update_sigma < self.bn_sigma_thre, axis=0) & (self.latent_sigma < self.bn_sigma_thre)
        else:
            crit = self.latent_sigma < self.bn_sigma_thre

        if ret_mask:
            return crit
        else:
            return jnp.arange(self.n_latents)[crit]
    
    def open_choice_latent(self, ret_mask: bool = False, consider_latent_bn: bool = False):
        """get the (0-based) indices of latents with open choice bottleneck (and also considered as open latent (see `open_latent` method) )"""

        if consider_latent_bn:
            crit = (self.choice_sigma < self.bn_sigma_thre) & self.open_latent(ret_mask=True, consider_update_bn=True)
        else:
            crit = self.choice_sigma < self.bn_sigma_thre

        if ret_mask:
            return crit
        else:
            return jnp.arange(self.n_latents)[crit]

    
    def forward(self, model_in: None|Array = None, key: PRNGKey|None = None):
        """run the model on inputs

        Args:
            model_in (_type_, optional): _description_. Defaults to None.
        
        Implementation:
            if `forward_func` is not set specifically, this utilize a forward function generated by `transform_hkRNN` which in turn calls the `haiku.dynamic_unroll` function

        Raises:
            ValueError: _description_

        Returns:
            Tuple: (model_outputs, updated latent states)
        """
        if model_in is None and self.dataset is None:
            raise ValueError("please provide inputs data for running the model")
        elif model_in is None and self.dataset is not None:
            model_in = self.dataset.xs
        else: # model_in is not None
            pass

        if key is None:
            key = jax.random.PRNGKey(jax.device_put(np.random.randint(2**16)))

        return self.forward_func(self.params, key, jax.device_put(model_in))
    

    def agent(self, 
            random_seed:int = 0, 
            params: Params | None = None, 
            n_sess: int = 1,
            additional_inputs: None | list | tuple | Array = None,
            init_ch_probs = None
        ):
        """make a RL agent instance from this network model

        Args:
            random_seed (int, optional): random seed for the agent instance. Defaults to 0. can also be modified later with method `infuse_seed`
            params (Params | None, optional): allows to override the model's parameter. Defaults to None.

        Returns:
            hkNetwork_agent: 
        """
        if params is None:
            params = self.params
        
        # if has_haiku_static_attrs(self.make_network, 'choice_selection'):
        #     # when `choice_selection` method is available, it is possible to calculate the initial choice probabilities from latent init
        #     # this feature should be optional, since the 1st choice was never a target for the model training

        #     def _choice_selection(state):
        #         """get choice logit from state
                
        #         Args:
        #         state: previous state of the hidden units of the RNN model.
                
        #         Returns:
        #         y_hat: output of RNN
        #         new_state: state of the hidden units of the RNN model
        #         """
        #         core = self.make_network()
        #         y_hat, penalty = core.choice_selection(state)
        #         return y_hat, penalty

        #     _, ch_selct = hk.transform(_choice_selection)
        #     y_hat, _ = ch_selct(params, jax.random.PRNGKey(random_seed), np.tile(self.latent_inits, (n_sess,1)))
        #     init_ch_probs = jax.nn.softmax(y_hat, axis=-1)
        # else:
        #     init_ch_probs = None
        
        return hkNetwork_multiAgent(
            self.make_network, params, 
            random_seed = int(random_seed), 
            n_sess=n_sess, 
            additional_inputs=additional_inputs, 
            init_ch_probs=init_ch_probs
        )
    
    def _set_patch_state_ids(self, patch_state_ids: List[int]):
        """call `set_patch_state_ids` on the disrnn module

        Use 0-based latent index

        Args:
            patch_state_ids (List[int]): _description_
        """

        def patched_module():
            disrnn = self.make_network()
            disrnn.set_patch_state_ids(patch_state_ids)
            return disrnn
        
        return patched_module
    
    def patched_agent(self, 
            patch_state_ids: List[int], 
            exo_states: None|Array = None, 
            n_sess:int = 1,
            random_seed:int = 0, 
            params: Params | None = None, 
            ):
        """
        Docstring for patched_agent
        
        :param self: Description
        :param patch_state_ids: Use 0-based latent index #TODO - rename it 
        :type patch_state_ids: List[int]
        :param exo_states: Exogeneous states to patch the internal state, of shape (N_trials, N_sessions, N_mod_states). Axis 0 is time (trial), axis 1 is session, axis 2 represent different latent variables to be patched. Each trial the agent played with the environment will use one row of the exogeneous states to patch the internal latent variable's activation
        :type exo_states: None | Array
        :param n_sess: Description
        :type n_sess: int
        :param random_seed: Description
        :type random_seed: int
        :param params: Description
        :type params: Params | None
        """
        
        assert has_haiku_static_attrs(self.make_network, ['set_patch_state_ids', 'step_with_exo_state']), "the haiku module definition is not compatiable with `patchable_hkRNNCore`"
        
        if params is None:
            params = self.params

        return patched_hkNetwork_multiAgent(
            make_network=self.make_network, 
            params=params, 
            patch_state_ids=patch_state_ids,
            n_sess=n_sess,
            exo_states=exo_states,
            random_seed = int(random_seed))
 
    # TODO there's newer version of activation patching forward available in `functions/activation_patching_forward.py`. I should consider how to incorporate that into here
    def forward_with_state_patching(self, model_in, patch_state_ids: List[int], exo_states):
        """evaluate a RNN model with certain latent state patched

        Args:
            make_network (Callable[[], patchable_hkRNNCore]): a closure function that returns a patchable_hkRNNCore instance
            params (Params): model parameters
            patch_state_ids (List[int]): the 0-based index of latents that will be patched
            xs (Inputs): model inputs of shape (N_trial, N_sess, N_obs)
            exo_state (States): exogeneous states to replace updated value of patched latents on each trial. The shape of the argument will be broadcasted to match a shape of (N_trial, N_sess, N_patched_latents)

        Returns:
            Tuple[Outputs, States]: model outputs and update latent states on each trial for each session
        """
        return patched_forward(self.make_network, self.params, patch_state_ids, model_in, exo_states)

    def _eval_dataset(self, 
            dataset: trainingDataset, 
            output_latents_tuple = None, 
            rand_key = None,
            output_label: None | List[str] = None,
            ):

        if output_latents_tuple is None:
            output_latents_tuple = self.forward(dataset.xs, key=rand_key)

        output_label = output_label or dataset.output_label or self.output_label

        return assemble_eval_dataset(dataset, output_latents_tuple, output_label)

    def augmented_dataset(self, 
            dataset: None | trainingDataset = None, 
            output_label: None | List[str] = None,
            optional_features: Iterable[str] = ['jac_chMLP', 'updMLP_out', 'prev_latents', 'next_choice_prob'],
            out_feat_name: None | List[str] = None,
            alt_params: None | Params = None,
            random_seed: int = 0,
            ) -> xr.Dataset:
        
        """augment a trainingDataset with disRNN model related variables

        ### Basic features

        - inputs: from the trainingDataset
        - outputs: from trainingDataset
        - latents: Updated latents for each trial
        - choice_logits: Model output, the prediction for next trial choice
        
        ### Optional features

        - jac_chMLP: the Jacobian matrix for the choice MLP function: latents -> choice_logits
        - prev_latents: the latent values before update; one of the inputs for the update MLPs
        - updMLP_out: output of update MLPs; learning rate and target

        Raises:
            ValueError: when cannot find dataset

        ### Returns:
            xarray.Dataset: the augmented dataset
        """
        
        if dataset is None:
            output_label = output_label or self.output_label
            if self.dataset is None:
                raise ValueError("please provide inputs data for running the model")
            dataset = self.dataset
        else:
            output_label = output_label or dataset.output_label or self.output_label

        if alt_params:
            params = alt_params
        else:
            params = self.params
        
        out, updated_latents = self.forward_func(params, jax.random.PRNGKey(random_seed), dataset.xs)
        ds = self._eval_dataset(
            dataset, 
            output_latents_tuple=(out, updated_latents),
            output_label=output_label)
        
        data_vars = {}
        coords = {}
        if out_feat_name: # override out_feat_name
            coords['out_feat'] = out_feat_name

        latents = ds['latents'].values
        model_in = ds['inputs'].values

        if 'jac_chMLP' in optional_features:
            # choice MLP takes updated latents as input
            chMLP = self.build_choiceMLP_func(convert_pR = False)
            jac_chMLP = jax.vmap(jax.vmap(jax.jacfwd(chMLP)))(ds['latents'].values)
            data_vars['jac_chMLP'] = (
                ["tri", "sess", "choice", "latent"], 
                jac_chMLP,
                {'description': "the Jacobian matrix for the choice MLP function: latents -> choice_logits"})

        if np.any(np.isin(['updMLP_out', 'prev_latents'], optional_features)):

            lat_inits = self.latent_inits 
            lat_inits = jnp.repeat(lat_inits.reshape(1,1,-1), latents.shape[1], axis=1)
            prev_latents = jnp.concatenate((lat_inits, latents[:-1, :, :]), axis=0)
            
            if 'prev_latents' in optional_features:
                data_vars['prev_latents'] = (
                    ["tri", "sess", "latent"], 
                    prev_latents,
                    {'description': "the latent values before update; one of the inputs for the update MLPs"})
            
            if 'updMLP_out' in optional_features:
                updMLPs = self.build_updateMLP_func()
                updMLP_out = jax.vmap(jax.vmap(updMLPs))(model_in, prev_latents)
                coords['upd_par'] = ['learning rate', 'target']
                data_vars['updMLP_out'] = (
                    ["tri", "sess", "latent", "upd_par"], 
                    updMLP_out,
                    {'description': "output of update MLPs; learning rate and target"})
                
        if 'next_choice_prob' in optional_features:
            data_vars['next_choice_prob'] = (
                ["tri", "sess", "choice"],
                jax.device_get(jax.nn.softmax(ds['choice_logits'].values, axis=-1)),
                {'description': "model predicted probability for each choices on next trial"}
            )
            
                
        
        dats = ds.assign(variables=data_vars).assign_coords(coords=coords)

        return dats
    

    def model_graph(self, 
            input_feature_name: None | List[str] = None, 
            latent_id_name: Dict[int, str] | Callable[[int], str] = dict(),
            output_feature_name: None | str = None,
            shrink = True,
            include_plot_info: bool = True,
            inputs_order: List[str] | ndarray | None = None,
            latents_order: List[int] | ndarray | None = None,
            choice_selection_weight: List[float] | ndarray | None = None,
            ) -> ig.Graph:
        """create a compact graph representation of the disRNN model with attributes representing the model information (rather than attributes for visualization)

        including inputs, latents, and output (next choice)

        igraph refercence: https://python.igraph.org/en/main/tutorial.html

        Args:
            input_feature_name (_type_, optional): _description_. Defaults to None.
            shrink (bool, optional): whether eliminate nodes with no edges. Defaults to True.
            choice_selection_weight(List[float] | None, optional): the weight in choice selection for latents with open choice bottleneck (of length `N_open_ch_lat`)

        Raises:
            ValueError: _description_

        Returns:
            _type_: _description_
        """

        input_sz = self.obs_dim
        latent_sz = self.n_latents
        output_sz = 1

        # ===== creating the graph representing latent update

        # build adjacency matrix: directed edges are represented as from row index to column index
        # size: (input_sz + latent_sz, input_sz + latent_sz)
        # [zeros, thresholded update sigma]
        # connections are only for those with a update sigma less than bn_close_thre
        adj = np.hstack((
            np.zeros((input_sz + latent_sz, input_sz)), 
            self.update_sigma < self.bn_sigma_thre
        ))

        # create the graph from adjacency matrix
        g:ig.Graph = ig.Graph.Adjacency(adj, mode = 'directed') # [inputs, outputs]

        # label inputs
        input_labs = input_feature_name or self.input_feature_name
        if len(input_labs) != input_sz:
            raise ValueError(f"size of input_feature_name ({len(input_labs)}) does not match with size of input features ({input_sz})")
        
        if isinstance(latent_id_name, dict):
            lat_labs = [
                latent_id_name[i+1] if i+1 in latent_id_name else f"lat{i+1}" 
                for i in range(latent_sz)]
        elif isinstance(latent_id_name, Callable):
            lat_labs = list(map(latent_id_name, range(1, latent_sz+1)))
        else:
            raise ValueError('unknown latent_id_name argument: {latent_id_name}')

        g.vs["label"] = input_labs + lat_labs
        g.vs["name"] = input_labs + lat_labs
        g.vs["color"] = list(range(input_sz)) + [input_sz for _ in range(latent_sz)]
        g.vs['type'] = ['input'] * input_sz + ['latent'] * latent_sz
        g.vs['lat_id'] = [np.nan] * input_sz + [i+1 for i in range(latent_sz)] 

        # ===== adding vertex and edges to represent choice selection

        output_lab = output_feature_name or 'next_choice'
        # only consider single output node for now
        g.add_vertex(
            name = output_lab,
            label = output_lab,
            color = input_sz + 1,
            type = "output",
            lat_id = np.nan
            )
        ch_slt_edges = [(lat_labs[i], output_lab) for i in self.open_choice_latent()]
        if choice_selection_weight is not None:
            g.add_edges(ch_slt_edges, attributes={'weight': choice_selection_weight})
        else:
            g.add_edges(ch_slt_edges)

        # ===== generate shrinked graph where closed latents are removed
        if shrink:
            # g_shrink = g_lat.copy()

            # delete orphan nodes: those are not part of the learnt model
            g.delete_vertices(np.arange(g.vcount())[np.array(g.degree()) == 0])
        
        # ===== add attributes for plotting purpose
        # TODO: move the part to generate plotting attributes into `plt_graph`
        if include_plot_info:
            df_node = g.get_vertex_dataframe()
            # ===== label edge type
            for e in g.es:
                s_type = g.vs[e.source]["type"]
                t_type = g.vs[e.target]["type"]
                if (s_type, t_type) == ('input', 'latent'):
                    e['e_type'] = 'input'
                elif (s_type, t_type) == ('latent', 'latent'):
                    e['e_type'] = 'recurrent'
                    s_lat = g.vs[e.source]["lat_id"]
                    t_lat = g.vs[e.target]["lat_id"]
                    e['curve_sign'] = 1 if s_lat >= t_lat else -1
                elif (s_type, t_type) == ('latent', 'output'):
                    e['e_type'] = 'output'
                else:
                    e["e_type"] = None

            # ===== order inputs, latents 
            inputs_order = inputs_order or input_labs
            latents_order = latents_order or g.vs(type='latent')['lat_id']

            df_order = pd.DataFrame(dict(
                label = inputs_order + [g.vs(lat_id = id)['label'][0] for id in latents_order] + [output_lab],
                type = ['input'] * len(inputs_order) + ['latent'] * len(latents_order) + ['output']
            ))
            df_order['in_grp_order'] = df_order.groupby('type').cumcount() + 1

            # add column 'in_grp_order'
            df_node = df_node.merge(df_order, 'inner', on = ['label', 'type'])
            
            # ===== calculate the columnar layout of nodes
            
            df_node['x'] = df_node['type'].map({
                'input': 0,
                'latent': 1,
                'output': 2,
            })
            df_node["y"] = ( # y center to the middle of each type
                df_node.groupby("type")["in_grp_order"].transform("mean")
                - df_node["in_grp_order"]
            )
            df_node.sort_index(inplace=True)

            g.vs['in_grp_order'] = df_order['in_grp_order'].tolist()
            g.vs['x'] = df_node['x'].tolist()
            g.vs['y'] = df_node['y'].tolist()

        return g
    
    def plt_lats_dist(
            self,
            lat_names: Dict[int, str] | None = None,
        ):
        """plot distributions of each latent

        Args:
            lat_names (Dict[int, str] | None, optional): _description_. Defaults to None.

        Returns:
            ggplot: _description_
        """

        import plotnine as pn

        lats = self._cache_latent_samples.melt()

        lats_stats = lats.groupby('variable').agg(['mean', 'median']).reset_index()
        lats_stats.columns = ['variable', 'mean', 'median']
        lats_stats_f = lats_stats[lats_stats['variable'].isin(self.np.open_latent(consider_update_bn=True))]

        lat_names_str = {str(k):f"L{k+1}" for k in range(self.n_latents)}
        if lat_names is not None:
            lat_names_str.update({str(k):v for k,v in lat_names.items()})

        return (
            pn.ggplot(lats[lats['variable'].astype(int).isin(self.np.open_latent(consider_update_bn=True))], pn.aes('value'))
            + pn.facet_wrap('variable', scales='free_x', labeller=pn.as_labeller(lat_names_str))
            + pn.geom_histogram() 
            + pn.geom_vline(pn.aes(xintercept='mean'), data=lats_stats_f, color = 'red')
            + pn.scale_y_continuous(expand=(0, 0))
            + pn.ggtitle('Latents mean and distribution')
            )

    def plt_graph(self, 
            graph: ig.Graph | nx.Graph | None = None, 
            backend: Literal['igraph', 'networkx'] = 'igraph', 
            visual_style: dict | None = None, 
            **kws
        ):
        """plot the model's graph representation
        
        Argument
        ---
        graph: (optional) If not provided, the graph will be generated as `self.model_graph(**kws)`
        """

        if graph is None:
            g = self.model_graph(**kws)
        else:
            g = graph

        match backend.lower():
            case 'igraph':
                if isinstance(g, nx.Graph):
                    g = ig.Graph.from_networkx(g)
                fig, ax = plt_graph_with_igraph(g, visual_style=visual_style)
            case 'networkx':
                if isinstance(g, ig.Graph):
                    g = g.to_networkx()
                fig, ax = plt_graph_with_networkx(g, visual_style=visual_style)
            case _:
                raise ValueError('unsupported backend')
        return fig, ax

    def plot_bottlenecks(self, obs_names = None):
                
        latent_dim = self.latent_sigma.shape[0]
        obs_dim = self.update_sigma.shape[0] - latent_dim

        if obs_names is None:
            if obs_dim == 2:
                obs_names = ['Choice', 'Reward']
            else: 
                obs_names = np.arange(1, obs_dim+1)

        latent_names = self.latent_sigma_order + 1
        fig = plt.subplots(1, 3, figsize=(10, 5))
        plt.subplot(1, 3, 1)
        plt.imshow(np.swapaxes([1 - self.sorted_latent_sigma], 0, 1), cmap='Oranges')
        plt.clim(vmin=0, vmax=1)
        plt.yticks(ticks=range(latent_dim), labels=latent_names)
        plt.xticks(ticks=[])
        plt.ylabel('Latent #')
        plt.title('Latent Bottlenecks')

        plt.subplot(1, 3, 2)
        plt.imshow(1 - self.sorted_update_sigma.T, cmap='Oranges')
        plt.clim(vmin=0, vmax=1)
        plt.yticks(ticks=range(latent_dim), labels=latent_names)
        xlabels = np.concatenate((np.array(obs_names), latent_names))
        plt.xticks(
            ticks=range(len(xlabels)),
            labels=xlabels,
            rotation='vertical',
        )
        plt.ylabel('Latent #')
        plt.title('Update MLP Bottlenecks')

        plt.subplot(1, 3, 3)
        plt.imshow(np.swapaxes([1 - self.sorted_choice_sigma], 0, 1), cmap='Oranges')
        plt.clim(vmin=0, vmax=1)
        plt.yticks(ticks=range(latent_dim), labels=latent_names)
        plt.xticks(ticks=[])
        plt.ylabel('Latent #')
        plt.title('Choice Bottlenecks')
        plt.colorbar()

        return fig

    def plt_latent_bottleneck(self, sorted_latent=True, ax=None, show_colorbar: bool|Axes = True, cbar_kws = None):
        if sorted_latent:
            lat = self.np.sorted_latent_sigma
            lats_ind = self.np.latent_sigma_order + 1
        else:
            lat = self.np.latent_sigma
            lats_ind = np.arange(self.n_latents) + 1
            
        lat_names = [f'l{i}' for i in lats_ind]
        
        if ax is None:
            fig, ax = plt.subplots(figsize=(2, 6))

        if cbar_kws is None:
            cbar_kwargs={'label': 'openness',} if show_colorbar else None
        else:
            cbar_kwargs = {'label': 'openness',}.update(cbar_kws)
        
        sns.heatmap(1 - lat.reshape(-1, 1),
                   cmap='Oranges',
                   vmin=0, vmax=1,
                   cbar=True if show_colorbar else False,
                   cbar_kws=cbar_kwargs,
                   cbar_ax=show_colorbar if isinstance(show_colorbar, Axes) else None, 
                   linewidths=1,
                   linecolor='white',
                   square=True,
                   ax=ax)
        
        ax.set_yticks(np.arange(len(lat_names)) + 0.5)
        ax.set_yticklabels(lat_names)
        ax.set_xticks([])
        
        ax.set_xlabel('')
        ax.set_ylabel('latents')
        
        ax.grid(False)
        sns.despine(ax=ax, left=True, bottom=True)
        
        if ax is None:
            return fig, ax
        return ax.figure, ax

    def plt_update_bottleneck(self, sorted_latent=True, input_label=None, ax=None, show_colorbar=True, xtick_rot = 0):
        if input_label is None:
            input_label = self.input_feature_name
            
        if sorted_latent:
            upd = self.np.sorted_update_sigma
            lats_ind = self.np.latent_sigma_order + 1
        else:
            upd = self.np.update_sigma
            lats_ind = np.arange(self.n_latents) + 1
            
        lat_names = [f'l{i}' for i in lats_ind]
        
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 6))
        
        sns.heatmap(np.transpose(1 - upd), 
                   cmap='Oranges',
                   vmin=0, vmax=1,
                   cbar=show_colorbar,
                   cbar_kws={
                       'label': 'openness',
                   } if show_colorbar else None,
                   linewidths=1,
                   linecolor='white',
                   square=True,
                   ax=ax)
        
        # Customize axes
        ax.set_yticks(np.arange(len(lat_names)) + 0.5)
        ax.set_yticklabels(lat_names)
        ax.set_xticks(
            np.arange(len(input_label) + len(lat_names)) + 0.5,
            input_label + lat_names,
            rotation = xtick_rot,
        )
        
        ax.set_xlabel('inputs')
        ax.set_ylabel('latents')
        
        # Remove grid and adjust theme
        ax.grid(False)
        sns.despine(ax=ax, left=True, bottom=True)
        
        if ax is None:
            return fig, ax
        return ax.figure, ax
    
    def plt_choice_bottleneck(self, sorted_latent=True, ax=None, show_colorbar=True):
        if sorted_latent:
            ch = self.np.sorted_choice_sigma
            lats_ind = self.np.latent_sigma_order + 1
        else:
            ch = self.np.choice_sigma
            lats_ind = np.arange(self.n_latents) + 1
            
        lat_names = [f'l{i}' for i in lats_ind]
        
        if ax is None:
            fig, ax = plt.subplots(figsize=(2, 6))
        
        sns.heatmap(1 - ch.reshape(-1, 1),
                   cmap='Oranges',
                   vmin=0, vmax=1,
                   cbar=show_colorbar,
                   cbar_kws={
                       'label': 'openness',
                   } if show_colorbar else None,
                   linewidths=1,
                   linecolor='white',
                   square=True,
                   ax=ax)
        
        ax.set_yticks(np.arange(len(lat_names)) + 0.5)
        ax.set_yticklabels(lat_names)
        ax.set_xticks([])
        
        ax.set_xlabel('')
        ax.set_ylabel('latents')
        
        ax.grid(False)
        sns.despine(ax=ax, left=True, bottom=True)
        
        if ax is None:
            return fig, ax
        return ax.figure, ax
  
    def plt_combine_heatmaps(self, sorted_latent=True, xtick_rot=0):

        ws = [1, self.np.update_sigma.shape[0], 1]

        fig, axs = plt.subplots(
            1,3, 
            sharey=False, 
            width_ratios=ws, 
            layout='compressed',
            figsize = [sum(ws)/8*6, 6])
        
        # Plot each component using the modified plotting functions
        self.plt_latent_bottleneck(sorted_latent=sorted_latent, ax=axs[0], show_colorbar=False)
        self.plt_update_bottleneck(sorted_latent=sorted_latent, ax=axs[1], show_colorbar=False, xtick_rot = xtick_rot)
        self.plt_choice_bottleneck(sorted_latent=sorted_latent, ax=axs[2], show_colorbar=True)
        
        # plt.tight_layout()
        return fig, axs

    # ====== calculation results saved as cached properties
    @cached_property
    def cache_forward(self):
        """cached result of evaluating self model on self dataset
        """
        return self.forward(key=jax.random.PRNGKey(0))

    @cached_property
    def cache_eval_dataset(self):
        """cached xarray.Dataset describing evaluation of the model on the dataset"""

        return self._eval_dataset(self.dataset, self.cache_forward, output_label=self.output_label)  # type: ignore

    @cached_property
    def _cache_latent_samples(self) -> pd.DataFrame:
        """dataframe of updated latents after evaluated on associated dataset

        only kept existing trials

        Index: (tri, sess)
        Columns: np.arange(self.n_latents)

        Returns:
            pd.DataFrame: _description_
        """
        df = ds2df(self.cache_eval_dataset[['inputs', 'outputs', 'latents']], omit=['latent', 'latents'])
        return df.filter(regex=r'^\d+$')[np.all(df.filter(regex='inputs:|outputs:') > -1, axis=1)]

    def _suggest_lat_grid(self, lat_id: int, N_points = 20):
        """suggest mesh grid for latent `lat_id` based on empirical range obsevered from evaluation on assoicated dataset

        Args:
            lat_id (int): specify which latent. 1-based index for latent
            N_points (int, optional): _description_. Defaults to 20.

        Returns:
            _type_: _description_
        """
        lat_empirical_range = lambda lat_ind: min_max(self.cache_eval_dataset['latents'].isel(latent=lat_ind).values)
        return np.linspace(*lat_empirical_range(lat_id-1), num = N_points)
    
    def _suggest_obs_grid(self, obs: str | int, N_points = 20, force_resample = False):
        """suggest mesh grid for an observation feature

        Args:
            obs (str | int): ether index of observation feature or name of it
            N_points (int, optional): _description_. Defaults to 20.
            force_resample (bool, optional): _description_. Defaults to False.

        Returns:
            _type_: _description_
        """

        if isinstance(obs, str):
            obs_ind = list(self.input_feature_name).index(obs)
        else:
            obs_ind = obs
        
        uniqs = np.unique(self.dataset.xs[:,:,obs_ind])
        uniqs = uniqs[uniqs > -1] # remove NAs

        if force_resample or len(uniqs) >= N_points:
            # consider as continuous
            return np.linspace(*min_max(uniqs), num = N_points)
        else:
            # consider as discrete
            return uniqs
        
    def _default_obs_grid_dict(self, lat_ind, N_grid_points: int = 20):
        # open_obs_idx = np.nonzero(self.np.open_update_bn()[:self.obs_dim, lat_id-1])[0]
        open_obs_idx = jax.device_get(self.LV_upd_open_obs(lat_ind))
        obs_grid = {self.input_feature_name[obs]:self._suggest_obs_grid(obs, N_points=N_grid_points) for obs in open_obs_idx}
        return obs_grid
    
    def _default_lat_input_grid_dict(self, lat_ind, N_grid_points: int = 20):
        # open_lat_in_idx = np.nonzero(self.np.open_update_bn()[self.obs_dim:, lat_id-1])[0]
        open_lat_in_idx = jax.device_get(self.LV_upd_open_lat(lat_ind))
        lat_input_grid = {f"L{lat_in+1}":self._suggest_lat_grid(lat_in+1, N_points=N_grid_points) for lat_in in open_lat_in_idx}
        return lat_input_grid



    def cal_updMLP_grid(self, 
            lat_id, 
            obs_grid: Dict[str, ndarray|List[float]] | None = None, 
            lat_input_grid: Dict[str, ndarray|List[float]] | None = None,
            N_grid_points: int = 20,
        ) -> pd.DataFrame:
        """evaluate an update MLP over observation and latent input grids

        If the grids for latent input is not provided, it will be generated as equally spaced points from minimum to maximum of the empirical value

        Args:
            lat_id (int): specify which latent's update MLP. 1-based index for latent
            obs_grid (Dict[str, np.ndarray | List[float]]): Grids for selected observation features. Keyed by input_feature_name
            lat_input_grid (Dict[str, np.ndarray | List[float]]): Grids for selected latents (as part of the input to updateMLP). Keyed by latent name (f"L{lat_id}")

        Returns:
            pd.DataFrame: DataFrame with columns for each grid variable (obs_grid keys, lat_input_grid keys)
                plus 'lr' (learning rate), 'target', 'intercept' (lr * target), and 'slope' (1 - lr).
        """

        if obs_grid is None:
            # open_obs_idx = np.nonzero(self.np.open_update_bn()[:self.obs_dim, lat_id-1])[0]
            open_obs_idx = jax.device_get(self.LV_upd_open_obs(lat_id - 1))
            obs_grid = {self.input_feature_name[obs]:self._suggest_obs_grid(obs, N_points=N_grid_points) for obs in open_obs_idx}
        
        if lat_input_grid is None:
            # open_lat_in_idx = np.nonzero(self.np.open_update_bn()[self.obs_dim:, lat_id-1])[0]
            open_lat_in_idx = jax.device_get(self.LV_upd_open_lat(lat_id - 1))
            lat_input_grid = {f"L{lat_in+1}":self._suggest_lat_grid(lat_in+1, N_points=N_grid_points) for lat_in in open_lat_in_idx}

        obs_defaults = jnp.zeros(self.obs_dim)
        # latent input default is also 0
        lat_input_defaults = jnp.zeros(self.n_latents)

        # index of each item in the dicts
        obs_idx = np.array(list(map(
            list(self.input_feature_name).index, 
            obs_grid.keys())), dtype = int)
        lat_in_idx = np.array(list(map(
            [f"L{i+1}" for i in range(self.n_latents)].index, 
            lat_input_grid.keys())), dtype = int)

        # make grids
        grids = np.meshgrid(*obs_grid.values(), *lat_input_grid.values())
        grid_flatten = np.column_stack([g.ravel() for g in grids])
        obs_g_f = grid_flatten[:, :len(obs_grid)]
        lat_in_g_f = grid_flatten[:, len(obs_grid):]

        # evaluate across grids
        updMLP = self.build_updateMLP_func(lat_id, )
        eval_grid = lambda partial_obs, partial_lat_input: updMLP(obs_defaults.at[obs_idx].set(partial_obs), lat_input_defaults.at[lat_in_idx].set(partial_lat_input))
        res = jax.vmap(eval_grid)(obs_g_f, lat_in_g_f)

        df = pd.DataFrame(np.hstack((grid_flatten, res)), columns = [*obs_grid.keys(), *lat_input_grid.keys(), 'lr', 'target'])

        df['intercept'] = df['lr'] * df['target']
        df['slope'] = 1 - df['lr']

        return df
class db_disRNN_analyzer(disRNN_model_analyzer):

    def __init__(self, db_sess, mt_id, step, bn_close_thre=0.9) -> None:            
        mt, rec = get_mt_model_by_step(db_sess, mt_id, step)

        super().__init__(mt.eval_model.model_haiku, rec.parameter, mt.sessions[0].train_dataset.xs, bn_close_thre)

        self.db_record = rec