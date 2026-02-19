# %%
from copy import deepcopy
from typing import Any, List, Tuple, Optional, Dict, Callable, NamedTuple, Union, Literal
from abc import ABC, abstractmethod
from dataclasses import field
import warnings

import flax.linen as nn
from flax.core import FrozenDict
import jax
from jax import lax
import jax.numpy as jnp
import optax
import optax.tree_utils as otu
import numpy as np
from jax.experimental.x64_context import enable_x64
from flax import struct


from disRNN_MP.metrics import BerLL_prob
from disRNN_MP.dataset import trainingDataset
from disRNN_MP.typing import States, Outputs

# %%

class boundedParam(NamedTuple):
    """represent a bounded parameter
    when only lb (lower bound) and ub (upper bound) are provided, the parameter init value will be uniformly sampled from the specified range of the specified shape (by default it is zero-dimension).

    When a init function is provided, it will be called to determine a init value

    When init is a constant number (or array), then the init value will be fixed at that constant value
    """
    name: str
    lb: Union[float, jax.Array]
    ub: Union[float, jax.Array]
    shape: tuple = ()
    init: Optional[Union[Callable, jax.typing.ArrayLike]] = None

    def init_fun(self):
        """generate the init_fun that can be passed to flax's Module.param
        Returns:
            Callable: (rng,) -> jax.Array
        """
        if self.init is None: 
            fn = _init_unif(self.shape, self.lb, self.ub)
        elif callable(self.init):
            fn = self.init(self.shape, self.lb, self.ub)
        else:
            fn = _init_const(self.shape, self.init)
        
        return fn


def listPar2dict(pars:List[boundedParam]) -> Dict[str, boundedParam]:
    if len(pars) == 0:
        return dict()
    else:
        return {p.name: p for p in pars}


def _init_const(shape, val):
    return lambda rng: jnp.ones(shape) * val

# =========== fixed shape bounded initializer functions
# all takes args: shape, min, max

def _init_unif(shape, min: jax.typing.ArrayLike = -1, max: jax.typing.ArrayLike = 1):
    return lambda rng: jax.random.uniform(rng, shape, minval=min, maxval=max)

# %%

# Helper function for optax L-BFGS optimization with box constraints
def _run_opt_bounded(init_params, loss_fun, opt, max_iter, tol, lb, ub):
    """Run optax L-BFGS optimization with box constraints.
    
    Args:
        init_params: Initial parameter pytree
        loss_fun: Loss function to minimize
        opt: optax optimizer (e.g., optax.lbfgs())
        max_iter: Maximum number of iterations
        tol: Tolerance for gradient norm stopping criterion
        lb: Lower bounds pytree (same structure as init_params)
        ub: Upper bounds pytree (same structure as init_params)
    
    Returns:
        Tuple of (final_params, final_state)
    """
    value_and_grad_fun = optax.value_and_grad_from_state(loss_fun)

    def step(carry):
        params, state = carry
        value, grad = value_and_grad_fun(params, state=state)
        updates, state = opt.update(
            grad, state, params, value=value, grad=grad, value_fn=loss_fun
        )
        params = optax.apply_updates(params, updates)
        params = optax.projections.projection_box(params, lb, ub)
        return params, state

    def continuing_criterion(carry):
        _, state = carry
        iter_num = otu.tree_get(state, 'count')
        grad = otu.tree_get(state, 'grad')
        err = otu.tree_norm(grad)
        return (iter_num == 0) | ((iter_num < max_iter) & (err >= tol))

    init_carry = (init_params, opt.init(init_params))
    final_params, final_state = jax.lax.while_loop(
        continuing_criterion, step, init_carry
    )
    return final_params, final_state


class RLmodel(nn.Module, ABC):
    """ ## An abstract class for traditional type of RL models of behavior

    Usually the models have a smaller set of "meaningfully" defined parameters (often bounded), like learning rate bounded to between 0 and 1, which allows the Models to be fitted via L-BFGSB.

    Subclasses of this should implement the model by specifying how values are initialized (`init_values`), updated (`value_update`) and being used to take action (`choice_selection`).

    Parameters required by a specific model should be listed out in `_default_paramSpecs` property. The parameters should be defined to be independent of scales (number of trials and sessions).    

    ### Implementation details

    the initialization of parameters does not depend on data shape
    """

    N_actions: int
    """number of possible actions"""

    N_values: int
    """number of values to track across iterations for each session"""
    N_obs_feat: int = 2
    """Number of observation features. 
    For a typical RL behavioral model, this includes last choice and outcome (N_obs_feat = 2)
    """
    param_specs: List[boundedParam] = struct.field(default_factory = list)
    """specification of parameters (name, bounds, shape, init)"""
    

    @property
    @nn.nowrap
    def paramSpecs(self) -> Dict[str, boundedParam]:
        """init self.paramSpecs from default and modification
        class default is found from `self._default_paramSpecs`
        modification is taken from argument `params_specs`
        """
        defaults = listPar2dict(self._default_paramSpecs)
        defaults.update(listPar2dict(self.param_specs))
        return defaults

    def setup(self) -> None:
        """customized setup function (from nn.Module) which initiate all parameters specified in `self.paramSpecs`"""
        for k_name, v_par in self.paramSpecs.items():
            setattr(self, k_name, self.param(k_name, v_par.init_fun()))

    @property
    @abstractmethod
    @nn.nowrap
    def _default_paramSpecs(self) -> List[boundedParam]:
        """the class-level default paramSpecs for the RL model"""
        pass

    @abstractmethod
    def init_values(self, n_sess: int) -> jax.Array:
        """initialize values given input shape
        usually handle parallel set of values for multiple sessions
        (n_sess) -> init_value
        init_value: (N_sess, N_values)
        """
        raise NotImplementedError()
    
    @abstractmethod
    def value_update(self, obs: jax.Array, values: jax.Array) -> jax.Array:    
        """function to calculate updated value for a single trial on a session
        (obs, values) -> updated_values
        obs: (N_sess, N_input_feats)
        values: (N_sess, N_values)
        updated_values: (N_sess, N_values)
        """
        raise NotImplementedError()
    
    @abstractmethod
    def choice_selection(self, values: jax.Array) -> jax.Array:
        """determine probabilities for every choice based on values for a single trial on a session
        (values) -> ch_probs
        values: (N_sess, N_values)
        ch_probs: (N_sess, N_actions)
        """
        raise NotImplementedError()
    
    def __call__(self, inputs: jax.Array):
        """
        inputs: (N_trials, N_sess, N_input_feats)
        """

        # the learning sequence on a single trial
        def _learn_seq(carry, obs):
            """the learning sequence on a single trial for multiple sessions
            (carry, obs) -> (carry, [carry, ch_probs])
            carry: (N_sess, N_values)
            obs: (N_sess, N_input_feats)
            ch_probs: (N_sess, N_actions)
            """
            # update state based on observation
            carry = self.value_update(obs, carry)

            # choice selection
            ch_probs = self.choice_selection(carry)

            return carry, jnp.concatenate((carry, ch_probs), axis=1)
        
        # _learn_seq_sess = jax.vmap(_learn_seq, in_axes=0, out_axes=(0, 0))
        # (carry, obs) -> carry, [carry, ch_probs]
        # carry: (N_sess, N_values)
        # obs: (N_sess, N_input_feats)
        # ch_probs: (N_sess, N_actions)
        
        values = self.init_values(inputs.shape[1]) # (N_sess, N_values)
        N_values = values.shape[1]
        _, out = lax.scan(_learn_seq, init = values, xs = inputs) # out: (N_trials, N_sess, N_values + N_actions)

        # all_values, all_ch_probs = out[:,:,:N_values], out[:,:,N_values:]
        all_values, all_ch_probs = jnp.split(out, N_values, axis=2)

        return all_values, all_ch_probs

    
    def _get_dummy_inputs(self, n_sess: int) -> jax.Array:
        """obtain a dummy input array for the purpose of initialize underlying flax module"""

        return jnp.zeros((1, n_sess, self.N_obs_feat))

LL_label = jax.Array
LL_probs = jax.Array
norm_BerLL_prob = lambda label, logits: BerLL_prob(label, logits, norm = True)
class RLmodelWrapper:
    """wrap a classic RL model definition and provide fit, forward, and metric methods

    model: RLmodel (definition)
    dataset: trainingDataset instance for fitting and as the default dataset
    rng: random seed used to init parameters for fitting
    run_fitting: whether to call the `fit` method during init

    Once fitted (default), the instance will have the following attribute:
    - _opt: the optimization result
    - params: the fitted model parameters
    - minimum: the minimized loss function value after training (negative normalized loglikelihood)
    - forward (method): runing through a dataset with fitted parameters 
    - metric (method): measure performance against a dataset by exponential of normalized log-likelihood
    """

    def __init__(
            self, 
            model: RLmodel, 
            params: dict | None = None,
            dataset: trainingDataset | None = None, # dataset for fitting
            rng: jax.Array = jax.random.key(0), # random key mainly for fitting
            nLL: Callable[[LL_label, LL_probs], jax.Array] = norm_BerLL_prob,
            run_fitting: bool = False
        ) -> None:

        self.model = model
        self.nLL = nLL
        self.rng = rng

        if dataset is not None:
            self.dataset = dataset

        # params should always be independent of number of session
        par = self.model.init(rng, model._get_dummy_inputs(n_sess=1))['params'] if params is None else params
        
        self.init_params = par
        self.params = par

        if run_fitting:
            self.fit(self.rng)
        

    @property
    def dataset(self) -> trainingDataset:
        if hasattr(self, '_dataset') and self._dataset is not None:
            return self._dataset
        else:
            raise RuntimeError("please set the `dataset` property before use")
        
    @dataset.setter
    def dataset(self, val):
        if isinstance(val, trainingDataset):
            self._dataset = val
        else:
            raise ValueError(f"please provide an instance of `trainingDataset` rather than {type(val)}")
        

    def _params_bound(self, var_template) -> Tuple[dict, dict]:
        """get lower and upper boundary for each parameters for the fitting

        source of the lower and upper boundary info is from `self.paramSpecs`
        populate the lower and upper bounds for each element of each parameter (in the case that a parameter is a vector),
        the pytree definition of output is the same as that for var_template

        Args:
            var_template (_type_): a flax variable container

        Returns:
            lb: pytree for lower bound
            ub: pytree for upper bound
        """
        var_lb = deepcopy(var_template)
        var_ub = deepcopy(var_template)
        
        pspecs = self.model.paramSpecs
        for parm_key in var_template['params']:
            p = pspecs[parm_key] # type: ignore
            var_lb['params'][parm_key] = jnp.ones_like(var_template['params'][parm_key]) * p.lb
            var_ub['params'][parm_key] = jnp.ones_like(var_template['params'][parm_key]) * p.ub

        return var_lb['params'], var_ub['params']

    def fit(
            self, 
            rng: jax.Array, 
            optimizer: Literal['optax', 'jaxopt'] = 'optax',
            **kwargs
        ):
        """Fit the model with bundled dataset and the given random key.

        Minimize negative log-likelihood.

        Args:
            rng: Random key for initializing parameters
            optimizer: Optimizer to use. 'optax' (default) uses optax.lbfgs.
                       'jaxopt' (deprecated) uses jaxopt.LBFGSB.
            **kwargs: Passed to the optimizer. For optax, these are passed to
                      optax.lbfgs(). For jaxopt, these are passed to jaxopt.LBFGSB().

        Returns:
            Optimization result (format depends on optimizer choice)
        """
        if optimizer == 'optax':
            return self._fit_optax(rng, **kwargs)
        elif optimizer == 'jaxopt':
            warnings.warn(
                "optimizer='jaxopt' is deprecated and will be removed in a future version. "
                "Use optimizer='optax' (default) instead. "
                "To use jaxopt, install with: pip install disRNN-MP[legacy]",
                DeprecationWarning,
                stacklevel=2
            )
            return self._fit_jaxopt(rng, **kwargs)
        else:
            raise ValueError(f"Unknown optimizer: {optimizer}. Use 'optax' or 'jaxopt'.")

    def _fit_optax(self, rng: jax.Array, max_iter: int = 100, tol: float = 1e-3, **kwargs):
        """Fit the model using optax L-BFGS (default optimizer).
        
        Args:
            rng: Random key for initializing parameters
            max_iter: Maximum number of iterations (default: 100)
            tol: Tolerance for gradient norm stopping criterion (default: 1e-3)
                 Note: For bounded optimization, the gradient at boundaries may not 
                 be zero even at convergence. A tolerance of 1e-3 is typically sufficient.
            **kwargs: Passed to optax.lbfgs()
        
        Returns:
            Tuple of (final_params, final_state)
        """
        xs = self.dataset.xs
        ys = self.dataset.ys
        init_vars = self.model.init(rng, xs)

        def loss_fun(variables):
            _, all_ch_probs = self.model.apply(variables, xs)
            # negative log-likelihood (unnormalized)
            return -BerLL_prob(ys, all_ch_probs, norm=False)
        
        opt = optax.lbfgs(**kwargs)
        lb, ub = self._params_bound(init_vars)
        final_params, final_state = _run_opt_bounded(
            init_vars, loss_fun, opt, max_iter, tol, 
            lb={'params': lb}, ub={'params': ub}
        )
        
        self._opt_state = final_state
        self.params = FrozenDict(final_params['params'])
        self.minimum = otu.tree_get(final_state, 'value')

        return final_params, final_state

    def _fit_jaxopt(self, rng: jax.Array, **kwargs):
        """Fit the model using jaxopt L-BFGSB (deprecated, legacy optimizer).

        Requires jaxopt to be installed: pip install disRNN-MP[legacy]

        Args:
            rng: Random key for initializing parameters
            **kwargs: Passed to jaxopt.LBFGSB()

        Returns:
            jaxopt optimization result
        """
        try:
            import jaxopt
        except ImportError:
            raise ImportError(
                "jaxopt is required for optimizer='jaxopt'. "
                "Install with: pip install disRNN-MP[legacy]"
            )

        xs = self.dataset.xs
        ys = self.dataset.ys
        init_vars = self.model.init(rng, xs)

        def loss_fun(params):
            all_values, all_ch_probs = self.model.apply({'params': params}, xs)
            # negative normalized loglikelihood
            loss = -self.nLL(ys, all_ch_probs) 
            return loss
        
        vg = jax.jit(jax.value_and_grad(loss_fun))
        
        LBFGSB_arg = {'tol': 1e-5, 'maxls': 100, 'maxiter': 1e4}
        LBFGSB_arg.update(kwargs)
        LBFGSB_arg.update({'value_and_grad': True})
        minimizer = jaxopt.LBFGSB(vg, **LBFGSB_arg)

        with enable_x64():
            opt = minimizer.run(
                init_vars['params'], 
                bounds=self._params_bound(init_vars))
            
        self._opt = opt
        self.params = FrozenDict(opt.params)
        self.minimum = opt.state.value

        return opt
    
    def forward(self, data:Optional[trainingDataset] = None) -> Tuple[States, Outputs]:
        if data is None:
            data = self.dataset
        all_values, all_ch_probs = self.model.apply({'params': self.params}, data.xs)
        return all_values, all_ch_probs
    
    def metric(self, data:Optional[trainingDataset] = None):
        if data is None:
            data = self.dataset
        _, ch_probs = self.forward(data)
        return np.exp(self.nLL(data.ys, ch_probs))



class forgetQ(RLmodel):
    """the forgetting Q-learning model from (Lee et al. 2004)
    
    N_actions = N_values

    model parameters:   # boundedParam(name, lb, ub, shape, init)
        - boundedParam('decay_rate', 0, 1),
        - boundedParam('positive_evi', 0, 10),
        - boundedParam('negative_evi', -10, 0)
        - boundedParam('v0', -10, 10, shape = (self.N_values, ))
    """
    N_actions: int = 2
    N_values: int = field(init=False, )
    """N_values = N_actions"""
    N_obs_feat: int = field(default=2, init=False)
    """this model cannot handle more than choice and reward in observation"""

    def __post_init__(self):
        self.N_values = self.N_actions
        super().__post_init__()

    @property
    def _default_paramSpecs(self) -> List[boundedParam]:
        return [ # boundedParam(name, lb, ub, shape, init)
            boundedParam('decay_rate', 0, 1),
            boundedParam('positive_evi', 0, 10),
            boundedParam('negative_evi', -10, 0),
            boundedParam('v0', -10, 10, shape = (self.N_values, ))
        ]
    
    def init_values(self, n_sess: int) -> jax.Array:
        """init model values

        Args:
            n_sess (int): number of session

        Returns:
            jax.Array: init values of shape (N_sess,N_values)
        """
        # init values as 0
        # values = jnp.zeros((n_sess, self.N_values))

        # duplicate v0 for each session
        values = jnp.matmul(jnp.ones((n_sess,1)), self.v0.reshape(1,-1)) 
        # (N_sess,1) x (1,N_values) -> (N_sess,N_values)

        return values
    
    def value_update(self, obs: jax.Array, values: jax.Array) -> jax.Array:
        """
        obs: (N_sess, 2) obs[:,0] is last choice, obs[:,1] is last reward
        values: (N_sess, N_values) previous value

        Return
            updated_value: (N_sess, N_values)
        """
        # for backward compatability 
        value_1d = False
        if len(obs.shape) == 1:
            obs = obs.reshape(1,-1)
            value_1d = True

        
        N_actions = self.N_values
        N_sess = obs.shape[0]
        # choice one-hot vector (N_sess, N_actions)
        chOH = jax.nn.one_hot(obs[:, 0].astype(jnp.int32), N_actions)
        # evidence vector for the targets (N_sess, N_actions)
        evid = chOH * lax.select(obs[:,1].astype('bool'), jnp.repeat(self.positive_evi, N_sess) , jnp.repeat(self.negative_evi, N_sess)).reshape(-1,1)
        # value update
        values = self.decay_rate * values + evid
        
        if value_1d:
            return values[0,:]
        else:
            return values
    
    def choice_selection(self, values: jax.Array) -> jax.Array:
        """ 
        N_values = N_actions

        values: (N_sess, N_values) -> ch_probs: (N_sess, N_action)
        values: (N_values, ) -> ch_probs: (N_action, )

        Return
            ch_probs: (N_sess, N_values)
        """
        ch_probs = jax.nn.softmax(values, axis=-1) # (n_batch, n_action) or 
        return ch_probs
