from typing import Optional, Callable, Tuple, Any, Dict, List, NamedTuple, Union, Concatenate, ParamSpec
import time
from copy import deepcopy
import logging
import importlib
import warnings
import inspect
import re

import numpy as np
import haiku as hk
import jax
import jax.tree_util as jt
import jax.numpy as jnp
from jax.typing import ArrayLike
import optax
import chex
import flax.linen as nn

from ..dataset import trainingDataset
from ..metrics import BerLL_logit
from ..typing import Params, RandomKey, Inputs, Outputs, OptState, Loss, States, BatchSize, TrainStepFun, RNN_Apply_Fun, LossFun, Array, patchable_hkRNNCore

@jax.jit
def nan_in_dict(d):
    """Check a nested dict (e.g. hk.params) for nans.
    this function is jit-ed
    the output can be used as a python bool
    """
    def chk_nan(x): return jnp.any(jnp.isnan(x))
    return jnp.any(jnp.array(jt.tree_map(chk_nan, jt.tree_leaves(d))))

def breakpoint_if_nan(d):
    jax.lax.cond(
        nan_in_dict(d), 
        lambda: jax.debug.breakpoint(),
        lambda: None)

class RNNtransformed(NamedTuple):
    init: Callable[[RandomKey, Inputs], Params]
    apply: RNN_Apply_Fun
    get_initial_state: Callable[[Params, RandomKey, BatchSize], States]
    model_haiku: Callable[[], hk.RNNCore]

def transform_hkRNN(hkRNN: Callable[[], hk.RNNCore]) -> RNNtransformed:
    """transform a haiku RNN network to have an apply function that unroll the RNN over inputs
    Args:
        hkRNN: a function wraps a haiku RNN inherit from haiku.RNNCore
    Returns:
        model:
            .init: (rng, xs)
            .apply: (params, rng, xs) -> (ys, states)
    """
    # Haiku, step one: Define the batched network
    def unroll_network(xs):
        core = hkRNN()
        if 'observations' in xs:
            batch_size = xs['observations'].shape[1]
        else:
            batch_size = jnp.shape(xs)[1]
        state = core.initial_state(batch_size)
        ys, states = hk.dynamic_unroll(core, xs, state, return_all_states=True)
        return ys, states
    
    def get_init(batch_size: BatchSize):
        core = hkRNN()
        state = core.initial_state(batch_size)
        return state
    
    def get_haiku():
        return hkRNN

    # Haiku, step two: Transform the network into a pair of functions
    # (model.init and model.apply)
    init, apply = hk.transform(unroll_network)
    
    _, get_init_state = hk.transform(get_init)
    _, _get_haiku = hk.transform(get_haiku) 

    return RNNtransformed(init, apply, get_init_state, hkRNN)

def make_RNNtransformed(_hk_module: Union[Callable, str], *args, **kwargs):
    """creeate a RNNtransformed instance by a haiku module identifier string

    the function will first import the haiku module using the identifier string, then make a simple function wrapper that init the haiku module with other provided positional or keyword arguments. Finally, the function wrapper will be sent to `transform_hkRNN` to create RNNtransformed instance
    
    Args:
        hkRNN: a haiku RNN module
    Returns:
        model:
            .init: (rng, xs)
            .apply: (params, rng, xs) -> (ys, states)
    """
    if isinstance(_hk_module, str):
        module_path, func_name = _hk_module.rsplit('.', 1)
        _module = importlib.import_module(module_path)
        _hk_module = getattr(_module, func_name)
    if not isinstance(_hk_module, Callable):
        warnings.warn(f'it seems that argument module {_hk_module} is not a Callable')
    else:
        # if it can be recognized as callable, check if provided arguments are valid
        sig = inspect.signature(_hk_module)
        bound_args = sig.bind(*args, **kwargs)

    def model():
        return _hk_module(*args, **kwargs) # type: ignore
    
    return transform_hkRNN(model)

def eval_model(
    model_fun: Callable[[], hk.RNNCore],
    params: hk.Params,
    xs: jax.Array,
) -> Tuple[jax.Array, jax.Array]:
    """Run an RNN with specified params and inputs. Track internal state.

    Args:
        model_fun: A Haiku function that defines a network architecture
        params: A set of params suitable for that network
        xs: A batch of inputs [timesteps, episodes, features] suitable for the model

    Returns:
        y_hats: Network outputs at each timestep
        states: Network states at each timestep
    """

    tfd = transform_hkRNN(model_fun)

    key = jax.random.PRNGKey(np.random.randint(2**32))
    y_hats, states = tfd.apply(params, key, xs)

    return y_hats, states

def make_param_metric_expLL(model: RNNtransformed, n_action: int):
    key = jax.random.PRNGKey(np.random.randint(2**32))
    def _metric(params, xs, ys):
        model_outputs, _ = model.apply(params, key, xs)
        normLL = BerLL_logit(ys[..., 0], model_outputs[:, :, :n_action], norm = True)
        exp_normLL = jnp.exp(normLL)
        return exp_normLL
    
    return _metric

def compute_log_likelihood(dataset: trainingDataset, model_fun: Callable, params) -> float:
    """calculate exponential of normalized log-likelihood
    assume the model output logits for each action
    """
    n_action = dataset.n_action
    xs, actual_choices = next(dataset)
    model_outputs, _ = eval_model(model_fun, params, xs)
    normLL = BerLL_logit(actual_choices[..., 0], model_outputs[:, :, :n_action], norm = True)
    exp_normLL = jnp.exp(normLL)
    return float(exp_normLL)

def make_compute_log_likeli(model_fun: Callable, n_action: int = 2) -> Callable[[Params, Inputs, Outputs], Loss]:

    @jax.jit
    def compute(params, xs, ys):
        model_outputs, _ = eval_model(model_fun, params, xs)
        normLL = BerLL_logit(ys[..., 0], model_outputs[:, :, :n_action], norm = True)
        exp_normLL = jnp.exp(normLL)
        return exp_normLL

    return compute    


def make_loss_fun(
        model_apply: Callable[[Params, RandomKey, Inputs], Tuple[Outputs, States]], 
        loss_type: str = 'categorical', 
        penalty_scale: float = 0, 
        beta_scale: float = 1
    ) -> Callable[[Params, RandomKey, Inputs, Outputs], Loss]:
    """_summary_

    Args:
        model_apply (Callable): _description_
        loss_type (str, optional): _description_. Defaults to 'categorical'.
        penalty_scale (float, optional): _description_. Defaults to 0.
        beta_scale (float, optional): _description_. Defaults to 1.

    Returns:
        loss_fun: (params, random_key, xs, ys) -> loss
    """

    if penalty_scale != 0 and loss_type != 'penalized_categorical':
        logging.warning(
            f'penalty_scale({penalty_scale}) is only meaningful when loss_type is "penalized_categorical", rather than "{loss_type}"'
        )

    def categorical_log_likelihood(
        labels: jax.Array, output_logits: jax.Array
    ) -> jax.Array:
        # Mask any errors for which label is negative
        mask = jnp.logical_not(labels < 0)
        log_probs = jax.nn.log_softmax(output_logits)
        if labels.shape[2] != 1:
            raise ValueError(
                'Categorical loss function requires targets to be of dimensionality'
                ' (n_timesteps, n_episodes, 1)'
            )
        one_hot_labels = jax.nn.one_hot(
            labels[:, :, 0].astype(jnp.int32), num_classes=output_logits.shape[-1]
        )
        log_liks = one_hot_labels * log_probs
        masked_log_liks = jnp.multiply(log_liks, mask)
        loss = -jnp.nansum(masked_log_liks)
        return loss

    def categorical_loss(
        params, random_key, xs: jax.Array, labels: jax.Array
    ) -> jax.Array:
        """compute loss as negative log-likelihood for categorical distribution"""
        output_logits, _ = model_apply(params, random_key, xs)
        loss = categorical_log_likelihood(labels, output_logits)
        return loss

    def penalized_categorical_loss(
        params, random_key, xs, targets
    ) -> jax.Array:
        """Treats the last two elements of the model outputs as penalty."""
        # (n_steps, n_episodes, n_targets)
        model_output, _ = model_apply(params, random_key, xs)
        output_logits = model_output[:, :, :-2]
        penalties = model_output[:, :, -2:]     
        penalties = penalties.at[:, :, 0].multiply(beta_scale)

        penalty = jnp.sum(penalties)  # ()
        loss = (
            categorical_log_likelihood(targets, output_logits)
            + penalty_scale * penalty
        )
        return loss

    losses = {
        'categorical': categorical_loss,
        'penalized_categorical': penalized_categorical_loss
    }
    compute_loss = losses[loss_type]
    # compute_loss.__doc__ = """the function to compute loss
    #     Args:
    #         params: the model parameters
    #         random_key: random key
    # """

    return compute_loss


def make_train_step(
    model_apply: Callable[[Params, RandomKey, Inputs], Tuple[Outputs, States]],
    optimizer: optax.GradientTransformation = optax.adam(1e-3),
    penalty_scale: float = 0,
    beta_scale: float = 1,
    loss_type: str = 'categorical',
) -> TrainStepFun:
    """Trains a model for a fixed number of steps.

    Args:
        model_fun: A function that, when called, returns a Haiku RNN object
        dataset: A DatasetRNN, containing the data you wish to train on
        optimizer: The optimizer you'd like to use to train the network
        random_key: A jax random key, to be used in initializing the network
        opt_state: An optimzier state suitable for opt.
            If not specified, will initialize a new optimizer from scratch.
        params:  A set of parameters suitable for the network given by make_network.
            If not specified, will begin training a network from scratch.
        n_steps: An integer giving the number of steps you'd like to train for
            (default=1000)
        penalty_scale: scalar weight applied to bottleneck penalty, only apply to loss_fun = {'penalized_categorical'} (default = 0)
        beta_scale: scale MLP update rules penalty relative to global bottlenecks, only apply to loss_fun = {'penalized_categorical'} (default = 1)
        loss_fun: string specifying type of loss function (default='categorical')

    Returns:
        params: Trained parameters
        opt_state: Optimizer state at the end of training
        losses: Losses on both datasets
    """

    compute_loss = make_loss_fun(model_apply, 
        loss_type=loss_type, penalty_scale=penalty_scale, beta_scale=beta_scale)
    
    value_grad_loss = jax.value_and_grad(compute_loss, argnums=0)

    # Define what it means to train a single step
    def train_step(
        params: Params, random_key: RandomKey, opt_state: optax.OptState, xs:Inputs, ys:Outputs
    ) -> Tuple[float, Any, Any]:
        loss, grads = value_grad_loss(
            params, random_key, xs, ys
        )
        grads, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, grads)
        return loss, params, opt_state

    return train_step

def make_train_step_with_loss_fun(
        model_apply: Callable[[Params, RandomKey, Inputs], Tuple[Outputs, States]],
        optimizer: optax.GradientTransformation,
        make_loss_fun: Callable[..., LossFun],
        **kwargs
        ):
    loss_fun = make_loss_fun(model_apply, **kwargs)
    value_grad_loss = jax.value_and_grad(loss_fun, argnums=0)

    # Define what it means to train a single step
    def train_step(
        params: Params, random_key: RandomKey, opt_state: optax.OptState, xs:Inputs, ys:Outputs
    ) -> Tuple[float, Any, Any]:
        loss, grads = value_grad_loss(
            params, random_key, xs, ys
        )
        grads, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, grads)
        return loss, params, opt_state

    return train_step

def train_with_step_fun(
        step_fun: Callable,
        dataset: trainingDataset,
        params: Params,
        opt_state: optax.OptState,
        random_key: RandomKey,
        n_steps: int = 10,
    ):

    # Train the network!
    training_loss = []
    t_start = time.time()

    for step in jnp.arange(n_steps):
        random_key, key_i = jax.random.split(random_key, 2) # type: ignore
        # Train on training data
        xs, ys = next(dataset)

        loss, params, opt_state = step_fun(params, key_i, opt_state, xs, ys)

        if step % 10 == 9:
            # Check if anything has become NaN that should not be NaN
            if nan_in_dict(params):
                print(params)
                raise ValueError('NaN in params')
            if len(training_loss) > 0 and np.isnan(training_loss[-1]):
                raise ValueError('NaN in loss')

            # Log every 10th step
            training_loss.append(float(loss)) # type: ignore
            print((f'\rStep {step + 1} of {n_steps}; '
                   f'Loss: {loss:.4e}. '
                   f'(Time: {time.time()-t_start:.1f}s)'), end='')

    losses = {
        'training_loss': np.array(training_loss),
    }

    return params, opt_state, losses # type: ignore

def train_model(
    model_fun: Callable[[], hk.RNNCore],
    dataset: trainingDataset,
    optimizer: optax.GradientTransformation = optax.adam(1e-3),
    random_key: Optional[jax.Array] = None,
    opt_state: Optional[optax.OptState] = None,
    params: Optional[Params] = None,
    n_steps: int = 1000,
    penalty_scale: float = 0,
    beta_scale: float = 1,
    loss_fun: str = 'categorical',
) -> Tuple[hk.Params, optax.OptState, Dict[str, np.ndarray]]:
    """Trains a model for a fixed number of steps.

    Args:
      model_fun: A function that, when called, returns a Haiku RNN object
      dataset: A DatasetRNN, containing the data you wish to train on
      optimizer: The optimizer you'd like to use to train the network
      random_key: A jax random key, to be used in initializing the network
      opt_state: An optimzier state suitable for opt.
        If not specified, will initialize a new optimizer from scratch.
      params:  A set of parameters suitable for the network given by make_network.
        If not specified, will begin training a network from scratch.
      n_steps: An integer giving the number of steps you'd like to train for
        (default=1000)
      penalty_scale: scalar weight applied to bottleneck penalty, only apply to loss_fun = {'penalized_categorical'} (default = 0)
      beta_scale: scale MLP update rules penalty relative to global bottlenecks, only apply to loss_fun = {'penalized_categorical'} (default = 1)
      loss_fun: string specifying type of loss function (default='categorical')

    Returns:
      params: Trained parameters
      opt_state: Optimizer state at the end of training
      losses: Losses on both datasets
    """
    # PARSE INPUTS
    n_steps = int(n_steps)
    sample_xs, _ = next(dataset)  # Get a sample input, for shape
    model = transform_hkRNN(model_fun)

    if random_key is None:
        random_key = jax.random.PRNGKey(0)
    # If params have not been supplied, start training from scratch
    if params is None:
        random_key, key1 = jax.random.split(random_key)
        params = model.init(key1, sample_xs)
    # It an optimizer state has not been supplied, start optimizer from scratch
    if opt_state is None:
        opt_state = optimizer.init(params)
    
    train_step = jax.jit(
        make_train_step(model.apply, optimizer, penalty_scale, beta_scale, loss_type=loss_fun)
    )

    # Train the network!
    training_loss = []
    t_start = time.time()

    for step in jnp.arange(n_steps):
        random_key, key_i = jax.random.split(random_key, 2) # type: ignore
        # Train on training data
        xs, ys = next(dataset)

        loss, params, opt_state = train_step(params, key_i, opt_state, xs, ys)

        if step % 10 == 9:
            # Check if anything has become NaN that should not be NaN
            if nan_in_dict(params):
                print(params)
                raise ValueError('NaN in params')
            if len(training_loss) > 0 and np.isnan(training_loss[-1]):
                raise ValueError('NaN in loss')

            # Log every 10th step
            training_loss.append(float(loss)) # type: ignore
            print((f'\rStep {step + 1} of {n_steps}; '
                   f'Loss: {loss:.4e}. '
                   f'(Time: {time.time()-t_start:.1f}s)'), end='')

    losses = {
        'training_loss': np.array(training_loss),
    }

    return params, opt_state, losses # type: ignore

def get_initial_state(make_network: Callable[[], hk.RNNCore],
                      params: Optional[Any] = None) -> Any:
    """Get the default initial state for a network architecture.

    Args:
      make_network: A Haiku function that defines a network architecture
      params: Optional parameters for the Hk function. If not passed, will init
        new parameters. For many models this will not affect initial state

    Returns:
      initial_state: An initial state from that network
    """

    # The logic below needs a jax randomy key and a sample input in order to work.
    # But neither of these will affect the initial network state, so its ok to
    # generate throwaways
    random_key = jax.random.PRNGKey(np.random.randint(2**32))

    def unroll_network():
        core = make_network()
        state = core.initial_state(batch_size=1)

        return state

    model = hk.transform(unroll_network)

    if params is None:
        params = model.init(random_key)

    initial_state = model.apply(params, random_key)

    return initial_state




def evo_state(make_network: Callable[[], hk.RNNCore], params: hk.Params) -> Callable[[Inputs, States], Tuple[Outputs, States]]:
    """create a jit-ed function that can evolve network state by one step (seeing one observation)

    Returns:
        f(xs: (batch_sz, obs_sz), state: (batch_sz, latent_sz)) -> output: (batch_sz, out_sz), latents: (batch_sz, latent_sz)
        if one of the input has batch_sz as 1, while the other > 1, 
        they will be broadcasted to larger one
        both input arguments can also be 1D array if there's only one batch
    """

    def step_sub(xs, state):
        core = make_network()
        y_hat, new_state = core(xs, state)
        return y_hat, new_state

    model = hk.transform(step_sub)
    key = jax.random.PRNGKey(0)

    def m_apply(xs: jax.Array, state: jax.Array):
        if len(xs.shape) == 1:
            xs = jnp.reshape(xs, (1, -1))
        
        if len(state.shape) == 1:
            state = jnp.reshape(state, (1, -1))
        
        batch_sz = jnp.broadcast_shapes((xs.shape[0],), (state.shape[0],))[0]

        # broadcast first dim size to batch_sz, leave the other dims untouched
        xs = jnp.broadcast_to(xs, (batch_sz, *xs.shape[1:])) 
        state = jnp.broadcast_to(state, (batch_sz, *state.shape[1:]))

        return model.apply(params, key, xs, state)

    return jax.jit(m_apply)

def hk2flax(par: dict):
    """convert single layer of haiku network parameter container to flax.nn's
    'w' -> 'kernel'
    'b' -> 'bias'

    Args:
        par (dict): _description_

    Returns:
        _type_: _description_
    """
    par_ = deepcopy(par)
    par_['kernel'] = par_.pop('w')
    par_['bias'] = par_.pop('b')

    return par_

def hk2flaxSeq(pars: List[dict], layers: Optional[List[Callable]] = None):
    """convert a sequence of haiku params to sequential layers of flax.nn
    
    if provided a list of functions for each layer, the numbering of the parameters "layers_" will skip layers that is not nn.Module (e.g. activation function)
    """
    if layers is None:
        names = [f'layers_{i}' for i in range(len(pars))]
    else:
        names = [f'layers_{i}' for i in range(len(layers)) if isinstance(layers[i], nn.Module)]
    
    vals = map(hk2flax, pars)
    return dict(zip(names, list(vals)))


def _get_MLP_sorted_lyr_names(params, prefix) -> List[str]:
    # extract layer names from params with the given prefix
    lyr_names = [key for key in params if key.startswith(prefix)]

    # given the fact that layer name of haiku MLP ends with layer index
    lyr_names = sorted(lyr_names, key=lambda s: int(re.findall(r"\d+$", s)[0]))

    return lyr_names

def _get_MLP_lyr_seq(params, prefix, activation_func) -> List[Callable]:
    """get the sequence of calculation for disRNN choice MLP or update MLP 

    no final activation 

    The linear function is represented with flax.linen.Dense

    Args:
        params (_type_): _description_
        prefix (_type_): _description_
        activation_func (_type_): _description_

    Returns:
        List[Callable]: a sequence of linear and activation functions
    """
    lyr_names = _get_MLP_sorted_lyr_names(params, prefix)

    # get each layer's output size from the parameter
    lyr_out_sz = [params[lyr]['w'].shape[1] for lyr in lyr_names]

    _lyrs = [[nn.Dense(out_sz), activation_func] for out_sz in lyr_out_sz]
    # flatten the above list, and remove the final activation function to get "lyrs"
    lyrs: List[Callable] = sum(_lyrs, [])[:-1]

    return lyrs



def get_haiku_static_attrs(make_network:Callable[[], hk.Module], attr_names:Union[List[str], str]) -> dict:
    """obtain static attributes of a haiku module

    static means the attributes that does not depend on the value of parameters

    Args:
        make_network (_type_): the haiku function that returns a initialized haiku module
        attr_names (List[str]): a list of attribute names to obtain

    Returns:
        dict: a dictionary of the attributes keyed by attribute names
    """

    if isinstance(attr_names, str):
        attr_names = [attr_names]

    def tmp():
        core = make_network()
        return {an: getattr(core, an) for an in attr_names}

    initf, f = hk.transform(tmp)
    params = initf(None)

    return f(params, None)

def has_haiku_static_attrs(make_network:Callable[[], hk.Module], attr_names:Union[List[str], str]) -> bool:
    """check if the haiku module has attributes

    static means the attributes that does not depend on the value of parameters

    Args:
        make_network (_type_): the haiku function that returns a initialized haiku module
        attr_names (List[str]): a list of attribute names to obtain

    Returns:
        dict: a dictionary of the attributes keyed by attribute names
    """

    if isinstance(attr_names, str):
        attr_names = [attr_names]

    def tmp():
        core = make_network()
        return all([hasattr(core, n) for n in attr_names])

    initf, f = hk.transform(tmp)
    params = initf(None)

    return f(params, None)


def patched_forward(
        make_network: Callable[[], patchable_hkRNNCore], 
        params: Params, 
        patch_state_ids: List[int], 
        xs: Inputs, 
        exo_state: States) -> Tuple[Outputs, States]:
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
    xs_batch_sz = xs.shape[1]
    xs_feat_sz = xs.shape[2]
    # shape (N_trial, N_sess, N_obs)

    assert has_haiku_static_attrs(make_network, ['set_patch_state_ids', 'step_with_exo_state']), "the haiku module definition is not compatiable with `patchable_hkRNNCore`"
    assert exo_state.shape[-1] == len(patch_state_ids), f"the shape[-1] of exo_state ({exo_state.shape[-1]}) should match length of patch_state_ids ({len(patch_state_ids)})"

    # === prepare functions

    def get_init(batch_size):
        core = make_network()
        state = core.initial_state(batch_size)
        return state

    def patchable_step(xs: Array, state: Array, exo_state: Array):
        core = make_network()
        core.set_patch_state_ids(patch_state_ids)
        y_hat, new_state = core.step_with_exo_state(xs, state, exo_state)
        return y_hat, new_state

    _, tf_patchable_step = hk.transform(patchable_step)
    _, tf_get_init = hk.transform(get_init)
    # tf_patchable_step = jax.jit(tf_patchable_step)
    # lax.scan will jit the function

    rand_key = jax.random.PRNGKey(0)
    def step_for_scan(carry, x):
        inp = x[..., :xs_feat_sz]
        exo_state = x[..., xs_feat_sz:]
        y, new_state = tf_patchable_step(params, rand_key, inp, carry, exo_state)
        return new_state, (y, new_state) # to match lax.scan required function signature
    
    # === prepare inputs 

    if np.prod(exo_state.shape) == 0: # empty exo_state array
        xs_with_exo = xs
    else:
        if len(exo_state.shape) == 1:
            exo_state = exo_state.reshape(1,1,-1)
        elif len(exo_state.shape) == 2:
            exo_state = exo_state[jnp.newaxis, ...]
        exo_state = jnp.broadcast_to(exo_state, (*xs.shape[:2], exo_state.shape[-1]))
        xs_with_exo = jnp.concatenate((xs, exo_state), axis=2)

    lat_init = tf_get_init(params, jax.random.PRNGKey(0), xs_batch_sz)
    
    carry, (ys, states) = jax.lax.scan(step_for_scan, lat_init, xs_with_exo)

    return ys, states