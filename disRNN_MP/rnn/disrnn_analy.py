""" a low-level functional interface for analyzing disRNN models. """

from operator import mul
from typing import Callable, List, Optional, Union
from pathlib import Path
import re
from copy import copy
from typing_extensions import deprecated

import numpy as np
import jax
import jax.numpy as jnp
import haiku as hk
import flax.linen as nn
import polars as pl
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt

from .disrnn import _get_disrnn_prefix
from .utils import hk2flaxSeq, eval_model, evo_state, get_initial_state, _get_MLP_lyr_seq, _get_MLP_sorted_lyr_names, get_haiku_static_attrs
from .train import RNNtraining
from .disrnn import sort_bottlenecks


def disRNN_update_par_func(params: hk.Params, make_network: Callable, lat_id: int, include_upd_mlp_gate: bool = False) -> Callable:
    """reconstruct update MLP function for a latent

    1-based latent index

    Args:
        params (hk.Params): model parameter
        make_network (Callable): model haiku definition
        lat_id (int): 1-based latent number

    Raises:
        ValueError: if latent #`lat_id` does not exist

    Returns:
        Callable: (obs, latents) -> (learning rate, target)
    """ 

    _prefix = _get_disrnn_prefix(params)

    act_f = get_haiku_static_attrs(make_network, '_activation')['_activation']

    if pl.Series(params.keys()).str.contains("/~latent_update/").any():
        sub_prefix = "/~latent_update"
    else:
        sub_prefix = ""

    if include_upd_mlp_gate:
        multiplier = params[_prefix]['update_mlp_gates'][:, lat_id-1] # (N_obs + N_latent, )
    else:
        multiplier = jnp.ones_like(params[_prefix]['update_mlp_gates'][:, lat_id-1])

    # update MLP
    lyr_names = _get_MLP_sorted_lyr_names(params, _prefix + f"{sub_prefix}/latent{lat_id}_update_MLP/~")
    lyrs = _get_MLP_lyr_seq(params, _prefix + f"{sub_prefix}/latent{lat_id}_update_MLP/~", act_f)

    if len(lyr_names) == 0:
        raise ValueError(f"cannot find parameters for the latent's update MLP with specified latent id: {lat_id}")

    target_lyrs = copy(lyrs)
    target_lyr_names = copy(lyr_names)
    # add the final linear layer for target
    upd_target_lyr_name = _prefix + f'{sub_prefix}/latent{lat_id}_update_target'
    target_lyr_names.append(upd_target_lyr_name)
    upd_target_out_sz = params[upd_target_lyr_name]['w'].shape[1]
    target_lyrs.append(nn.Dense(upd_target_out_sz))

    lr_lyrs = copy(lyrs)
    lr_lyr_names = copy(lyr_names)
    # add the final linear layer for lr
    upd_lr_lyr_name = _prefix + f'{sub_prefix}/latent{lat_id}_update_lr'
    lr_lyr_names.append(upd_lr_lyr_name)
    upd_lr_out_sz = params[upd_lr_lyr_name]['w'].shape[1]
    lr_lyrs.append(nn.Dense(upd_lr_out_sz))


    MLP_target_vars = {'params': hk2flaxSeq(
        [params[lyr_name] for lyr_name in target_lyr_names], target_lyrs)} # type: ignore

    MLP_lr_vars = {'params': hk2flaxSeq(
        [params[lyr_name] for lyr_name in lr_lyr_names], lr_lyrs)} # type: ignore

    def updMLP_target(obs, latents) -> jax.Array:
        """updateMLP target output
        latents is 1D vector
        """
        out = nn.Sequential(target_lyrs).apply(MLP_target_vars, jnp.hstack((jnp.array(obs), jnp.array(latents))) * multiplier)[1] 
        return out # type: ignore

    def updMLP_lr(obs, latents) -> jax.Array:
        """updateMLP learning rate output
        latents is 1D vector
        """
        return jax.nn.sigmoid(nn.Sequential(lr_lyrs).apply(MLP_lr_vars, jnp.hstack((jnp.array(obs), jnp.array(latents))) * multiplier)[1])

    def updMLP_lr_target(obs, latents):
        return jnp.hstack((updMLP_lr(obs, latents), updMLP_target(obs, latents)))
    
    return updMLP_lr_target


def disRNN_update_pars_func(params: hk.Params, make_network: Callable, include_upd_mlp_gate: bool = False):
    """reconstruct update MLP function for latents

    Args:
        params (hk.Params): model parameter
        make_network (Callable): model haiku definition

    Returns:
        Callable: (obs, latents) -> lr_target of size (N_latent, 2): 1st column are learning rates, 2nd column are targets
    """

    lat_sz = get_haiku_static_attrs(make_network, '_latent_size')['_latent_size']

    funcs = [disRNN_update_par_func(params, make_network, latID, include_upd_mlp_gate=include_upd_mlp_gate) for latID in range(1, lat_sz+1)]
    def updMLPs(obs, latents):
        res = jnp.zeros((lat_sz, 2))
        for i in np.arange(lat_sz):
            res = res.at[i,:].set(funcs[i](obs, latents))
        return res # (latent_size, updMLP ouput size), output order: lr, target
    
    return updMLPs

@deprecated("use `get_haiku_static_attrs(make_network, '_activation')['_activation']` instead")
def disRNN_activation_func(make_network: Callable, params: Optional[hk.Params] = None) -> Callable:
    """get the activation function from a disRNN model haiku

    readout the instance's `_activation` attribute

    Args:
        make_network (Callable): model haiku for disRNN
        params (Optional[hk.Params], optional): when parameter is not needed for determining activation function, this argument can be left empty. Defaults to None.

    Returns:
        Callable: _description_
    """

    def _get_activation_func():
            core = make_network()
            return core._activation
    _init, get_activation_func = hk.transform(_get_activation_func)

    if params is None:
        params = _init(jax.random.PRNGKey(0))

    activation = get_activation_func(params, None)

    return activation

def disRNN_choiceMLP_func(
        params: hk.Params, 
        make_network: Optional[Callable] = None, 
        activate_func: Optional[Callable] = jax.nn.relu,
        convert_pR: bool = False):
    """generate a function to calculate the function: latents -> chMLP output or p(R)

    reassemble the choice_MLP and choice_logit network from model parameters

    Args:
        params (hk.Params): haiku module parameters
        activate_func (Callable): activation function for the MLP network. Default to jax.nn.relu
        convert_pR (bool): whether make a function that output model predicted p(R). Default to False, which keep the model output as it is

    Returns:
        function(latents) -> p(R)
            input and output are vectors of same length
    """

    _prefix = _get_disrnn_prefix(params)

    if make_network is None:
        activation = activate_func
    else:
        # getting activation function from make_network
        activation = get_haiku_static_attrs(make_network, '_activation')['_activation']

    if pl.Series(params.keys()).str.contains("/~choice_selection/").any():
        sub_prefix = "/~choice_selection"
    else:
        sub_prefix = ""

    lyr_names = _get_MLP_sorted_lyr_names(params, _prefix + f'{sub_prefix}/choice_MLP/~')
    lyrs = _get_MLP_lyr_seq(params, _prefix + f'{sub_prefix}/choice_MLP/~', activation)

    # add the final logit layer
    lyr_names.append(_prefix + f'{sub_prefix}/choice_logits')
    ch_logit_out_sz = params[_prefix + f'{sub_prefix}/choice_logits']['w'].shape[1]
    lyrs.append(nn.Dense(ch_logit_out_sz))

    if convert_pR:
        # add additional layer to convert logits to p(R)
        lyrs.append(jax.nn.softmax) # logit -> p(R)

    MLP_vars = {'params': hk2flaxSeq(
        [params[lyr_name] for lyr_name in lyr_names], lyrs)} # type: ignore
    
    if 'choice_mlp_gates' in params[_prefix]:
        lat_mult = params[_prefix]['choice_mlp_gates']
    else:
        lat_mult = 1
    
    def ch_MLP(latents):
            """choice selection function: latents -> logits for each choice
            latents is 1D vector
            """
            return nn.Sequential(lyrs).apply(MLP_vars, latents * lat_mult)
    
    def ch_MLP_pR(latents):
            """choice selection function: latents -> p(R)
            latents is 1D vector
            """
            return nn.Sequential(lyrs).apply(MLP_vars, latents * lat_mult)[1]

    if convert_pR:
        return ch_MLP_pR
    else:
        return ch_MLP


def choice_selection_gradient_func(params: hk.Params):
    """generate a function to calculate the gradient of function latents -> p(R)

    reassemble the choice_MLP and choice_logit network from model parameters

    Args:
        params (hk.Params): haiku module parameters

    Returns:
        function(latents) -> grad 
            input and output are vectors of same length
    """

    ch_MLP = disRNN_choiceMLP_func(params, convert_pR = True)

    return jax.grad(ch_MLP)

def make_MP_phase_df(
        make_network: Callable[[], hk.RNNCore], 
        params: hk.Params,
        model_in: np.ndarray, 
    ):
    """make the dataframe that is used for making phase space figures

    for each row:
    - "chs", "rews" are model input for the current timestamp
    - "latents": updated latents after observing the inputs

    Args:
        train_data (RNNtraining): training data
        model_in (Optional[np.ndarray], optional): inputs for evaluate the model (N_trial, N_sess, N_features). Defaults to None.

    Returns:
        pl.Dataframe: the dataframe
    """

    # model prediction and state
    res = eval_model(make_network, params, model_in) # type: ignore

    # serialize latent states: (N_states, latent_sz)
    obs_latents = res[1].reshape(-1, res[1].shape[2])

    # the gradient of choice selection function at each observed latent
    latent_grads = jax.vmap(choice_selection_gradient_func(
        params), in_axes=0, out_axes=0)(obs_latents)

    stepf_ = evo_state(make_network, params) # type: ignore
    next_states = jax.vmap(stepf_, in_axes=(0, None), out_axes=0)(
        jnp.expand_dims(jnp.array([[0, 0], [0, 1], [1, 0], [1, 1]]), 1), obs_latents)[1]
    # sz: (4, N_states, latent_sz)

    # serialize and split model input to `chs` and `rews`
    chs, rews = np.hsplit(model_in.reshape(-1, model_in.shape[2]), 2)
    # which model prediction prefer right side

    if isinstance(res[0], dict) and 'prediction' in res[0]:
        pred_logits = res[0]['prediction']
    else:
        pred_logits = res[0]
    
    # predR = (pred_logits[:, :, 1] > pred_logits[:, :, 0]).reshape(-1)
    predR = jax.nn.softmax(pred_logits, axis=2)[:,:,1].reshape(-1)
    
    # generate sess_id and tri_id
    sessid = np.repeat(np.arange(model_in.shape[1]).reshape(1,-1), model_in.shape[0], axis=0)
    triid = np.repeat(np.arange(model_in.shape[0]).reshape(-1,1), model_in.shape[1], axis=1)

    df = pl.DataFrame({
        'sess_id': sessid.reshape(-1),
        'tri_id': triid.reshape(-1),
        'chs': np.array(chs.flatten()),
        'rews': np.array(rews.flatten()),
        'predR': np.array(predR),
        'latents': np.array(obs_latents),
        'latent_grads': np.array(latent_grads),
        'next_states': np.array(jnp.swapaxes(next_states, 0, 1)),
    })

    df = df.with_columns(
        rew_c=pl.col('rews').replace_strict({0: '-', 1: '+'}, default=""),
        ch_c=pl.col('chs').replace_strict({0: 'L', 1: 'R'}, default="")
    ).with_columns(
        ch_rew=pl.col('ch_c') + pl.col('rew_c')
    )

    rdf = df.to_pandas()
    rdf['next_states'] = [*df['next_states']]

    return rdf


def make_MP_phase_df_fromRNNtraining(train_data: RNNtraining, model_in: Optional[np.ndarray] = None, params: Optional[hk.Params] = None):
    """make the dataframe that is used for making phase space figures

    Args:
        train_data (RNNtraining): training data
        model_in (Optional[np.ndarray], optional): inputs for evaluate the model (N_trial, N_sess, N_features). Defaults to None.

    Returns:
        pl.Dataframe: the dataframe
    """

    if model_in is None:
        model_in = np.array(next(train_data.datasets[0])[0])

    if params is None:
        if train_data.params is None:
            raise ValueError("train_data.params is empty")
        else:
            params = train_data.params

    # model prediction and state
    res = eval_model(train_data.eval_model, params, model_in) # type: ignore

    # serialize latent states: (N_states, latent_sz)
    obs_latents = res[1].reshape(-1, res[1].shape[2])

    # the gradient of choice selection function at each observed latent
    latent_grads = jax.vmap(choice_selection_gradient_func(
        params), in_axes=0, out_axes=0)(obs_latents)

    stepf_ = evo_state(train_data.eval_model, params) # type: ignore
    next_states = jax.vmap(stepf_, in_axes=(0, None), out_axes=0)(
        jnp.expand_dims(jnp.array([[0, 0], [0, 1], [1, 0], [1, 1]]), 1), obs_latents)[1]
    # sz: (4, N_states, latent_sz)

    # serialize and split model input to `chs` and `rews`
    chs, rews = np.hsplit(model_in.reshape(-1, model_in.shape[2]), 2)
    # which model prediction prefer right side

    if isinstance(res[0], dict) and 'prediction' in res[0]:
        pred_logits = res[0]['prediction']
    else:
        pred_logits = res[0]
    
    predR = (pred_logits[:, :, 1] > pred_logits[:, :, 0]).reshape(-1)

    df = pl.DataFrame({
        'chs': chs.flatten(),
        'rews': rews.flatten(),
        'predR': np.array(predR),
        'latents': np.array(obs_latents),
        'latent_grads': np.array(latent_grads),
        'next_states': np.array(jnp.swapaxes(next_states, 0, 1)),
    })

    df = df.with_columns(
        rew_c=pl.col('rews').replace_strict({0: '-', 1: '+'}, default=""),
        ch_c=pl.col('chs').replace_strict({0: 'L', 1: 'R'}, default="")
    ).with_columns(
        ch_rew=pl.col('ch_c') + pl.col('rew_c')
    )

    return df


def colored_latent_scatter(df: pl.DataFrame, latents: List[int], color: str) -> go.Figure:
    """make a plotly scatter figure of latent states with color

    Args:
        df (pl.DataFrame): the output of `make_MP_phase_df`
        latents (List[int]): 0-based index of latent
        color (str): _description_

    Returns:
        go.Figure: _description_
    """

    assert len(latents) >= 2 and len(latents) <= 3

    
    if len(latents) == 2:
        pltd = df.select(
            x=pl.col('latents').list[latents[0]],
            y=pl.col('latents').list[latents[1]],
            color=pl.col(color),
        )

        fig = px.scatter(pltd, x='x', y='y', color='color',
                         labels={
                             'x': f"latent_{latents[0]+1}",
                             'y': f"latent_{latents[1]+1}",
                             'color': color
                         })
        
    else:
        pltd = df.select(
            x=pl.col('latents').list[latents[0]],
            y=pl.col('latents').list[latents[1]],
            z=pl.col('latents').list[latents[2]],
            color=pl.col(color),
        )

        fig = px.scatter_3d(pltd, x='x', y='y', z='z', color='color',
                            labels={
                                'x': f"latent_{latents[0]+1}",
                                'y': f"latent_{latents[1]+1}",
                                'z': f"latent_{latents[2]+1}",
                                'color': color
                            })
        

    fig.update_traces(marker_size=2, opacity=.4,)
    # since marker size is small, make the legend icon not depend on marker size
    fig.update_layout(legend={'itemsizing': 'constant'})

    return fig

def MP_phase_dynamic_figure(df: pl.DataFrame, latents: List[int], to_html: Union[Path, str]):
    """generate phase space figure
    scatter plot of 2 selected latents as x and y with color

    df: dataframe with column `latents`, `next_states`, color(str)
    """

    if not isinstance(to_html, str) and not isinstance(to_html, Path):
        raise ValueError('to_html argument has to be string or Path')

    color = 'ch_rew'
    fig = colored_latent_scatter(df, latents, color)
    fig.update_traces(hoverinfo="none", hovertemplate=None)

    # next_states = np.stack(df['next_states'], axis = 0)[:, :, latents]
    # pltd['color'].unique()
    for l1cond in ['L-', 'L+', 'R-', 'R+']:
        fig.update_traces({
            'customdata': np.stack(df.filter(df[color] == l1cond)['next_states'], axis=0)[:, :, latents] # type: ignore
        },
            selector={'name': l1cond})
    
    fd = Path(__file__).parent
    if len(latents) == 2:
        with open(fd / '2d_arrow_annotation.js', 'r') as f:
            js_callback = f.read()
    else:
        with open(fd / '3d_dynamic_callback.js', 'r') as f:
            js_callback = f.read()

    fig.write_html(to_html, post_script=js_callback)

    return fig


def plot_update_1d(step_fun: Callable, refer_state, unit_i: int, observations, titles):
    """plot the update function for state `unit_i` respectively for each `observations`

    Args:
        step_fun (Callable): the step function of the network 
            (xs: (batch_sz, obs_sz), state: (batch_sz, latent_sz)) -> output: (batch_sz, out_sz), latents: (batch_sz, latent_sz)
        refer_state (1D array): the reference state. Latents except `unit_i` will not change from it
        unit_i (int): index of the latent to be plotted
        observations (_type_): a list of observations. there will be a subplot for each observation
        titles (_type_): the title for each subplot

    Returns:
        _type_: _description_
    """
    lim = 3
    state_bins = np.linspace(-lim, lim, 20)
    colormap = plt.get_cmap('viridis', 3) # type: ignore
    colors = colormap.colors

    fig, ax = plt.subplots(
        1, len(observations), figsize=(len(observations) * 4, 5.5)
    )
    plt.subplot(1, len(observations), 1)
    plt.ylabel('Updated Activity')

    for observation_i in range(len(observations)):
        # for each subplot
        observation = observations[observation_i]

        # states: (N_bins, N_latents)
        states = jnp.tile(refer_state, (state_bins.shape[0], 1))
        states = states.at[:, unit_i].set(state_bins)
        _, next_states = step_fun(
                jnp.array(observation), states
            )

        plt.subplot(1, len(observations), observation_i + 1)

        plt.plot((-3, 3), (-3, 3), '--', color='grey')
        plt.plot((-3, 3), (0, 0), color='black')
        plt.plot((0, 0), (-3, 3), color='black')
        
        plt.plot(state_bins, next_states[:, unit_i], color=colors[1])

        plt.title(titles[observation_i])
        plt.xlim(-lim, lim)
        plt.ylim(-lim, lim)
        plt.xlabel('Previous Activity')

        if isinstance(ax, np.ndarray):
            ax[observation_i].set_aspect('equal')
        else:
            ax.set_aspect('equal')
    return fig


def plot_update_2d(step_fun: Callable, refer_state, unit_i, unit_input, observations, titles):
    lim = 3

    state_bins = np.linspace(-lim, lim, 20)
    colormap = plt.get_cmap('viridis', len(state_bins))
    colors = colormap.colors

    fig, ax = plt.subplots(
        1, len(observations), figsize=(len(observations) * 2 + 10, 5.5)
    )
    plt.subplot(1, len(observations), 1)
    plt.ylabel('Updated Latent ' + str(unit_i + 1) + ' Activity')

    for observation_i in range(len(observations)):
        observation = observations[observation_i]
        plt.subplot(1, len(observations), observation_i + 1)

        plt.plot((-3, 3), (-3, 3), '--', color='grey')
        plt.plot((-3, 3), (0, 0), color='black')
        plt.plot((0, 0), (-3, 3), color='black')

        for si_i in np.arange(len(state_bins)):
            # states: (N_bins, N_latents)
            states = jnp.tile(refer_state, (state_bins.shape[0], 1))
            states = states.at[:, unit_i].set(state_bins)
            states = states.at[:, unit_input].set(state_bins[si_i])
            _, next_states = step_fun(
                    jnp.array(observation), states
                )
            
            # delta_states = np.zeros(shape=(len(state_bins), 1))
            # for s_i in np.arange(len(state_bins)):
            #     state = refer_state
            #     state[0, unit_i] = state_bins[s_i]
            #     state[0, unit_input] = state_bins[si_i]
            #     _, next_state = step_hk(params, key, observation, state)
            #     next_state = np.array(next_state)
            #     delta_states[s_i] = next_state[0, unit_i]

            # plt.plot(state_bins, delta_states, color=colors[si_i])
            plt.plot(state_bins, next_states[:, unit_i], color=colors[si_i])

        plt.title(titles[observation_i])
        plt.xlim(-lim, lim)
        plt.ylim(-lim, lim)
        plt.xlabel('Latent ' + str(unit_i + 1) + ' Activity')

        if isinstance(ax, np.ndarray):
            ax[observation_i].set_aspect('equal')
        else:
            ax.set_aspect('equal')
    return fig


def plot_update_rules(params, make_network):
    """Generates visualizations of the update ruled of a disRNN.

    right now this only works with the case where observation is of size 2 (choice and reward only)
    """

    params_disrnn = params[_get_disrnn_prefix(params)]

    # def step(xs, state):
    #     core = make_network()
    #     output, new_state = core(jnp.expand_dims(jnp.array(xs), axis=0), state)
    #     return output, new_state

    step_fun = evo_state(make_network, params)

    initial_state = np.array(get_initial_state(make_network))
    reference_state = np.zeros(initial_state.shape)

    latent_sigmas = 2*jax.nn.sigmoid(
        jnp.array(params_disrnn['latent_sigmas_unsquashed'])
        )
    update_sigmas = 2*jax.nn.sigmoid(
        np.transpose(
            params_disrnn['update_mlp_sigmas_unsquashed']
            )
        )
    latent_order = np.argsort(
        params_disrnn['latent_sigmas_unsquashed']
        )
    
    obs_dim = update_sigmas.shape[1] - update_sigmas.shape[0]

    figs = []

    # Loop over latents. Plot update rules
    for latent_i in latent_order:
        # If this latent's bottleneck is open
        if latent_sigmas[latent_i] < 0.5:
            # Which of its input bottlenecks are open?
            update_mlp_inputs = np.argwhere(update_sigmas[latent_i] < 0.9)
            choice_sensitive = np.any(update_mlp_inputs == 0)
            reward_sensitive = np.any(update_mlp_inputs == 1)
            # Choose which observations to use based on input bottlenecks
            if choice_sensitive and reward_sensitive:
                observations = ([0, 0], [0, 1], [1, 0], [1, 1])
                titles = ('Left, Unrewarded',
                        'Left, Rewarded',
                        'Right, Unrewarded',
                        'Right, Rewarded')
            elif choice_sensitive:
                observations = ([0, 0], [1, 0])
                titles = ('Choose Left', 'Choose Right')
            elif reward_sensitive:
                observations = ([0, 0], [0, 1])
                titles = ('Unreward', 'Rewarded')
            else:
                observations = ([0, 0],)
                titles = ('All Trials',)
            # Choose whether to condition on other latent values
            latent_sensitive = update_mlp_inputs[update_mlp_inputs > (obs_dim-1)] - obs_dim
            # Doesn't count if it depends on itself (this'll be shown no matter what)
            latent_sensitive = np.delete(
                latent_sensitive, latent_sensitive == latent_i
            )
            if not latent_sensitive.size:  # Depends on no other latents
                fig = plot_update_1d(step_fun, reference_state, latent_i, observations, titles)
            else:  # It depends on latents other than itself.
                fig = plot_update_2d(
                    step_fun, reference_state,
                    latent_i,
                    latent_sensitive[np.argmax(latent_sensitive)],
                    observations,
                    titles,
                )
            
            fig.suptitle(f"latent {latent_i + 1}", y = 0.95)
            if len(latent_sensitive) > 1:
                print(
                    'WARNING: This update rule depends on more than one '
                    + 'other latent. Plotting just one of them'
                )

            figs.append(fig)

    return figs




def plot_bottlenecks(params, sort_latents=True, obs_names=None):
    """Plot the bottleneck sigmas from an disRNN."""
    
    latent_sigmas, update_sigmas, latent_sigma_order = sort_bottlenecks(params, sort_latents=sort_latents)

    latent_dim = latent_sigmas.shape[0]
    obs_dim = update_sigmas.shape[1] - latent_dim

    if obs_names is None:
        if obs_dim == 2:
            obs_names = ['Choice', 'Reward']
        elif obs_dim == 5:
            obs_names = ['A', 'B', 'C', 'D', 'Reward']
        else: 
            obs_names = np.arange(1, obs_dim+1)

    latent_names = np.arange(1, latent_dim + 1)[latent_sigma_order]
    fig = plt.subplots(1, 2, figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(np.swapaxes([1 - latent_sigmas], 0, 1), cmap='Oranges')
    plt.clim(vmin=0, vmax=1)
    plt.yticks(ticks=range(latent_dim), labels=latent_names)
    plt.xticks(ticks=[])
    plt.ylabel('Latent #')
    plt.title('Latent Bottlenecks')

    plt.subplot(1, 2, 2)
    plt.imshow(1 - update_sigmas, cmap='Oranges')
    plt.clim(vmin=0, vmax=1)
    plt.colorbar()
    plt.yticks(ticks=range(latent_dim), labels=latent_names)
    xlabels = np.concatenate((np.array(obs_names), latent_names))
    plt.xticks(
        ticks=range(len(xlabels)),
        labels=xlabels,
        rotation='vertical',
    )
    plt.ylabel('Latent #')
    plt.title('Update MLP Bottlenecks')
    return fig