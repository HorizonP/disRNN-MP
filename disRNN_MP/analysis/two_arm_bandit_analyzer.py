
import logging
from typing import Dict, List, Literal
from functools import cached_property
import inspect

from matplotlib import pyplot as plt
import numpy as np
import jax
import jax.numpy as jnp
from jax.tree_util import tree_reduce
import pandas as pd
import xarray as xr
import seaborn as sns
from xarray.plot import FacetGrid as xr_FacetGrid

from disRNN_MP.typing import Array
from disRNN_MP.rnn.disrnn_analy import disRNN_choiceMLP_func, disRNN_update_par_func, make_MP_phase_df, plot_update_1d, plot_update_2d
from disRNN_MP.utils import ds2df
from .disrnn_analyzer import disRNN_model_analyzer
from disRNN_MP.rnn.utils import get_haiku_static_attrs, evo_state, get_initial_state, eval_model
from disRNN_MP.dataset import trainingDataset
from disRNN_MP.metrics import BerLL_logit, bic

def _plt_2d_updMLP_out(xda: xr.DataArray):
    """plot learning rate and target as a heatmap 
    x and y axis are input latents value
    1st facet row is learning rate, 2nd facet row is target
    observations mapped to columns
    """
    wxda = xda.unstack('grid')
    fg: xr_FacetGrid = wxda.plot(row='updMLP_out', col='obs', add_colorbar=False, ) # type: ignore
    for row in range(fg.axs.shape[0]):
        if row == 0:
            for col in range(fg.axs.shape[1]):
                quadmesh = fg.axs[row,col].collections[0]
                quadmesh.set_clim(vmin=0, vmax=1)
                quadmesh.set_cmap('viridis')
        fg.fig.colorbar(fg.axs[row,0].collections[0], ax = fg.axs[row,:])
    return fg

def _plt_1d_updMLP_out(xda: xr.DataArray, map_obs: Literal['hue', 'facet'] = 'hue'):
    wxda = xda.unstack('grid')
    
    if map_obs == 'hue':
        fg: xr_FacetGrid = wxda.plot(row='updMLP_out', hue='obs', sharey='row') # type: ignore
    elif map_obs == 'facet':
        fg: xr_FacetGrid = wxda.plot(row='updMLP_out', col='obs', sharey='row') # type: ignore
    else:
        raise ValueError(f'unknown argument {map_obs} for `map_obs`')

    return fg

def _plt_0d_updMLP_out(xda: xr.DataArray):
    fg = sns.FacetGrid(xda.to_dataframe('value').reset_index(), row='updMLP_out', )
    fg.map(sns.barplot, 'obs', 'value', 'obs', palette = "pastel")
    return fg

def _scale_plt_to_ori(fig, scale_factor):
    """scales every axes in the figure toward origin (bottom-left) by a `scale_factor`
    """
    for ax in fig.axes:
        pos = ax.get_position()  # Bbox object
        new_x0 = pos.x0 * scale_factor
        new_y0 = pos.y0 * scale_factor
        new_width = pos.width * scale_factor
        new_height = pos.height * scale_factor
        ax.set_position([new_x0, new_y0, new_width, new_height])
    
    return fig
    


class two_arm_bandit_analyzer(disRNN_model_analyzer):
    """analyze disRNN model trained on 2-arm bandit tasks
    assumptions: 
     (1) observation vector is of at least length 2: the 1st two features are choice and reward
     (2) only 2 possible actions for choice, coded as "L" and "R"
    """

    @classmethod
    def exmp_obs_inputs(cls, choice: bool, reward: bool, name_type: Literal['short', 'long'] = 'short'):
        """generate example inputs and description for each combination of inputs

        Args:
            choice (bool): whether use choice input
            reward (bool): whether use choice input
            name_type (Literal['short', 'long'], optional): _description_. Defaults to 'short'.

        Returns:
            _type_: _description_
        """
        obsD = { # key is str(choice)+str(reward)
            str(True)+str(True) : np.array([[0,0], [1,0], [0,1], [1,1]]),
            str(True)+str(False) : np.array([[0,0], [1,0]]),
            str(False)+str(True) : np.array([[0,0], [0,1]]),
            str(False)+str(False) : np.array([[0,0]]),
        }
        if name_type == 'short':
            cond_name_D = { # key is str(choice)+str(reward)
                str(True)+str(True) : ["L-", "R-", "L+", "R+"],
                str(True)+str(False) : ["L", "R"],
                str(False)+str(True) : ["-", "+"],
                str(False)+str(False) : ["all trial"],
            }
        else:
            cond_name_D = { # key is str(choice)+str(reward)
                str(True)+str(True) : [
                    'Left, Unrewarded',
                    'Left, Rewarded',
                    'Right, Unrewarded',
                    'Right, Rewarded'],
                str(True)+str(False) : ['Choose Left', 'Choose Right'],
                str(False)+str(True) : ['Unreward', 'Rewarded'],
                str(False)+str(False) : ["all trial"],
            }

        obs = obsD[str(choice)+str(reward)]
        cond_name = cond_name_D[str(choice)+str(reward)]

        return (obs, cond_name)
    
    @cached_property
    def cache_dpR_dLats(self):
        """cached results of the derivatives of function model prediction p(R) = f(updated_latents) w.r.t each latents"""

        _, latents = self.cache_forward
        chMLP = self.build_choiceMLP_func(convert_pR=True)

        return jnp.apply_along_axis(jax.grad(chMLP), axis=2, arr = latents)
    
    @cached_property
    def cache_flatten_dpR_dLats(self):
        """cached flattened dpR_dLats
        of shape (N_obs, N_latent)
        N_obs represents number of all valid trials (not nan and > -1)
        """
        ds = self.cache_eval_dataset.assign(dpR_dLats = (["tri", "sess", "latent"], self.cache_dpR_dLats))

        df = ds2df(ds[['inputs', 'outputs', 'dpR_dLats']])
        return df.filter(regex='dpR_dLats')[np.all(df.filter(regex='inputs:|outputs:') > -1, axis=1)].values
    
    @property
    def choiceFun(self):
        """function to perform choice selection"""
        return disRNN_choiceMLP_func(self.params, convert_pR = True)
    
    @property
    def grad_dVR(self):
        r"""This is the coefficients of each latent for "choice selection as logistic regression" 

        logit(p(R)) = a^T \cdot z

        This makes sense only when the choice MLP has been reduced to a linear projection of latent state
        """
        assert len(self.choice_mlp_shape) == 1, 'this property is valid only when choice MLP has been reduced to a linear projection of latent state'

        chMLP = self.build_choiceMLP_func(convert_pR=False)
        f_dVR = lambda lats: jnp.diff(chMLP(lats))[0]

        return jax.grad(f_dVR)(jnp.zeros(self.n_latents))


    def likelihood(self, ds: trainingDataset|None = None, normalize: bool = False, exponentiate: bool = False):
        """calculate the likelihood of a dataset given the model

        Args:
            ds (trainingDataset | None, optional): the dataset to run on. Defaults to None, which uses self.dataset (and cached evaluation results).
            normalize (bool, optional): whether calculate normalized (log) likelihood. Defaults to False.
            exponentiate (bool, optional): whether calculate likelihood or log-likelihood. Defaults to False, which is log-likelihood.
        """
        if ds is None:
            ds = self.dataset
            yh, _ = self.cache_forward
        else:
            yh, _ = self.forward(ds.xs)
        
        log_likelihood = BerLL_logit(ds.ys, yh['prediction'], jax.device_put(normalize))

        if exponentiate:
            return jnp.exp(log_likelihood)
        else:
            return log_likelihood
        
    def BIC(self, ds: trainingDataset|None = None, likelihood: float|None = None, k_type: Literal["IB", "params", "capacity"] = "params"):
        """calculate the Bayesian Information criterion on a dataset
        BIC = k ln(N) - 2 ln(L)

        this implicitly depends on bn_close_thre

        Args:
            k_type: methods to calculate number of parameters.

                (1) "IB": sum of number of open IB
                (2) "params": effective number of parameters when only considering open bottlenecks
                (3) "capacity": consider total number of parameters regardless of IB open or not

        """
        if ds is None:
            ds = self.dataset
        
        if k_type == "IB":
            N_params = self.N_open_update_bn + len(self.open_latent()) + len(self.open_choice_latent())
        elif k_type == "params":
            N_params = self.N_effective_params
        elif k_type == "capacity":
            N_params = tree_reduce(np.add, jax.tree_util.tree_map(lambda x: np.prod(x.shape), self.params)) 
        else:
            raise ValueError(f"unknown k_type {k_type}")
        
        N_data = ds.n_observations
        if likelihood is None:
            ll = self.likelihood(ds)
        else:
            ll = likelihood

        return bic(N_params, N_data, ll)


    def df_model_runthrough(self, model_in = None):
        if model_in is None:
            if self.dataset is None:
                raise ValueError("please provide dataset!")
            else:
                model_in = self.dataset.xs

        return make_MP_phase_df(self.make_network, self.params, model_in)
        
    def augmented_dataset(self, *args, **kwargs) -> xr.Dataset:
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
        - dVR: V(R) - V(L)

        Raises:
            ValueError: when cannot find dataset

        ### Returns:
            xarray.Dataset: the augmented dataset
        """

        sig = inspect.signature(super().augmented_dataset)
        # Bind the received arguments to the signature
        bound_args = sig.bind(*args, **kwargs)
        if 'optional_features' not in bound_args.arguments:
            _default_opt_feat = True
        else:
            _default_opt_feat = False

        # Apply default values to missing arguments (if any)
        bound_args.apply_defaults()
        if _default_opt_feat: # also apply modified default to `optional_features`
            bound_args.arguments['optional_features'].append('dVR')

        # Extract the optional_features argument, if provided
        optional_features = bound_args.arguments.get('optional_features')

        res = super().augmented_dataset(*args, **kwargs)

        if optional_features and 'dVR' in optional_features:
            res['dVR'] = res['choice_logits'][...,1] - res['choice_logits'][...,0]

        if optional_features and 'weighted_latents' in optional_features:
            res['weighted_latents'] = res['latents'] * xr.DataArray(self.np.grad_dVR, dims='latent')
        
        return res


    def latentUpdateRule_pltd_1d(self, unit_i, observations, lim = 2, n_samp = 20):
        """generate data for plotting the update rule of a latent of the model

        compute the updated value given different previous value of a latent

        Args:
            unit_i (_type_): the index of the latent to be analyzed
            observations (_type_): _description_
            lim (int, optional): calculate in the domain of [-lim, lim]. Defaults to 2.
            n_samp (int, optional): number of samples in the domain. Defaults to 20.

        Returns:
            _type_: _description_
        """

        unit_i = int(unit_i)

        initial_state = np.array(get_initial_state(self.make_network))
        reference_state = np.zeros(initial_state.shape)

        state_bins = np.linspace(-lim, lim, n_samp)    

        # states: (N_bins, 1, N_latents)
        states = jnp.tile(reference_state, (state_bins.shape[0], 1, 1))
        states = states.at[:, 0, unit_i].set(state_bins)

        # (N_state_bins, N_observations, N_latents) -> (N_state_bins, N_observations)
        res = jax.vmap(self.stepFun, in_axes=(None, 0), out_axes=0)(jnp.array(observations), states)[1][:,:,unit_i]

        return np.array(res)
    

    def latentUpdateRule_pltd_2d(self, unit_i, observations, dep_unit_i, lim = 2, n_samp = 20, n_dep_samp = 20):

        unit_i = int(unit_i)

        initial_state = np.array(get_initial_state(self.make_network))
        reference_state = np.zeros(initial_state.shape)

        state_bins = np.linspace(-lim, lim, n_samp)
        dep_state_bins = np.linspace(-lim, lim, n_dep_samp)

        res = []
        for si_i in np.arange(n_dep_samp):
            # states: (N_bins, 1, N_latents)
            states = jnp.tile(reference_state, (state_bins.shape[0], 1, 1))
            states = states.at[:, 0, unit_i].set(state_bins)
            states = states.at[:, 0, dep_unit_i].set(dep_state_bins[si_i])

            # (N_state_bins, N_observations, N_latents) -> (N_state_bins, N_observations)
            res.append(jax.vmap(self.stepFun, in_axes=(None, 0), out_axes=0)(jnp.array(observations), states)[1][:,:,unit_i]) 

        return np.stack(res, axis = 2)
        
    def plot_update_rules(self):
        """Generates visualizations of the update ruled of a disRNN.

        right now this only works with the case where observation is of size 2 (choice and reward only)
        """

        latent_sigmas = self.latent_sigma
        latent_order = self.latent_sigma_order

        figs = []

        # Loop over latents. Plot update rules
        for latent_i in latent_order:
            # If this latent's bottleneck is open
            if latent_sigmas[latent_i] < 0.5:
                fig = self.plt_update_rule(latent_i + 1)
                figs.append(fig)

        return figs
    
    def plt_update_rule(self, lat_id: int, ):
        step_fun = self.stepFun

        initial_state = np.array(self.latent_inits)
        reference_state = np.zeros(initial_state.shape)

        update_sigmas = self.update_sigma
        obs_dim = self.obs_dim

        latent_i = lat_id - 1


        # Which of its input bottlenecks are open?
        update_mlp_inputs = np.argwhere(update_sigmas[:,latent_i] < 0.9)
        choice_sensitive = np.any(update_mlp_inputs == 0)
        reward_sensitive = np.any(update_mlp_inputs == 1)
        # Choose which observations to use based on input bottlenecks
        observations, titles = self.exmp_obs_inputs(choice_sensitive, reward_sensitive)
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
        
        fig.suptitle(f"latent {lat_id}", y = 0.95)
        if len(latent_sensitive) > 1:
            print(
                'WARNING: This update rule depends on more than one '
                + 'other latent. Plotting just one of them'
            )
        
        return fig

    def cal_updMLP_grid_nlat(self, 
            lat_id: int, 
            input_lat_ids: List[int],
            input_obs: List[str],
            latent_data: pd.DataFrame | Array | None = None,
            n_grid_point: int = 100, 
            return_type: Literal['tensor', 'long', 'wide'] = 'tensor'):
        """calculate updateMLP output by a grid of inputs

        all latent indices are 1-based to match 

        Args:
            lat_id (int): the latent whose update MLP to be calculated
            input_lat_ids (List[int]): latent inputs to the update MLP for building input grid
            input_obs (List[str]): Over which observation dimensions to sweep
            latent_data (pd.DataFrame | Array | None, optional): a sample of latent data to calculate empirical range and mean for each latent. Expect 2D array (obs x lats) or a dataframe similar to a output of `df_model_runthrough`. Defaults to None, which will call `df_model_runthrough` with default settings
            n_grid_point (int, optional): _description_. Defaults to 100.

        Returns:
            _type_: _description_
        """
        
        # get empirical latent data, which is used to calculate the ranges and mean for each latent
        if isinstance(latent_data, Array):
            assert len(latent_data.shape) == 2
            lats = latent_data
        else:
            if isinstance(latent_data, pd.DataFrame):
                latdf = latent_data
            else:
                logging.info("latent data comes from `self.df_model_runthrough`")
                latdf = self.mdrth
            
            lats = np.vstack(latdf['latents'][latdf['chs'] >= 0])

        lat_lr_target = self.build_updateMLP_func(lat_id = lat_id)

        obs, cond = self.exmp_obs_inputs('choice' in input_obs, 'reward' in input_obs)
        
        lat_mean = np.mean(lats, 0)
        if len(input_lat_ids) > 0: 
            # latent whose update depends on other latents

            lat_names = ['lat'+str(i_lat)+"_l1" for i_lat in input_lat_ids]

            axes_vals = [np.linspace(np.min(lats[:, i-1]), np.max(lats[:, i-1]), n_grid_point) for i in input_lat_ids]

            grids = np.meshgrid(*axes_vals)
            grid_flatten = np.hstack([g.reshape(-1,1) for g in grids])

            lat_inputs = np.tile(lat_mean, (n_grid_point**len(input_lat_ids), 1))
            lats_ind = np.array(input_lat_ids, dtype = int) - 1
            lat_inputs[:, lats_ind] = grid_flatten

            res = jax.vmap(jax.vmap(lat_lr_target, in_axes=[None, 0]), in_axes=[0, None])(obs, lat_inputs)

            index = pd.MultiIndex.from_arrays([g.flatten() for g in grids], names = lat_names)

            xda = xr.DataArray(res, dims=["obs", "grid", "updMLP_out"], coords = {
                'grid': index,
                'obs': cond,
                'updMLP_out': ['lr', 'target']
                })
            
            if return_type == 'tensor':
                return xda 
            
            df = xda.to_dataframe("value")
            df.drop(lat_names, axis= 1, inplace=True) # somehow there are duplicate columns for the grids
            df.reset_index(inplace=True)
            
            # tmp_name = ['_lat'+str(i_lat)+"_l1" for i_lat in input_lat_ids]

            # # Rename the index levels to avoid conflicts
            # df.index = df.index.set_names(['obs', *tmp_name, 'updMLP_out'])

            # # Now reset the index
            # df = df.reset_index().drop(tmp_name, axis= 1)

            # make it wider
            # df = df.pivot(index=[c for c in df.columns if c not in ['updMLP_out', 'value']], columns='updMLP_out', values='value').reset_index()

        else:
            lat_inputs = lat_mean
            res = jax.vmap(lat_lr_target, in_axes=[0, None])(obs, lat_inputs)

            xda = xr.DataArray(res, dims=["obs", "updMLP_out"], coords = {
                'obs': cond,
                'updMLP_out': ['lr', 'target']
            })

            if return_type == 'tensor':
                return xda 
            
            df = xda.to_dataframe("value")
            df.reset_index(inplace=True)

            # df = pd.DataFrame(res, columns=['lr', 'target'])

            # df['obs'] = cond
        
        if return_type == 'long':
            return df
        else:
            return df.pivot(index=[c for c in df.columns if c not in ['updMLP_out', 'value']], columns='updMLP_out', values='value').reset_index()

    def plt_updMLP_out(self, 
            lat_id: int, 
            input_lat_ids: List[int] | None = None,
            input_obs: List[str] | None = None,
            latent_data: pd.DataFrame | Array | None = None,
            n_grid_point: int = 100, ):
        # TODO update the lower-level functions to use those in `functions.aug_input_disrnn_analyzer`
        
        if input_obs is None or input_lat_ids is None:
            op_upd = self.np.open_update_bn()[:, lat_id-1] 
            # boolean array indicating which update MLP input bottleneck is open for latent `lat_id`

            if input_obs is None:
                input_obs = list(np.array(['choice', 'reward'])[op_upd[:2]])
            
            if input_lat_ids is None:
                input_lat_ids = list(np.arange(1, self.n_latents+1)[op_upd[2:]])
        
        xda = self.cal_updMLP_grid_nlat(
            lat_id=lat_id,
            input_lat_ids = input_lat_ids,
            input_obs = input_obs,
            latent_data = latent_data,
            n_grid_point = n_grid_point,
            return_type = 'tensor',
            )
        
        match len(input_lat_ids):
            case 0:
                fg = _plt_0d_updMLP_out(xda)
                fig = fg.fig
            case 1:
                fg = _plt_1d_updMLP_out(xda, map_obs='hue')
                fig = fg.fig
            case 2:
                fg = _plt_2d_updMLP_out(xda)
                fig = fg.fig
            case _:
                raise NotImplementedError('this function does not support plotting with more than 2 latent dependencies')
        
        # add title for the figure
        # fig.subplots_adjust(top=0.9) 
        # subplots_adjust doesn't work very well when there's a colorbar
        # fig.suptitle(f"update MLP output for latent {lat_id}", y=1.02) # this doesn't work well with quarto

        _scale_plt_to_ori(fig, 0.95)
        fig.suptitle(f"update MLP output for latent {lat_id}", x=0.5*0.95)
            
        return fig
    
    @property
    def mdrth(self):
        """cached df_model_runthrough result"""
        if not hasattr(self, '_mdrth'):
            self._mdrth = pd.DataFrame(self.df_model_runthrough())
        return self._mdrth
    
    def plt_choice_slct_grad(self, 
            sorted_latent=True, 
            PAL_CH = {'L': '#c8884c', 'R': '#c24ea9'},
            use_dVR = True,
            ax=None,):
        
        if sorted_latent:
            lats_ind = self.np.latent_sigma_order
            # Reorder the variables according to sigma order
            order = lats_ind
        else:
            # Use original order
            order = list(range(self.n_latents))
        
        if use_dVR and len(self.choice_mlp_shape) == 1:
            df = pd.DataFrame({'variable': np.arange(self.n_latents), 'value': self.np.grad_dVR})
        else:
            df = pd.DataFrame(self.cache_flatten_dpR_dLats).melt()

        def ci_low(x):
            return x.quantile(0.025)
        
        def ci_high(x):
            return x.quantile(0.975)
        
        # Calculate summary statistics
        summary = df.groupby('variable')['value'].agg(['mean', 'std', ci_low, ci_high]).reset_index()
        summary['low_err'] = summary['mean'] - summary['ci_low']
        summary['high_err'] = summary['ci_high'] - summary['mean']

        summary['choice'] = summary['mean'].apply(lambda x: 'R' if x > 0 else 'L')
        # Ensure proper ordering
        summary['variable'] = pd.Categorical(summary['variable'], 
                                           categories=order,
                                           ordered=True)
        
        summary = summary.sort_values('variable').rename(columns={'variable': 'latents', 'mean': 'stats of gradient'})
        
        if ax is None:
            fig, ax = plt.subplots(figsize=(6, 6))
        
        # Create barplot using summary data
        sns.barplot(data=summary, 
                   x='stats of gradient',
                   y='latents',
                   hue='choice',
                   palette=PAL_CH,
                   ax=ax)
        
        if not use_dVR or len(self.choice_mlp_shape) != 1:
            # Add error bars
            ax.errorbar(summary['stats of gradient'], 
                    range(len(summary)), 
                    xerr=summary[['low_err', 'high_err']].T,
                    fmt='none',
                    color='black',
                    capsize=3)
        
        ax.axvline(x=0, color='black', linestyle='-', alpha=0.5)
        
        ax.set_yticks(range(len(order)))
        ax.set_yticklabels([f'l{int(i)+1}' for i in order])  # Add l prefix and 1-based indexing
        ax.set_xticks([])
        
        ax.grid(False)
        sns.despine(ax=ax, left=True, bottom=True)
        ax.margins(y=0)
        
        if ax is None:
            return fig, ax
        return ax.figure, ax
    
    def plt_combine_heatmaps_with_ch_grad(self, sorted_latent=True, PAL_CH={'L': '#c8884c', 'R': '#c24ea9'}):

        ws = [1, self.np.update_sigma.shape[0], 1, 2]

        fig, axs = plt.subplots(
            1,4, 
            sharey=False, 
            width_ratios=ws, 
            layout='compressed',
            figsize = [sum(ws)/8*6, 6])
        
        # fig = plt.figure(figsize=([sum(ws), 6], 6), layout = 'compressed')
        # gs = GridSpec(2, 4, height_ratios=[20, 1], figure=fig)
        
        # # Create all subplots
        # axs = [fig.add_subplot(gs[0, i]) for i in range(4)]
        # # Create colorbar axis
        # cax = fig.add_subplot(gs[1, :3]) 
        
        # Plot each component using the modified plotting functions
        self.plt_latent_bottleneck(sorted_latent=sorted_latent, ax=axs[0], show_colorbar=False)
        self.plt_update_bottleneck(sorted_latent=sorted_latent, ax=axs[1], show_colorbar=False)
        self.plt_choice_bottleneck(sorted_latent=sorted_latent, ax=axs[2], show_colorbar=True)
        self.plt_choice_slct_grad(sorted_latent=sorted_latent, ax=axs[3], PAL_CH = PAL_CH)

        return fig, axs
    
    
