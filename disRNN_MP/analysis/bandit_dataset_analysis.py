from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, Iterable, Literal, Optional, List, Tuple, Union, overload
import re

import pandas as pd
import polars as pl
import numpy as np
import xarray as xr
import seaborn as sns
import itertools
from scipy.stats import binomtest, norm
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import plotnine as pn
from plotnine import (
    ggplot, 
    aes, 
    geom_point, 
    geom_line, 
    geom_pointrange, 
    theme_bw, 
    theme, 
    element_rect, 
    element_line, 
    facet_wrap, 
    facet_grid,
    scale_color_discrete, 
    labs, 
    position_nudge)

from disRNN_MP.dataset import trainingDataset, read_dataframe
from disRNN_MP.typing import Array

_build_terms = lambda prefix, lags: sorted([f"{prefix}_{l}" for prefix in prefix for l in lags])
"""build name for lagged variables

```
lambda prefix, lags: sorted([f"{prefix}_{l}" for prefix in prefix for l in lags])
```
"""

def _summarize_reg(df: pd.DataFrame, formula: str, fami_f: Callable):
    model = smf.glm(formula=formula, data=df, family=fami_f()).fit()

    # Convert the summary to a DataFrame similar to tidy(m) %>% as.data.table in R
    mfit = model.summary2().tables[1].reset_index().rename(columns={'index': 'term'})
    return mfit


def _describe_df_row(df: pd.DataFrame, columns: List[str]):
    return df.apply(
        lambda row: ", ".join(f"{col} = {row[col]}" for col in columns), 
        axis=1
    )


class BanditDataset:
    """ Hold the complete dataset for analysis purpose
    BanditDataset is more complete than trainingDataset, which means conversion from BanditDataset to trainingDataset is lossless, conversion back will loss one trial of data

    all trials are completed trials

    choice should be represented as integer 0 (L) and 1 (R)

    The dataframe is sorted by ['sessID', 'triID']
    """

    @classmethod 
    def from_dataframe(cls, 
            df: pd.DataFrame | str | Path, 
            choice_col = 'choice', 
            reward_col = 'reward', 
            sessID_col = 'sessID', 
            triID_col = 'triID',
            include_other_cols: bool | List[str] = False,
            nan_filling: None | float = None):
        """convert a dataframe to the format required by this class
        Args:
            df: a dataframe or a path to the dataframe
        """
        ori_names = [sessID_col, triID_col, choice_col, reward_col]
        new_names = ['sessID', 'triID', 'choice', 'reward']

        df = read_dataframe(df).reset_index(drop=True)
        # reset_index to make unique index to avoid index clash during join

        fdf = df[ori_names].rename(columns=dict(zip(ori_names, new_names)))
        if include_other_cols:
            if isinstance(include_other_cols, Iterable): 
                # include selected columns
                fdf = fdf.join(df[list(include_other_cols)])
            else: 
                # include all other cols
                fdf = fdf.join(df.loc[:, ~df.columns.isin(ori_names)])

        fdf.dropna(inplace=True)
        if nan_filling is not None:
            # remove NA-containing rows
            fdf = fdf[np.all(fdf != nan_filling, axis=1)]
        
        return cls(fdf.reset_index(drop=True))
    
    @classmethod
    def from_3darr(cls, 
        tensor: Array, 
        feat_names: List[str] = ['choice', 'reward'], 
        nan_filling: None | float = -1,
        sess_ids = None
        ):
        """create a BanditDataset from a 3D tensor of shape (N_tri, N_sess, N_feature)

        Args:
            tensor (Array): of shape (N_tri, N_sess, N_feature)
            feat_names (List[str], optional): the name for each feature dimension in the 3rd axis. Defaults to ['choice', 'reward'].
            nan_filling: the value used to fill NA values in the tensor. If this argument is set to any value other than None, this function will remove those NA-filled rows by identifying which row contains this value.
        """

        # through xarray.DataArray, convert the tensor to a dataframe multi-indexed by triID and sessID
        res: pd.DataFrame = (
            xr.DataArray(
                np.array(tensor), 
                dims=['triID', 'sessID', 'feat'], 
                coords={
                    'feat': feat_names,
                    'sessID': np.arange(tensor.shape[1]) if sess_ids is None else sess_ids,
                }
            )
            .to_dataframe("value")
            .unstack('feat')
        ) # type: ignore

        res.dropna(inplace=True)

        if nan_filling is not None:
            # remove NA-containing rows
            res = res[np.all(res != nan_filling, axis=1)]

        # tidy up column and row indices
        res.columns = res.columns.get_level_values(1).set_names(None)
        res.reset_index(inplace=True)

        return cls(res)

    @classmethod
    def from_trainingDataset(cls, 
        td: trainingDataset,
        feat_names: None | List[str] = None, 
        nan_filled_with: None | float = None,
        ):

        if nan_filled_with is None:
            nan_filled_with = td._na_as
        
        if feat_names is None:
            feat_names = [*td.input_feature_name, *td.output_feature_name]
        
        if ('choice' not in feat_names):
            print(f"rename feat_names[0]: {feat_names[0]} to 'choice'")
            feat_names[0] = 'choice'

        if ('reward' not in feat_names):
            print(f"rename feat_names[1]: {feat_names[1]} to 'reward'")
            feat_names[1] = 'reward'

        tensor = np.concatenate((td.xs, td.ys), axis=2)

        return cls.from_3darr(tensor, feat_names, nan_filling=nan_filled_with, sess_ids=td._sess_ids) 
        


    def __init__(self, df: pd.DataFrame) -> None:
        """init DanditDataset from a dataframe

        The dataframe has to contain the following columns: 'choice', 'reward', 'sessID', 'triID'

        'sessID' along (without combining together with other variables) must uniquely identify one session 

        recommended to use classmethod `from_dataframe` to init
        """
        assert {'choice', 'reward', 'sessID', 'triID'} <= set(df.columns), f"missing required columns (input cols: {set(df.columns)}) "

        self.df = df.sort_values(['sessID', 'triID'])

        assert ~np.any(self.df.duplicated(['sessID', 'triID'])), "combination of sessID and triID need to uniquely identify every trial"

        # make sure the index is always as expected
        self.df.reset_index(drop=True, inplace=True)

        self.choices = df['choice']
        self.rewards = df['reward']

    def __repr__(self) -> str:
        return super().__repr__() + "\n" + self.df.__repr__()

    def _repr_html_(self):
        return self.df._repr_html_()  # type: ignore

    def _to_3darr(self, values, fillna = -1) -> np.ndarray:
        pivot_df = self.df.pivot_table(index='triID', columns='sessID', values=values)  # type: ignore

        # Filling NaN values with -1
        pivot_df = pivot_df.fillna(fillna)

        # Convert to 3D NumPy array 
        array_3d = pivot_df.to_numpy().reshape(len(self.df['triID'].unique()), len(self.df['sessID'].unique()), -1)

        return array_3d
    
    def query(self, expr, **kws):
        subdf = self.df.query(expr, **kws)
        return self.__class__(subdf)
    
    def to_trainingD(self, in_feat: Optional[List] = None, out_feat: Optional[List] = None):
        """ make a trainingDataset from this instance
        
        this function will compute shift for output feature column
        
        
        """

        if in_feat is None:
            in_feat = ['choice', 'reward']
        
        if out_feat is None:
            out_feat = ['choice']

        xs = self._to_3darr(values=in_feat)[:-1,:,:] # last trial data
        ys = self._to_3darr(values=out_feat)[1:,:,:] # current trial data

        return trainingDataset(xs, ys, input_feature_name=in_feat)
    
    def to_file(self, filename):
        match Path(filename).suffix:
            case '.parquet':
                self.df.to_parquet(Path(filename))
            case '.csv':
                self.df.to_csv(Path(filename))
            case '.feather':
                self.df.to_feather(Path(filename))    
            case _:
                raise ValueError('unrecognized file extension')
            
    def _add_ch_rew_str_col(self, force = False):
        """add a `ch_rew` column if not exist, which represent a trial event by string such as 'L+' """
        if ('ch_rew' not in self.df.columns) or force:
            df = self.df
            self.df['ch_rew'] = np.char.add(np.array(['L', 'R'])[df['choice'].astype(int)], np.array(['-', '+'])[df['reward'].astype(int)])

        return self

    def _add_rle_col(self, var: str = 'choice', groupby: str | Iterable[str] = 'sessID', order_by: str | Iterable[str] = 'triID'):
        """calculate run length of a variable grouped by sesssion and ordered by trial sequence
        
        will add the calculated variable as column f"{var}_rle"

        :param var: column name of the variable to calculate run length of
        
        """
        col_name = f"{var}_rle"

        self.df[col_name] = pl.DataFrame(self.df).select(
            # assign a group id for each choice streak
            _rle_id := pl.col(var).rle_id().over(groupby, order_by=order_by), 

            # run length of the streak upto current trial
            pl.cum_count(var).over([groupby, _rle_id], order_by=order_by).alias(col_name), 
        )[:, col_name].to_pandas().astype(int)

        return self
    
    def cal_word_freq(self, seq_len: int, test_against_unif: bool = True):
        """calculate the frequency of all possible words of a fixed length in the entire behavioral dataset
        
        here the character is the choice-outcome event for one trial

        if `test_against_unif` is True, the frequency of each word will be hypothesis-tested against a trival binomial distribution
        """
        seq_len = int(seq_len)
        if seq_len <= 1:
            raise ValueError("sequence length must be longer than 1")
        
        self._add_ch_rew_str_col()
        
        beh = self.df

        # columns needed in the dataframe
        seq_cols = ['ch_rew'] + [f'ch_rew_{i}' for i in range(1, seq_len)]
        
        # number of possible different words given the length of word
        n_unique_word = 4 ** seq_len

        if not set(seq_cols) <= set(beh.columns):
            cols_to_drop = list(set(beh.columns).intersection(set(seq_cols[1:]))) # never drop 'ch_rew'
            beh = beh.drop(columns=cols_to_drop)

            # create shifted ch_rew
            beh = beh.join(beh.groupby('sessID')[['ch_rew']].shift(range(1,seq_len)), )

        self.df = beh

        uniq_ch_rew = beh['ch_rew'].unique()

        # Create all possible combinations of sequence columns
        unique_values = [uniq_ch_rew for _ in seq_cols]
        all_combinations = pd.DataFrame(itertools.product(*unique_values), columns=seq_cols)

        # Merge with the actual frequency table to ensure all combinations are included
        freq_table = beh[seq_cols].value_counts().reset_index(name='N')
        freq_table = all_combinations.merge(freq_table, on=seq_cols, how='left').fillna({'N': 0})

        # Add new columns to calculate stats similar to the R code
        freq_table['seq'] = freq_table[seq_cols[::-1]].astype(str).sum(axis=1)
        freq_table['unif_p'] = 1 / n_unique_word
        freq_table['total'] = np.sum(freq_table['N'])
        freq_table['freq'] = freq_table['N'] / freq_table['total']

        # Calculate binomial test for each row and use tidy-like approach
        if test_against_unif:
            ci_lower, ci_upper, p_values = [], [], []
            for _, row in freq_table.iterrows():
                test = binomtest(int(row['N']), int(row['total']), p=row['unif_p'])
                ci = sm.stats.proportion_confint(row['N'], row['total'], alpha=0.05, method='wilson')
                ci_lower.append(ci[0])
                ci_upper.append(ci[1])
                p_values.append(test.pvalue)

            # Append CI and p-values to the dataframe
            freq_table['ci_lower'] = ci_lower
            freq_table['ci_upper'] = ci_upper
            freq_table['p_value'] = p_values

        return freq_table
    
    def cal_conditional_pR(self, hist_seq_len: int, include_ci: bool = True):
        """calculate probability of right choice after each possible history sequence of fixed length

        Args:
            hist_seq_len (int): _description_
            include_ci (bool, optional): _description_. Defaults to True.

        Returns:
            _type_: _description_
        """

        hist_seq_len = int(hist_seq_len)
        if hist_seq_len <= 0:
            raise ValueError("sequence length must be longer than 0")
        
        self._add_ch_rew_str_col()
        self.df['ch_str'] = np.array(['L', 'R'])[self.df['choice'].astype(int)]
        
        beh = self.df

        # columns needed in the dataframe
        hist_seq_cols = [f'ch_rew_{i+1}' for i in range(hist_seq_len)]
        seq_cols = ['ch_str'] + hist_seq_cols

        # add ch_rew_1, ch_rew_2, ... columns
        beh = beh.join(beh.groupby('sessID')[['ch_rew']].shift(range(1,hist_seq_len+1)), )


        # Create all possible combinations of sequence columns
        unique_values = [beh[col].dropna().unique() for col in seq_cols]
        all_combinations = pd.DataFrame(itertools.product(*unique_values), columns=seq_cols)

        # Merge with the actual frequency table to ensure all combinations are included
        freq_table = beh[seq_cols].value_counts().reset_index(name='N')
        freq_table = all_combinations.merge(freq_table, on=seq_cols, how='left').fillna({'N': 0})

        # Add new columns to calculate stats similar to the R code
        freq_table['hist_seq'] = freq_table[seq_cols[-1:0:-1]].astype(str).sum(axis=1) # hist_seq only considers historic `ch_rew` columns

        def calc_r_percentage(group):
            total = group['N'].sum()
            r_count = group.loc[group['ch_str'] == 'R', 'N'].sum()
            p_r = (r_count / total) if total > 0 else 0
            return pd.Series({'pR': p_r, 'total_N': total, 'R_N': r_count})

        # Apply function to each group based on hist_seq
        grp_pR = freq_table.groupby('hist_seq').apply(calc_r_percentage).reset_index()

        if include_ci:
            grp_pR = grp_pR.join(
                grp_pR.apply(
                    lambda d: pd.Series(sm.stats.proportion_confint(d['R_N'], d['total_N'], alpha=0.05, method='wilson'), index=['ci_low', 'ci_high']), 
                    axis=1, result_type='expand'))

        return grp_pR

    def add_latent_values(self, latents: Array):
        """add latents columns provided through `latents` argument

        will add columns: latent_0, latent_1, ...

        Args:
            latents (Array): 3D array of shape (N_tri, N_sess, N_latents)
        """

        assert latents.shape[:-1] == (self.df['triID'].max()+1, self.df['sessID'].max()+1, )

        tmp = xr.DataArray(latents, dims=['triID', 'sessID', 'lat']).to_dataframe(name='latent').unstack('lat')
        tmp.columns = ['_'.join(map(str,c)) for c in tmp.columns] # generate column names: latent_0, latent_1, ...
        tmp = tmp.reset_index()

        self.df = self.df.merge(tmp, how = 'left', on=['triID', 'sessID'])

        return self

    def plt_session(self, 
            sessID: int, 
            latent_ids: List[int], # TODO a better API would to use latents column names to avoid misunderstanding of "id"
            latent_colors: List[str]|None = None, 
            latent_names: List[str]|None = None,
            figsize = (10, 6),

        ):
        """plot trial event and latent dynamics for a session

        Args:
            sessID (int): session to plot
            latent_ids (List[int]): list of latent ids to show in the plot. The numbering followings latents columns. If the latents are added via `add_latent_values`, it is supposed to be 0-based
            latent_colors (List[str] | None, optional): colors for each latent, follow order of `latent_ids`. Defaults to None.
            latent_names (List[str] | None, optional): names for each latent, follow order of `latent_ids`. Defaults to None.
            figsize (tuple, optional): _description_. Defaults to (10, 6).

        Returns:
            _type_: _description_
        """        

        assert np.any(self.df.columns.str.match(r'^latent_\d+')), "latent values are not found in the dataframe, need to run `add_latent_values` first"

        df = self.df[self.df['sessID'] == sessID]
        assert len(df) > 0, f"cannot find session with id {sessID}"

        df_long = df.melt(id_vars='triID', value_vars=df.columns[df.columns.str.match('latent_')]) # type: ignore

        lat_names = [f"latent_{i}" for i in latent_ids]
        df_long_sel = df_long[np.isin(df_long['variable'], lat_names)]

        # set name and order of latents in the plot legend
        df_long_sel['latent'] = pd.Categorical(df_long_sel['variable'], lat_names)
        if latent_names is not None:
            df_long_sel['latent'] = df_long_sel['latent'].map(dict(zip(lat_names, latent_names)))

        lat_val_range = (df_long_sel['value'].min(), df_long_sel['value'].max())

        # Assuming tmp and tmp_beh are pandas DataFrames
        sns.set_theme(style="whitegrid")

        # Create the figure and axis
        fig, ax = plt.subplots(figsize=figsize)

        # Add horizontal line at y = 0
        ax.axhline(0, linestyle='solid', color='black')

        # Plot the lines
        if latent_colors is not None:
            color_dict = dict(zip(lat_names, latent_colors))    
        else:
            color_dict = None
        
        sns.lineplot(
            data=df_long_sel, 
            x='triID', 
            y='value',
            hue='latent',
            palette=color_dict,
            alpha=0.7, 
            ax=ax
        )

        # Add point markers for behavior
        for _, row in df.iterrows():
            chs_value = lat_val_range[0]-0.5 if row['choice'] == 0 else lat_val_range[1]+0.5
            color = 'black' if row['reward'] else 'red'
            ax.plot(row['triID'], chs_value, '|', color=color, markersize=10, alpha=1)

        # Add text annotations for "Right" and "Left"
        ax.text(14, lat_val_range[1]+0.1, 'Right', fontsize=14, ha='left', verticalalignment='center')
        ax.text(14, lat_val_range[0]-0.1, 'Left', fontsize=14, ha='left', verticalalignment='center')

        # Set labels and title
        ax.set_xlabel('Trial')
        ax.set_ylabel('Latent value')
        ax.set_title(f'Example Session (id = {sessID})')

        # Create dummy handles for black and red ticks to add to the legend
        win_handle = mlines.Line2D([], [], color='black', marker='|', linestyle='None', markersize=10, label='Win')
        loss_handle = mlines.Line2D([], [], color='red', marker='|', linestyle='None', markersize=10, label='Loss')

        # Retrieve existing legend handles and labels from the latent lines
        handles, labels = ax.get_legend_handles_labels()

        # Add our new handles and labels for ticks
        handles.extend([win_handle, loss_handle])
        labels.extend(['Win', 'Loss'])
        ax.legend(handles=handles, labels=labels, loc='center left', bbox_to_anchor=(1, 0.5), framealpha=0)

        # Set axis limits
        ax.set_xlim(df['triID'].min(), df['triID'].max())

        # Remove gridlines to mimic theme_bw from ggplot
        ax.grid(False)

        # Adjust figure layout to accommodate the legend
        plt.tight_layout()

        return fig
    
    def cal_tri_hist(
            self, 
            lag: int|Iterable[int], 
            fit_by_grp: Iterable[str] | str | None = None, 
            dep_var: str = 'choice',
            history_items = ["choice", "oppo_choice"],
            family: Literal['binomial', 'gaussian'] = 'binomial',
            include_reg_data: bool = True,
            name: str = "serial correlation",
            obs_filter: None | str = None,
        ) -> 'TriHistRegResults':
        """perform trial history regression 

        assume matching pennies type of task where one and only one target is rewarding
        assume only two possible choices and represented as 0 and 1

        independent variables are calculated from 'choice' and 'reward' columns.
        lagged variables are grouped by sessions before being calculated

        Args:
            lag (int|Iterable[int]): specify how many past trials to use as independent variables. If it is a single integer, will use last 1 upto last `lag` trials (include both end) (`range(1, int(lag + 1))`). This argument also support custom iterable for specifying lags being used.
            dep_var (str): the dependent variable for the regression. By default it is 'choice' column
            fit_by_grp (str): The name of the column used to group the dataset. If provided, the regression will be performed separately by the grouping variable
        """
        if isinstance(lag, Iterable):
            lags = list(lag)
        else:
            lags = list(range(1, int(lag + 1)))
        
        if fit_by_grp is not None:
            if isinstance(fit_by_grp, str):
                fit_by_grp = [fit_by_grp]
            elif isinstance(fit_by_grp, Iterable):
                fit_by_grp = list(fit_by_grp)
            else:
                fit_by_grp = [str(fit_by_grp)]
        

        beh = self.df.copy()

        beh['oppo_choice'] = np.where(beh['reward'], beh['choice'], 1 - beh['choice'])

        # sess_grp = ['sessID'] if fit_by_grp is None else ['sessID'] + fit_by_grp
        sess_grp = 'sessID'
        beh = beh.join(beh.groupby(sess_grp, sort=True)[history_items].shift(lags, )).dropna()

        # Create labels similar to expand.grid in R
        terms = _build_terms(history_items, lags)

        # Fit the model using statsmodels GLM
        formula = f'{dep_var} ~ ' + ' + '.join(terms)

        match family.lower():
            case 'binomial':
                fami_f = sm.families.Binomial
            case 'gaussian':
                fami_f = sm.families.Gaussian
            case _:
                raise NotImplementedError('unimplemented family')

        def _tri_hist_reg(df: pd.DataFrame):
            model = smf.glm(formula=formula, data=df, family=fami_f()).fit()

            # Convert the summary to a DataFrame similar to tidy(m) %>% as.data.table in R
            mfit = model.summary2().tables[1].reset_index().rename(columns={'index': 'term'})
            return mfit

        if obs_filter is not None:
            beh.query(obs_filter, inplace=True)
        # fitting the glm models
        if fit_by_grp is not None:
            from tqdm.auto import tqdm
            tqdm.pandas()
            mfit = beh.groupby(fit_by_grp).progress_apply(_tri_hist_reg).reset_index(fit_by_grp)
        else:
            mfit = _tri_hist_reg(beh)

        # Extract prefix and nlag as in the R code
        # mfit['prefix'] = mfit['term'].apply(lambda x: re.sub(r'^(oppo_choice|choice).*', r'\1', x) if re.match(r'^(oppo_choice|choice)', x) else None)
        # mfit['nlag'] = mfit['term'].apply(lambda x: int(re.search(r'_(\d+)$', x).group(1)) if re.search(r'_(\d+)$', x) else None)

        mfit.reset_index(drop=True, inplace=True) # need one more time to avoid duplicated index
        mfit = mfit.join(mfit['term'].str.extract(fr"(?P<prefix>{'|'.join(history_items)})_(?P<nlag>\d+)$", )) 
        mfit['nlag'] = mfit['nlag'].astype(float)

        if include_reg_data:
            return TriHistRegResults(
                term_df=mfit, 
                formula = formula,
                name=name,
                dependent_var=dep_var,
                group_var=fit_by_grp,
                data = BanditDataset(beh))
        else:
            return TriHistRegResults(
                term_df=mfit, 
                formula = formula,
                name=name,
                dependent_var=dep_var,
                group_var=fit_by_grp,
                )
        

    def cal_mod_tri_hist(
            self, 
            lag: int|Iterable[int], 
            fit_by_grp: Iterable[str] | str | None = None, 
            dep_var: str = 'choice',
            history_items = ["choice", "oppo_choice"],
            modulator = ['choice', 'reward'],
            # build_mod_str: Callable | None = lambda mod1, mod2: mod1.map({'0.0': 'L', '1.0': 'R'}) + mod2.map({'0.0': '-', '1.0': '+'}),
            build_mod_str: Callable | None = lambda mod1, mod2: mod1.map({0: 'L', 1: 'R'}) + mod2.map({0: '-', 1: '+'}),
            family: Literal['binomial', 'gaussian'] = 'binomial',
            include_reg_data: bool = True,
            name: str = "modulated serial correlation",
            obs_filter: None | str = None,
    ):
        
        if isinstance(lag, Iterable):
            lags = list(lag)
        else:
            lags = list(range(1, int(lag + 1)))
        
        if fit_by_grp is not None:
            if isinstance(fit_by_grp, str):
                fit_by_grp = [fit_by_grp]
            elif isinstance(fit_by_grp, Iterable):
                fit_by_grp = list(fit_by_grp)
            else:
                fit_by_grp = [str(fit_by_grp)]

        match family.lower():
            case 'binomial':
                fami_f = sm.families.Binomial
            case 'gaussian':
                fami_f = sm.families.Gaussian
            case _:
                raise NotImplementedError('unimplemented family')

        # the smallest lag will be special: interacting with other lagged events
        lags = sorted(lags)

        beh = self.df.copy()

        beh['oppo_choice'] = np.where(beh['reward'], beh['choice'], 1 - beh['choice'])

        all_terms = list(set(modulator).union(history_items))
        beh.loc[:, all_terms] = beh[all_terms].astype(int)

        sess_grp = 'sessID'
        lagged = beh.groupby(sess_grp, sort=True)[all_terms].shift(lags, )
        beh = beh.join(lagged).dropna()

        # Create labels similar to expand.grid in R
        remotes = _build_terms(history_items, lags[1:])
        recents = _build_terms(history_items, lags[:1])
        mods = _build_terms(modulator, lags[:1])

        # Fit the model using statsmodels GLM

        formula = f"{dep_var} ~ {' + '.join(recents)} + {''.join([f"C({term}):" for term in mods])}({' + '.join(remotes)})"

        if obs_filter is not None:
            beh.query(obs_filter, inplace=True)
        
        if fit_by_grp is not None:
            mfit = beh.groupby(fit_by_grp).apply(
                _summarize_reg, include_groups=False, formula=formula, fami_f = fami_f # type: ignore
                ).reset_index(fit_by_grp)  
        else:
            mfit = _summarize_reg(beh, formula=formula, fami_f = fami_f)
        
        mfit.reset_index(drop=True, inplace=True) # need one more time to avoid duplicated index

        mfit = mfit.join(mfit['term'].str.extract(fr"(?P<prefix>{'|'.join(history_items)})_(?P<nlag>\d+)$", )) 
        mfit['nlag'] = mfit['nlag'].astype(float)

        # extract modulator values
        mfit = mfit.join(mfit['term'].str.extract(".+".join([fr"C\({m}\)\[(?P<{m}>[\.\d]+)\]" for m in mods])))
        mfit[mods] = mfit[mods].astype(float)

        if callable(build_mod_str):
            mfit['mod'] = build_mod_str(*mfit[mods].to_dict('series').values())
        
        if include_reg_data:
            return TriHistRegResults(
                term_df=mfit, 
                formula = formula,
                name=name,
                dependent_var=dep_var,
                group_var=fit_by_grp,
                modulators=mods,
                data = BanditDataset(beh))
        else:
            return TriHistRegResults(
                term_df=mfit, 
                formula = formula,
                name=name,
                dependent_var=dep_var,
                group_var=fit_by_grp,
                modulators=mods,
                )



@dataclass
class TriHistRegResults:
    """class to hold trial-history regression result

    can also include a reference regression results for plotting
    for example, when showing 
    """

    term_df: pd.DataFrame
    """a dataframe that describes coefficents for each term in the regression"""
    formula: str
    """the exact formula used for the regression"""
    name: str = 'trial-history regression'
    dependent_var: str = 'choice'
    """the dependent variable in the regression"""
    group_var: List[str] | None = None
    """the grouping variable for the regression (if exists)"""
    modulators: List[str] | None = None
    data: Union[None, BanditDataset] = None
    """the original trial data used to fit the regression model"""
    # ref_result: Union[None, 'TriHistRegResults'] = None
    

    _ref_result: Union[None, 'TriHistRegResults'] = field(init=False, repr=False, default=None)

    def __post_init__(self):
        self._signif_diff = None

        if self.name is None:
            self.name = 'trial-history regression'
        
        if self.modulators is not None and 'mod' not in self.term_df:
            self.term_df['mod'] = _describe_df_row(self.term_df, self.modulators)

    @property
    def ref_result(self):
        """optional reference trial history regression result"""
        return self._ref_result
    
    @ref_result.setter
    def ref_result(self, val):
        if isinstance(val, type(self)):
            if self.group_var is not None:
                assert val.group_var == self.group_var, "the reference result should be grouped in the exactly same way as this result"
            self._ref_result = val
        elif val is not None:
            raise ValueError(f'ref_result should be either None or TriHistRegResults, received {type(val)}')
        
    def _all_group_vars(self) -> List[str]:
        """include 'prefix', 'nlag', group_var, 'mod' if exists"""
        vars = ['prefix', 'nlag']
        if self.group_var is not None:
            vars += self.group_var
        
        if self.modulators is not None:
            vars += self.modulators
            if 'mod' in self.term_df:
                vars += ['mod']
        
        return vars

    def set_name(self, name):
        self.name = name
        return self
    
    @property
    def signif_diff(self):
        if self._signif_diff is None:
            self._signif_diff = self.tst_signif_diff()    
        return self._signif_diff
    
    def tst_signif_diff(self):
        """hypothesis test of whether the coefficient of self is significantly different from that of reference result
        
        return a dataframe indexed by possible terms (and group_var if exist)
        """
        assert self.ref_result is not None, "no reference result for testing significance"
        tmpdf = pd.concat([
            self.term_df.assign(grp = 'self'),
            self.ref_result.term_df.assign(grp = 'ref')])
        
        grpbys = self._all_group_vars()
        gdf = tmpdf.groupby(grpbys)

        assert (gdf.size() == 2).all(), f"grouping by {grpbys} should make each group has exactly 2 rows, but not:\n{gdf.size()}"
        
        zscore = gdf.apply(lambda g: np.abs(g['Coef.'].iloc[0] - g['Coef.'].iloc[1])/ np.sqrt((g['Std.Err.'] ** 2).sum()))
        pval = zscore.map(lambda x: 2*(1-norm.cdf(x))) # type: ignore # 2-tailed test

        return pval
        
    
    def plot(self, 
            term_df_filter: Callable | None = None,
            use_ref_result: bool = True,
            color_name: str | None = None,
            test_signif: float|None = 0.05,
            mod_to_facet: bool = False,
            facet_grid_kwargs: dict | None = None,
            aes_override_kwargs: dict | None = None,
            geom_line_kwargs: dict = dict(),
            geom_pointrange_kwargs: dict = dict(),
        ):

        aes_kwargs = dict(x='nlag', y='Coef.')
        label_kwargs = dict(x = "lagged trial")
        signif_layer = None
        pltd = self.term_df
        
        # combine reference regression results
        if use_ref_result and self.ref_result:

            if self.name == self.ref_result.name:
                self_result_name = 'self ' + self.name
                ref_result_name = 'reference ' + self.ref_result.name
            else:
                self_result_name = self.name
                ref_result_name = self.ref_result.name

            pltd = pd.concat([
                self.term_df.assign(reg = self_result_name),
                self.ref_result.term_df.assign(reg = ref_result_name)]
            ).assign(
                reg = lambda df: pd.Categorical(df['reg'], categories=[ref_result_name, self_result_name], ordered=True) 
            )
            

            if test_signif:
                pval = self.signif_diff
                signif = self.term_df.merge(
                    pval.to_frame('p_value').reset_index(), 
                    how='left', 
                    on = self._all_group_vars()) 
                signif = signif[(~signif['nlag'].isna()) & (signif['p_value'] < test_signif)]

                if mod_to_facet and ('mod' in signif) and (signif.groupby(signif['mod'].isna()).ngroups == 2):
                    signif1, signif2 = [x for _, x in signif.groupby(signif['mod'].isna())]
                    signif_layer = [
                        pn.geom_point(
                            data = signif1,
                            # mapping = pn.aes(color = None, shape = None),
                            mapping=pn.aes(x='nlag', y='Coef.'),
                            inherit_aes = False,
                            color = 'black',
                            shape = "*", 
                            position = pn.position_nudge(x=0.1, y=0.02),
                        ), 
                        
                        pn.geom_point(
                            data = signif2.drop(columns = 'mod'),
                            # mapping=pn.aes(color = None, shape = None),
                            mapping=pn.aes(x='nlag', y='Coef.'),
                            inherit_aes = False,
                            color = 'black',
                            shape = "*", 
                            position = pn.position_nudge(x=0.1, y=0.02),
                        )
                    ]

                else:
                    signif_layer = pn.geom_point(
                        data = signif,
                        shape = "*", 
                        color = 'black',
                        mapping=pn.aes(x='nlag', y='Coef.'),
                        inherit_aes = False,
                        position = pn.position_nudge(x=0.1, y=0.02))
            
        # heuristic of aesthetic settings categorized by whether has modulators and whether plotting reference result
        if use_ref_result and self.ref_result and self.modulators:
            if mod_to_facet:
                aes_kwargs.update(color = 'reg')
                label_kwargs.update(color = "regression models")
            else:
                aes_kwargs.update(color = 'mod', linetype = 'reg', shape = 'reg')
                label_kwargs.update(color = "modulators", linetype = 'regression models', shape = 'regression models')
        elif use_ref_result and self.ref_result:
            aes_kwargs.update(color = 'reg', )
            label_kwargs.update(color = "regression models")
        elif self.modulators:
            aes_kwargs.update(color = 'mod')
            label_kwargs.update(color = "modulators")
        
        label_kwargs.update(color = color_name) if color_name is not None else None

        facet_args = dict(rows = 'prefix', cols = self.group_var)
        if facet_grid_kwargs is not None:
            facet_args.update(facet_grid_kwargs)
        
        if aes_override_kwargs is not None:
            aes_kwargs.update(aes_override_kwargs)

        if term_df_filter is not None:
            pltd = pltd.loc[term_df_filter]
            # potentially applying to ref_result's term_df

        pltd = pltd[~pltd['nlag'].isna()]

        if mod_to_facet and ('mod' in pltd) and (pltd.groupby(pltd['mod'].isna()).ngroups == 2):
            facet_args.update(cols = 'mod')
            pltd1, pltd2 = [x for _, x in pltd.groupby(pltd['mod'].isna())]
            min_lag = pltd['nlag'].min()
            p  = (
                pn.ggplot(pltd1, pn.aes(**aes_kwargs)) +
                    pn.facet_grid(**facet_args, labeller=pn.labeller(cols = lambda x: f"lag-{int(min_lag)} trial: {x}")) +  # type: ignore
                    pn.geom_hline(yintercept = 0, linetype = 'dashed', color = 'gray') +
                    pn.geom_pointrange(pn.aes(ymin = '[0.025', ymax = '0.975]'), **geom_pointrange_kwargs) +
                    pn.geom_line(**geom_line_kwargs) +
                    pn.geom_pointrange( # this layer is for the lag-1 (modulator) self coefficient to be plotted in all facets
                        data=pltd2.drop(columns = 'mod'), 
                        mapping=pn.aes(
                            ymin = '[0.025', ymax = '0.975]',
                            **{k: None for k, v in aes_kwargs.items() if v == 'mod'}), # reset aesthetic settings for 'mod'
                        **geom_pointrange_kwargs
                    ) +
                    signif_layer + # type: ignore
                    pn.labs(**label_kwargs) + 
                    pn.scale_color_discrete() + 
                    pn.theme_bw() +
                    pn.theme(
                        dpi = 600,
                        panel_background = pn.element_rect(fill = (1,1,1,0)),
                        panel_grid = pn.element_line(color = (0.85, 0.85, 0.85, 0.3))
                    )
            )
        else:
            p  = (
                pn.ggplot(pltd, pn.aes(**aes_kwargs)) +
                    pn.facet_grid(**facet_args) +
                    pn.geom_hline(yintercept = 0, linetype = 'dashed', color = 'gray') +
                    pn.geom_pointrange(pn.aes(ymin = '[0.025', ymax = '0.975]'), **geom_pointrange_kwargs) +
                    pn.geom_line(**geom_line_kwargs) +
                    signif_layer + # type: ignore
                    pn.labs(**label_kwargs) + 
                    pn.scale_color_discrete() + 
                    pn.theme_bw() +
                    pn.theme(
                        dpi = 600,
                        panel_background = pn.element_rect(fill = (1,1,1,0)),
                        panel_grid = pn.element_line(color = (0.85, 0.85, 0.85, 0.3))
                    )
            )
            
        return p
        
