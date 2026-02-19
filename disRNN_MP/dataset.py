from typing import Collection, Optional, Tuple, Union, List, overload, Literal
from collections.abc import Sized
from pathlib import Path
import logging
from warnings import warn

import numpy as np
import jax
import jax.numpy as jnp
import pandas as pd
import xarray as xr

from .typing import Array

def match_patterns(pats, strings: Collection[str]):
    """match each pattern to the collection of strings, return all unique matches"""
    strs_ = pd.Series(strings) # type: ignore
    matches = pd.concat(map(lambda pat: strs_[strs_.str.match(pat)], pats))
    return matches.drop_duplicates().to_list()

def read_dataframe(df: pd.DataFrame|Path|str):
    """read a file to create a dataframe
    if it is already dataframe, it will be directly returned

    not complete list of supported format:
    - csv
    - feather/arrow
    - parquet
    """
    if not isinstance(df, pd.DataFrame):
        fp = Path(df)
        match fp.suffix:
            case '.csv':
                df = pd.read_csv(fp)
            case '.feather' | '.arrow':
                df = pd.read_feather(fp)
            case '.parquet':
                df = pd.read_parquet(fp)
            case _:
                df = pd.read_table(fp)
    
    return df

def _proc_train_sess_samp(
        n_sess_sample: None | int | float | str, 
        n_sess: int, 
        seed: int = 0) -> Tuple[np.ndarray, np.ndarray]:
    """internal function to processing the `n_sess_sample` or `train_sess_samp` argument appeared in several functions

    Args:
        n_sess_sample (None | int | float | str): _description_
        n_sess (int): number of total sessions
        seed (int, optional): random seed for the sampling. Defaults to 0.

    Raises:
        ValueError: unknown n_sess_sample argument
    
    Return: 
        Tuple[np.ndarray, np.ndarray]: training session index, testing session index
    """

    if n_sess == 1:
        logging.warning(
            "when there's only 1 session avilable, either train or test indices will be empty")
        
    if isinstance(n_sess_sample, (int, float)):
        if n_sess_sample > n_sess:
            raise ValueError(f'tried to choose {n_sess_sample} sessions out of {n_sess} sessions')

        # when n_sess_sample <= n_sess:
        
        np.random.seed(seed)
        if n_sess_sample < 1:  # interpret the argument as proportion
            sz_ = round(n_sess * n_sess_sample)
        else:
            if n_sess_sample == 1:
                logging.warning('when n_sess_sample = 1, it is interpreted as sampling 1 session')
            sz_ = round(n_sess_sample)

        train_ind = np.random.choice(n_sess, size=sz_, replace=False)
        test_ind = np.setdiff1d(np.arange(n_sess), train_ind)
    
    elif isinstance(n_sess_sample, str):
        match n_sess_sample:
            case "every-other":
                train_ind = np.arange(0, n_sess, 2)
                test_ind = np.setdiff1d(np.arange(n_sess), train_ind)
            case "every-other2":
                train_ind = np.arange(1, n_sess, 2)
                test_ind = np.setdiff1d(np.arange(n_sess), train_ind)
            case _:
                raise ValueError(f"unknown n_sess_sample keyword: {n_sess_sample}")
        
    else:
        train_ind = np.arange(n_sess)
        test_ind = train_ind
        logging.info(
            "it is instructed to use all sessions as both training and testing dataset")
    
    return train_ind, test_ind
class trainingDataset:
    """Holds a dataset for training an RNN or fitting a model on bandit task, consisting of inputs and targets.

        Both inputs and targets are stored as [timestep, episode, feature]
        Serves them up in batches
    """

    n_observations: int
    """total number of time steps across all episode"""

    @classmethod
    def from_df(cls,
        df: Union[pd.DataFrame, Path, str], 
        x_vars: List[str], 
        y_vars: List[str],
        regex_match: bool = False,
        tri_cutoff: int = 1000,
        sess_select: Optional[list|np.ndarray] = None,
        tri_ind_var: str = 'ntri_', 
        sess_ind_var: str = 'sess.id',
        **kwargs,
        ):

        df = read_dataframe(df)

        if regex_match:
            x_vars = match_patterns(x_vars, df.columns)
            y_vars = match_patterns(y_vars, df.columns)

        _df_vars_to_da = lambda vars: _df_to_xr_3darray(df, vars, tri_cutoff=tri_cutoff, sess_select=sess_select, tri_ind_var=tri_ind_var, sess_ind_var=sess_ind_var)
        xv_da = _df_vars_to_da(x_vars)
        yv_da = _df_vars_to_da(y_vars)

        # nparr = data_tensor_from_df(
        #     df=df, vars=x_vars + y_vars, return_xarray=False, regex_match=regex_match, tri_cutoff=tri_cutoff, sess_select=sess_select,
        #     tri_ind_var=tri_ind_var, sess_ind_var=sess_ind_var)
        # nparr = xr.concat((xv_da, yv_da), dim = 'var').to_numpy().astype("float")

        init_kwargs = {
            'input_feature_name': x_vars,
            'output_feature_name': y_vars,
            'sess_ids': xv_da['sess'].to_numpy(),
        }
        init_kwargs.update(kwargs) # override default

        return trainingDataset(
            xs=xv_da.to_numpy().astype("float"), 
            ys=yv_da.to_numpy().astype("float"), 
            **init_kwargs  # type: ignore
        )
        
        # return makeDataset_nparr(
        #     nparr, 
        #     input_feature_name = x_vars, 
        #     output_feature_name = y_vars,
        #     N_y_feat=len(y_vars))
        


    def __init__(self,
                 xs: Array,
                 ys: Array,
                 batch_size: Optional[int] = None,
                 input_feature_name: None | List[str] | Tuple[str, ...] = None,
                 output_label: None | List[str] | Tuple[str, ...] = None,
                 n_action: Optional[int] = None,
                 fill_na: float = -1,
                 output_feature_name: None | List[str] | Tuple[str, ...] = None,
                 sess_ids = None
                 ):
        """Do error checking and bin up the dataset into batches.

        Args:
          xs: Values to become inputs to the network.
            Should have dimensionality [timestep, episode, feature]
          ys: Values to become output targets for the RNN.
            Should have dimensionality [timestep, episode, feature]
          batch_size: The size of the batch (number of episodes) to serve up each
            time next() is called. If not specified, all episodes in the dataset 
            will be served
          feature_name: (optional) the name for each feature
          n_action: (optional) number of possible actions, default is the number of unique values of `ys`
          fill_na: (optional) the number used for filling NaN values
        """

        if batch_size is None:
            batch_size = xs.shape[1]

        # Error checking
        # Do xs and ys have the same number of timesteps?
        if xs.shape[0] != ys.shape[0]:
            msg = ('number of timesteps in xs {} must be equal to number of timesteps'
                   ' in ys {}.')
            raise ValueError(msg.format(xs.shape[0], ys.shape[0]))

        # Do xs and ys have the same number of episodes?
        if xs.shape[1] != ys.shape[1]:
            msg = ('number of sessions in xs {} must be equal to number of sessions'
                   ' in ys {}.')
            raise ValueError(msg.format(xs.shape[1], ys.shape[1]))
        
        # batch size has to be less or equal to dataset size
        if batch_size > xs.shape[1]:
            raise ValueError(f'batch-size of {batch_size} is larger than number of sessions {xs.shape[1]}')

        if input_feature_name is None:
            input_feature_name = ["in_f" + str(i) for i in range(xs.shape[2])]
        else:
            if len(input_feature_name) != xs.shape[2]:
                raise ValueError(
                    f"input_feature_name length ({len(input_feature_name)}) does not match input feature dim ({xs.shape[2]})")
        
        if output_feature_name is None:
            output_feature_name = ["out_f" + str(i) for i in range(ys.shape[2])]
        else:
            if len(output_feature_name) != ys.shape[2]:
                raise ValueError(
                    f"output_feature_name length ({len(output_feature_name)}) does not match output feature dim ({ys.shape[2]})")
        
        if n_action is None:
            # ignore nan when count unique type of actions
            n_action = np.sum(np.logical_not(np.isnan(np.unique(ys))))
        
        if output_label is None:
            output_label = [f'choice {i}' for i in range(n_action)]

        if (np.any(xs == fill_na) or np.any(ys == fill_na)) and (np.any(np.isnan(xs)) or np.any(np.isnan(ys))):
            warn(f"the original dataset already contains elements equal to `fill_na` ({fill_na}) as well as NaN elements, filling NaN with `fill_na` will cause confusion")
        
        # fill NA 
        xs = np.nan_to_num(xs, copy=True, nan=fill_na, posinf=fill_na, neginf=fill_na)
        ys = np.nan_to_num(ys, copy=True, nan=fill_na, posinf=fill_na, neginf=fill_na)

        # Property setting
        self.xs = jnp.array(xs)
        self.ys = jnp.array(ys)
        self._na_as = fill_na
        self._batch_size = batch_size
        self._dataset_size = self.xs.shape[1]
        self._idx = 0
        self.n_batches = self._dataset_size // self._batch_size
        self.input_feature_name = input_feature_name
        self.output_feature_name = output_feature_name
        self.n_action = n_action
        self.output_label = output_label
        self.n_observations = int(np.sum(np.any(np.concatenate([xs, ys], axis=-1) != fill_na, axis=-1)))
        self._sess_ids = sess_ids

    def __iter__(self):
        return self
    
    def set_seed(self, seed):
        pass

    @property
    def shape(self):
        return (self.xs.shape, self.ys.shape)

    def __next__(self) -> Tuple[jax.Array, jax.Array]:
        """Return a batch of data, including both xs and ys.

        Returns:
          x, y: next input (x) and target (y) in sequence.
        """

        # Define the chunk we want: from idx to idx + batch_size
        start = self._idx
        end = start + self._batch_size
        # Check that we're not trying to overshoot the size of the dataset
        assert end <= self._dataset_size

        # Update the index for next time
        if end == self._dataset_size:
            self._idx = 0
        else:
            self._idx = end

        # Get the chunks of data
        x, y = self.xs[:, start:end], self.ys[:, start:end]

        return x, y
    
    def to_xr_dataset(self):
        """convert trainingDataset to xarray.Dataset
        
        DataArrays: 
            inputs: xs
            outputs: ys

        Coordinates:
            tri
            sess
            in_feat: input_feature_name
            out_feat: output_feature_name
        
        """
        # convert NA-representing numbers back to NA
        xs = jax.device_get(self.xs.at[np.where(self.xs == self._na_as)].set(jnp.nan))
        ys = jax.device_get(self.ys.at[np.where(self.ys == self._na_as)].set(jnp.nan))
        data_vars = {
            'inputs': (["tri", "sess", "in_feat"], xs),
            'outputs': (["tri", "sess", "out_feat"], ys),
        }

        coords = {
            'sess': np.arange(self.xs.shape[1]) if self._sess_ids is None else self._sess_ids,
            'tri': np.arange(self.xs.shape[0]),
            'in_feat': self.input_feature_name,
            'out_feat': self.output_feature_name,
        }

        return xr.Dataset(data_vars=data_vars, coords = coords)

    

class randSampTrainingDataset(trainingDataset):

    def __init__(self, 
            xs: Array, ys: Array, 
            batch_size: int | None = None, 
            input_feature_name: Sized | None = None, 
            n_action: int | None = None, 
            fill_na: float = -1, random_seed: Optional[int] = None):
        super().__init__(xs, ys, batch_size, input_feature_name, n_action, fill_na)

        if random_seed is None:
            random_seed = np.random.random_integers(0, 1e6)
        self.rng = np.random.default_rng(random_seed)

    def set_seed(self, seed):
        self.rng = np.random.default_rng(seed)

    def __next__(self) -> Tuple[jax.Array, jax.Array]:
        samp_ind = self.rng.choice(self._dataset_size, self._batch_size, replace=False)

        x, y = self.xs[:, samp_ind], self.ys[:, samp_ind]
        return x, y






def makeDataset_nparr(nparr: np.ndarray, N_y_feat: int = 1, **kwargs):
    """
    validate and generate an instance of trainingDataset class from a numpy 3d array

    shape: (n_trials x n_sessions x (n_observation_features + n_y_features))
    assume the axis2[:-N_y_feat] are all input features, while axis[-N_y_feat:] is target to predict
    """

    assert len(nparr.shape) == 3  # make sure 3d array

    return trainingDataset(
        xs=nparr[:, :, :-N_y_feat], ys=nparr[:, :, -N_y_feat:], **kwargs)




def train_test_datasets(
        dat_or_path: np.ndarray|str|Path, 
        n_sess_sample: Optional[int|float|str]=None, 
        min_sess: Optional[int]=41, 
        seed: int=0, 
        in_feat_name: Optional[List[str]]=None,
        input_ind: Union[List[int], Tuple[int], slice] = slice(-1),
        output_ind: Union[List[int], Tuple[int], slice] = [-1],
        train_batch_samp: Optional[str] = None,
        batch_size: Optional[int] = None,
        n_action: Optional[int] = None,
        fill_na: float = -1,
        sess_ids: list|np.ndarray|None=None,
        ):
    """validate, generate and split the train and test datasets from a numpy 3d array

    shape: (n_trials x n_sessions x (n_observation_features + 1))
    by default assume the axis2[:-1] are all input features, while axis[-1] is target to predict

    ## Args:
        - dat_or_path (np.ndarray | str | Path): _description_
        - n_sess_sample (Optional[int | float | str], optional): 
            - Defaults to None.
            - None -- use all sessions as train, the same set of sessions as test 
            - numbers between 0 and 1 (not include) -- choose a fraction of sessions as train, the other as test
            - str -- special way to split. 
                - "every-other" will use ::2 as train, 1::2 as test
                - "every-other2" will use 1::2 as train, ::2 as test
        - min_sess (Optional[int], optional): the minimum number of session to use for train and test, if not enough, duplicate the data to reach this minimum. Defaults to 41. If None, do not use this feature
        - seed (int, optional): _description_. Defaults to 0.
        - in_feat_name (Optional[List[str]], optional): _description_. Defaults to None.
        - input_ind (Union[List[int], Tuple[int], slice], optional): _description_. Defaults to slice(-1).
        - output_ind (Union[List[int], Tuple[int], slice], optional): _description_. Defaults to [-1].
        - train_batch_samp: whether and how to generate batches for training dataset. None means use all data for training at once, "random" means randomly sample batches
        - sess_ids: the ID for each session. If provided, must match the length of session dimension of the data, and will be attached to the training and testing dataset for future reference

    Raises:
        ValueError: _description_

    Returns:
        _type_: _description_
    """

    if isinstance(dat_or_path, (Path, str)):
        dat = np.load(dat_or_path)
    else:
        dat = dat_or_path

    dat = dat.astype(np.float64)
    logging.info(f"raw data has a shape of {dat.shape}")

    assert len(dat.shape) == 3  # make sure 3d array

    if sess_ids is not None:
        sess_ids = np.asarray(sess_ids).reshape((-1, )) # convert to numpy 1d array
        assert sess_ids.shape[0] == dat.shape[1], f"sess_id shape ({sess_ids.shape}) does not match data's session dimension ({dat.shape[1]})"

    # TODO make sure choices are coded starting from 0

    # TODO make sure target is not contained in input

    def dup_sess(dat, sess_ids=None):
        # duplicate the data to make sure it satisfy the minimum number of sessions (min_sess)
        if min_sess is not None and dat.shape[1] < min_sess:
            n_rep = np.ceil(min_sess / dat.shape[1]).astype("int")
            dat = np.tile(dat, (1, n_rep, 1))
            # log.info(f"data is duplicated to have a shape of {dat.shape} to meet min_sess requirement")

            if sess_ids is not None: # match `sess_ids` to `dat`
                sess_ids = np.tile(sess_ids, (n_rep, ))
        return dat, sess_ids

    train_ind, test_ind = _proc_train_sess_samp(n_sess_sample, dat.shape[1], seed = seed)
    # ===== making the training dataset

    train_d, train_sess_ids = dup_sess(dat[:, train_ind, :], sess_ids[train_ind] if sess_ids is not None else None)

    if train_batch_samp is None:
        train_dt = trainingDataset(
            train_d[:, :, input_ind], train_d[:, :, output_ind], 
            input_feature_name=in_feat_name, 
            batch_size = batch_size, 
            n_action=n_action, 
            fill_na=fill_na, 
            sess_ids=train_sess_ids
        )
    elif train_batch_samp.lower() == "random":
        train_dt = randSampTrainingDataset(
            train_d[:, :, input_ind], train_d[:, :, output_ind], 
            input_feature_name=in_feat_name, batch_size = batch_size, n_action=n_action, fill_na=fill_na)
    else:
        raise ValueError(f"unrecognized train_batch_samp value: {train_batch_samp}. It has to be either None or 'random'")

    
    logging.info(f"training dataset (after duplicate to meet min_sess {min_sess}) has a shape of {train_dt.shape}")

    # ===== making the testing dataset

    test_d, test_sess_ids = dup_sess(dat[:, test_ind, :], sess_ids[test_ind] if sess_ids is not None else None)
    test_dt = trainingDataset(
        test_d[:, :, input_ind], test_d[:, :, output_ind], 
        input_feature_name=in_feat_name, 
        n_action=n_action, 
        fill_na=fill_na,
        sess_ids=test_sess_ids
    )
    logging.info(f"testing dataset (after duplicate to meet min_sess {min_sess}) has a shape of {test_dt.shape}")
    logging.info(f"training session indices: {sorted(train_ind)}")
    logging.info(f"testing session indices: {sorted(test_ind)}")

    return train_dt, test_dt


def _df_to_xr_3darray(
        df: pd.DataFrame, 
        vars: List[str], 
        tri_cutoff: int|float = 1000,
        sess_select: None | List | np.ndarray = None,
        tri_ind_var: str = 'ntri_', 
        sess_ind_var: str = 'sess.id',
    ) -> xr.DataArray:
    """convert a pandas dataframe to 3d xarray DataArray with trials, sessions, features dimenions

    trial dimension will be sorted ascendingly according to `tri_ind_var`

    Args:
        df (pd.DataFrame): input pandas dataframe
        vars (List[str]): selected df columns to be used as features
        tri_cutoff (int, optional): cut-off of trial index. Defaults to 1000.
        sess_select (None | List | np.ndarray, optional): if provided, select only these sessions. Defaults to None.
        tri_ind_var (str, optional): df column name for trial index. Defaults to 'ntri_'.
        sess_ind_var (str, optional): df column name for session ID. Defaults to 'sess.id'.

    Returns:
        xr.DataArray: 3d xarray DataArray with trials, sessions, features dimenions
    """
    
    if sess_select is None:
        sess_select = df[sess_ind_var].unique()

    return (
        df[(df[tri_ind_var]<=tri_cutoff) & (df[sess_ind_var].isin(sess_select))] # select trials and sessions
        .set_index([tri_ind_var, sess_ind_var])
        [vars] # select df columns as variables
        .to_xarray()
        .to_dataarray('var')
        .rename({tri_ind_var: 'tri', sess_ind_var: 'sess'})
        .transpose('tri', 'sess', 'var')
        .sortby('tri')
    )


@overload
def data_tensor_from_df(
        df: Union[pd.DataFrame, Path, str], 
        vars: List[str], 
        return_xarray: Literal[True],
        regex_match: bool = False,
        tri_cutoff: int = 1000,
        sess_select: Optional[list|np.ndarray] = None,
        tri_ind_var: str = 'ntri_', 
        sess_ind_var: str = 'sess.id',
    ) -> xr.DataArray:
    ...

@overload
def data_tensor_from_df(
        df: Union[pd.DataFrame, Path, str], 
        vars: List[str], 
        return_xarray: Literal[False] = ...,
        regex_match: bool = False,
        tri_cutoff: int = 1000,
        sess_select: Optional[list|np.ndarray] = None,
        tri_ind_var: str = 'ntri_', 
        sess_ind_var: str = 'sess.id',
    ) -> np.ndarray:
    ...

def data_tensor_from_df(
        df: Union[pd.DataFrame, Path, str], 
        vars: List[str], 
        return_xarray: bool = False,
        regex_match: bool = False,
        tri_cutoff: int = 1000,
        sess_select: Optional[list|np.ndarray] = None,
        tri_ind_var: str = 'ntri_', 
        sess_ind_var: str = 'sess.id',
    ) -> np.ndarray|xr.DataArray:
    """create a 3D tensor for RNN network input or output

    Args:
        df (Union[pd.DataFrame, Path, str]): specify a long dataframe
        vars (List[str]): name of columns to be included in the 3rd dimension. They will be ordered along the 3rd dimension as specified in this argument
        regex_match(bool, optional): whether `vars` are meant to be a list of patterns against which the column names will be matched
        tri_cutoff (int, optional): use trials from 1 to `tri_cutoff`. Defaults to 1000.
        sess_select (Optional[list | np.ndarray], optional): keys of the sessions to be used. Defaults to None, which uses all sessions
        tri_ind_var (str, optional): which variable in the dataframe represent trial index. Defaults to 'ntri_'.
        sess_ind_var (str, optional): which variable represent session keys. Defaults to 'sess.id'.
        return_xarray (bool, optional): whether return a xarray.dataarray type or numpy.ndarray type

    Returns:
        np.ndarray|xr.DataArray: (N_tri, N_sess, N_vars)
    """

    if not isinstance(df, pd.DataFrame):
        fp = Path(df)
        match fp.suffix:
            case '.csv':
                df = pd.read_csv(fp)
            case '.feather' | '.arrow':
                df = pd.read_feather(fp)
            case '.parquet':
                df = pd.read_parquet(fp)
            case _:
                df = pd.read_table(fp)

    if regex_match:
        # match each element in vars against column names and collect them into a list
        matches = list(map(lambda pat: df.columns[df.columns.str.match(pat)].to_series(), vars))

        # concat them to a 1d array and remove duplicate matches
        vars = pd.concat(matches).drop_duplicates(keep="first").to_list()

    logging.debug(f"[data_tensor_from_df] columns selected for 3rd dimension are: ${vars}")

    nparr = _df_to_xr_3darray(df = df, vars = vars, tri_cutoff = tri_cutoff, sess_select=sess_select, tri_ind_var=tri_ind_var, sess_ind_var=sess_ind_var)

    if return_xarray:
        return nparr
    else:
        return nparr.to_numpy().astype("float")


def train_test_datasets_from_df(
        df: Union[pd.DataFrame, Path, str], x_vars: List[str], y_vars: List[str], 
        train_sess_sample: Optional[Union[int, float, str]], seed: int = 0, min_sess: Optional[int] = 41, 
        tri_cutoff:int = 1000, tri_ind_var: str = 'ntri_', sess_ind_var: str = 'sess.id',
        train_batch_samp: Optional[str] = None, batch_size: Optional[int] = None, **kwargs):
    """create a pair of train and test dataset from a long dataframe by random sampling sessions

    sessions will be randomly splited to create train and test dataset
    trials from first upto `tri_cutoff` will be included in the function's output. Shorter session will be padded with -1

    Args:
        df (Union[pd.DataFrame, Path, str]): specify a long dataframe
        x_vars (List[str]): name of columns to be used for making model inputs
        y_vars (List[str]): name of columns to be used for making target model outputs
        train_sess_sample (Union[int, float]): how many sessions for training dataset, the rest will be testing dataset. If a float number (0<x<1) is provided, will use it as the training portion
        seed (int, optional): random seed for train/test splitting. Defaults to 0.
        min_sess (int, optional): _description_. Defaults to 41.
        tri_cutoff (int, optional): _description_. Defaults to 1000.
        tri_ind_var (str, optional): _description_. Defaults to 'ntri_'.
        sess_ind_var (str, optional): _description_. Defaults to 'sess.id'.

    Returns:
        _type_: _description_
    """
    
    # convert dataframe to tensor of shape (N_trial, N_sess, N_x_feat + N_y_feat)
    xarr = data_tensor_from_df(
        df = df,
        vars = x_vars + y_vars,
        tri_cutoff = tri_cutoff,
        tri_ind_var = tri_ind_var,
        sess_ind_var = sess_ind_var,
        return_xarray=True
    )

    nparr = xarr.to_numpy().astype("float")
    sess_ids = xarr['sess'].to_numpy()

    return train_test_datasets(
        nparr, 
        n_sess_sample=train_sess_sample, 
        min_sess=min_sess, 
        seed=seed, 
        in_feat_name=x_vars, input_ind=slice(len(x_vars)), 
        output_ind=slice(len(x_vars), None),
        train_batch_samp = train_batch_samp, batch_size = batch_size,
        sess_ids=sess_ids,
        **kwargs
    )



def stratified_train_test_datasets_from_df(
        df: Union[pd.DataFrame, Path, str], 
        x_vars: List[str], 
        y_vars: List[str], 
        group_vars: List[str],
        train_sess_sample: Optional[Union[int, float, str]], 
        seed: int = 0,
        tri_cutoff:int = 1000, 
        tri_ind_var: str = 'ntri_', 
        sess_ind_var: str = 'sess.id',
    ):
    """make stratified training and testing datasets pair from a dataframe

    The sessions are first stratified by the `group_vars` before being split into train and test by `train_sess_sample` and `seed` arguments. Train and test data from each stratium are collected to the final train and test data

    Implementation details:


    Args:
        df (Union[pd.DataFrame, Path, str]): dataframe or path of dataframe 
        x_vars (List[str]): _description_
        y_vars (List[str]): _description_
        group_vars (List[str]): _description_
        train_sess_sample (Optional[Union[int, float, str]]): _description_
        seed (int, optional): _description_. Defaults to 0.
        tri_cutoff (int, optional): _description_. Defaults to 1000.
        tri_ind_var (str, optional): _description_. Defaults to 'ntri_'.
        sess_ind_var (str, optional): _description_. Defaults to 'sess.id'.
    """

    df = read_dataframe(df)

    train_dfs = []
    test_dfs = []

    for _, grp in df.groupby(group_vars):

        sess_ids = grp[sess_ind_var].unique()

        train_ind, test_ind = _proc_train_sess_samp(train_sess_sample, len(sess_ids), seed=seed)

        train_dfs.append(grp[grp[sess_ind_var].isin(sess_ids[train_ind])]) 
        test_dfs.append(grp[grp[sess_ind_var].isin(sess_ids[test_ind])])

    train_df = pd.concat(train_dfs, axis=0)
    test_df = pd.concat(test_dfs, axis=0)

    _make_td = lambda df:  trainingDataset.from_df(
        df = df,
        x_vars=x_vars,
        y_vars=y_vars,
        tri_cutoff=tri_cutoff,
        sess_ind_var=sess_ind_var,
        tri_ind_var=tri_ind_var
    )
    
    return _make_td(train_df), _make_td(test_df)

        