from typing import Any, Hashable, Iterable, Mapping, MutableMapping, overload, Callable, List
from collections import ChainMap
import inspect
import logging

import cloudpickle

import numpy as np
import jax
import jax.numpy as jnp
import pandas as pd
import xarray as xr

def setup_jl():
    """import and return juliacall's Main

    Returns:
        module: juliacall's Main
    """

    from juliacall import Main as jl # type: ignore
    return jl


def pickle_to_file(obj, fp):
    with open(fp, 'wb') as handle:
        cloudpickle.dump(obj, handle)


def unpickle(fp):
    with open(fp, 'rb') as handle:
        res = cloudpickle.load(handle)
    
    return res

######### ============= some dictionary utility functions

def pop_exist(x:dict, key):
    """pop item from dict only when it exists
    return empty dict when not exists
    """
    if key in x:
        val = x.pop(key)
        return {key: val}
    else:
        return dict()
    
def extract_keys(x:dict, keys: list):
    """extract the keys from original dict to new dict
    the original dict will be modified
    """
    return dict(ChainMap(*[pop_exist(x, key) for key in keys]))

def update_r_dicts(d:MutableMapping, u:Mapping) -> MutableMapping:
    """update nested MutableMapping object

    Args:
        d (MutableMapping): the dictionary-like object to be updated, can be nested
        u (Mapping): the update to be applied

    Returns:
        MutableMapping: reference to the updated input `d`
    """
    for k, v in u.items():
        if isinstance(v, Mapping):
            d[k] = update_r_dicts(d.get(k, {}), v)
        else:
            d[k] = v
    return d

# def update_r_path(d: MutableMapping, path: Iterable[Hashable], val: Any):
#     path = iter(path)
#     try:
#         k = next(path)
#         if k not in d:
#             d[k] = {}  # Create a new dict if the key doesn't exist.
#         d[k] = update_r_path(d[k], path, val)
#         return d
#     except StopIteration as e:
#         return val

def update_r_path(d: MutableMapping, path: Iterable[Hashable], val: Any) -> MutableMapping:
    """update nested MutableMapping object with a path and value

    Args:
        d (MutableMapping): the dictionary-like object to be updated, can be nested
        path (Iterable[Hashable]): a path point to the place to be updated
        val (Any): a value to update with

    Returns:
        MutableMapping: in-place updated dictionary-like object
    """
    key, *rest = path
    if not rest:
        d[key] = val
    else:
        if key not in d or not isinstance(d[key], MutableMapping):
            d[key] = {}
        update_r_path(d[key], rest, val)
    return d


@overload
def update_r(d:MutableMapping, u:Mapping) -> MutableMapping:
    """update nested MutableMapping object with another Mapping

    Args:
        d (MutableMapping): the dictionary-like object to be updated, can be nested
        u (Mapping): the update to be applied

    Returns:
        MutableMapping: reference to the updated input `d`
    """
    ...

@overload
def update_r(d: MutableMapping, path: Iterable[Hashable], val: Any) -> MutableMapping:
    """update nested MutableMapping object with a path and value

    Args:
        d (MutableMapping): the dictionary-like object to be updated, can be nested
        path (Iterable[Hashable]): a path point to the place to be updated
        val (Any): a value to update with

    Returns:
        MutableMapping: in-place updated dictionary-like object
    """
    ...


def update_r(*args, **kwargs) -> MutableMapping:

    N_arg = len(args) + len(kwargs)
    if N_arg == 2:
        return update_r_dicts(*args, **kwargs)
    elif N_arg == 3:
        return update_r_path(*args, **kwargs)
    else:
        raise TypeError("illegal number of arguments")
    


def get_from_varargs(args_ind: int, kwargs_candidates: list, args, kwargs):
    """find an argument from *args, **kwargs

    Args:
        args_ind (int): the index of the wanted argument
        kwargs_candidates (list): potential names for the argument when passed as kwargs

    Raises:
        ValueError: _description_
        ValueError: _description_

    Returns:
        _type_: _description_
    """
    
    if args and args_ind < len(args):
        return args[args_ind]
    elif kwargs:
        for cand in kwargs_candidates:
            if cand in kwargs:
                return kwargs[cand]
        
        raise ValueError("cannot find the arg looked for")
    else:
        raise ValueError("please provide non-empty *args, **kwargs")

def _1st_arg_name(func:Callable):
    """get the first argument name of a function"""
    names = iter(inspect.signature(func).parameters.keys())
    return next(names) # the first argument name

def _select_func_by_1st_arg_type(type_func_dict:dict, args, kwargs):
    """dispatch functions based on the 1st argument's type
    """
    # get potential names of the 1st argument
    names = [_1st_arg_name(func) for func in type_func_dict.values()]
    
    # get the value of the 1st argument
    arg_1st = get_from_varargs(0, names, args, kwargs)

    if type(arg_1st) in type_func_dict:
        # run the corresponding function
        return type_func_dict[type(arg_1st)](*args, **kwargs)
    else:
        # check if the 1st arg's type is subclass of any type listed in the dict
        for key in type_func_dict:
            if isinstance(arg_1st, key):
                return type_func_dict[key](*args, **kwargs)
        raise ValueError(f"unknown 1st argument type: {type(arg_1st)}")


def isequal_pytree(x, y) -> bool:
    """
    Check equality of two pytrees by comparing their structures and leaf values.

    ### Parameters:
    - `x`: First pytree to compare.
    - `y`: Second pytree to compare.

    ### Returns:
    - `bool`: True if both pytrees have the same structure and identical leaves; False otherwise.

    ### Example:
    ```python
    # Assuming appropriate imports and pytree definitions
    result = isequal_pytree(pytree1, pytree2)
    print(result)  # True if pytrees are identical.
    ```

    ### Notes:
    - Utilizes JAX's `tree_structure` and `tree_leaves`.
    - Suitable for nested combinations of lists, tuples, dicts, etc.
    """
    if jax.tree_util.tree_structure(x) == jax.tree_util.tree_structure(y):
        x_leaves = jax.tree_util.tree_leaves(x)
        y_leaves = jax.tree_util.tree_leaves(y)

        for i in range(len(x_leaves)):
            if not hasattr(x_leaves[i], 'shape') or not hasattr(y_leaves[i], 'shape'):
                logging.info(f"[isequal_pytree] x_leave: {x_leaves[i]}, y_leave: {y_leaves[i]}, one of them does not have shape attribute")
                return False
            elif x_leaves[i].shape == y_leaves[i].shape and bool(jnp.all(x_leaves[i] == y_leaves[i])):
                continue
            else:
                return False
        return True
    return False



def flatten_multiindex(indi: pd.MultiIndex|pd.Index, name_ind_sep: str = "_", lev_sep: str = ":") -> list[str]:
    """
    Flatten a pandas MultiIndex into a list of strings with custom formatting.
    Also works for pandas Index

    Authored by chatGPT

    Parameters
    ----------
    indi : pd.MultiIndex
        The MultiIndex to flatten.
    name_ind_sep : str, optional
        Separator between the level name and the level value (default is "_").
    lev_sep : str, optional
        Separator between the different levels in the flattened index (default is ":").

    Returns
    -------
    list[str]
        A list of flattened index strings.
    
    Examples
    --------
    >>> mi = pd.MultiIndex.from_tuples(
    ...     [('updMLP_out', 0, 'target'),
    ...      ('updMLP_out', 1, 'target'),
    ...      ('updMLP_out', 2, 'target'),
    ...      ('updMLP_out', 3, 'target'),
    ...      ('updMLP_out', 4, 'target'),
    ...      ('updMLP_out', 5, 'target'),
    ...      ('updMLP_out', 6, 'target'),
    ...      ('updMLP_out', 7, 'target')],
    ...     names=[None, 'latent', 'upd_par']
    ... )
    >>> flatten_multiindex(mi)
    ['updMLP_out:latent_0:upd_par_target',
     'updMLP_out:latent_1:upd_par_target',
     'updMLP_out:latent_2:upd_par_target',
     'updMLP_out:latent_3:upd_par_target',
     'updMLP_out:latent_4:upd_par_target',
     'updMLP_out:latent_5:upd_par_target',
     'updMLP_out:latent_6:upd_par_target',
     'updMLP_out:latent_7:upd_par_target']
    
    >>> idx = pd.Index(['monCh', 'rew'], name='in_feat')
    >>> flatten_multiindex(idx)
    ['in_feat_monCh', 'in_feat_rew']
    """
    if isinstance(indi, pd.MultiIndex):
        # Process each tuple in the MultiIndex.
        flattened = [
            lev_sep.join(
                (f"{name}{name_ind_sep}{val}" if name is not None else str(val))
                for val, name in zip(tup, indi.names)
            )
            for tup in indi
        ]
    else:
        # For a single-level Index, treat each element as a whole value.
        if indi.name is not None:
            flattened = [f"{indi.name}{name_ind_sep}{val}" for val in indi]
        else:
            flattened = [str(val) for val in indi]
    return flattened

def ds2df(ds: xr.Dataset, index_dims: None | List[Hashable] = None, omit: List = [], omit_var_names: List = [], omit_dim_names: List = []):
    """
    Convert an xarray Dataset with variables that share some common dimensions
    into a pandas DataFrame where the common dimensions form the row MultiIndex
    and the remaining (extra) dimensions for each variable are expanded to columns.
    
    Parameters
    ----------
    ds: The input dataset.
    index_dims: the dimension used as index. if None, default to dimensions that are shared across all variables
    omit: list of names to omit in the output column names, will match against both variables and dimensions
    omit_var_names: list of variable names to omit in the final column names
    omit_dim_names: list of dimension names to omit in the final column names
    
    Returns
    -------
    df_final: A DataFrame whose rows are indexed by the common dimensions and whose columns are variables combined with unstacked remaining dimensions.
    """
    # Get a list of variable names.
    var_names:List[str] = list(ds.data_vars)
    if not var_names:
        raise ValueError("Dataset has no data variables.")
    
    if index_dims is None:
        # Determine the common dimensions. We use the ordering from the first variable.
        first = ds[var_names[0]]
        common_dims = [d for d in first.dims if all(d in ds[var].dims for var in var_names)]
        index_dims = common_dims

    omit_var_names = list(set(omit).union(omit_var_names))
    omit_dim_names = list(set(omit).union(omit_dim_names))
    
    dfs = []

    for var in var_names:
        
        # Convert the DataArray to a pandas Series whose index is a MultiIndex
        # (common dims + extra dims, in order).
        s = ds[var].to_dataframe()

        # move extra columns to index if there's any
        extra_cols = [c for c in s.columns if c != var]
        s.set_index(extra_cols, append=True, inplace=True)

        df_var = s.unstack([ind for ind in s.index.names if ind not in index_dims])
        cols = df_var.columns

        if var in omit_var_names:
            cols = cols.droplevel(0) # the first level will always be the variable name
        
        cols.names = [n if n not in omit_dim_names else None for n in cols.names]
        
        df_var.columns = flatten_multiindex(cols)
            
        
        dfs.append(df_var)
    
    # Concatenate all the per-variable DataFrames along columns.
    df_final = pd.concat(dfs, axis=1)
    df_final.sort_index(inplace=True)
    return df_final


def min_max(x):
    return np.array([np.nanmin(x), np.nanmax(x)])



from flax.serialization import msgpack_serialize, msgpack_restore

def msgpack_serialize_to_file(pytree, fp):
    byt = msgpack_serialize(pytree)
    with open(fp, 'wb') as handle:
        handle.write(byt)

def msgpack_restore_from_file(fp):
    with open(fp, 'rb') as handle:
        byt = handle.read()
    return msgpack_restore(byt)
