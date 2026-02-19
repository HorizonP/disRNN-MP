import jax


import functools
import importlib
from copy import copy
from typing import Any


def _pkl_instantiate(container: Any) -> Any:
    """
    Instantiates an object from a Hydra-compatible configuration dictionary.

    This function supports instantiating Python objects from a dictionary that contains
    a special '_target_' key, indicating the object to be instantiated. It recursively
    processes nested dictionaries.

    ### Parameters:
    - `config`: A dictionary potentially containing the '_target_' key and other parameters for instantiation.

    ### Returns:
    - The instantiated Python object or, if '_target_' is not present, the processed dictionary.

    ### Usage:
    - Intended to be used with Hydra configurations for dynamic object instantiation in complex applications.

    ### Notes:
    - If '_target_' is a string, it's assumed to be a class or function path and is imported from the module
    - If '_target_' is an object, it's directly invoked with the remaining configuration.
    - Supports deep instantiation for nested configuration structures.
    - If a dictionary container has '_partial_' key and its value is True, the object will be instantiated by `jax.tree_util.Partial`
    - If a dictionary container has '_args_' key and its value is a list, each of its elements will be processed by this function and the processed list will be passed as func(*_args_)
    """
    if isinstance(container, dict):
        container = copy(container) # only copy current level

        # depth first processing any nested dictionary
        for key, val in container.items():
            container[key] = _pkl_instantiate(val)

        # after processing all of its items, process the dict itself
        if '_target_' not in container:
            # this is a normal dict
            return container
        else:
            targ = container.pop('_target_')

            if isinstance(targ, str):
                # return instantiate(config)

                paths = targ.split('.')
                # parent_module = importlib.import_module(paths[0])

                # recursively load module or objects after the initial module
                # this method will be robust to class methods
                # for example, it works for: ModA.submodB.classC.classmethodD
                # targ = functools.reduce(getattr, paths[1:], parent_module)

                i = 1
                while i < len(paths):
                    try:
                        module = importlib.import_module('.'.join(paths[:-i]))
                        break
                    except (ModuleNotFoundError, ) as e:
                        i += 1
                        continue

                # recursively load the attribute from first non-module object
                targ = functools.reduce(getattr, paths[(-i):], module)

            if_partial = False
            posi_args = []
            if '_partial_' in container:
                if_partial = container.pop('_partial_')

            if '_args_' in container:
                posi_args = _pkl_instantiate(container.pop('_args_'))

            if if_partial:
                    return jax.tree_util.Partial(targ, *posi_args, **container)
            else:
                return targ(*posi_args, **container)
    elif isinstance(container, list):
        container = copy(container) # only copy current level
        for i in range(len(container)):
            container[i] = _pkl_instantiate(container[i])
        return container
    else:
        return container