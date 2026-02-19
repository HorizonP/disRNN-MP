import abc
import importlib
import inspect
from pathlib import Path
from typing import Any, Deque, List, MutableMapping, Optional, Mapping, Tuple, Union, Callable
import numpy as np
import jax
import pandas as pd
import haiku as hk
from jax.typing import ArrayLike
from optax import OptState

from pydantic import ValidationError, create_model


Array = Union[np.ndarray, jax.Array] # remove scalar types from jax.typing.ArrayLike
"""both numpy and jax array"""

NestedDict = MutableMapping[str, Array | 'Params']
Params = NestedDict
"""haiku style parameter: nested dictionary str -> Array"""

ListLike = Union[Array, List, Deque, pd.Series]
PathLike = Union[Path, str]

RandomKey = jax.Array
Inputs = Array | NestedDict
Outputs = jax.Array
Loss = ArrayLike
States = jax.Array
BatchSize = int
TrainStepFun = Callable[[Params, RandomKey, OptState, Inputs, Outputs], Tuple[Loss, Params, OptState]]
RNN_Apply_Fun = Callable[[Params, RandomKey, Inputs], Tuple[Outputs, States]]
LossFun = Callable[[Params, RandomKey, Inputs, Outputs], Loss]


def create_pydantic_model(sig: inspect.Signature, obj_name:Optional[str] = None):
    """create pydantic model from a `inspect.Signature`
    - arguments without type annotation will be assigned 'Any' type
    - arguments that does not exist in the signature will be foridden during checking by the returned model
    """

    if obj_name is None:
        obj_name = 'anonymous'
    
    # Prepare field definitions with types, using `Any` for parameters without annotations
    fields = {
        name: (param.annotation if param.annotation != param.empty else Any, ...)
        for name, param in sig.parameters.items()
    }
    
    # Create a custom configuration to forbid extra fields
    class Config:
        extra = 'forbid'
    
    # Dynamically create the Pydantic model with custom configuration
    return create_model(obj_name, __config__=Config, **fields) # type: ignore

def validate_instantiatable(inst_dict: Mapping['str', Any]):
    """validate a dictionary to be passed to `hydra.utils.instantiate`"""

    for _, value in inst_dict.items():
        if isinstance(value, dict) and '_target_' in value:
            raise ValueError('validate_params cannot recursively validate nested instantiatable dictionaries')
    
    inst_dict = dict(inst_dict)

    target_path = inst_dict.pop('_target_')
    module_path, func_name = target_path.rsplit('.', 1)
    module = importlib.import_module(module_path)
    func = getattr(module, func_name)
    # Inspect the function signature
    sig = inspect.signature(func)
    print(f"the function's signature is {sig}")

    # Create a Pydantic model for this function
    DynamicModel = create_pydantic_model(sig, func_name)

    # Validate the parameters
    try:
        validated_params = DynamicModel(**inst_dict)
        print("Parameters are valid:", validated_params)
    except ValidationError as e:
        print("Type validation error:", e)


class patchable_hkRNNCore(hk.RNNCore):

    @abc.abstractmethod
    def set_patch_state_ids(self, patch_state_ids):
        """set which latents to be patched

        Args:
            patch_state_ids (List[int]): index of latents to patch
        """

    @abc.abstractmethod
    def step_with_exo_state(self, observations: Array, prev_latents: Array, exo_states: Array) -> Tuple[dict[str, jax.Array] | jax.Array, jax.Array]:
        """similar to __call__ method but with the ability to substitute some of the latents by `exo_states`

        Args:
            observations (Array): _description_
            prev_latents (Array): _description_
            exo_states (Array): _description_

        Returns:
            Tuple[dict[str, jax.Array], jax.Array]: _description_
        """