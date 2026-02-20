from typing import Tuple, List, Sequence

import numpy as np
import jax
import haiku as hk

from .disrnn_def import hkDisRNN
from ..dataset import trainingDataset
from .train import RNNtraining
from .utils import transform_hkRNN, RNNtransformed

def make_disrnn_funcs(latent_size, update_mlp_shape, choice_mlp_shape, target_size=None, sample_dataset=None):
    """create a pair of network definition functions for disRNN
    normal and eval mode models with same hyperparameters
    Return (normal_model, eval_model)
    """

    if target_size is not None:
        pass
    elif sample_dataset is not None:
        md_input, md_target = next(sample_dataset)
        target_size = len(np.unique(md_target))
    else:
        raise Exception(
            "either provide a sample dataset or specify target_size")

    def disrnn_module():
        model = hkDisRNN(latent_size=latent_size,
                         update_mlp_shape=update_mlp_shape,
                         choice_mlp_shape=choice_mlp_shape,
                         target_size=target_size,
                         eval_mode=False)  # type: ignore
        return model

    def disrnn_eval():
        model = hkDisRNN(latent_size=latent_size,
                         update_mlp_shape=update_mlp_shape,
                         choice_mlp_shape=choice_mlp_shape,
                         target_size=target_size,
                         eval_mode=True)  # type: ignore
        return model

    return (disrnn_module, disrnn_eval)


def make_transformed_disrnn(
        latent_size: int,
        update_mlp_shape: Sequence[int],
        choice_mlp_shape: Sequence[int],
        target_size: int = 2,
        eval_mode = False) -> RNNtransformed:
    def disrnn_module():
        model = hkDisRNN(
            latent_size=latent_size,
            update_mlp_shape=update_mlp_shape,
            choice_mlp_shape=choice_mlp_shape,
            target_size=target_size,
            eval_mode=eval_mode
        )  # type: ignore
        return model

    return transform_hkRNN(disrnn_module)

def disRNN_training(datasets: Tuple[trainingDataset, trainingDataset], latent_size, update_mlp_shape, choice_mlp_shape, optimizer):
    """A wrapper function that allows defining RNNtraining class for disRNN model in one line with Hydra

    Args:

    Returns:
        RNNtraining: a constructed RNNtraining instance for the disRNN model specified
    """

    train_dataset, _ = datasets

    md_input, md_target = next(train_dataset)
    target_size = len(np.unique(md_target))

    disrnn_module, disrnn_eval = make_disrnn_funcs(
        latent_size, update_mlp_shape, choice_mlp_shape, target_size=target_size)

    return RNNtraining(model = disrnn_module, eval_model= disrnn_eval, datasets = datasets, optimizer = optimizer)


def make_GRU(hidden_size):
    def model():
        return hk.DeepRNN([hk.GRU(hidden_size), hk.Linear(output_size=2)])
    return model


def make_LSTM(hidden_size):
    def model():
        return hk.DeepRNN([hk.LSTM(hidden_size), hk.Linear(output_size=2)])
    return model

def make_deepLSTM(hidden_sizes, activation=jax.nn.relu):
    def model():
        layers = []
        for sz in hidden_sizes:
            layers.append(hk.LSTM(sz))
            layers.append(activation)
        layers.append(hk.Linear(output_size=2))
        return hk.DeepRNN(layers)
    return model
