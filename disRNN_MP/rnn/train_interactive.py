from typing import Callable, List, Optional, Tuple, Any

import numpy as np
import plotly.graph_objects as go

from .train import RNNtraining, train_session
from disRNN_MP.rnn.utils import Params, RandomKey

#====================== train_rnn_interactive class definition

# TODO add loss function trace to the figure, reuse part of the code for disRNN dashboard
def init_likelihood_plotly():
    fig = go.FigureWidget()
    fig.add_scatter(name="training likelihood", y=[], mode="lines")
    fig.add_scatter(name="testing likelihood", y=[], mode="lines")
    return fig


class RNNtraining_interactive(RNNtraining):
    """extend RNNtraining class with a plotly FigureWidget
    To be used in Jupyter notebook: train and test log-likelihood will be updated to the figure during training

    Attribute:
        likelihood_plot: go.FigureWidget
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.likelihood_plot = init_likelihood_plotly()

    def update_ll_plot(self):
        fig = self.likelihood_plot
        fig.data[0].y = np.array(self.train_history.train_ll)  # type: ignore
        fig.data[1].y = np.array(self.train_history.test_ll) # type: ignore
    
    def train(self, 
            name: Optional[str] = None,
            random_key: Optional[RandomKey] = None,
            n_block: int = 10, 
            steps_per_block: int = 100,
            penalty_scale: float = 0,
            beta_scale: float = 1,
            loss_type: str = 'categorical',
            block_update_funcs: Optional[List[Callable]] = None,
        ):
        
        sess = self._create_train_sess(
            name=name, 
            random_key=random_key, 
            n_block=n_block,
            steps_per_block=steps_per_block,
            penalty_scale=penalty_scale,
            beta_scale=beta_scale,
            loss_type=loss_type,
            block_update_funcs=block_update_funcs)
        
        sess.block_update_funcs.append(RNNtraining_interactive.update_ll_plot)
        
        self.train_with_sess(sess)