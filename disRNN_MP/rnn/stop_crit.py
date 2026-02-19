from typing import Literal
import numpy as np
from jax.tree_util import tree_map
import jax.numpy as jnp
from disRNN_MP.rnn.train_db import ModelTrainee, stopCritTest
from disRNN_MP.rnn.disrnn import get_bottlenecks_latent, get_bottlenecks_update
from disRNN_MP.analysis.disrnn_static_analy import disRNN_params_analyzer

def disrnn_bottlenecks_converged(params: list, thres = 1e-3, n_last = 5):
    params = params[-min(n_last, len(params)):] # cap params list at most `n_last` elements
    upd = get_bottlenecks_update(params)
    latb = get_bottlenecks_latent(params)
    allb = np.hstack((upd.reshape((upd.shape[0], -1)),latb.reshape((latb.shape[0], -1))))

    max_var = np.max(np.abs(np.mean(np.diff(allb, axis = 0), axis=0)))

    print(f"maximum median variability for last {n_last} step is {max_var}")
    return max_var < thres


def disrnn_bottlenecks_converged2(params: list, thres = 1e-2):
    assert len(params) == 2
    upd = get_bottlenecks_update(params)
    latb = get_bottlenecks_latent(params)
    allb = np.hstack((upd.reshape((upd.shape[0], -1)),latb.reshape((latb.shape[0], -1))))

    # rule:
    # diff over time series for each bottleneck
    # take median of the diffs for each channel
    # the max across channels should be less than the threshold
    max_var = np.max(np.abs(allb[1,:] - allb[0,:]))
    print(f"maximum median variability is {max_var}")
    return max_var < thres


class disrnnConvTest_meanDiff(stopCritTest):

    def __init__(self, thres = 1e-3, n_last = 21, min_steps = 4000, max_steps: int|float = np.inf) -> None:
        self.thres = thres
        self.n_last = n_last
        self.min_steps = min_steps
        self.max_steps = max_steps

    def bind(self, mt: ModelTrainee) -> None:
        super().bind(mt)
        self.param_analyzer = disRNN_params_analyzer(mt.model.model_haiku)

    def disrnn_bottlenecks_converged(self, params: list):

        params = params[-min(self.n_last, len(params)):] # cap params list at most `n_last` elements

        upd = jnp.stack(list(map(self.param_analyzer.get_update_sigma, params)))
        latb = jnp.stack(list(map(self.param_analyzer.get_latent_sigma, params)))
        chb = jnp.stack(list(map(self.param_analyzer.get_choice_sigma, params)))
        allb = jnp.hstack((upd.reshape((upd.shape[0], -1)), latb, chb))

        max_var = jnp.max(jnp.abs(jnp.mean(jnp.diff(allb, axis = 0), axis=0)))

        print(f"maximum median variability for last {self.n_last} step is {max_var}")
        return bool(max_var < self.thres)
    
    def test(self) -> bool:
        
        mt = self.trainee
        sess_step = mt.state.step - mt.sessions[mt.curr_sess_ind or 0].start_step + 1
        if len(mt.records) < self.n_last or sess_step < self.min_steps:
            return False
        elif self.max_steps is not None and sess_step >= self.max_steps:
            return True
        else:
            params = [rec.parameter for rec in mt.records[-self.n_last:]]
            return self.disrnn_bottlenecks_converged(params)
    
    def _test(self) -> bool:
        """this is the old test method which only tested convergence on latent and update bottleneck due to historical reason"""
        mt = self.trainee
        sess_step = mt.state.step - mt.sessions[mt.curr_sess_ind or 0].start_step + 1
        if len(mt.records) < self.n_last or sess_step < self.min_steps:
            return False
        elif self.max_steps is not None and sess_step >= self.max_steps:
            return True
        else:
            params = [rec.parameter for rec in mt.records[-self.n_last:]]
            return disrnn_bottlenecks_converged(params, self.thres, self.n_last)