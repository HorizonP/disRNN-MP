from typing import List, Callable, Optional, Tuple, overload
from dataclasses import dataclass, field
import logging
import time

import numpy as np
import jax
import optax
import haiku as hk

from ..dataset import trainingDataset
from .utils import compute_log_likelihood, RNN_Apply_Fun, TrainStepFun, RandomKey, transform_hkRNN, make_train_step, Params, nan_in_dict, make_compute_log_likeli

#====================== train_rnn helper classs definition
@dataclass(frozen=False)
class train_block_history:
    """a dataframe-like structure optimized for append
    block_id start from 0
    """
    block_id: List = field(default_factory=list)
    params: List = field(default_factory=list)
    train_ll: List = field(default_factory=list)
    test_ll: List = field(default_factory=list)

    def __str__(self) -> str:
        
        if self.curr_id == -1:
            return f"train_block_history(<empty>)"
        else:
            return f"train_block_history(block_id:[{self.block_id[0]}...{self.block_id[self.curr_id()]}], params:[{type(self.params[0])}], train_ll:[...{self.train_ll[self.curr_id()]:.4f}], test_ll:[...{self.test_ll[self.curr_id()]:.4f}])"

    def __repr__(self) -> str:
        return self.__str__()
    
    def __len__(self) -> int:
        return len(self.block_id)
    
    def curr_id(self):
        return len(self.block_id)-1

    def append(self, params, train_ll, test_ll):
        self.block_id.append(len(self.block_id))
        self.params.append(params)
        self.train_ll.append(train_ll)
        self.test_ll.append(test_ll)

@dataclass(frozen=False)
class train_session:
    name: str
    train_fun: Callable # a function of how to train the model with certain objective
    random_key: jax.Array
    n_block: int = 10 
    steps_per_block: int = 100
    block_update_funcs: List[Callable] = field(default_factory=list)
    block_ids: List = field(default_factory=list)

    def __str__(self) -> str:
        s = f"train_session(name: {self.name}, steps: {self.n_block} x {self.steps_per_block}, "
        s += f"block_ids: [{self.block_ids[0]}...{self.block_ids[-1]}], "
        s += f"train_fun: {self.train_fun.__repr__()}"
        s += f")"
        return s
    
    def __repr__(self) -> str:
        return self.__str__()


#====================== train_rnn class definition
# @dataclass(frozen=False) # the dataclass is only used for its convienence of defining many attributes
class RNNtraining:
    """the class that store generated model parameters and training data, simlar to `flax.trainState`

    `model`: a function wraps haiku module and functions
    `eval_model`: model function for evaluation purpose. if not provided, it will be the same as `model`
    
    """

    def __init__(self,
            model: Callable,
            datasets: Tuple[trainingDataset, trainingDataset],
            eval_model: Optional[Callable] = None,
            optimizer: optax.GradientTransformation = optax.adam(1e-3),
            init_opt_state: Optional[optax.OptState] = None,
            init_params: Optional[Params] = None,
            init_random_key: Optional[RandomKey] = None,
            na_rerun_chance: int = 5,
        ) -> None:

        self.model = model
        self.datasets = datasets
        self.optimizer = optimizer
        self._na_rerun_chance = na_rerun_chance

        if eval_model is None:
            self.eval_model = self.model
        else:
            self.eval_model = eval_model
        
        if init_random_key is None:
            self._rand_key = jax.random.PRNGKey(0)
        else:
            self._rand_key = init_random_key
        
        self._transformedModel = transform_hkRNN(self.model)

        sample_xs = next(self.datasets[0])[0]
        if init_params is None:
            self.params = self._transformedModel.init(self._rand_key, sample_xs)
        else: 
            # better to try apply model with the params to see if it is valid
            self.params = init_params

        if init_opt_state is None:
            self.opt_state = self.optimizer.init(self.params)
        else:
            # better to try apply model with the params to see if it is valid
            self.opt_state = init_opt_state

        self._metric_fun = make_compute_log_likeli(self.eval_model, self.datasets[0].n_action)

        self.loss_trace = np.array([], dtype='float')
        self.train_ll:float = 0
        self.test_ll:float = 0
        self.train_history = train_block_history()
        self.train_sessions: List[train_session] = []
        
    def metric(self, params: Params, dataset: trainingDataset) -> float:
        xs, ys = next(dataset)
        return float(jax.device_get(self._metric_fun(params, xs, ys)))

    def train_CustomStep(self):
        """an alternative version of train that use a custom step function"""
        raise NotImplementedError()
    
    def _create_train_sess(self, 
            name: Optional[str] = None,
            random_key: Optional[RandomKey] = None,
            n_block: int = 10, 
            steps_per_block: int = 100,
            penalty_scale: float = 0,
            beta_scale: float = 1,
            loss_type: str = 'categorical',
            block_update_funcs: Optional[List[Callable]] = None,
        ):
        if name is None:
            if loss_type == 'categorical':
                name = f"train_sess with categorical loss function"
            elif loss_type == 'penalized_categorical':
                name = f"train_sess with penalized categorical loss function (penalty = {penalty_scale}, beta = {beta_scale})"
            else:
                Warning(f"unknown loss_type: {loss_type}")
                name = f"train_sess with loss_type {loss_type}"
        if random_key is None:
            self._rand_key, random_key = jax.random.split(self._rand_key)

        if block_update_funcs is None:
            block_update_funcs = []
        
        train_step = make_train_step(
            self._transformedModel.apply,
            self.optimizer, 
            penalty_scale = penalty_scale,
            beta_scale = beta_scale,
            loss_type = loss_type)

        sess = train_session(
            name=name,
            train_fun=train_step,
            n_block=n_block,
            steps_per_block=steps_per_block,
            random_key=random_key, # type: ignore
            block_update_funcs=block_update_funcs
        )

        return sess
    
    def add_train_sess(self, *args, **kwargs) -> None:
        self.train_sessions.append(self._create_train_sess(*args, **kwargs))

    def _last_checkpoint(self):
        """determine which train_session and which block it was at"""

        if len(self.train_history) == 0:
            return (-1, self.train_sessions[0])
        else:
            for ts in self.train_sessions:
                ts.n_block

    def run(self) -> None:
        pass

    
    def train_with_sess(self, sess:train_session):

        self.train_sessions.append(sess)

        train_dataset, test_dataset = self.datasets

        keys = jax.random.split(sess.random_key, num = sess.n_block)

        jit_train_step = jax.jit(sess.train_fun)

        def train_block(params, rand_key, opt_state):
            losses = np.zeros(sess.steps_per_block)
            for i_step in range(sess.steps_per_block):
                # evolve random key 
                rand_key, key_i = jax.random.split(rand_key, 2) # type: ignore
                # Train on training data
                xs, ys = next(train_dataset)

                loss, params, opt_state = jit_train_step(params, key_i, opt_state, xs, ys)
                losses[i_step] = loss
            
            return losses, params, opt_state
        t_start = time.time()
        for i_blk in range(sess.n_block):
            
            train_OK = False
            n_attempt = 0
            rand_key = keys[i_blk]
            while not train_OK and n_attempt <= self._na_rerun_chance:
                losses, params, opt_state = train_block(self.params, rand_key, self.opt_state)

                if nan_in_dict(params):
                    n_attempt += 1
                    _, rand_key = jax.random.split(rand_key, 2) # try next with new seed
                    logging.warning(f"NaN is caught in params on attempt {n_attempt}")
                else:
                    train_OK = True
            
            if not train_OK:
                msg = f"unable to resolve NaN in params issue after {n_attempt} attempts"
                logging.error(msg)
                raise ValueError(msg)
            
            # if params learned in current block is OK, then update self
            self.params = params # type: ignore
            self.opt_state = opt_state # type: ignore
            self.loss_trace = np.concatenate((self.loss_trace, losses)) # type: ignore
    
            self.train_ll = self.metric(self.params, train_dataset)
            self.test_ll = self.metric(self.params, test_dataset)
            
            # update train_history
            self.train_history.append(self.params, self.train_ll, self.test_ll)

            # update current train_session's block ids
            sess.block_ids.append(self.train_history.curr_id())

            # if provided, run other updating functions
            if len(sess.block_update_funcs) > 0:
                for fun in sess.block_update_funcs:
                    fun(self)
            
            print(f"block {i_blk} is done with loss: {losses[-1]:.4e} (Time: {time.time()-t_start:.1f}s)")


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
        
        self.train_with_sess(sess)


