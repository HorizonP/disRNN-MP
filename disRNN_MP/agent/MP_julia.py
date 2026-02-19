from pathlib import Path
import os
import numpy as np
from typing import List, Sequence
import pandas as pd

from .. import utils
from .abstract import RL_Environment, RL_multiEnvironment


class EnvironmentBanditsMP_julia(RL_Environment):
    """Env for 2-armed bandit task with matching pennies algorithm
    
    Arguments:
        max_depth: the maximum depth of computer algorithm
        alpha: the p-value significance level for statistically testing bias
        ch_code: how to code the L and R choice
    
    """

    def _setup_jl_code(self):
        jl = utils.setup_jl()
        script_dir = Path(os.path.dirname(__file__))
        jl.include(str(script_dir / "matching_pennies.jl"))

        self._jl = jl
        

    def __init__(self, max_depth: int = 4, alpha: float = 0.05, ch_code: Sequence = (0,1)):

        # setup julia
        self._setup_jl_code()

        self._max_depth = max_depth
        self._alpha = alpha
        self._choices = ch_code  # encode left and right choices as 0 and 1
        # could also include params to select which algorithm to use

        self.new_session()
        
        
    def new_session(self) -> None:
        self._MPT = self._jl.MatchingPenneisTask(
            self._max_depth, self._alpha, choices=self._choices)

    @property
    def history(self):
        return self._jl.pytable(self._MPT.history)

    @property
    def biasCount(self):
        return self._jl.pytable(self._MPT.biasCount)

    @property
    def n_actions(self) -> int:
        return len(self._choices)

    @property
    def reward_probs(self) -> np.ndarray:
        if len(self.history) == 0:
            pR = 0.5
        else:
            pR = self.history.com_pR.iloc[-1]
        return np.array([1-pR, pR])

    @property
    def reward_probs_all(self) -> np.ndarray:
        pR = self.history.com_pR.to_numpy(dtype="float")
        pR = np.expand_dims(pR, 1)
        return np.hstack(((1-pR), pR))

    def step(self, choice):
        """Step the model forward given chosen action."""

        reward = self._jl.run_step(self._MPT, choice)

        # Return the reward
        return float(reward)


class multiEnvironmentBanditsMP_julia(RL_multiEnvironment):
    """Env for 2-armed bandit task with matching pennies algorithm
    
    Arguments:
        max_depth: the maximum depth of computer algorithm
        alpha: the p-value significance level for statistically testing bias
        ch_code: how to code the L and R choice
    
    """

    def _setup_jl_code(self):
        # os.environ['PYTHON_JULIACALL_THREADS'] = 'auto'
        jl = utils.setup_jl()
        script_dir = Path(os.path.dirname(__file__))
        jl.include(str(script_dir / "matching_pennies.jl"))

        self._jl = jl
        

    def __init__(self, n_sess: int = 1, max_depth: int = 4, alpha: float = 0.05, ch_code: Sequence = (0,1)):
        """init multiple MP task environments

        Args:
            n_sess (int, optional): number of session to run simultanesouly. Defaults to 1. It's safe to ignore this setting if use this instance through `RL_watcher`
            max_depth (int, optional): _description_. Defaults to 4.
            alpha (float, optional): _description_. Defaults to 0.05.
            ch_code (Sequence, optional): _description_. Defaults to (0,1).
        """

        # setup julia
        self._setup_jl_code()

        self.n_sess = n_sess
        self._max_depth = max_depth
        self._alpha = alpha
        self._choices = ch_code  # encode left and right choices as 0 and 1
        # could also include params to select which algorithm to use

        self._sess_ids = None

        self.new_session()

    def set_random_seed(self, seed: int):
        self._jl.seval(f"Random.seed!({seed})")
        return self
        
        
    def new_session(self) -> None:
        self._MPTs = [self._jl.MatchingPenneisTask(
            self._max_depth, self._alpha, choices=self._choices) for _ in range(self.n_sess)]

        # self._MPTs = self._jl.make_MPTs(self._max_depth, self._alpha, choices=self._choices, n=self.n_sess)

    @property
    def sess_ids(self) -> List[int] | np.ndarray:
        """sessID for each of the sessions

        self.sess_ids[i] ~ self._MPTs[i]

        For perserving the sessIDs received when calling `eval_on_dataset`

        Returns:
            _type_: _description_
        """
        if self._sess_ids is None:
            return list(range(self.n_sess))
        else:
            return self._sess_ids

    @sess_ids.setter
    def sess_ids(self, val):
        if val is not None:
            assert len(val) == self.n_sess, f"length of the value {len(val)} does not match attribute `n_sess` {self.n_sess}"
        self._sess_ids = val


    @property
    def history(self) -> pd.DataFrame:
        sess_ids = self.sess_ids
        df = pd.concat([
            self._jl.pytable(self._MPTs[i].history).assign(sess_id = sess_ids[i])
            for i in range(self.n_sess)
        ])

        # make sure no duplicate index
        if 'tri_id' in df.columns:
            df = df.reset_index(drop=True)
        else:
            df = df.reset_index(names="tri_id")

        df['choice'] = df['choice'].astype(int)
        df['reward'] = df['reward'].astype(int)
        df['com_pR'] = df['com_pR'].astype(float)

        return df

    @property
    def biasCount(self):
        sess_ids = self.sess_ids
        df = pd.concat([
            self._jl.pytable(self._MPTs[i].biasCount).assign(sess_id = sess_ids[i])
            for i in range(self.n_sess)
        ])
        for col in ['signif_tris', 'detected_on', 'matched_on']:
            df[col] = df[col].map(np.array)

        return df.astype({
            'histseq': str,
            'p_val': np.float64,
        })
    
    @property
    def trialBias(self):
        """computer algorithm output given the ***recent history*** of each trial (without knowning current trial monkey choice)"""

        # add histseq column to trialBias df
        for i in range(self.n_sess):
            self._jl.add_histseq_col(self._MPTs[i])

        sess_ids = self.sess_ids
        df = pd.concat([
            self._jl.pytable(self._MPTs[i].trialBias).assign(sess_id = sess_ids[i])
            for i in range(self.n_sess)
        ])

        # make sure no duplicate index
        if 'tri_id' in df.columns:
            df = df.reset_index(drop=True)
        else:
            df = df.reset_index(names="tri_id")

        return df.astype({
            "detected": int,
            "depth": int,
            "magnitude": float,
            'which_alg': int,
            'histseq': str,
        })


    @property
    def n_actions(self) -> int:
        return len(self._choices)

    # @property
    # def reward_probs(self) -> np.ndarray:
    #     if len(self.history) == 0:
    #         pR = 0.5
    #     else:
    #         pR = self.history.com_pR.iloc[-1]
    #     return np.array([1-pR, pR])

    @property
    def reward_probs_all(self) -> np.ndarray:

        # here the index will be triID for each session
        pR = self.history.pivot(columns='sess', values='com_pR').to_numpy()
        return np.stack((1-pR, pR), axis=-1)

    def step(self, choice) -> np.ndarray:
        """Step the model forward given chosen action."""

        reward = np.array(self._jl.broadcast(self._jl.run_step, self._MPTs, choice))
        # reward = np.array(self._jl.batch_run_step(self._MPTs, choice.astype('int')), dtype='float')

        # Return the reward
        return reward

    def eval_on_dataset(self, df:pd.DataFrame, ch_col: str = 'choice', rew_col: str = 'reward', tri_id_col: str|None = 'tri_id', sess_id_col: str|None = 'sess_id'):
        """calculate MP task variables on an existing behavioral dataset

        

        Args:
            df (pd.DataFrame): _description_
            ch_col (str, optional): _description_. Defaults to 'choice'.
            rew_col (str, optional): _description_. Defaults to 'reward'.
            tri_id_col (str | None, optional): Which column is for trial index. Defaults to 'tri_id'. The purpose is for keeping trials sorted. If it is none, this function will use the order of trials as it is
            sess_id_col (str | None, optional): _description_. Defaults to 'sess_id'.
        """

        # reset any intermediate variables
        self.new_session()

        df = df.rename(columns={
            ch_col: 'choice',
            rew_col: 'reward',
        }).astype({
            'choice': np.int64, 
            'reward': np.int64,
        })

        if tri_id_col is not None:
            df = df.sort_values(tri_id_col).drop(columns=tri_id_col)

        def eval_one_sess(mpt_i, df):
            jl = self._jl
            jl.seval("append!")(self._MPTs[mpt_i].history, df, cols = jl.Symbol("union"))
            jl.cal_MP_algs(self._MPTs[mpt_i])

        if sess_id_col is None:
            self.n_sess = 1
            self.new_session()

            eval_one_sess(0, df)

        else:
            # update n_sess attribute to reflect number of sessions
            sess_ids = sorted(df[sess_id_col].unique().tolist())
            self.n_sess = len(sess_ids)
            self.sess_ids = sess_ids
            self.new_session()

            for sid, group in df.groupby(sess_id_col):
                eval_one_sess(list(self.sess_ids).index(sid), group)

