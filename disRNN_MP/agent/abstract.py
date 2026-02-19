# define abstract classes
from typing import Any, Optional
from abc import ABC, abstractmethod
import numpy as np
import pandas as pd

from ..typing import Array
class RL_Environment(ABC):
    """
    the RL environment should be able to receive agent's choice through `step` and feedback with outcome
    the environment should also record the history of interactions
    """
    @property
    @abstractmethod
    def n_actions(self) -> int:
        """number of possible actions"""
        raise NotImplementedError
    
    @abstractmethod
    def step(self, choice) -> float:
        """one step of the environment: take agent's choice, return reward as float 0 or 1"""
        raise NotImplementedError
    
    @property
    @abstractmethod
    def reward_probs_all(self) -> np.ndarray:
        """get reward probability for all possible actions on all past trials
        return a numpy ndarray of shape (N_trials, N_actions)
        """
        raise NotImplementedError
    
    @abstractmethod
    def new_session(self) -> None:
        """reset all records and start a new empty session"""
        raise NotImplementedError
    
    @property
    @abstractmethod
    def history(self) -> pd.DataFrame:
        """obtain the history of interaction with agents

        Returns:
            pandas.DataFrame: a DataFrame of shape (N_trials, N_cols) including column "choice" and "reward" at least
            the remaining columns are environment specific
        """
        raise NotImplementedError


class RL_agent(ABC):
    """a state-machine representing a agent in a RL environment

    Attributes:
        random_seed (int): the starting random seed. For each trial new random seed will be generated stemming from this one

    Raises:
        NotImplementedError: _description_

    Returns:
        _type_: _description_
    """

    random_seed: int
    latents: Any # the current latent variables/state of the agent (e.g. values for RL models)

    @abstractmethod
    def __init__(self, random_seed:int = 0) -> None:
        # initialize private random number generator for the agent
        self.infuse_seed(random_seed)
    
    def _update_rng(self):
        # self.random_key, _ = jax.random.split(self.random_key, 2)
        self.rng = self._rng_seq.spawn(1)[0]

    @property
    @abstractmethod
    def choice_probs(self) -> Array:
        pass
    
    @abstractmethod
    def update(self, choice, reward) -> None:
        """update agent's latent variables and next random seed after observe itself's choice and reward"""
        self._update_rng()
    
    @abstractmethod
    def new_session(self) -> None:
        """reset all records and start a new empty session"""
        raise NotImplementedError

    def get_choice(self) -> int:
        """random sample a choice
        number of possible actions is determined by shape[0] of the choice probabilities.
        Actions are represented as 0, 1, 2, ...
        """
        p_chs = np.array(self.choice_probs)
        p_chs = p_chs/np.sum(p_chs) # make sure it sums to 1
        # print(np.random.default_rng(self.rng).random())
        rng = np.random.default_rng(self.rng)
        ch = rng.choice(p_chs.shape[0], p = p_chs)

        # ch = jax.random.choice(self.random_key, p_chs.shape[0], p = p_chs)
        return ch
    
    def infuse_seed(self, random_seed):
        """perform the routine in order to use a newly updated random_seed"""

        self._rng_seq = np.random.SeedSequence(random_seed)
        self.random_seed = random_seed
        self._update_rng()

    
class RL_multiEnvironment(ABC):
    """
    this class can emulate multiple RL_environment at once, processing batched agents choice
    """

    n_sess: int
    "number of sessions to run simultaneously"

    @property
    @abstractmethod
    def n_actions(self) -> int:
        """number of possible actions"""
        raise NotImplementedError
    
    @abstractmethod
    def step(self, choice:np.ndarray) -> np.ndarray:
        """one step of the environment: take agent's choice, return reward as float 0 or 1
        both choice and output are 1D array
        """
        raise NotImplementedError
    
    @property
    @abstractmethod
    def reward_probs_all(self) -> np.ndarray:
        """get reward probability for all possible actions on all past trials
        return a numpy ndarray of shape (N_trials, N_sessions, N_actions)
        """
        raise NotImplementedError
    
    @abstractmethod
    def new_session(self) -> None:
        """reset all records and start a new empty session"""
        raise NotImplementedError
    
    @property
    @abstractmethod
    def history(self) -> pd.DataFrame:
        """obtain the history of interaction with agents

        Returns:
            pandas.DataFrame: a DataFrame of shape (N_trials * N_sessions, N_cols) including column "choice", "reward", "tri_id", "sess_id" at least
            the remaining columns are environment specific
        """
        raise NotImplementedError
    

class RL_multiAgent(ABC):
    """a state-machine representing a agent in a RL environment

    Attributes:
        random_seed (int): the starting random seed. For each trial new random seed will be generated stemming from this one

    Raises:
        NotImplementedError: _description_

    Returns:
        _type_: _description_
    """

    random_seed: int
    n_sess: int
    "number of sessions to run simultaneously"
    latents: Array 
    """the current trial latent variables/state of the agent (e.g. values for RL models)
    of shape (N_sess, N_latents)
    """

    @abstractmethod
    def __init__(self, random_seed:int = 0) -> None:
        # initialize private random number generator for the agent
        self.infuse_seed(random_seed)
    
    def _update_rng(self):
        """ Generate next random number in the sequence (not for public use)
        this function will be called by the abstract method `update`

        by storing the intermediate random number, repeated calls of `get_choice` will deterministically return the same random choice sample in one trial
        """
        # self.random_key, _ = jax.random.split(self.random_key, 2)
        self.rng = self._rng_seq.spawn(1)[0]
        # the numpy.random.SeedSequence has state: repeatedly calling spawn will return different random sequences

    @property
    @abstractmethod
    def choice_probs(self) -> Array:
        """of shape (N_sessions, N_possible_choices)"""
        pass
    
    @abstractmethod
    def update(self, choice:Array, reward:Array) -> None:
        """update agent's latent variables and next random seed after observe itself's choice and reward
        both choice and reward should be 1D array of shape (N_sess, )
        """
        # NOTE for future overhaul: as an abstract class of RL agent, the update should take "observation" input rather than specific "choice" and "reward", as in some tasks, choice and reward may not be the solely observation
        self._update_rng()
    
    @abstractmethod
    def new_session(self) -> None:
        """reset all records and start a new empty session"""
        raise NotImplementedError

    def get_choice(self) -> np.ndarray:
        """random sample a choice
        number of possible actions is determined by shape[1] of the choice probabilities.
        Actions are represented as 0, 1, 2, ...

        ### Implementation details

        this function will get `self.choice_probs` as numpy array to generate choice result on CPU memory

        """
        p_chs = np.array(self.choice_probs)
        p_chs = p_chs/np.sum(p_chs,1).reshape(-1,1) # make sure it sums to 1 for each row

        n_actions = p_chs.shape[1]

        # reset random sequence to the cached one
        # so that the following results are random but repeatable
        rng = np.random.default_rng(self.rng)

        # ch = rng.choice(n_actions, p = p_chs)
        ch = np.apply_along_axis(lambda p: rng.choice(n_actions, p = p), axis=1, arr=p_chs)
        # here each call of rng.choice will also evolve the state of rng, so they will be different samples for different sessions

        # ch = jax.random.choice(self.random_key, p_chs.shape[0], p = p_chs)
        return ch
    
    def infuse_seed(self, random_seed):
        """perform the routine in order to use a newly updated random_seed"""

        self._rng_seq = np.random.SeedSequence(random_seed)
        self.random_seed = random_seed
        self._update_rng()