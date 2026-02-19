from copy import copy

from tqdm import tqdm

import numpy as np
import pandas as pd
import xarray as xr

from ..analysis.bandit_dataset_analysis import BanditDataset
from .abstract import RL_Environment, RL_agent, RL_multiAgent, RL_multiEnvironment


def run_experiment(
        agent: RL_agent, environment: RL_Environment,
        n_trials: int, n_sessions: int = 1):
    """Runs a behavioral experiment from a given agent and environment.

    Args:
        agent: An agent object
        environment: An environment object
        n_trials: The number of steps in the session you'd like to generate
        n_sessions: number of sessions to generate

    Returns:
        experiment: A BanditSession holding choices and rewards from the session
    """
    # choices = np.zeros(n_trials)
    # rewards = np.zeros(n_trials)
    hists = []
    states = []
    for sess in np.arange(n_sessions):
        
        environment.new_session()
        agent.new_session()
        states.append([])

        for trial in np.arange(n_trials):
            # record the agent's states that give rise to current trial choice
            states[sess].append({
                'sessID': sess, 'triID': trial,
                'latent': agent.latents, 'pChoices': agent.choice_probs
            })

            # First agent makes a choice
            choice = agent.get_choice()
            # Then environment computes a reward
            reward = environment.step(choice)
            # Finally agent learns
            agent.update(choice, reward)
        
        hist = copy(environment.history)
        hist = hist.assign(sessID = sess, triID = np.arange(n_trials))
        hists.append(hist)
    
    return BanditDataset(pd.concat(hists)), states


class RL_watcher:
    """oversee running of interaction between agent and environment, and log the behavioral history and internal variables"""

    agent: RL_multiAgent
    environ: RL_multiEnvironment   

    def __init__(self, agent: RL_multiAgent, environ: RL_multiEnvironment) -> None:
        self.agent = agent
        self.environ = environ
        assert self.agent.n_sess == self.environ.n_sess, f"The number of session for agent and environment {(self.agent.n_sess, self.environ.n_sess)} does not match"
        
    def interact(self):
        """agent interact and learn by one trial"""
        # First age makes a choice
        choice = self.agent.get_choice()
        # Then env computes a reward
        reward = self.environ.step(choice)
        # Finally age learns
        self.agent.update(choice, reward)

        return choice, reward
    
    def run_experiment(self, n_trials: int):
        """simulate `n_sessions` sessions of experiments between agent and environment simultaneously

        the number of sessions is determined by agent and environment parameter `n_sess`, which has to match

        This function will modify the `n_sess` attribute for both agent and environment to reflect the `n_sessions` argument

        Args:
            n_trials (int): number of trials to simulate
            n_sessions (int): number of sessions

        Returns:
            _type_: _description_
        """

        # self.agent.n_sess = n_sessions
        # self.environ.n_sess = n_sessions

        assert self.agent.n_sess == self.environ.n_sess, f"The number of session for agent and environment {(self.agent.n_sess, self.environ.n_sess)} does not match"

        # reset agent and environment
        self.agent.new_session()
        self.environ.new_session()

        prev_latents = []
        latents = []
        pChoices = []
        for tri in tqdm(range(n_trials)):

            # record the age's states that give rise to current trial choice
            # states.append({
            #     'latent': self.agent.latents, 'pChoices': self.agent.choice_probs
            # })

            prev_latents.append(self.agent.latents)
            pChoices.append(self.agent.choice_probs)

            self.interact()

            latents.append(self.agent.latents)
        
        prev_latents = np.stack(prev_latents, 0)
        pChoices = np.stack(pChoices, 0)
        latents = np.stack(latents, 0)

        tri_df = BanditDataset.from_dataframe(self.history, sessID_col='sess_id', triID_col = 'tri_id', )
        internal_vars = xr.Dataset(
            data_vars={
                'prev_latents': (["tri", "sess", "latent"], prev_latents),
                'pChoices': (
                    ["tri", "sess", "choice"], 
                    pChoices, 
                    {'description': "agent's probability for each choice on current trial"}),
                'latents': (["tri", "sess", "latent"], latents),
            }, 
            coords={
                'tri': np.arange(prev_latents.shape[0]),
                'sess': np.arange(prev_latents.shape[1]),
                'latent': np.arange(prev_latents.shape[2]),
                # 'choice': 
            },
        )

        return tri_df, internal_vars
    

    @property
    def history(self):
        """the observed behavior of agent and environment"""
        return self.environ.history