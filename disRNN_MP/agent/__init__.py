"""
this module contains functions to run models as agent to interact with various environments

### interface for agent

### interface for environment

"""

from .MP_julia import EnvironmentBanditsMP_julia
from .agents import RLmodel_agent, hkNetwork_agent
from .interact import run_experiment