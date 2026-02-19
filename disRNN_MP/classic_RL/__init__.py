"""
this module contains a base class for making classic RL models, and functions to fit the models

"""

from .classic_RL import forgetQ, boundedParam, RLmodelWrapper, RLmodel
from .forgetQ_hetero import forgetQ_perSession