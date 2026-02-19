

## Classic RL model related

### define a classic RL model

### specify parameters of a classic RL model without fitting to data

Use the `init` argument of `boundedParam` to specify the parameter value when initializing a classic RL model instance. For example, 

```python
from disRNN_MP.classic_RL import forgetQ, boundedParam

rlmd = forgetQ(param_specs=[ 
    boundedParam('decay_rate', 0, 1, init = 0.5),
    boundedParam('positive_evi', -10, 10, init = 1),
    boundedParam('negative_evi', -10, 10, init = -2),
])
```

Then, depending on the goal of using the RL model, there are two ways to utilize the specified parameters and model.

First, if the goal is to run the model through a dataset:

```python
from disRNN_MP.classic_RL import RLmodelWrapper

# here fQ_1334 is a dataset
mdw = RLmodelWrapper(rlmd, fQ_1334, run_fitting = False)

# example usages:
mdw.metric()
mdw.forward()
```

Second, if the goal is to use the model as an agent to play against an RL environment

```python
from disRNN_MP.agents import RLmodel_agent

# for specifying size only
dummy_dat = jnp.zeros((100,3,2)) # 100 trials, 3 sessions, 2 observation features
params = rlmd.init(jax.random.key(89), dummy_dat)

age = RLmodel_agent(rlmd, params, jax.random.key(10))

```

### fit forgetting Q-learning model to behavioral data

```python
from disRNN_MP.classic_RL import forgetQ, boundedParam, RLmodelWrapper
from disRNN_MP.agent import RLmodel_agent, run_experiment, EnvironmentBanditsMP_julia
from disRNN_MP.dataset import makeDataset_nparr
```

1. Declare a forgetting Q-learning model. You can specify hyperparameter and/or parameter bounds. For example:
```python
rlmd = forgetQ(param_specs=[ # boundedParam(name, lb, ub, shape, init)
    boundedParam('decay_rate', 0, 1),
    boundedParam('positive_evi', -10, 10),
])
```

2. the dataset format for the fitting need to be `trainingDataset`

3. then fit the model by wrapping model definition, dataset, random seed and optionally other hyperparameter for the fitting process

```python
mdw = RLmodelWrapper(rlmd, trainD, rng=jax.random.key(seed))
```

## Simulate behavioral data with a task environment

### Simulate fitted forgetting Q-learning model with matching pennies task

1. declare the task environment instance by:

```python
env = EnvironmentBanditsMP_julia()
```



## RNN model related

### 

### evaluate a haiku RNN model

`rnn.utils.evo_state`: evolve a single step with given state and observation



