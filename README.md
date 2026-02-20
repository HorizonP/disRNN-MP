# disRNN-MP

A Python package for training, simulating, and analyzing disentangled RNN models on behavioral data from the matching pennies task. Built on JAX/Haiku.

## Overview

**disRNN-MP** provides tools for modeling sequential decision-making behavior using disentangled recurrent neural networks (disRNN; Miller et al. 2023). The disRNN architecture incorporates information bottlenecks—inspired by variational autoencoders—that encourage the model to learn sparse, interpretable representations where each latent dimension corresponds to a distinct factor of variation in behavior.

This package was developed to study monkey behavior in the matching pennies task, a competitive game that probes adaptive decision-making under uncertainty. While standard RNNs achieve state-of-the-art predictive performance on this task, their distributed representations resist interpretation. The disRNN achieves comparable predictive accuracy while producing sparse networks amenable to mechanistic analysis.

Although designed around the matching pennies task, the package's modular and hierarchical class design makes it readily adaptable to other trial-based behavioral tasks. The abstract agent/environment interfaces, flexible data containers, and task-agnostic training infrastructure can be extended to new experimental paradigms with minimal modification.

### Key Capabilities

- **Train** disentangled RNN models or fit classic RL models (e.g., forgetting Q-learning) on trial-based behavioral data
- **Simulate** agent-environment interactions to generate synthetic data and compare model predictions
- **Analyze** trained models and behavioral datasets with built-in statistical and visualization tools

## The Matching Pennies Task

The matching pennies task is a two-player competitive game widely used in cognitive neuroscience and psychology to study decision-making. On each trial:

1. An agent (e.g., a monkey) and a computer opponent each independently choose one of two targets
2. The agent wins if both players choose the same target; otherwise, the computer wins
3. The computer opponent tracks statistical regularities in the agent's choice history and exploits detected biases to minimize the agent's reward rate

This adversarial structure creates rich, dynamic behavioral patterns as agents adapt their strategies in response to the opponent's exploitation. The task has been instrumental in understanding reinforcement learning and prefrontal cortex function.

**Key references:**
- Lee, D., Conroy, M. L., McGreevy, B. P., & Barraclough, D. J. (2004). Reinforcement learning and decision making in monkeys during a competitive game. *Cognitive Brain Research, 22*, 45–58.
- Barraclough, D. J., Conroy, M. L., & Lee, D. (2004). Prefrontal cortex and decision making in a mixed-strategy game. *Nature Neuroscience, 7*, 404–410.

## Package Capabilities

### 1. Model Training and Fitting

**Generic and Disentangled RNN Training** (`disRNN_MP.rnn`)

- `hkDisRNN` — Haiku-based disRNN with configurable information bottlenecks
- `ModelTrainee` — Database-backed training jobs with automatic checkpointing and resumption
- `trainingSession` — Flexible training schedules with customizable optimizers and stopping criteria
- Hydra-based configuration for reproducible experiments

**Classic RL Model Fitting** (`disRNN_MP.classic_RL`)
- `forgetQ` — Forgetting Q-learning model (Lee et al., 2004)
- `RLmodelWrapper` — Unified interface for fitting RL models with L-BFGS optimization
- Extensible `RLmodel` base class for custom model implementations

### 2. Simulation

**Agent-Environment Interaction** (`disRNN_MP.agent`)
- `hkNetwork_multiAgent` / `RLmodel_multiAgent` — Convert trained models to simulating agents
- `EnvironmentBanditsMP_numba` — Numba-accelerated matching pennies environment
- `RL_watcher` — Orchestrates experiments and collects behavioral + latent data
- Vectorized multi-session simulation for efficient synthetic data generation

### 3. Analysis

**Behavioral Analysis** (`disRNN_MP.analysis`)
- `BanditDataset` — Container for behavioral data with built-in analysis methods
  - Trial history regression
  - Conditional reward probability
  - Choice sequence statistics
- `disRNN_model_analyzer` — Examine trained disRNN structure
  - Bottleneck compression metrics
  - Active latent identification
  - Forward pass evaluation


## Installation

```bash
# Clone the repository
git clone https://github.com/HorizonP/disRNN-MP.git
cd disRNN-MP

# Install in editable mode
pip install -e .

# Optional: for legacy jaxopt optimizer support
pip install -e ".[legacy]"
```

### Requirements

- Python ≥ 3.11
- JAX with GPU/TPU support recommended (CPU execution supported but slower)
- See `pyproject.toml` for full dependency list

**Note:** JAX installation may require platform-specific steps for GPU support. See the [JAX installation guide](https://github.com/google/jax#installation).

### Quickstart

- [Resumable model training with a database backend](./usage_example/training_ModelTrainee.ipynb)
- more quickstart examples are planned in the future

## Package Structure

| Submodule | Description |
|-----------|-------------|
| `disRNN_MP.rnn` | disRNN model definitions, training infrastructure, database ORM |
| `disRNN_MP.classic_RL` | Classic RL models (forgetQ) and fitting utilities |
| `disRNN_MP.agent` | Agent/environment abstractions, simulation orchestration |
| `disRNN_MP.analysis` | Behavioral dataset analysis, model analyzers |
| `disRNN_MP.dataset` | Data loading and `trainingDataset` (XArray-based) |
| `disRNN_MP.metrics` | Evaluation metrics (log-likelihood, BIC) |

## Citation

If you use this package in your research, please cite:

```bibtex
@inproceedings{liu2024discovering,
  title={Discovering cognitive models in a competitive mixed-strategy game},
  author={Liu, Peiyu and Miller, Kevin J and Seo, Hyojung},
  booktitle={Cognitive Computational Neuroscience},
  year={2024},
  address={Boston, MA},
  url={https://2024.ccneuro.org/pdf/68_Paper_authored_Liu-et-al-CCN2024-authored.pdf}
}
```

*A manuscript describing the methods and findings in detail is currently in preparation.*

For the disRNN method itself, please also cite:

```bibtex
@inproceedings{miller2023cognitive,
  title={Cognitive model discovery via disentangled RNNs},
  author={Miller, Kevin J and Eckstein, Maria K and Botvinick, Matthew M and Kurth-Nelson, Zeb},
  booktitle={Advances in Neural Information Processing Systems},
  volume={36},
  year={2023},
  url={https://proceedings.neurips.cc/paper_files/paper/2023/hash/c194ced51c857ec2c1928b02250e0ac8-Abstract-Conference.html}
}
```

## References

- Miller, K. J., Eckstein, M. K., Botvinick, M. M., & Kurth-Nelson, Z. (2023). Cognitive model discovery via disentangled RNNs. *Advances in Neural Information Processing Systems, 36*. https://proceedings.neurips.cc/paper_files/paper/2023/hash/c194ced51c857ec2c1928b02250e0ac8-Abstract-Conference.html
- Lee, D., Conroy, M. L., McGreevy, B. P., & Barraclough, D. J. (2004). Reinforcement learning and decision making in monkeys during a competitive game. *Cognitive Brain Research, 22*, 45–58.
- Barraclough, D. J., Conroy, M. L., & Lee, D. (2004). Prefrontal cortex and decision making in a mixed-strategy game. *Nature Neuroscience, 7*, 404–410.
- Seo, H., Barraclough, D. J., & Lee, D. (2007). Dynamic signals related to choices and outcomes in the dorsolateral prefrontal cortex. *Cerebral Cortex, 17*(suppl_1), i110–i117. https://doi.org/10.1093/cercor/bhm064

## License

MIT License. See [LICENSE](LICENSE) for details.