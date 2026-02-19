# AGENTS

## Scope
- Subproject: `disRNN_MatchingPennies`.
- Note: This directory is a git submodule. Always run git commands and status checks from within this folder when working on the subproject.
- Follow repo-wide guidance in `../AGENTS.md` for general rules.

## Python Environment
- `disRNN_MatchingPennies` targets Python 3.11 or higher in `pyproject.toml`.
- This workspace uses pixi to manage Python environments
  - `pixi.toml` is the configuration file
- Before running terminal commands, evaluate the pixi shell-hook:
  - `eval "$(pixi shell-hook --shell zsh -e default)"`
  - Replace `zsh` with your active shell.
- Use virtual environments when mixing projects.

## Install / Build
- `disRNN_MatchingPennies` (editable): `python -m pip install -e disRNN_MatchingPennies`.
- Dependencies are listed in `disRNN_MatchingPennies/pyproject.toml`.

## Tests (pytest)
- Run full suite: `pytest disRNN_MatchingPennies/tests`.
- Run single file: `pytest disRNN_MatchingPennies/tests/test_agents.py`.
- Run single test: `pytest disRNN_MatchingPennies/tests/test_agents.py::test_get_choice_repeatable_within_trial`.
- Some tests set `XLA_PYTHON_CLIENT_PREALLOCATE` to avoid GPU preallocation.

## Test Environment Tips
- Tests import JAX/Haiku; ensure deps are installed.
- CPU execution is the default; GPU usage is optional.
- Use `XLA_PYTHON_CLIENT_PREALLOCATE=False` if GPU memory issues arise.
- If tests or JAX operations hang without output, try running python with -u (unbuffered) to see progress.
- Preserve `# %%` cell markers in test files.

## Docs / Notebooks
- Matching-pennies docs use MkDocs + mkdocstrings.
- Local preview (from `disRNN_MatchingPennies/`): `mkdocs serve`.

## Scripts and CLI
- Standalone scripts live in `disRNN_MatchingPennies/scripts/`.
- CLI entrypoints live in `disRNN_MatchingPennies/disRNN_MP/cli/`.
- Run scripts from the package root to keep relative paths working.

## JAX / Haiku / Flax Guidelines
- Keep random keys explicit (`jax.random.PRNGKey` / `split`).
- `jax.random.PRNGKey` and `jax.random.key` are old/new styles and often incompatible.
- Follow existing module patterns (Haiku uses `PRNGKey`, Flax uses `key`) and test when unsure.
- Prefer `optax` for optimization; `jaxopt` is deprecated for new work in this submodule.
- Avoid mixing numpy and jax arrays notebook cells.
- Avoid mixing numpy and jax arrays inside the same computation chain.
- For torch, keep tensors on consistent devices.
- Respect existing device flags (`device`, `send_to_cpu`).

## File/Module Placement
- Put new analysis utilities under `disRNN_MatchingPennies/disRNN_MP/analysis/`.
- Put agent logic under `disRNN_MatchingPennies/disRNN_MP/agent/`.
- Put RNN training modules under `disRNN_MatchingPennies/disRNN_MP/rnn/`.

## Helpful Commands Cheatsheet
- Run disRNN tests: `pytest disRNN_MatchingPennies/tests`.
- Run a single disRNN test: `pytest disRNN_MatchingPennies/tests/test_agents.py::test_get_choice_repeatable_within_trial`.
- Install disRNN editable: `python -m pip install -e disRNN_MatchingPennies`.
- Serve docs: `mkdocs serve` (from `disRNN_MatchingPennies/`).

## Notes for Agents
- Keep compatibility with JAX/Haiku code paths.
- Preserve `# %%` cell markers in tests/notebooks.
- When completing significant changes to the codebase (criterion: you generated a plan and asked for approval), you MUST ask the user whether they would like a walkthrough document summarizing the changes (to be placed in the `docs/LLM agent changes walkthrough` folder) before concluding the task.

## Folder Structure
- `disRNN_MP/`: Main source code package.
- `scripts/`: Example scripts demonstrating the capabilities of the package.
- `usage_example/`: Usage examples of functions and classes defined in the disRNN_MP package, mirroring its internal structure.
- `tests/`: Test suite (pytest).
- `docs/`: Documentation and LLM agent walkthroughs.
- `data/`: Data files and model checkpoints.
- `pyproject.toml`: Project metadata, build configuration, and dependencies.
- `requirements.txt`: Python dependencies list.
- `prep_juliapkg.py`: Utility script to configure Julia dependencies.
- `juliapkg.json`: Julia package dependencies and configuration.
- `mkdocs.yml`: Configuration for MkDocs documentation.
- `README.md`: General project overview.
- `LICENSE`: Project license.
