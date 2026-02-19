# Recipes Docs Scaffold - Walkthrough

## Summary

Added a `docs/recipes/` section with short recipe-card templates for analysis workflows and wired them into MkDocs navigation.

## Changes Made

### `disRNN_MatchingPennies/docs/recipes/`

- Added `README.md` workflow index with links to each recipe.
- Added recipe cards for setup, data loading, training, evaluation, visualization, reproducibility, and common pitfalls.
- Each recipe includes explicit expected inputs/outputs and a minimal example block.

### `disRNN_MatchingPennies/docs/NAVIGATION.md`

- Added a new “Recipes” section with links to all recipe cards.

## Usage Notes

- Recipes are intentionally short and meant to be expanded with concrete snippets.
- The structure is MkDocs-ready via `NAVIGATION.md`.
