# %%
from disRNN_MP.analysis.hierClust.square_hier_clust import SquareMatrixAnalyzer, SquareHierarchicalClustering

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import polars as pl
import pandas as pd


# %%
# Load tidy data
flights = sns.load_dataset("flights")

# Pivot to matrix form: months × years
flights_pivot = flights.pivot(index="year", columns="month", values="passengers")

flights_years = (
    flights
    .groupby("year")
    .agg(
        mean_psgn = pd.NamedAgg("passengers", "mean"),
        var_psgn = pd.NamedAgg("passengers", "var"),
        min_psgn = pd.NamedAgg("passengers", "min"),
        max_psgn = pd.NamedAgg("passengers", "max"),
    )
    .reset_index()
)

# %%
shc = SquareHierarchicalClustering(
    matrix=np.corrcoef(flights_pivot),
    dataframe=flights_years,
    distance_trans=lambda x: 1 - x,
)
# shc.reset_index()
# %%
shc.fcluster(t=0.06)
shc.df
# %% testing filter and basic plot

(
    shc
    .filter(pl.col('mean_psgn') > 200)
    .plot(xticklabel='year', yticklabel='year')
)
# %%
