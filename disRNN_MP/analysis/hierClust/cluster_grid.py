"""This module provide a class to improve seaborn's clustermap 
"""
# %%
from types import SimpleNamespace
from typing import Literal

import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.axes import Axes
from scipy.cluster import hierarchy
from seaborn.matrix import ClusterGrid as sns_ClusterGrid
from seaborn.utils import despine



def _plot_dendro_scipy(ax: Axes, linkage, type: Literal['row', 'col'], tree_kws: dict|None = None):
    """
    """
    kws = {} if tree_kws is None else tree_kws.copy()
    kws['orientation'] = 'left' if type == 'row' else 'top'

    dendro_dict = hierarchy.dendrogram(
        linkage,
        ax=ax,    # draw into the cleared Seaborn axis
        no_plot=False,
        no_labels=True,           # supress labels to keep it clean like Seaborn
        **kws # orientation will be either left or top depends on type
    )

    if type == 'row':
        ax.invert_yaxis()
    ax.axis('off')

    leaves = dendro_dict["leaves"]
    return SimpleNamespace(reordered_ind=leaves)


class ClusterGrid(sns_ClusterGrid):

    def __init__(self, 
            metric='euclidean', method='average', row_linkage=None, col_linkage=None,  # new arguments
            figsize=(10, 10), dendrogram_ratio=.2, colors_ratio=0.03, cbar_pos=(.02, .8, .05, .18), # superclass arguments with new default
            **kwargs) -> None:
        
        super().__init__(
            **kwargs, 
            figsize=figsize, dendrogram_ratio=dendrogram_ratio, 
            colors_ratio=colors_ratio, cbar_pos=cbar_pos
        )

        self.metric = metric
        self.method = method
        self.row_linkage = row_linkage
        self.col_linkage = col_linkage

    def _cal_linkage(self, axis, method, metric):
        data = self.data2d.to_numpy(copy=False)
        if axis == 1:
            data = data.T
        return hierarchy.linkage(data, method=method, metric=metric)

    def _resolve_linkage(self, axis, metric:str|None=None, method:str|None=None, linkage_override:np.ndarray|None=None):
        """Return linkage matrix, computing it if not provided, 
        and also update the corresponding linkage attribute
        
        when all of metric, method, and linkage_override are None, will use (or calculate first) default linkage set in __init__
        """

        match axis:
            case 0:
                linkage_attr = 'row_linkage'
            case 1:
                linkage_attr = 'col_linkage'
            case _:
                raise ValueError(f'only accept 0 or 1 as axis, got {axis}')

        if linkage_override is not None:
            link = linkage_override
        
        elif metric is None and method is None:
            link = getattr(self, linkage_attr) 
            link = self._cal_linkage(axis, method=self.method, metric=self.metric) if link is None else link

        else:
            metric = metric or self.metric
            method = method or self.method
            link = self._cal_linkage(axis, method=method, metric=metric)
        
        # update corresponding linkage attribute
        setattr(self, linkage_attr, link)
        return link

    def plot_dendrograms(self, row_cluster: bool, col_cluster: bool, metric = None, method = None,
                         row_linkage = None, col_linkage = None, tree_kws = None):
        # Plot the row dendrogram
        if row_cluster:
            resolved_row = self._resolve_linkage(
                axis=0, metric=metric, method=method, linkage_override=row_linkage
            )
            self.dendrogram_row = _plot_dendro_scipy(
                type = 'row', linkage=resolved_row, ax=self.ax_row_dendrogram,
                tree_kws=tree_kws
            ) # type: ignore
        else:
            self.ax_row_dendrogram.set_xticks([])
            self.ax_row_dendrogram.set_yticks([])
        # PLot the column dendrogram
        if col_cluster:
            resolved_col = self._resolve_linkage(
                axis=1, metric=metric, method=method, linkage_override=col_linkage
            )
            self.dendrogram_col = _plot_dendro_scipy(
                type = 'col', linkage=resolved_col, ax=self.ax_col_dendrogram,
                tree_kws=tree_kws
            ) # type: ignore
        else:
            self.ax_col_dendrogram.set_xticks([])
            self.ax_col_dendrogram.set_yticks([])
        # despine(ax=self.ax_row_dendrogram, bottom=True, left=True)
        # despine(ax=self.ax_col_dendrogram, bottom=True, left=True)

    def plot(self, metric=None, method=None, colorbar_kws=None, row_cluster=True, col_cluster=True,
             row_linkage=None, col_linkage=None, tree_kws=None, **kws):
        # inlcude defaults
        return super().plot(
            metric=metric, method=method, colorbar_kws=colorbar_kws, row_cluster=row_cluster, col_cluster=col_cluster,
            row_linkage=row_linkage, col_linkage=col_linkage, tree_kws=tree_kws, **kws)
        

def clustermap(
    data, *,
    pivot_kws=None, method=None, metric=None,
    z_score=None, standard_scale=None, figsize=(10, 10),
    cbar_kws=None, row_cluster=True, col_cluster=True,
    row_linkage=None, col_linkage=None,
    row_colors=None, col_colors=None, mask=None,
    dendrogram_ratio=.2, colors_ratio=0.03,
    cbar_pos=(.02, .8, .05, .18), tree_kws=None,
    **kwargs
):
    """
    Plot a matrix dataset as a hierarchically-clustered heatmap.

    This function requires scipy to be available.

    Parameters
    ----------
    data : 2D array-like
        Rectangular data for clustering. Cannot contain NAs.
    pivot_kws : dict, optional
        If `data` is a tidy dataframe, can provide keyword arguments for
        pivot to create a rectangular dataframe.
    method : str, optional
        Linkage method to use for calculating clusters. See
        :func:`scipy.cluster.hierarchy.linkage` documentation for more
        information.
    metric : str, optional
        Distance metric to use for the data. See
        :func:`scipy.spatial.distance.pdist` documentation for more options.
        To use different metrics (or methods) for rows and columns, you may
        construct each linkage matrix yourself and provide them as
        `{row,col}_linkage`.
    z_score : int or None, optional
        Either 0 (rows) or 1 (columns). Whether or not to calculate z-scores
        for the rows or the columns. Z scores are: z = (x - mean)/std, so
        values in each row (column) will get the mean of the row (column)
        subtracted, then divided by the standard deviation of the row (column).
        This ensures that each row (column) has mean of 0 and variance of 1.
    standard_scale : int or None, optional
        Either 0 (rows) or 1 (columns). Whether or not to standardize that
        dimension, meaning for each row or column, subtract the minimum and
        divide each by its maximum.
    figsize : tuple of (width, height), optional
        Overall size of the figure.
    cbar_kws : dict, optional
        Keyword arguments to pass to `cbar_kws` in :func:`heatmap`, e.g. to
        add a label to the colorbar.
    {row,col}_cluster : bool, optional
        If ``True``, cluster the {rows, columns}.
    {row,col}_linkage : :class:`numpy.ndarray`, optional
        Precomputed linkage matrix for the rows or columns. See
        :func:`scipy.cluster.hierarchy.linkage` for specific formats.
    {row,col}_colors : list-like or pandas DataFrame/Series, optional
        List of colors to label for either the rows or columns. Useful to evaluate
        whether samples within a group are clustered together. Can use nested lists or
        DataFrame for multiple color levels of labeling. If given as a
        :class:`pandas.DataFrame` or :class:`pandas.Series`, labels for the colors are
        extracted from the DataFrames column names or from the name of the Series.
        DataFrame/Series colors are also matched to the data by their index, ensuring
        colors are drawn in the correct order.
    mask : bool array or DataFrame, optional
        If passed, data will not be shown in cells where `mask` is True.
        Cells with missing values are automatically masked. Only used for
        visualizing, not for calculating.
    {dendrogram,colors}_ratio : float, or pair of floats, optional
        Proportion of the figure size devoted to the two marginal elements. If
        a pair is given, they correspond to (row, col) ratios.
    cbar_pos : tuple of (left, bottom, width, height), optional
        Position of the colorbar axes in the figure. Setting to ``None`` will
        disable the colorbar.
    tree_kws : dict, optional
        Parameters for the :class:`scipy.hierarchy.dendrogram`
        that is used to plot the dendrogram tree.
    kwargs : other keyword arguments
        All other keyword arguments are passed to :func:`heatmap`.

    Returns
    -------
    :class:`ClusterGrid`
        A :class:`ClusterGrid` instance.

    See Also
    --------
    heatmap : Plot rectangular data as a color-encoded matrix.

    Notes
    -----
    The returned object has a ``savefig`` method that should be used if you
    want to save the figure object without clipping the dendrograms.

    To access the reordered row indices, use:
    ``clustergrid.dendrogram_row.reordered_ind``

    Column indices, use:
    ``clustergrid.dendrogram_col.reordered_ind``


    """

    plotter = ClusterGrid(data=data, pivot_kws=pivot_kws, figsize=figsize,
                          row_colors=row_colors, col_colors=col_colors,
                          z_score=z_score, standard_scale=standard_scale,
                          mask=mask, dendrogram_ratio=dendrogram_ratio,
                          colors_ratio=colors_ratio, cbar_pos=cbar_pos)

    return plotter.plot(metric=metric, method=method,
                        colorbar_kws=cbar_kws,
                        row_cluster=row_cluster, col_cluster=col_cluster,
                        row_linkage=row_linkage, col_linkage=col_linkage,
                        tree_kws=tree_kws, **kwargs)

# %% TEST 
# import numpy as np
# import seaborn as sns
# import matplotlib.pyplot as plt

# # Load tidy data
# flights = sns.load_dataset("flights")

# # Pivot to matrix form: months × years
# flights_pivot = flights.pivot(index="month", columns="year", values="passengers")

# cmap = plt.get_cmap('viridis')
# col_colors = [cmap(i) for i in np.linspace(0,1,num=flights_pivot.shape[1])]
# # %% Original Clustermap
# sns.clustermap(flights_pivot, cmap="viridis", standard_scale=1, col_colors=col_colors)
# plt.show()
# # %%
# clustermap(flights_pivot, cmap="viridis", standard_scale=1, col_colors=col_colors)
# plt.show()
# # %% new clustermap
# clustermap(flights_pivot, cmap="viridis", standard_scale=1, col_colors=col_colors, method='single')
# plt.show()
# %%
