"""
This module provide a class specialized in representing calculations and plots associated with hierarchical clustering on a square matrix (usually a pairwise )


"""

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from scipy.cluster import hierarchy as ha
from typing import List, Tuple, Dict, Any, Literal
from functools import cached_property
import seaborn as sns
from matplotlib.patches import Rectangle
import pandas as pd
from matplotlib.axes import Axes
import seaborn as sns
import numpy as np
from matplotlib.patches import Rectangle
from typing import List, Tuple, Dict, Any
from typing import Sequence, Literal, Callable
import seaborn as sns
import numpy as np
from scipy.spatial.distance import squareform
from scipy.cluster.hierarchy import linkage, dendrogram, leaves_list

from dataclasses import dataclass, asdict, InitVar, field
import polars as pl
from .cluster_grid import ClusterGrid, clustermap
from .linkage_prune import LinkagePruner

import pandas as pd

def is_auto_generated(df:pd.DataFrame, axis: Literal['index', 'columns'] = 'index'):
    """
    Checks if a DataFrame's index or column is auto-generated
    (i.e., strictly equals the default RangeIndex: 0, 1, ..., n-1).

    Parameters:
    -----------
    df : pd.DataFrame
        The DataFrame to inspect.
    target : str, optional
        The name of the column to check. If None (default), checks the DataFrame's index.

    Returns:
    --------
    bool
        True if the target is equivalent to a default auto-generated RangeIndex.
    """
    assert axis in ('index', 'columns'), f"illegal arg for axis {axis}"

    indx = getattr(df, axis)
    _length = df.shape[0 if axis == 'index' else 1]

    return (
        isinstance(indx, pd.RangeIndex) 
        and indx.start == 0 
        and indx.step == 1 
        and indx.stop == _length
    )


def _assert_indices_in_bounds(rows: np.ndarray, cols: np.ndarray, shape: tuple[int, int]):
    """Raise IndexError if any (row, col) is outside matrix bounds."""
    n_rows, n_cols = shape
    if (
        (rows < 0).any() or (cols < 0).any()
        or (rows >= n_rows).any() or (cols >= n_cols).any()
    ):
        raise IndexError(f"indices contain out-of-bound positions for matrix of shape {shape}")


@dataclass
class SquareMatrixAnalyzer:
    """a data structure with aligned square matrix and dataframe

    The matrix and dataframe index will always be reset and dropped to default integer index upon initialization
    
    upon initialization and using its method will guarantee alignment, but modifying its attribute directly does not. It is user's responsibility to maintain the alignment in those case
    """

    matrix: InitVar[np.ndarray]
    dataframe: InitVar[pl.DataFrame | pd.DataFrame | None]

    _df_index_col: str = '__index__'
    mat: np.ndarray = field(init=False)
    df: pl.DataFrame = field(init=False)

    def __post_init__(self, matrix: np.ndarray, dataframe: pl.DataFrame | pd.DataFrame | None):

        assert matrix.shape[0] == matrix.shape[1], f"matrix should be square, but received a {matrix.shape} matrix"

        if dataframe is None:
            dataframe = pl.DataFrame({self._df_index_col: np.arange(matrix.shape[0])})
        elif isinstance(dataframe, pd.DataFrame):
            dataframe = pl.from_pandas(dataframe).with_row_index(name=self._df_index_col)
        else:
            # to override existing column if exists
            dataframe = dataframe.with_columns(pl.row_index(name=self._df_index_col))

        assert matrix.shape[0] == dataframe.shape[0], f"matrix ({matrix.shape[0]} rows) does not match dataframe ({dataframe.shape[0]} rows)"

        self.mat = matrix
        self.df = dataframe

    def filter(self, *predicates: pl.Expr, _copy = True):
        
        # filter dataset by dataframe

        ind = (
            self.df
            .filter(*predicates)
            .get_column(self._df_index_col)
            .to_list()
        )

        return self.reorder(ind, _copy = _copy)
    
    def reorder(self, ind: Sequence[int], _copy = False):

        df = self.df[ind]
        mat = self.mat[np.ix_(ind, ind)] # note the `ix_` call

        if _copy:

            kws = asdict(self)
            kws.pop('mat')
            kws.pop('df')
            
            return self.__class__(mat, df, **kws)
        else:
            self.df = df.with_columns(pl.row_index(name=self._df_index_col))
            self.mat = mat
            return self
    
    def long_form(self, indices = None, row_name = 'row', col_name = 'col', value_name = 'value'):
        """return a long-form dataframe of the matrix
        
        optionally provide `indices` as an array-like of shape (n, 2) to select
        only specific (row, col) pairs
        """

        if indices is None:
            rows, cols = np.indices(self.mat.shape)
            rows = rows.ravel()
            cols = cols.ravel()
        else:
            idx = np.asarray(indices)
            if idx.ndim != 2 or idx.shape[1] != 2:
                raise ValueError("indices must be a sequence/array of shape (n, 2)")

            if not np.issubdtype(idx.dtype, np.integer):
                idx = idx.astype(int)

            rows, cols = idx[:, 0], idx[:, 1]

            _assert_indices_in_bounds(rows, cols, self.mat.shape)

        return pl.DataFrame({
            row_name: rows,
            col_name: cols,
            value_name: self.mat[rows, cols]
        })
    
    
    def get_mat_and_df(self):
        return (self.mat, self.df)
    
    #TODO make a method to perform hierarchical clustering (calculate linkage and sort accordingly) and return the class
    
def square_mat_linkage(smat, linkage_method):
    condensed = squareform(smat, checks=False)
    # when input to linkage is condensed distance vector, metric is ignored
    return linkage(condensed, method=linkage_method)


@dataclass
class SquareHierarchicalClustering(SquareMatrixAnalyzer):
    """hierarchical clustering of a pairwise matrix
    
    The matrix and associated dataframe will be reordered by the linkage result upon initialization
    """
    linkage_method: Literal['average', 'single', 'complete', 'weighted'] = 'average'
    distance_trans: Callable[[np.ndarray], np.ndarray] = lambda x: x
    visual_trans: Callable[[np.ndarray], np.ndarray] = lambda x: x

    # TODO need to make `distance_trans` immutable after init, since cached properties depend on them

    _cluster_id_col: str = 'cluster_id'
    cluster_thre: float | None = field(default=None)
    criterion: Literal['distance', ] = 'distance' #TODO complete the type

    heatmap_kws: dict = field(default_factory=dict)
    dendrogram_kws: dict = field(default_factory=dict)

    def __post_init__(self, 
            matrix: np.ndarray, 
            dataframe: pl.DataFrame | pd.DataFrame | None
        ):

        super().__post_init__(matrix, dataframe)
        self._reorder_by_linkage()
        # after above reordering, leaves_list(self.linkage) should generate np.arange(self.mat.shape[0])

    def _reorder_by_linkage(self):
        """reorder matrix by linkage
        need to calculate distance and linkage on original matrix data
        since it will be reordered later, has to calculate again later

        This function is calculated during init, so cannot call other method
        """
        distance = self.distance_trans(self.mat)
        linkage = square_mat_linkage(distance, self.linkage_method)
        leaves = leaves_list(linkage)

        # make no copy to avoid infinite loop when called in __post_init__
        self.reorder(leaves, _copy = False)

    @cached_property
    def distance(self):
        return self.distance_trans(self.mat)

    @property
    def visual_matrix(self):
        return self.visual_trans(self.mat)

    @cached_property
    def linkage(self):
        """linkage result of the distance matrix"""

        condensed = squareform(self.distance, checks=False)
        # when input to linkage is condensed distance vector, metric is ignored
        return linkage(condensed, method=self.linkage_method)
    
    @cached_property
    def _linkage_pruner(self):
        """a helper class to calculate pruned linkage tree for a subset of observations (for fast plotting) """
        return LinkagePruner(self.linkage, leaf_order=np.arange(self.mat.shape[0]))
    

    def fcluster(self, t, criterion=None, **kwargs):
        """
        a wrapper of scipy hierarchical clustering `fcluster` function

        calling this function will update the instance for threshold and cluster labels

        :param t: threshold
        :param criterion: Description
        :param kwargs: scipy `fcluster` keyword args
        """

        if criterion is not None:
            self.criterion = criterion

        self.cluster_thre = t
        clust_id = ha.fcluster(self.linkage, t=t, criterion=self.criterion, **kwargs)

        self.df = self.df.with_columns(**{self._cluster_id_col: clust_id})

        return self
    
    def plot_dendrogram(self):
        pass

    def annot_blocks(self):
        pass
        return self

    def annot_clusters(self):
        pass
    
    def _label_visual_matrix(self, xticklabel: str | None = None, yticklabel: str | None = None):
        """return visual_matrix as dataframe with selected dataframe column labels"""

        visual_data = pd.DataFrame(self.visual_matrix)
        visual_data.columns = pd.Index(self.df[xticklabel], name=xticklabel) if xticklabel is not None else self.df.get_column(self._df_index_col).to_numpy() 
        visual_data.index = pd.Index(self.df[yticklabel], name=yticklabel) if yticklabel is not None else self.df.get_column(self._df_index_col).to_numpy() # type: ignore
        
        return visual_data

    def plot_heatmap(self,
            xticklabel=None, 
            yticklabel=None,
            **kwargs
        ):
        visual_data = self._label_visual_matrix(xticklabel=xticklabel, yticklabel=yticklabel)


        g = sns.heatmap(
            visual_data,
            xticklabels = True if xticklabel is not None else False,
            yticklabels = True if yticklabel is not None else False,
            **kwargs
        )

        return g

    def plot(self, 
            xticklabel=None, 
            yticklabel=None,
            subsample_factor: int = 1,
            **kwargs
        ):
        """
        Docstring for plot
        
        :param self: Description
        :param xticklabel: Description
        :param yticklabel: Description
        :param subsample_factor: Description
        :param kwargs: Description
        """

        assert subsample_factor.is_integer() and subsample_factor >= 1, f"please specify a whole number for `subsample_factor` (currently is {subsample_factor})"

        visual_data = self._label_visual_matrix(xticklabel=xticklabel, yticklabel=yticklabel)

        _linkage = self.linkage

        if subsample_factor > 1:
            visual_data = visual_data.iloc[::subsample_factor, ::subsample_factor]
            _linkage, _ = self._linkage_pruner.prune(np.arange(self.mat.shape[0])[::subsample_factor])


        g = clustermap(
            visual_data,
            method=None,
            metric=None,

            row_linkage=_linkage,
            col_linkage=_linkage,

            # IMPORTANT: don’t set z_score/standard_scale when visualizing distances
            z_score=None, 
            standard_scale=None,

            xticklabels = True if xticklabel is not None else False,
            yticklabels = True if yticklabel is not None else False,
            **kwargs
        )

        g.ax_heatmap.tick_params(axis="x", labelrotation=90, labelsize=7)
        g.ax_heatmap.tick_params(axis="y", labelsize=7)

        return g
