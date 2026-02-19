from typing import Optional, Collection, List, Tuple

import numpy as np
import jax
import jax.numpy as jnp
import igraph as ig
import pandas as pd
from tqdm import tqdm

from disRNN_MP.rnn.disrnn import _get_disrnn_prefix, get_bottlenecks_update
from disRNN_MP.rnn.utils import Params
from disRNN_MP.typing import Array

from disRNN_MP.rnn.disrnn_analy import plot_bottlenecks

def updateM2graph(updm:Array, thres = 0.5, input_feature_name: Optional[List[str]] = None) -> Tuple[ig.Graph, ig.Graph]:
    """convert the disRNN's update bottleneck matrix to a graph

    create a graph by thresholding the update matrix and use it as the adjacency matrix for the graph.
    also label the input names.

    Args:
        updm (Array): (shape: (latent_sz, latent_sz + input_sz)) update bottleneck matrix in the scale of 0-1 with close to 0 means open
        thres (float, optional): bottleneck openness higher than the threshold will be considered as an edge for the graph. Defaults to 0.5.
        input_feature_name (Optional[List[str]], optional): names to label input features. Defaults to None, which will set "choice" and "reward" when there are only 2 inputs, and "in{i}" when otherwise

    Raises:
        ValueError: when length of `input_feature_name` does not match with input size

    Returns:
        Tuple[full graph, compact graph]: full graph contains all nodes (including empty ones), compact graph only contains connected nodes
    """
    # pad the update matrix with zeros at top: input variables does not receive any other inputs

    input_sz = updm.shape[1]-updm.shape[0]
    latent_sz = updm.shape[0]

    adj = jnp.vstack((
        jnp.zeros((input_sz, latent_sz + input_sz)), 
        1 - updm # close to 1 means open
    ))

    # thresholding it
    adj = adj > thres

    # create the graph from adjacency matrix
    g_lat = ig.Graph.Adjacency(np.transpose(np.array(adj))) # [inputs, outputs]

    # label inputs
    if input_feature_name:
        if len(input_feature_name) == input_sz:
            input_labs = input_feature_name
        else:
            raise ValueError(f"size of input_feature_name ({len(input_feature_name)}) does not match with size of input features ({input_sz})")
    else:
        if input_sz == 2:
            input_labs = ["choice", "reward"]
        else:
            input_labs = [f"in{i}" for i in range(input_sz)]

    g_lat.vs["label"] = input_labs + [f"lat{i+1}" for i in range(latent_sz)]
    g_lat.vs["color"] = list(range(input_sz)) + [input_sz for _ in range(latent_sz)]
    g_lat.vs['type'] = ['input'] * input_sz + ['latent'] * latent_sz
    g_lat.vs['lat_id'] = [np.nan] * input_sz + [i+1 for i in range(latent_sz)] 

    g_shrink = g_lat.copy()

    # delete orphan nodes: those are not part of the learnt model
    g_shrink.delete_vertices(np.arange(g_lat.vcount())[np.array(g_lat.degree()) == 0])

    return (g_lat, g_shrink)



def group_isomorphic(graphs: Collection[ig.Graph]) -> np.ndarray:
    """group a collection of graphs into isomorphic graphs

    the isomorphism respects vertex attribute "color"

    Args:
        graphs (Collection[ig.Graph]): _description_

    Returns:
        np.ndarray: np array of group ids for the graphs
    """

    # grp = np.ones(len(graphs), dtype = int) * -1
    pgdf = pd.DataFrame(data = {
        'graph': np.array(graphs),
        'group': np.ones(len(graphs), dtype = int) * -1,
    })

    pgdf['group'][0] = 0
    for i in tqdm(range(1, len(pgdf)), "calculating isomorphism"):
        g = pgdf['graph'].iloc[i]

        # get one representative graph from each isomorphic group
        repr_df = pgdf.iloc[:i].groupby('group').first().reset_index()
        for j, row in repr_df.iterrows():
            g2 = row['graph']
            if g.isomorphic_bliss(g2, color1 = g.vs['color'], color2 = g2.vs['color']):
                pgdf['group'][i] = row['group']
                break
        if pgdf['group'][i] == -1:
            pgdf['group'][i] = repr_df['group'].max() + 1
    
    return pgdf['group'].to_numpy()


def cal_subisomorph(gA: ig.Graph, gB: ig.Graph) -> Tuple[bool, list]:
    """gA *has* gB as sub-network
    gB < gA
    """
    
    type_match = np.array(gB.vs['color']).reshape(-1,1) == np.array(gA.vs['color']).reshape(1,-1)
    domains = [list(np.arange(type_match.shape[1])[type_match[j]]) for j in range(type_match.shape[0])]
    A_issub_B, A_mapto_B = gA.subisomorphic_lad(gB, domains = domains, return_mapping = True)
    return (A_issub_B, A_mapto_B)