from typing import Tuple
from matplotlib.axes import Axes
from matplotlib.figure import Figure, SubFigure
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import networkx as nx
import igraph as ig

def _get_both_fig_ax(fig: Figure | SubFigure | None = None, ax: Axes | None = None) -> Tuple[Figure|SubFigure, Axes]:

    if ax is None and fig is None:
        fig, ax = plt.subplots()
    elif ax is None and fig is not None:
        ax = fig.subplots()
    elif ax is not None and fig is None:
        fig = ax.get_figure()

    if fig is None or ax is None:
        raise ValueError('cannot get figure and/or axes')
    
    return fig, ax 


def plt_graph_with_igraph(g, fig: Figure | SubFigure | None = None, ax: Axes | None = None, visual_style: dict | None = None):
    
    

    # find out latent to latent connection
    for e in g.es:
        s_type = g.vs[e.source]["type"]
        t_type = g.vs[e.target]["type"]
        if s_type == "latent" and t_type == "latent":
            e["e_type"] = "lat_lat"
            s_lat = g.vs[e.source]["lat_id"]
            t_lat = g.vs[e.target]["lat_id"]
            e['curve_sign'] = 1 if s_lat >= t_lat else -1
        else:
            e["e_type"] = None


    # Extract vertex-level attributes into a DataFrame
    df_vertices = pd.DataFrame({
        "type":   g.vs["type"],
        "lat_id": g.vs["lat_id"]
    })

    # Map types to x-coordinates
    def map_type_to_x(t):
        if t == "input":
            return 0
        elif t == "latent":
            return 1
        elif t == "output":
            return 2
        else:
            return np.nan

    df_vertices["x"] = df_vertices["type"].apply(map_type_to_x)

    # Sort so that within each type, we order by descending lat_id
    df_vertices = df_vertices.sort_values(by=["type", "lat_id"], ascending=[True, False])

    # in_grp_order = 1..n() within each type
    df_vertices["in_grp_order"] = df_vertices.groupby("type").cumcount() + 1

    # y = mean(in_grp_order) - in_grp_order
    df_vertices["y"] = (
        df_vertices.groupby("type")["in_grp_order"].transform("mean")
        - df_vertices["in_grp_order"]
    )

    # (Optional) Re-index to match igraph’s vertex ordering
    df_vertices = df_vertices.reset_index(drop=True)


    # We'll add the index of the vertex in g.vs:
    df_vertices["original_index"] = range(g.vcount())

    # Re-sort by type + lat_id (descending)
    df_vertices = df_vertices.sort_values(by=["type", "lat_id"], ascending=[True, False])

    # Create the in-group columns
    df_vertices["in_grp_order"] = df_vertices.groupby("type").cumcount() + 1
    df_vertices["y"] = (
        df_vertices.groupby("type")["in_grp_order"].transform("mean")
        - df_vertices["in_grp_order"]
    )

    # Now reorder by the original igraph order:
    df_vertices = df_vertices.sort_values(by="original_index")

    # Finally get the layout
    layout_coords = list(zip(df_vertices["x"], df_vertices["y"]))

    # Create an igraph Layout from our coords
    layout = ig.Layout(coords=layout_coords)

    # Build visual style

    # defaults
    _visual_style = {
        # "bbox": (600, 600),
        # "margin": 40,
        "vertex_size": 0.02,
        "vertex_label_dist": 1,
        "vertex_label_angle": 0.5* np.pi,
        "vertex_label_color": 'red',
        # "edge_arrow_size": 0.7,
        # "edge_color": [
        #     "red" if et == "lat_lat" else "black" 
        #     for et in g.es["e_type"]
        # ],
        "edge_curved": [
            0.4 * e['curve_sign'] if e["e_type"] == "lat_lat" else 0.0
            for e in g.es
        ]
    }

    if visual_style is not None:
        _visual_style.update(visual_style)

    fig, ax = _get_both_fig_ax(fig, ax)
    
    ig.plot(g, target = ax, layout = layout, **_visual_style)

    return fig, ax




def _get_edgelist_attr(nxg: nx.Graph, edgelist, attr: str, default = None):
    """get attribute for each edge in an edgelist as a list"""
    res = []
    for u, v in edgelist:
        res.append(
            nxg[u][v].get(attr, default)
        )
    return res



def plt_graph_with_networkx(nxg, fig: Figure | SubFigure | None = None, ax: Axes | None = None, visual_style: dict | None = None):
    

    # default visual style
    _visual_style = {
        'edge_default_width': 1,
        'edge_default_color': 'black',
        'edge_width_attribute': 'width',
        'edge_color_attribute': 'color',

        'node_label_attribute': 'name',
        'node_color_attribute': 'color',
        'node_label_default_color': 'black',
        'node_label_position_offset_x': 0,
        'node_label_position_offset_y': 0,
    }

    if visual_style is not None:
        _visual_style.update(visual_style)

    type_to_x = {
        'input': 0,
        'latent': 1,
        'output': 2
    }

    nodes_dict = dict(nxg.nodes(data=True))
    df_node = pd.DataFrame.from_dict(nodes_dict, orient='index')
    df_node.sort_values(by=["type", "lat_id"], ascending=[True, False], inplace = True)
    df_node["in_grp_order"] = df_node.groupby("type").cumcount() + 1
    df_node['x'] = df_node['type'].map(type_to_x)
    df_node["y"] = ( # y center to the middle of each type
        df_node.groupby("type")["in_grp_order"].transform("mean")
        - df_node["in_grp_order"]
    )
    df_node.sort_index(inplace=True)
    pos = df_node[['x', 'y']].apply(tuple, axis=1).to_dict()

    lab_pos = (
        df_node[['x', 'y']]
        .assign(
            x = lambda df: df['x'] + _visual_style['node_label_position_offset_x'],
            y = lambda df: df['y'] + _visual_style['node_label_position_offset_y'],
        )
        .apply(tuple, axis=1).to_dict()
    )

    fig, ax = _get_both_fig_ax(fig, ax)

    plt.figure(fig, layout = 'constrained')

    # Draw nodes
    nx.draw_networkx_nodes(nxg, pos, node_color = df_node['color'].values, cmap = 'Set1')

    # Draw labels using the 'name' attribute
    labels = nx.get_node_attributes(nxg, _visual_style['node_label_attribute'])
    nx.draw_networkx_labels(
        nxg, lab_pos, labels,
        font_color=_visual_style['node_label_default_color'],
    )



    # ======= draw edges
    default_width = _visual_style['edge_default_width']    # width for edges with no 'weight'
    ew_attr = _visual_style['edge_width_attribute']
    default_color = _visual_style['edge_default_color'] # color for edges with no 'weight'
    ec_attr = _visual_style['edge_color_attribute']


    # Separate edges: find edges connecting two latent nodes
    latent_edges = [
        (u, v) for u, v in nxg.edges()
        if nxg.nodes[u].get('type') == 'latent' and nxg.nodes[v].get('type') == 'latent'
    ]

    output_edges = [
        (u, v) for u, v in nxg.edges()
        if nxg.nodes[u].get('type') == 'latent' and nxg.nodes[v].get('type') == 'output'
    ]

    # Other edges remain normal
    other_edges = [edge for edge in nxg.edges() if edge not in latent_edges and edge not in output_edges]

    # Draw normal edges (input edges)
    nx.draw_networkx_edges(
        nxg, pos, 
        edgelist = other_edges, 
        width = _get_edgelist_attr(nxg, other_edges, ew_attr, default_width),
        edge_color = _get_edgelist_attr(nxg, other_edges, ec_attr, default_color),
        min_source_margin=10, min_target_margin=10
    )
    # Draw curved edges for latent-to-latent connections
    nx.draw_networkx_edges(
        nxg, pos, 
        edgelist=latent_edges,
        width = _get_edgelist_attr(nxg, latent_edges, ew_attr, default_width),
        edge_color = _get_edgelist_attr(nxg, latent_edges, ec_attr, default_color),
        min_source_margin=10, min_target_margin=10,
        connectionstyle='arc3, rad=0.5'  # adjust rad for desired curvature
    )

    # ======= Draw output edges as straight ones with optional weights and colors

    # build width & color lists
    edge_widths = _get_edgelist_attr(nxg, output_edges, ew_attr, default_width)
    edge_colors = _get_edgelist_attr(nxg, output_edges, ec_attr, default_color)

    # now draw them
    nx.draw_networkx_edges(
        nxg, pos,
        edgelist=output_edges,
        width=edge_widths,
        edge_color=edge_colors,
        min_source_margin=10,
        min_target_margin=10,
        arrows=True             # show arrows if directed
    )

    ax.axis('off')

    return fig, ax