import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import igraph as ig


def tidy_to_graph(df: pd.DataFrame, node_cols: list, bipartite: bool = True):
    """
    Converts two columns from a tidy table into NODES and EDGES data frames.

    bipartite=True  (default) — two-mode graph: the two columns contain different
                    kinds of entities (e.g. countries vs. tasters) with no overlapping
                    values. Nodes are colored and shaped by column.
    bipartite=False — one-mode graph: both columns contain the same kind of entity
                    (e.g. home team vs. away team). All values are merged into a
                    single deduplicated node table with a uniform style.
    """

    if bipartite:
        NODES = df[node_cols].melt().value_counts().to_frame().reset_index()
        NODES['weight'] = NODES['count'] / NODES['count'].sum()
        NODES.index.name = 'node_id'

        color_map = {node_cols[0]: '#dce9f2', node_cols[1]: '#d9f0a3'}
        shape_map = {node_cols[0]: 'circle',   node_cols[1]: 'rectangle'}
        NODES['color'] = NODES['variable'].map(color_map)
        NODES['shape'] = NODES['variable'].map(shape_map)

    else:
        combined = df[node_cols].melt()['value'].dropna()
        NODES = combined.value_counts().rename('count').reset_index()
        NODES['weight'] = NODES['count'] / NODES['count'].sum()
        NODES['variable'] = 'node'
        NODES.index.name = 'node_id'
        NODES['color'] = '#dce9f2'
        NODES['shape'] = 'circle'

    EDGES = df[node_cols].value_counts().to_frame().reset_index()
    EDGES['weight'] = EDGES['count'] / EDGES['count'].sum()
    EDGES.index.name = 'edge_id'

    node_id_map = NODES.reset_index().set_index('value').node_id
    EDGES['source'] = EDGES[node_cols[0]].map(node_id_map)
    EDGES['target'] = EDGES[node_cols[1]].map(node_id_map)
    EDGES = EDGES.query("count > 0")

    return NODES, EDGES


def graph_to_igraph(NODES: pd.DataFrame, EDGES: pd.DataFrame, directed: bool = False):
    """
    Converts NODES and EDGES data frames into an igraph Graph object.
    """
    g = ig.Graph(edges=EDGES[['source', 'target']].values.tolist(), directed=directed)

    g.vs['label']     = NODES['value'].to_list()
    g.vs['count']     = NODES['count'].to_list()
    g.vs['weight']    = NODES['weight'].to_list()
    g.vs['node_type'] = NODES['variable'].to_list()
    g.vs['shape']     = NODES['shape'].tolist()
    g.vs['color']     = NODES['color'].tolist()
    g.es['weight']    = EDGES['weight'].tolist()

    return g


def graphplot(my_g, layout_type='rt_circular', node_size: int = None, edge_size: int = None, title=None, figsize=(20, 20)):
    """
    Plots the graph.
    """
    v_s = pd.Series(my_g.vs['weight']) * node_size if node_size else None
    e_w = pd.Series(my_g.es['weight']) * edge_size if edge_size else None

    fig, ax = plt.subplots(figsize=figsize)
    ig.plot(
        my_g,
        target=ax,
        layout=layout_type,
        edge_width=e_w,
        vertex_size=v_s,
        vertex_frame_width=1,
        vertex_frame_color="gray",
        vertex_label=my_g.vs["label"],
        vertex_label_size=12,
        edge_color='gray'
    )
    plt.title(title, fontsize=20)
    plt.show()
