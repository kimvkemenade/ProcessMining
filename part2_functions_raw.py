#!/usr/bin/env python
# coding: utf-8


import pandas as pd
import pm4py
import matplotlib.pyplot as plt
from typing import Optional, Set, FrozenSet
import numpy as np
import networkx as nx
from pm4py.algo.evaluation.generalization import algorithm as generalization_evaluator
from pm4py.algo.evaluation.simplicity import algorithm as simplicity_evaluator
from pm4py.objects.process_tree.obj import ProcessTree, Operator


XES_PATH = r"BPI2017Denied(3).xes"


event_log = pm4py.read_xes(XES_PATH)
net, initial_marking, final_marking = pm4py.discover_petri_net_inductive(event_log)
tree = pm4py.discover_process_tree_inductive(event_log)




def get_leaf_activities(node):
    if node.label is not None:  # leaf activity
        return {node.label}
    acts = set()
    for child in node.children:
        acts |= get_leaf_activities(child)
    return acts

def extract_parallel_pairs(node):
    pairs = set()
    if node.operator == Operator.PARALLEL:
        children_sets = [get_leaf_activities(c) for c in node.children]
        for i in range(len(children_sets)):
            for j in range(i+1, len(children_sets)):
                for a in children_sets[i]:
                    for b in children_sets[j]:
                        pairs.add(frozenset([a, b]))
    for child in node.children:
        pairs |= extract_parallel_pairs(child)
    return pairs

parallel_pairs = extract_parallel_pairs(tree)
print("Detected parallel pairs:", parallel_pairs)

# Define ParallelPairs as a type alias for better readability
ParallelPairs = Set[FrozenSet[str]]

def compute_handover_matrix(
    log_df: pd.DataFrame,
    case_col: str = "case_id",
    activity_col: str = "activity",
    resource_col: str = "resource",
    timestamp_col: Optional[str] = None,
    parallel_pairs: Optional[ParallelPairs] = None,
) -> pd.DataFrame:
    """
    Compute mean number of handovers (from resource r1 to r2) per case.

    Parameters
    ----------
    log_df : pandas.DataFrame
        Event log: one row per event.
    case_col, activity_col, resource_col : str
        Column names
    timestamp_col : str or None
        If provided, rows PER CASE are sorted by this column. If None, it assumes
        the rows are already ordered in case order.
    parallel_pairs : set of frozensets or None
        Set containing unordered pairs of activities that are parallel.
        If None, no activity is considered parallel.

    Returns
    -------
    handover_df : pandas.DataFrame
        DataFrame indexed / columns by resource names, with values = mean handovers per case.
    """

    if parallel_pairs is None:
        parallel_pairs = set()

    # distinct resources
    resources = sorted(log_df[resource_col].dropna().unique().tolist())
    r_index = {r: i for i, r in enumerate(resources)}
    n = len(resources)

    # total handover counts across all cases (directed)
    total_counts = np.zeros((n, n), dtype=float)

    # number of cases (denominator)
    cases = log_df[case_col].unique()
    n_cases = len(cases)
    if n_cases == 0:
        raise ValueError("Empty log or no cases found.")

    # group by case
    grouped = log_df.groupby(case_col)

    for case_id, group in grouped:
        # ensure events in this case are ordered
        if timestamp_col is not None:
            group_sorted = group.sort_values(timestamp_col, kind="mergesort")
        else:
            # preserve original order within group
            group_sorted = group

        # iterate consecutive pairs
        events = group_sorted[[activity_col, resource_col]].values
        # events is list of rows [activity, resource]
        for i in range(len(events) - 1):
            act_i, res_i = events[i]
            act_j, res_j = events[i + 1]

            # skip missing resources or activities
            if pd.isna(res_i) or pd.isna(res_j) or pd.isna(act_i) or pd.isna(act_j):
                continue

            # same resource -> not a handover
            if res_i == res_j:
                continue

            # skip if activities are model-declared parallel
            if frozenset([act_i, act_j]) in parallel_pairs:
                continue

            # otherwise increment handover count res_i -> res_j
            if res_i in r_index and res_j in r_index:
                total_counts[r_index[res_i], r_index[res_j]] += 1.0
    print(total_counts)
    # mean per case
    mean_per_case = total_counts / np.array(total_counts).sum()

    handover_df = pd.DataFrame(mean_per_case, index=resources, columns=resources)
    return handover_df, total_counts

handover_df, total_counts = compute_handover_matrix(
    log_df= pm4py.read_xes(XES_PATH),                 
    case_col="case:concept:name",
    activity_col="concept:name",
    resource_col="org:resource",
    timestamp_col="time:timestamp",
    parallel_pairs=parallel_pairs
)


np.amax(np.array(total_counts), axis=1)


def handover_network_from_matrix(
    handover_df: pd.DataFrame,
    threshold: float = 0.0,
    directed: bool = True,
    show_unconnected_nodes: bool = False
) -> nx.Graph | nx.DiGraph:
    """
    Create a NetworkX graph from a handover DataFrame.
    Only edges with weight >= threshold are included.

    Parameters
    ----------
    handover_df : pandas.DataFrame
        index and columns are resource names; values are mean handovers per case
    threshold : float
        Minimum edge weight to keep
    directed : bool
        If False, returns an undirected graph (weight = mean of both directions)
    show_unconnected_nodes : bool
        If True, nodes without meaningful edges are included.

    Returns
    -------
    G : networkx.DiGraph or Graph
    """
    G = nx.DiGraph() if directed else nx.Graph()

    if directed:
        for src in handover_df.index:
            for dst in handover_df.columns:
                w = handover_df.at[src, dst]
                if w >= threshold and w > 0:
                    G.add_edge(src, dst, weight=float(w))
        if show_unconnected_nodes:
            for node in handover_df.index:
                G.add_node(node)
    else:
        visited = set()
        for a in handover_df.index:
            for b in handover_df.columns:
                if (a, b) in visited or (b, a) in visited:
                    continue
                w = (handover_df.at[a, b] + handover_df.at[b, a]) / 2.0
                if w >= threshold and w > 0:
                    G.add_edge(a, b, weight=float(w))
                visited.add((a, b))
        if show_unconnected_nodes:
            for node in handover_df.index:
                G.add_node(node)

    return G


# Get network with threshold 0.02 mean handovers per case
G = handover_network_from_matrix(handover_df, threshold=0.001, show_unconnected_nodes=False)


pos = nx.spring_layout(G, seed=42)
weights = [G[u][v]['weight'] for u,v in G.edges()]
nx.draw(G, pos, with_labels=True, node_size=100, width=[max(0.1, w*5) for w in weights], font_size=8)
edge_labels = nx.get_edge_attributes(G, 'weight')
nx.draw_networkx_edge_labels(G, pos, edge_labels={k: f"{v:.2f}" for k, v in edge_labels.items()}, font_size=6)
plt.show()




