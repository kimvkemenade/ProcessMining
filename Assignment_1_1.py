# Imports and setup
import os
import pandas as pd
import numpy as np
import pm4py
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances, pairwise_distances
import networkx as nx
from pm4py.objects.conversion.log import converter as log_converter


XES_PATH = r"BPI2017Denied(3).xes"
OUTPUT_DIR = "outputs_task_similarity_nb"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def import_files(XES_PATH):
    """Import event log and set up output directory."""
    log = pm4py.read_xes(XES_PATH)
    print(f"Log imported with {len(log)} traces.")
    return log


def convert_to_df(log):
    """Convert event log to pandas DataFrame."""
    df = log_converter.apply(log, variant=log_converter.Variants.TO_DATA_FRAME)
    col_map = {
    'case:concept:name': 'case_id',
    'concept:name': 'activity',
    'org:resource': 'resource',
    'time:timestamp': 'timestamp'
    }
    for old, new in col_map.items():
        if old in df.columns:
            df = df.rename(columns={old: new})
    
    req = ['case_id', 'activity', 'resource']
    missing = [c for c in req if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    use_cols = ['case_id', 'activity', 'resource', 'timestamp'] if 'timestamp' in df.columns else ['case_id', 'activity', 'resource']
    df = df[use_cols].dropna(subset=['case_id','activity','resource']).copy()
    return df

def create_counts(df):
    """Create activity-resource count matrix."""
    counts = (
    df.groupby(['resource','activity','case_id'])
      .size()
      .rename('count')
      .reset_index()
    )
    mean_per_case = (
        counts.groupby(['resource','activity'])['count']
            .mean()
            .rename('mean_per_case')
            .reset_index()
    )
    matrix = mean_per_case.pivot(index='resource', columns='activity', values='mean_per_case').fillna(0.0)
    return matrix

def compute_distance(matrix, metric = 'minkowski', minkowski_p = 2):
    """Compute distance matrix based on different metrics."""
    if matrix.shape[0] == 0:
        raise ValueError('Empty resource-activity matrix.')

    metric_l = metric.lower()
    if metric_l == 'cosine_similarity':
        weights = pd.DataFrame(cosine_similarity(matrix.values), index=matrix.index, columns=matrix.index)
    elif metric_l in ('cosine_distance', 'cosine'):
        weights = pd.DataFrame(1.0 - cosine_similarity(matrix.values), index=matrix.index, columns=matrix.index)
    elif metric_l == 'euclidean':
        weights = pd.DataFrame(euclidean_distances(matrix.values), index=matrix.index, columns=matrix.index)
    elif metric_l == 'minkowski':
        weights = pd.DataFrame(pairwise_distances(matrix.values, metric='minkowski', p=minkowski_p), index=matrix.index, columns=matrix.index)
    elif metric_l == 'hamming':
        weights = pd.DataFrame(pairwise_distances(matrix.values, metric='hamming'), index=matrix.index, columns=matrix.index)
    elif metric_l == 'pearson':
        corr = np.corrcoef(matrix.values)
        weights = pd.DataFrame(corr, index=matrix.index, columns=matrix.index).fillna(0.0)
    else:
        raise ValueError('Unsupported metric')

    return weights

def build_network(weights, measure_preference = 'similarity', threshold_value = 0.6):
    use_similarity = (measure_preference == 'similarity')

    G = nx.Graph()
    for r in weights.index:
        G.add_node(r)

    nodes = list(weights.index)
    for i in range(len(nodes)):
        for j in range(i+1, len(nodes)):
            u, v = nodes[i], nodes[j]
            w = float(weights.loc[u, v])
            if use_similarity:
                if w >= threshold_value:
                    G.add_edge(u, v, weight=w)
            else:
                if w <= threshold_value:
                    G.add_edge(u, v, weight=w)

    print(f"Nodes: {G.number_of_nodes()}, Edges: {G.number_of_edges()}")

    plt.figure(figsize=(10,8))
    pos = nx.spring_layout(G, seed=42)
    edge_w = [1 + 2 * max(0.0, (d.get('weight', 1.0) - threshold_value)) if use_similarity else 1 for _,_,d in G.edges(data=True)]
    nx.draw_networkx_nodes(G, pos, node_size=300, node_color='#4C78A8')
    nx.draw_networkx_edges(G, pos, width=edge_w, alpha=0.7)
    nx.draw_networkx_labels(G, pos, font_size=8)
    plt.axis('off')
    plt.show()
    return G


def main(XES_PATH, metric = 'minkowski', minkowski_p = 2, measure_preference = 'similarity', threshold_value = 0.6):
    """Main function to execute the steps.
    XES_PATH: Path to the XES event log file.
    metric (optional): Distance metric to use.
    minkowski_p (optional): Parameter for Minkowski distance (if applicable).
    measure_preference (optional): 'similarity' or 'distance' for network construction.
    threshold_value (optional): Threshold to filter weak connections.
    """
    log = import_files(XES_PATH)
    df = convert_to_df(log)
    counts = create_counts(df)
    matrix = compute_distance(counts, metric, minkowski_p)
    G = build_network(matrix, measure_preference, threshold_value)

    print("Process Completed")
    return G 

