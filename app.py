import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances, pairwise_distances
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
import pm4py
from pm4py.objects.conversion.log import converter as log_converter
import os
import io
from part2_functions_raw import compute_handover_matrix, handover_network_from_matrix, extract_parallel_pairs, event_log
from pm4py.objects.process_tree.obj import Operator
from pm4py.visualization.petri_net import visualizer as pn_visualizer

st.set_page_config(page_title="Resource Clustering Analysis", layout="wide")

# Initialize session state
if 'matrix' not in st.session_state:
    st.session_state.matrix = None
if 'df' not in st.session_state:
    st.session_state.df = None
if 'handover_matrix' not in st.session_state:
    st.session_state.handover_matrix = None
if 'file_loaded' not in st.session_state:
    st.session_state.file_loaded = False



def compute_similarity_weights(matrix, metric='cosine_distance', minkowski_p=2):
    """Compute pairwise similarity/distance weights"""
    metric_l = metric.lower()
    
    if metric_l == 'cosine_similarity':
        weights = pd.DataFrame(cosine_similarity(matrix.values), index=matrix.index, columns=matrix.index)
    elif metric_l in ('cosine_distance', 'cosine'):
        weights = pd.DataFrame(1.0 - cosine_similarity(matrix.values), index=matrix.index, columns=matrix.index)
    elif metric_l == 'euclidean':
        weights = pd.DataFrame(euclidean_distances(matrix.values), index=matrix.index, columns=matrix.index)
    elif metric_l == 'minkowski':
        weights = pd.DataFrame(pairwise_distances(matrix.values, metric='minkowski', p=minkowski_p), 
                             index=matrix.index, columns=matrix.index)
    elif metric_l == 'hamming':
        weights = pd.DataFrame(pairwise_distances(matrix.values, metric='hamming'), 
                             index=matrix.index, columns=matrix.index)
    elif metric_l == 'pearson':
        corr = np.corrcoef(matrix.values)
        weights = pd.DataFrame(corr, index=matrix.index, columns=matrix.index).fillna(0.0)
    else:
        raise ValueError(f'Unsupported metric: {metric}')
    
    return weights

def build_network(weights, threshold_value, use_similarity=False):
    """Build network graph from weights"""
    G = nx.Graph()
    for r in weights.index:
        G.add_node(r)
    
    nodes = list(weights.index)
    for i in range(len(nodes)):
        for j in range(i+1, len(nodes)):
            u, v = nodes[i], nodes[j]
            w = float(weights.loc[u, v])
            if use_similarity and w >= threshold_value:
                G.add_edge(u, v, weight=w)
            elif not use_similarity and w <= threshold_value:
                G.add_edge(u, v, weight=w)
    
    return G

def plot_elbow_curve(data, max_k=10):
    """Plot elbow curve for K-means"""
    inertias = []
    K_range = range(2, min(max_k + 1, len(data)))
    
    for k in K_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(data)
        inertias.append(kmeans.inertia_)
    
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(K_range, inertias, 'bo-')
    ax.set_xlabel('Number of Clusters (K)')
    ax.set_ylabel('Inertia')
    ax.set_title('Elbow Method for Optimal K')
    ax.grid(True, alpha=0.3)
    
    return fig

def plot_clustered_network(G, cluster_labels, resource_list, title):
    """Plot network with cluster coloring"""
    resource_cluster = dict(zip(resource_list, cluster_labels))
    
    fig, ax = plt.subplots(figsize=(10, 8))
    pos = nx.spring_layout(G, seed=42)
    
    node_colors = [resource_cluster.get(node, 0) for node in G.nodes()]
    
    nx.draw(G, pos, with_labels=False, node_color=node_colors, 
            cmap=plt.cm.jet, node_size=150, font_size=8, 
            font_color='black', ax=ax)
    
    ax.set_title(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    return fig

# Streamlit UI
st.title("🔍 Resource Clustering Analysis")
st.markdown("Analyze resource behavior patterns using activity profiles and handover matrices")
            
resource_df = pd.read_csv("outputs_task_similarity_nb/resource_activity_mean_per_case.csv")
resource_df = resource_df.set_index('resource')
handover_df = pd.read_csv('outputs_task_similarity_nb/handover_matrix.csv')
handover_df = handover_df.set_index('Unnamed: 0')
handover_matrix = handover_df

st.session_state.matrix = resource_df
st.session_state.handover_matrix = handover_matrix
st.session_state.file_loaded = True
            

# Main content - only show if file is loaded
if st.session_state.handover_matrix is not None and st.session_state.matrix is not None:
    matrix = st.session_state.matrix
    handover_matrix = st.session_state.handover_matrix
    
    st.sidebar.success(f"✅ {len(matrix)} resources, {len(matrix.columns)} activities")
    
    # Tabs for different analyses
    tab1, tab2, tab3 = st.tabs(["📊 Resource Activity Analysis", "🔄 Handover Matrix Analysis", "📈 Process Mining Analysis"])
    
    # Tab 1: Resource Activity Analysis
    with tab1:
        st.header("Resource Activity Clustering")
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.subheader("Network Settings")
            
            metric = st.selectbox(
                "Similarity Metric",
                ['cosine_distance', 'cosine_similarity', 'euclidean', 'minkowski', 'hamming', 'pearson'],
                help="Distance/similarity metric for comparing resources"
            )
            
            use_similarity = metric in ['cosine_similarity', 'pearson']
            
            if metric == 'minkowski':
                minkowski_p = st.slider("Minkowski p-parameter", 1, 5, 2)
            else:
                minkowski_p = 2
            
            threshold = st.slider(
                f"Edge Threshold ({'similarity' if use_similarity else 'distance'})",
                0.0, 20.0, 0.85 if metric == 'cosine_distance' else 0.6,
                0.01,
                help="Minimum weight for edges to appear in network"
            )
            
            st.subheader("Clustering Settings")
            
            clustering_method = st.radio("Clustering Method", ['K-Means', 'Hierarchical'])
            
            show_elbow = st.checkbox("Show Elbow Plot", value=False)
            
            if show_elbow:
                max_k = st.slider("Max K for Elbow", 3, 15, 10)
            
            n_clusters = st.slider("Number of Clusters", 2, min(10, len(matrix)), 5)
            
            if clustering_method == 'Hierarchical':
                linkage_method = st.selectbox("Linkage Method", 
                                             ['ward', 'complete', 'average', 'single'])
        
        with col2:
            # Compute weights and build network
            weights = compute_similarity_weights(matrix, metric, minkowski_p)
            G = build_network(weights, threshold, use_similarity)
            
            st.metric("Network Nodes", G.number_of_nodes())
            st.metric("Network Edges", G.number_of_edges())
            
            # Show elbow plot if requested
            if show_elbow:
                st.subheader("Elbow Method")
                elbow_fig = plot_elbow_curve(matrix.values, max_k)
                st.pyplot(elbow_fig)
        
        # Perform clustering
        st.subheader("Clustering Results")
        
        if clustering_method == 'K-Means':
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(matrix.values)
            method_title = f"K-Means Clustering (K={n_clusters})"
        else:
            linked = linkage(matrix.values, method=linkage_method)
            cluster_labels = fcluster(linked, n_clusters, criterion='maxclust')
            method_title = f"Hierarchical Clustering (K={n_clusters}, {linkage_method})"
        
        # Plot clustered network
        fig = plot_clustered_network(G, cluster_labels, matrix.index.tolist(), method_title)
        st.pyplot(fig)
        
        # Show cluster distribution
        cluster_dist = pd.Series(cluster_labels).value_counts().sort_index()
        st.subheader("Cluster Distribution")
        st.bar_chart(cluster_dist)
    
    # Tab 2: Handover Matrix Analysis
    with tab2:
        st.header("Handover Matrix Clustering")
        
        if handover_matrix is not None:
            col1, col2 = st.columns([1, 2])
            
            with col1:
                st.subheader("Network Settings")
                
                ho_threshold = st.slider(
                    "Handover Threshold",
                    0.0, 
                    float(handover_matrix.max().max()),
                    0.001,
                    0.0005,
                    help="Minimum handover frequency to show edge"
                )
                
                st.subheader("Clustering Settings")
                
                ho_clustering_method = st.radio(
                    "Clustering Method", 
                    ['K-Means', 'Hierarchical'],
                    key='ho_clustering'
                )
                
                ho_show_elbow = st.checkbox("Show Elbow Plot", value=False, key='ho_elbow')
                
                if ho_show_elbow:
                    ho_max_k = st.slider("Max K for Elbow", 3, 15, 10, key='ho_max_k')
                
                ho_n_clusters = st.slider(
                    "Number of Clusters", 
                    2, 
                    min(10, len(handover_matrix)), 
                    5,
                    key='ho_n_clusters'
                )
                
                if ho_clustering_method == 'Hierarchical':
                    ho_linkage_method = st.selectbox(
                        "Linkage Method",
                        ['ward', 'complete', 'average', 'single'],
                        key='ho_linkage'
                    )
            
            with col2:
                # Build handover network
                G_handover = handover_network_from_matrix(handover_matrix, threshold=ho_threshold)
                
                st.metric("Network Nodes", G_handover.number_of_nodes())
                st.metric("Network Edges", G_handover.number_of_edges())
                
                # Show elbow plot if requested
                if ho_show_elbow:
                    st.subheader("Elbow Method")
                    ho_elbow_fig = plot_elbow_curve(handover_matrix.values, ho_max_k)
                    st.pyplot(ho_elbow_fig)
            
            # Perform clustering
            st.subheader("Clustering Results")
            
            if ho_clustering_method == 'K-Means':
                ho_kmeans = KMeans(n_clusters=ho_n_clusters, random_state=42, n_init=10)
                ho_cluster_labels = ho_kmeans.fit_predict(handover_matrix.values)
                ho_method_title = f"K-Means Clustering (K={ho_n_clusters})"
            else:
                ho_linked = linkage(handover_matrix.values, method=ho_linkage_method)
                ho_cluster_labels = fcluster(ho_linked, ho_n_clusters, criterion='maxclust')
                ho_method_title = f"Hierarchical Clustering (K={ho_n_clusters}, {ho_linkage_method})"
            
            # Plot clustered network
            ho_fig = plot_clustered_network(
                G_handover, 
                ho_cluster_labels, 
                handover_matrix.index.tolist(),
                ho_method_title
            )
            st.pyplot(ho_fig)
            
            # Show cluster distribution
            ho_cluster_dist = pd.Series(ho_cluster_labels).value_counts().sort_index()
            st.subheader("Cluster Distribution")
            st.bar_chart(ho_cluster_dist)
        else:
            st.warning("⚠️ Handover matrix could not be computed. Timestamp data may be missing.")
    # Tab 3: Process Mining Analysis
    with tab3:
        st.header("Process Mining Evaluation")

        # Discover process model
        model_type = st.radio("Choose Discovery Algorithm", ["Inductive Miner", "Alpha Miner"])
        if model_type == "Inductive Miner":
            net, im, fm = pm4py.discover_petri_net_inductive(event_log)
        else:
             net, im, fm = pm4py.discover_petri_net_alpha(event_log)

        try:

            st.subheader("Discovered Petri Net")
            gviz = pn_visualizer.apply(net, im, fm)
            png_bytes = gviz.pipe(format='png')
            st.image(png_bytes, use_container_width=True)
        except Exception as e:
            st.error(f"Could not render Petri net: {e}")

        st.subheader("Parallel Activities Detection")
        tree = pm4py.discover_process_tree_inductive(event_log)
        parallel_pairs = extract_parallel_pairs(tree)
        st.write(f"Detected parallel pairs: {len(parallel_pairs)}")
        st.dataframe([list(p) for p in parallel_pairs])

        st.subheader("Handover Matrix from Log")
        log_df = pm4py.convert_to_dataframe(event_log)
        ho_df, total_counts = compute_handover_matrix(
            log_df,
            case_col="case:concept:name",
            activity_col="concept:name",
            resource_col="org:resource",
            timestamp_col="time:timestamp",
            parallel_pairs=parallel_pairs
        )
        st.dataframe(ho_df)

        threshold = st.slider("Network Threshold", 0.0, float(ho_df.values.max()), 0.001, 0.0001, format="%.4f")

        show_unconnected = st.checkbox("Show unconnected nodes", value=False)

        G = handover_network_from_matrix(ho_df, threshold=threshold, show_unconnected_nodes=show_unconnected)
        st.metric("Network Nodes", G.number_of_nodes())
        st.metric("Network Edges", G.number_of_edges())

        st.subheader("Network Visualization")
        fig, ax = plt.subplots(figsize=(8, 6))
        pos = nx.spring_layout(G, seed=42)
        nx.draw(G, pos, with_labels=True, node_size=200, font_size=8, ax=ax)
        st.pyplot(fig)

else:
    pass