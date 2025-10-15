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

# Helper functions
def load_xes_file(uploaded_file):
    """Load XES file with fallback mechanisms"""
    with open("temp.xes", "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    log = None
    for variant in (None, 'iterparse', 'etree'):
        try:
            if variant is None:
                log = pm4py.read_xes("temp.xes")
            else:
                log = pm4py.read_xes("temp.xes", variant=variant)
            break
        except Exception as e:
            continue
    
    os.remove("temp.xes")
    return log

def create_resource_activity_matrix(log):
    """Convert log to resource-activity matrix"""
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
    
    use_cols = ['case_id', 'activity', 'resource']
    if 'timestamp' in df.columns:
        use_cols.append('timestamp')
    
    df = df[use_cols].dropna(subset=['case_id','activity','resource']).copy()
    
    counts = df.groupby(['resource','activity','case_id']).size().rename('count').reset_index()
    mean_per_case = counts.groupby(['resource','activity'])['count'].mean().rename('mean_per_case').reset_index()
    matrix = mean_per_case.pivot(index='resource', columns='activity', values='mean_per_case').fillna(0.0)
    
    return matrix, df

def compute_handover_matrix(df):
    """Compute handover matrix from event log"""
    if 'timestamp' not in df.columns:
        return None
    
    df = df.sort_values(['case_id', 'timestamp'])
    
    handovers = []
    for case_id, group in df.groupby('case_id'):
        resources = group['resource'].tolist()
        for i in range(len(resources) - 1):
            handovers.append({'from': resources[i], 'to': resources[i+1], 'case_id': case_id})
    
    if not handovers:
        return None
    
    handover_df = pd.DataFrame(handovers)
    handover_counts = handover_df.groupby(['from', 'to', 'case_id']).size().reset_index(name='count')
    handover_mean = handover_counts.groupby(['from', 'to'])['count'].mean().reset_index(name='mean_per_case')
    
    all_resources = sorted(set(handover_mean['from'].unique()) | set(handover_mean['to'].unique()))
    handover_matrix = pd.DataFrame(0.0, index=all_resources, columns=all_resources)
    
    for _, row in handover_mean.iterrows():
        handover_matrix.at[row['from'], row['to']] = row['mean_per_case']
    
    return handover_matrix

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
    
    nx.draw(G, pos, with_labels=True, node_color=node_colors, 
            cmap=plt.cm.jet, node_size=300, font_size=8, 
            font_color='black', ax=ax)
    
    ax.set_title(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    return fig

# Streamlit UI
st.title("🔍 Resource Clustering Analysis")
st.markdown("Analyze resource behavior patterns using activity profiles and handover matrices")

# Sidebar configuration
st.sidebar.header("⚙️ Configuration")

uploaded_file = st.sidebar.file_uploader("Upload XES File", type=['xes'])

# Load button
if uploaded_file is not None:
    load_button = st.sidebar.button("🚀 Load and Process File", type="primary", use_container_width=True)
    
    if load_button:
        with st.spinner("Loading XES file..."):
            log = load_xes_file(uploaded_file)
            if log is None:
                st.error("Failed to load XES file")
                st.stop()
            
            matrix, df = create_resource_activity_matrix(log)
            handover_matrix = compute_handover_matrix(df)
            
            st.session_state.matrix = matrix
            st.session_state.df = df
            st.session_state.handover_matrix = handover_matrix
            st.session_state.file_loaded = True
            
        st.sidebar.success(f"✅ Loaded {len(matrix)} resources and {len(matrix.columns)} activities")
        st.rerun()

# Main content - only show if file is loaded
if st.session_state.file_loaded and st.session_state.matrix is not None:
    matrix = st.session_state.matrix
    df = st.session_state.df
    handover_matrix = st.session_state.handover_matrix
    
    st.sidebar.success(f"✅ {len(matrix)} resources, {len(matrix.columns)} activities")
    
    # Tabs for different analyses
    tab1, tab2 = st.tabs(["📊 Resource Activity Analysis", "🔄 Handover Matrix Analysis"])
    
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
                0.0, 1.0, 0.85 if metric == 'cosine_distance' else 0.6,
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
                    0.001,
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

else:
    st.info("👈 Please upload an XES file and click 'Load and Process File' to begin analysis")
    
    st.markdown("""
    ### Features:
    - **Resource Activity Analysis**: Cluster resources based on activity patterns
    - **Handover Matrix Analysis**: Cluster resources based on work handover patterns
    - **Multiple Metrics**: Cosine, Euclidean, Minkowski, Hamming, Pearson
    - **Clustering Methods**: K-Means and Hierarchical clustering
    - **Elbow Method**: Find optimal number of clusters
    - **Interactive Visualization**: Adjust thresholds and parameters in real-time
    """)