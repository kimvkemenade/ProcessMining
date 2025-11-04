"""
Clustering module for applying clustering algorithms to social networks.
"""
import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
from sklearn.cluster import KMeans, AgglomerativeClustering
import pandas as pd


class Clustering:
    """Applies clustering algorithms to network data."""
    
    def __init__(self, resource_matrix, resource_graph, handover_matrix, handover_graph):
        """
        Initialize clustering analyzer.
        
        Args:
            resource_matrix (pd.DataFrame): Resource similarity/distance matrix
            resource_graph (nx.Graph): Resource similarity network
            handover_matrix (pd.DataFrame): Handover matrix
            handover_graph (nx.Graph): Handover network
        """
        self.resource_matrix = resource_matrix
        self.resource_graph = resource_graph
        self.handover_matrix = handover_matrix
        self.handover_graph = handover_graph

    def elbow_method(self, data_type='handover', k_range=range(2, 11)):
        """
        Compute and plot elbow method to help determine optimal number of clusters.
        
        Args:
            data_type (str): 'handover' or 'resources'
            k_range (range): Range of k values to test
            
        Returns:
            list: Inertia values for each k
        """
        data = self.handover_matrix if data_type == 'handover' else self.resource_matrix
        
        inertias = []
        k_values = list(k_range)
        
        for k in k_values:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=15)
            kmeans.fit(data.values)
            inertias.append(kmeans.inertia_)
        
        # Plot elbow curve
        plt.figure(figsize=(10, 6))
        plt.plot(k_values, inertias, 'bo-', linewidth=2, markersize=8)
        plt.xlabel('Number of Clusters (k)', fontsize=12)
        plt.ylabel('Inertia (Within-cluster sum of squares)', fontsize=12)
        plt.title(f'Elbow Method for {data_type.capitalize()} Data', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
        
        print(f"Elbow method completed for {data_type} data")
        return inertias

    def cluster_kmeans(self, n_clusters=4, data_type='handover'):
        """
        Perform K-means clustering.
        
        Args:
            n_clusters (int): Number of clusters
            data_type (str): 'handover' or 'resources'
            
        Returns:
            np.ndarray: Cluster labels for each node
        """
        data = self.handover_matrix if data_type == 'handover' else self.resource_matrix
        
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        clusters = kmeans.fit_predict(data.values)
        
        # Print cluster distribution
        unique, counts = np.unique(clusters, return_counts=True)
        print(f"\n✓ K-means clustering (k={n_clusters}) on {data_type} data:")
        for cluster_id, count in zip(unique, counts):
            print(f"  Cluster {cluster_id}: {count} resources")
        
        return clusters

    def cluster_hierarchical(self, n_clusters=4, data_type='handover', linkage='ward'):
        """
        Perform hierarchical clustering.
        
        Args:
            n_clusters (int): Number of clusters
            data_type (str): 'handover' or 'resources'
            linkage (str): Linkage method ('ward', 'average', 'complete', 'single')
            
        Returns:
            np.ndarray: Cluster labels for each node
        """
        data = self.handover_matrix if data_type == 'handover' else self.resource_matrix
        
        hierarchical = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage)
        clusters = hierarchical.fit_predict(data.values)
        
        # Print cluster distribution
        unique, counts = np.unique(clusters, return_counts=True)
        print(f"\n✓ Hierarchical clustering (k={n_clusters}, linkage={linkage}) on {data_type} data:")
        for cluster_id, count in zip(unique, counts):
            print(f"  Cluster {cluster_id}: {count} resources")
        
        return clusters

    def plot_clustered_network(self, clusters, data_type='handover', 
                               threshold=0.0001, directed=True, 
                               show_unconnected=False, output_path=None):
        """
        Visualize network with nodes colored by cluster assignment.
        
        Args:
            clusters (np.ndarray): Cluster labels for each node
            data_type (str): 'handover' or 'resources'
            threshold (float): Minimum edge weight to include
            directed (bool): Whether to create directed graph
            show_unconnected (bool): Whether to show isolated nodes
            output_path (str, optional): Path to save figure
            
        Returns:
            nx.Graph: Clustered network
        """
        if data_type == 'handover':
            matrix = self.handover_matrix
            original_graph = self.handover_graph
        else:
            matrix = self.resource_matrix
            original_graph = self.resource_graph
        
        # Create node to cluster mapping
        node_to_cluster = dict(zip(matrix.index, clusters))
        n_clusters = len(np.unique(clusters))
        
        # Create color map
        cmap = plt.cm.get_cmap('tab10')
        colors = [cmap(i / max(n_clusters - 1, 1)) for i in range(n_clusters)]
        
        # Rebuild graph with threshold
        G = nx.DiGraph() if directed else nx.Graph()
        
        if directed:
            for src in matrix.index:
                for dst in matrix.columns:
                    w = matrix.at[src, dst]
                    if w >= threshold and w > 0:
                        G.add_edge(src, dst, weight=float(w))
            if show_unconnected:
                for node in matrix.index:
                    G.add_node(node)
        else:
            visited = set()
            for a in matrix.index:
                for b in matrix.columns:
                    if (a, b) in visited or (b, a) in visited:
                        continue
                    w = (matrix.at[a, b] + matrix.at[b, a]) / 2.0
                    if w >= threshold and w > 0:
                        G.add_edge(a, b, weight=float(w))
                    visited.add((a, b))
            if show_unconnected:
                for node in matrix.index:
                    G.add_node(node)
        
        # Create node colors based on cluster
        node_colors = [colors[node_to_cluster[node]] for node in G.nodes()]
        
        # Plot
        plt.figure(figsize=(14, 10))
        pos = nx.spring_layout(G, seed=42, k=0.5)
        edge_weights = [G[u][v]['weight'] for u, v in G.edges()]
        
        # Draw nodes colored by cluster
        nx.draw_networkx_nodes(G, pos, node_size=600, node_color=node_colors, 
                              alpha=0.9, edgecolors='black', linewidths=1.5)
        
        # Draw edges
        if data_type == 'handover':
            nx.draw_networkx_edges(G, pos, width=[w * 100 for w in edge_weights], 
                                  alpha=0.5, arrowsize=15, arrowstyle='->')
        else:
            nx.draw_networkx_edges(G, pos, width=[max(0.5, w * 5) for w in edge_weights], 
                                  alpha=0.5)
        
        # Draw labels
        nx.draw_networkx_labels(G, pos, font_size=9, font_weight='bold')
        
        # Create legend
        legend_elements = [plt.Line2D([0], [0], marker='o', color='w', 
                                     markerfacecolor=colors[i], markersize=10, 
                                     label=f'Cluster {i}')
                          for i in range(n_clusters)]
        plt.legend(handles=legend_elements, loc='upper right', fontsize=10)
        
        plt.title(f'Clustered {data_type.capitalize()} Network (k={n_clusters})', 
                 fontsize=14, fontweight='bold')
        plt.axis('off')
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"✓ Clustered network plot saved to {output_path}")
        
        plt.show()
        
        return G
    
    def get_cluster_assignments(self, clusters, data_type='handover'):
        """
        Get a DataFrame showing which resources belong to which cluster.
        
        Args:
            clusters (np.ndarray): Cluster labels
            data_type (str): 'handover' or 'resources'
            
        Returns:
            pd.DataFrame: Resource-cluster assignments
        """
        matrix = self.handover_matrix if data_type == 'handover' else self.resource_matrix
        
        df = pd.DataFrame({
            'resource': matrix.index,
            'cluster': clusters
        })
        
        return df.sort_values('cluster')