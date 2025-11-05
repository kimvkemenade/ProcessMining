"""
Resource Similarity module for computing resource-activity profiles and similarity networks.
"""
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import pairwise_distances
import matplotlib.pyplot as plt
import networkx as nx


class ResourceSimilarity:
    """Computes resource similarity based on activity profiles."""
    
    def __init__(self, data):
        """
        Initialize resource similarity analyzer.
        
        Args:
            data (pd.DataFrame): Event log DataFrame
        """
        self.data = data
        self.matrix = None
        self.distance_matrix = None

    def create_activity_matrix(self):
        """
        Create resource-activity matrix showing mean activity counts per case.
        
        Returns:
            pd.DataFrame: Resource-activity matrix (resources x activities)
        """
        # Count activities per resource per case
        counts = (
            self.data.groupby(['resource', 'activity', 'case_id'])
            .size()
            .rename('count')
            .reset_index()
        )
        
        # Compute mean per case for each resource-activity pair
        mean_per_case = (
            counts.groupby(['resource', 'activity'])['count']
            .mean()
            .rename('mean_per_case')
            .reset_index()
        )
        
        # Pivot to matrix form
        matrix = mean_per_case.pivot(
            index='resource', 
            columns='activity', 
            values='mean_per_case'
        ).fillna(0.0)
        
        self.matrix = matrix
        print(f"Resource-activity matrix created: {matrix.shape[0]} resources × {matrix.shape[1]} activities")
        return matrix
    
    def compute_distance(self, metric='minkowski', minkowski_p=2):
        """
        Compute pairwise distances between resource profiles.
        
        Args:
            metric (str): Distance metric ('minkowski', 'hamming', 'pearson')
            minkowski_p (int): Parameter for Minkowski distance
            
        Returns:
            pd.DataFrame: Distance matrix (lower values = more similar)
        """
        if self.matrix is None:
            raise ValueError("Must create activity matrix first using create_activity_matrix()")
        
        if self.matrix.shape[0] == 0:
            raise ValueError('Empty resource-activity matrix.')

        metric_lower = metric.lower()

        if metric_lower == 'minkowski':
            distances = pairwise_distances(
                self.matrix.values, 
                metric='minkowski', 
                p=minkowski_p
            )
            distance_df = pd.DataFrame(
                distances, 
                index=self.matrix.index, 
                columns=self.matrix.index
            )
        elif metric_lower == 'hamming':
            distances = pairwise_distances(
                self.matrix.values, 
                metric='hamming'
            )
            distance_df = pd.DataFrame(
                distances, 
                index=self.matrix.index, 
                columns=self.matrix.index
            )
        elif metric_lower == 'pearson':
            # For Pearson, we want correlation (similarity), then convert to distance
            # Distance = 1 - correlation
            corr = np.corrcoef(self.matrix.values)
            distance_df = pd.DataFrame(
                1 - corr,  # Convert correlation to distance
                index=self.matrix.index, 
                columns=self.matrix.index
            ).fillna(0.0)
        else:
            raise ValueError(f'Unsupported metric: {metric}. Use "minkowski", "hamming", or "pearson"')

        self.distance_matrix = distance_df
        print(f"Distance matrix computed using {metric} metric")
        return distance_df
    
    def build_network(self, threshold=0.4):
        """
        Build similarity network from distance matrix.
        
        Args:
            threshold (float): Maximum distance threshold (edges with distance <= threshold are included)
            
        Returns:
            nx.Graph: Resource similarity network
        """
        if self.distance_matrix is None:
            raise ValueError("Must compute distance matrix first using compute_distance()")
        
        G = nx.Graph()
        
        # Add all resources as nodes
        for r in self.distance_matrix.index:
            G.add_node(r)

        # Add edges for similar resources
        nodes = list(self.distance_matrix.index)
        for i in range(len(nodes)):
            for j in range(i + 1, len(nodes)):
                u, v = nodes[i], nodes[j]
                distance = float(self.distance_matrix.loc[u, v])

                # Lower distance = more similar, so we keep edges with distance <= threshold
                if distance <= threshold:
                    G.add_edge(u, v, weight=distance)

        print(f"Resource similarity network: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
        return G
    
    def plot_network(self, G, output_path=None):
        """
        Visualize the resource similarity network.
        
        Args:
            G (nx.Graph): Network to plot
            output_path (str, optional): Path to save figure
        """
        plt.figure(figsize=(12, 10))
        pos = nx.spring_layout(G, seed=42)
        
        # Draw network
        nx.draw_networkx_nodes(G, pos, node_size=500, node_color='#4C78A8', alpha=0.9)
        nx.draw_networkx_edges(G, pos, width=1.5, alpha=0.5)
        nx.draw_networkx_labels(G, pos, font_size=10, font_weight='bold')
        
        plt.title('Resource Similarity Network\n(Based on Activity Profiles)', fontsize=14, fontweight='bold')
        plt.axis('off')
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"Network plot saved to {output_path}")
        
        plt.show()