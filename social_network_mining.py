"""
Social Network Mining for Process Mining
=========================================
This module implements organizational mining approaches for deriving and analyzing
social networks from event logs.

Main functionalities:
1. Resource-Activity Similarity Network
2. Handover-of-Work Network
3. Clustering Analysis (K-Means and Hierarchical)
"""

import os
import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from typing import Tuple, Dict, Set, Optional, List
from collections import defaultdict

import pm4py
from pm4py.objects.conversion.log import converter as log_converter
from pm4py.algo.discovery.inductive import algorithm as inductive_miner
from pm4py.objects.conversion.process_tree import converter as tree_to_petri_converter
from pm4py.objects.process_tree.obj import Operator

from sklearn.metrics.pairwise import (
    cosine_similarity,
    euclidean_distances,
    pairwise_distances
)
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster


class EventLogProcessor:
    """Handles event log import and preprocessing."""
    
    def __init__(self, xes_path: str):
        """
        Initialize EventLogProcessor.
        
        Parameters
        ----------
        xes_path : str
            Path to the XES event log file
        """
        self.xes_path = xes_path
        self.log = None
        self.df = None
        
    def import_log(self) -> object:
        """
        Import event log from XES file.
        
        Returns
        -------
        log : pm4py EventLog
            Imported event log
        """
        self.log = pm4py.read_xes(self.xes_path)
        print(f"✓ Log imported with {len(self.log)} traces")
        return self.log
    
    def convert_to_dataframe(self) -> pd.DataFrame:
        """
        Convert event log to pandas DataFrame with standardized column names.
        
        Returns
        -------
        df : pd.DataFrame
            DataFrame with columns: case_id, activity, resource, timestamp
        """
        if self.log is None:
            raise ValueError("Log not imported. Call import_log() first.")
        
        # Convert to DataFrame
        self.df = log_converter.apply(
            self.log,
            variant=log_converter.Variants.TO_DATA_FRAME
        )
        
        # Standardize column names
        col_map = {
            'case:concept:name': 'case_id',
            'concept:name': 'activity',
            'org:resource': 'resource',
            'time:timestamp': 'timestamp'
        }
        for old, new in col_map.items():
            if old in self.df.columns:
                self.df = self.df.rename(columns={old: new})
        
        # Validate required columns
        required_cols = ['case_id', 'activity', 'resource']
        missing = [c for c in required_cols if c not in self.df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")
        
        # Select and clean data
        use_cols = ['case_id', 'activity', 'resource', 'timestamp'] \
                   if 'timestamp' in self.df.columns else \
                   ['case_id', 'activity', 'resource']
        
        self.df = self.df[use_cols].dropna(
            subset=['case_id', 'activity', 'resource']
        ).copy()
        
        print(f"✓ DataFrame created with {len(self.df)} events")
        return self.df


class ResourceActivityNetwork:
    """Generates social networks based on resource-activity similarities."""
    
    def __init__(self, df: pd.DataFrame):
        """
        Initialize ResourceActivityNetwork.
        
        Parameters
        ----------
        df : pd.DataFrame
            Event log DataFrame with columns: case_id, activity, resource
        """
        self.df = df
        self.resource_activity_matrix = None
        self.distance_matrix = None
        self.network = None
        
    def create_resource_activity_matrix(self) -> pd.DataFrame:
        """
        Create resource-activity matrix showing mean activity counts per case.
        
        Returns
        -------
        matrix : pd.DataFrame
            Resource-activity matrix (rows=resources, columns=activities)
        """
        # Count activities per resource per case
        counts = (
            self.df.groupby(['resource', 'activity', 'case_id'])
            .size()
            .rename('count')
            .reset_index()
        )
        
        # Calculate mean per case
        mean_per_case = (
            counts.groupby(['resource', 'activity'])['count']
            .mean()
            .rename('mean_per_case')
            .reset_index()
        )
        
        # Pivot to matrix form
        self.resource_activity_matrix = mean_per_case.pivot(
            index='resource',
            columns='activity',
            values='mean_per_case'
        ).fillna(0.0)
        
        print(f"✓ Resource-activity matrix created: "
              f"{self.resource_activity_matrix.shape}")
        return self.resource_activity_matrix
    
    def compute_distance_matrix(
        self,
        metric: str = 'minkowski',
        minkowski_p: int = 2
    ) -> pd.DataFrame:
        """
        Compute pairwise distance/similarity matrix between resources.
        
        Parameters
        ----------
        metric : str
            Distance metric: 'minkowski', 'euclidean', 'hamming',
            'cosine', 'cosine_distance', 'cosine_similarity', 'pearson'
        minkowski_p : int
            Parameter for Minkowski distance (default: 2 for Euclidean)
        
        Returns
        -------
        distance_matrix : pd.DataFrame
            Pairwise distance/similarity matrix
        """
        if self.resource_activity_matrix is None:
            raise ValueError("Resource-activity matrix not created.")
        
        if self.resource_activity_matrix.shape[0] == 0:
            raise ValueError("Empty resource-activity matrix.")
        
        matrix = self.resource_activity_matrix.values
        index = self.resource_activity_matrix.index
        
        metric_lower = metric.lower()
        
        if metric_lower == 'cosine_similarity':
            weights = cosine_similarity(matrix)
        elif metric_lower in ('cosine_distance', 'cosine'):
            weights = 1.0 - cosine_similarity(matrix)
        elif metric_lower == 'euclidean':
            weights = euclidean_distances(matrix)
        elif metric_lower == 'minkowski':
            weights = pairwise_distances(
                matrix,
                metric='minkowski',
                p=minkowski_p
            )
        elif metric_lower == 'hamming':
            weights = pairwise_distances(matrix, metric='hamming')
        elif metric_lower == 'pearson':
            corr = np.corrcoef(matrix)
            weights = corr
            weights = np.nan_to_num(weights, nan=0.0)
        else:
            raise ValueError(f"Unsupported metric: {metric}")
        
        self.distance_matrix = pd.DataFrame(
            weights,
            index=index,
            columns=index
        )
        
        print(f"✓ Distance matrix computed using {metric}")
        return self.distance_matrix
    
    def build_network(
        self,
        measure_preference: str = 'similarity',
        threshold: float = 0.6
    ) -> nx.Graph:
        """
        Build social network from distance/similarity matrix.
        
        Parameters
        ----------
        measure_preference : str
            'similarity' or 'distance' - determines edge filtering
        threshold : float
            Threshold value for filtering weak connections
        
        Returns
        -------
        G : nx.Graph
            NetworkX graph representing the social network
        """
        if self.distance_matrix is None:
            raise ValueError("Distance matrix not computed.")
        
        use_similarity = (measure_preference.lower() == 'similarity')
        
        # Initialize graph
        G = nx.Graph()
        for resource in self.distance_matrix.index:
            G.add_node(resource)
        
        # Add edges based on threshold
        nodes = list(self.distance_matrix.index)
        for i in range(len(nodes)):
            for j in range(i + 1, len(nodes)):
                u, v = nodes[i], nodes[j]
                weight = float(self.distance_matrix.loc[u, v])
                
                # Add edge if it meets threshold criteria
                if use_similarity:
                    if weight >= threshold:
                        G.add_edge(u, v, weight=weight)
                else:
                    if weight <= threshold:
                        G.add_edge(u, v, weight=weight)
        
        self.network = G
        print(f"✓ Network built: {G.number_of_nodes()} nodes, "
              f"{G.number_of_edges()} edges")
        return G
    
    def visualize_network(
        self,
        figsize: Tuple[int, int] = (12, 10),
        save_path: Optional[str] = None
    ):
        """
        Visualize the social network.
        
        Parameters
        ----------
        figsize : tuple
            Figure size (width, height)
        save_path : str, optional
            Path to save the figure
        """
        if self.network is None:
            raise ValueError("Network not built.")
        
        plt.figure(figsize=figsize)
        pos = nx.spring_layout(self.network, seed=42, k=0.5)
        
        # Draw nodes
        nx.draw_networkx_nodes(
            self.network,
            pos,
            node_size=500,
            node_color='#4C78A8',
            alpha=0.9
        )
        
        # Draw edges
        edges = self.network.edges()
        weights = [self.network[u][v].get('weight', 1.0) for u, v in edges]
        
        nx.draw_networkx_edges(
            self.network,
            pos,
            width=[w * 2 for w in weights],
            alpha=0.5,
            edge_color='gray'
        )
        
        # Draw labels
        nx.draw_networkx_labels(
            self.network,
            pos,
            font_size=9,
            font_weight='bold'
        )
        
        plt.title('Resource Similarity Network', fontsize=14, fontweight='bold')
        plt.axis('off')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Network visualization saved to {save_path}")
        
        plt.show()


class HandoverNetwork:
    """Generates handover-of-work social networks from event logs."""
    
    def __init__(self, log: object):
        """
        Initialize HandoverNetwork.
        
        Parameters
        ----------
        log : pm4py EventLog
            Process mining event log
        """
        self.log = log
        self.process_tree = None
        self.concurrent_pairs = None
        self.handover_matrix = None
        self.network = None
        
    def discover_process_model(self):
        """
        Discover process tree using Inductive Miner.
        
        Returns
        -------
        process_tree : pm4py ProcessTree
            Discovered process tree
        """
        self.process_tree = inductive_miner.apply(
            self.log,
            variant=inductive_miner.Variants.IMf
        )
        print("✓ Process model discovered")
        return self.process_tree
    
    def extract_concurrent_activities(self) -> Set[Tuple[str, str]]:
        """
        Extract concurrent activity pairs from process tree.
        
        Returns
        -------
        concurrent_pairs : set
            Set of tuples representing concurrent activity pairs
        """
        if self.process_tree is None:
            raise ValueError("Process model not discovered.")
        
        def get_leaves(node):
            """Extract leaf activities from a process tree node."""
            leaves = []
            if node is None:
                return leaves
            if hasattr(node, 'label') and node.label is not None and node.label != '':
                leaves.append(node.label)
            if hasattr(node, 'children') and node.children:
                for child in node.children:
                    leaves.extend(get_leaves(child))
            return leaves
        
        def traverse(node):
            """Traverse tree to find concurrent operators."""
            if node is None:
                return
            
            # Check for parallel operator
            if hasattr(node, 'operator') and node.operator == Operator.PARALLEL:
                if hasattr(node, 'children') and node.children:
                    # Get activities from each branch
                    branches = [get_leaves(child) for child in node.children]
                    
                    # Create pairs from different branches
                    for i in range(len(branches)):
                        for j in range(i + 1, len(branches)):
                            for act_i in branches[i]:
                                for act_j in branches[j]:
                                    if act_i and act_j:
                                        pairs.add((act_i, act_j))
                                        pairs.add((act_j, act_i))
            
            # Recursively traverse children
            if hasattr(node, 'children') and node.children:
                for child in node.children:
                    traverse(child)
        
        pairs = set()
        traverse(self.process_tree)
        self.concurrent_pairs = pairs
        
        print(f"✓ Detected {len(self.concurrent_pairs)} concurrent activity pairs")
        return self.concurrent_pairs
    
    def compute_handover_matrix(self) -> pd.DataFrame:
        """
        Compute handover-of-work matrix.
        
        Returns
        -------
        handover_matrix : pd.DataFrame
            Matrix showing mean handovers between resources
        """
        if self.concurrent_pairs is None:
            raise ValueError("Concurrent activities not extracted.")
        
        # Count handovers
        counts = defaultdict(int)
        case_count = len(self.log)
        
        for trace in self.log:
            # Sort events by timestamp
            events = sorted(trace, key=lambda e: e['time:timestamp'])
            
            # Check consecutive events
            for i in range(len(events) - 1):
                activity1 = events[i].get('concept:name')
                resource1 = events[i].get('org:resource')
                activity2 = events[i + 1].get('concept:name')
                resource2 = events[i + 1].get('org:resource')
                
                # Skip if same resource or missing data
                if not resource1 or not resource2 or resource1 == resource2:
                    continue
                
                # Skip if activities are concurrent
                if (activity1, activity2) in self.concurrent_pairs:
                    continue
                
                counts[(resource1, resource2)] += 1
        
        # Get all unique resources
        all_resources = set()
        for (r1, r2) in counts.keys():
            all_resources.add(r1)
            all_resources.add(r2)
        resources = sorted(list(all_resources))
        
        # Create matrix
        n_resources = len(resources)
        resource_to_idx = {r: i for i, r in enumerate(resources)}
        matrix = np.zeros((n_resources, n_resources))
        
        # Fill matrix with mean handovers per case
        for (r1, r2), count in counts.items():
            i, j = resource_to_idx[r1], resource_to_idx[r2]
            matrix[i, j] = count / case_count
        
        self.handover_matrix = pd.DataFrame(
            matrix,
            index=resources,
            columns=resources
        )
        
        print(f"✓ Handover matrix created: {self.handover_matrix.shape}")
        print(f"✓ Total unique handover pairs: {len(counts)}")
        return self.handover_matrix
    
    def build_network(
        self,
        threshold: float = 0.0,
        directed: bool = True
    ) -> nx.Graph:
        """
        Build handover network from handover matrix.
        
        Parameters
        ----------
        threshold : float
            Minimum handover weight to include edge
        directed : bool
            Whether to create directed or undirected graph
        
        Returns
        -------
        G : nx.DiGraph or nx.Graph
            Handover network
        """
        if self.handover_matrix is None:
            raise ValueError("Handover matrix not computed.")
        
        G = nx.DiGraph() if directed else nx.Graph()
        
        if directed:
            # Add directed edges
            for src in self.handover_matrix.index:
                for dst in self.handover_matrix.columns:
                    weight = self.handover_matrix.at[src, dst]
                    if weight >= threshold and weight > 0:
                        G.add_edge(src, dst, weight=float(weight))
        else:
            # Add undirected edges (average both directions)
            visited = set()
            for res1 in self.handover_matrix.index:
                for res2 in self.handover_matrix.columns:
                    if (res1, res2) in visited or (res2, res1) in visited:
                        continue
                    
                    weight = (
                        self.handover_matrix.at[res1, res2] +
                        self.handover_matrix.at[res2, res1]
                    ) / 2.0
                    
                    if weight >= threshold and weight > 0:
                        G.add_edge(res1, res2, weight=float(weight))
                    
                    visited.add((res1, res2))
        
        self.network = G
        print(f"✓ Handover network built: {G.number_of_nodes()} nodes, "
              f"{G.number_of_edges()} edges")
        return G
    
    def visualize_network(
        self,
        figsize: Tuple[int, int] = (12, 10),
        save_path: Optional[str] = None
    ):
        """
        Visualize the handover network.
        
        Parameters
        ----------
        figsize : tuple
            Figure size (width, height)
        save_path : str, optional
            Path to save the figure
        """
        if self.network is None:
            raise ValueError("Network not built.")
        
        plt.figure(figsize=figsize)
        pos = nx.spring_layout(self.network, seed=42, k=0.5)
        
        # Draw nodes
        nx.draw_networkx_nodes(
            self.network,
            pos,
            node_size=500,
            node_color='#E8743B',
            alpha=0.9
        )
        
        # Draw edges with weights
        edges = self.network.edges()
        weights = [self.network[u][v].get('weight', 1.0) for u, v in edges]
        max_weight = max(weights) if weights else 1.0
        normalized_weights = [w / max_weight * 3 for w in weights]
        
        nx.draw_networkx_edges(
            self.network,
            pos,
            width=normalized_weights,
            alpha=0.6,
            edge_color='gray',
            arrows=isinstance(self.network, nx.DiGraph),
            arrowsize=15,
            arrowstyle='->'
        )
        
        # Draw labels
        nx.draw_networkx_labels(
            self.network,
            pos,
            font_size=9,
            font_weight='bold'
        )
        
        plt.title('Handover-of-Work Network', fontsize=14, fontweight='bold')
        plt.axis('off')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Network visualization saved to {save_path}")
        
        plt.show()


class NetworkClusterer:
    """Applies clustering algorithms to social networks."""
    
    def __init__(
        self,
        network: nx.Graph = None,
        feature_matrix: pd.DataFrame = None
    ):
        """
        Initialize NetworkClusterer.
        
        Parameters
        ----------
        network : nx.Graph, optional
            NetworkX graph to cluster
        feature_matrix : pd.DataFrame, optional
            Feature matrix for clustering (alternative to network)
        """
        self.network = network
        self.feature_matrix = feature_matrix
        self.cluster_labels = None
        self.n_clusters = None
        
    def kmeans_clustering(
        self,
        n_clusters: int = None,
        use_elbow: bool = False,
        max_k: int = 10,
        random_state: int = 42
    ) -> Dict[str, int]:
        """
        Apply K-Means clustering.
        
        Parameters
        ----------
        n_clusters : int, optional
            Number of clusters (required if use_elbow=False)
        use_elbow : bool
            Whether to use elbow method to determine optimal k
        max_k : int
            Maximum k to test for elbow method
        random_state : int
            Random state for reproducibility
        
        Returns
        -------
        cluster_assignments : dict
            Mapping from node/resource to cluster label
        """
        if self.feature_matrix is None:
            raise ValueError("Feature matrix not provided.")
        
        # Standardize features
        scaler = StandardScaler()
        X = scaler.fit_transform(self.feature_matrix.values)
        
        if use_elbow:
            # Use elbow method to find optimal k
            inertias = []
            silhouettes = []
            k_range = range(2, min(max_k + 1, len(X)))
            
            print("\nTesting K-Means with different k values...")
            for k in k_range:
                kmeans = KMeans(n_clusters=k, random_state=random_state, n_init=10)
                labels = kmeans.fit_predict(X)
                inertias.append(kmeans.inertia_)
                
                if k < len(X):
                    sil_score = silhouette_score(X, labels)
                    silhouettes.append(sil_score)
                else:
                    silhouettes.append(0)
                
                print(f"  k={k}: inertia={kmeans.inertia_:.2f}, "
                      f"silhouette={silhouettes[-1]:.3f}")
            
            # Plot elbow curve
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
            
            ax1.plot(k_range, inertias, 'bo-')
            ax1.set_xlabel('Number of Clusters (k)', fontsize=12)
            ax1.set_ylabel('Inertia', fontsize=12)
            ax1.set_title('Elbow Method', fontsize=14, fontweight='bold')
            ax1.grid(True, alpha=0.3)
            
            ax2.plot(k_range, silhouettes, 'ro-')
            ax2.set_xlabel('Number of Clusters (k)', fontsize=12)
            ax2.set_ylabel('Silhouette Score', fontsize=12)
            ax2.set_title('Silhouette Analysis', fontsize=14, fontweight='bold')
            ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.show()
            
            # Select k with best silhouette score
            best_k = k_range[np.argmax(silhouettes)]
            print(f"\n✓ Optimal k selected: {best_k}")
            n_clusters = best_k
        
        if n_clusters is None:
            raise ValueError("n_clusters must be specified or use_elbow=True")
        
        # Perform final clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
        labels = kmeans.fit_predict(X)
        
        self.cluster_labels = labels
        self.n_clusters = n_clusters
        
        # Create assignment dictionary
        cluster_assignments = dict(zip(self.feature_matrix.index, labels))
        
        print(f"✓ K-Means clustering completed with k={n_clusters}")
        return cluster_assignments
    
    def hierarchical_clustering(
        self,
        n_clusters: int = 5,
        linkage_method: str = 'ward',
        plot_dendrogram: bool = True
    ) -> Dict[str, int]:
        """
        Apply hierarchical clustering.
        
        Parameters
        ----------
        n_clusters : int
            Number of clusters
        linkage_method : str
            Linkage method: 'ward', 'complete', 'average', 'single'
        plot_dendrogram : bool
            Whether to plot dendrogram
        
        Returns
        -------
        cluster_assignments : dict
            Mapping from node/resource to cluster label
        """
        if self.feature_matrix is None:
            raise ValueError("Feature matrix not provided.")
        
        X = self.feature_matrix.values
        
        # Compute linkage
        Z = linkage(X, method=linkage_method)
        
        # Plot dendrogram
        if plot_dendrogram:
            plt.figure(figsize=(14, 7))
            dendrogram(
                Z,
                labels=self.feature_matrix.index.tolist(),
                leaf_font_size=9
            )
            plt.title(
                f'Hierarchical Clustering Dendrogram ({linkage_method})',
                fontsize=14,
                fontweight='bold'
            )
            plt.xlabel('Resource', fontsize=12)
            plt.ylabel('Distance', fontsize=12)
            plt.tight_layout()
            plt.show()
        
        # Cut tree to get clusters
        labels = fcluster(Z, n_clusters, criterion='maxclust') - 1
        
        self.cluster_labels = labels
        self.n_clusters = n_clusters
        
        # Create assignment dictionary
        cluster_assignments = dict(zip(self.feature_matrix.index, labels))
        
        print(f"✓ Hierarchical clustering completed with {n_clusters} clusters")
        return cluster_assignments
    
    def visualize_clustered_network(
        self,
        cluster_assignments: Dict[str, int],
        figsize: Tuple[int, int] = (12, 10),
        save_path: Optional[str] = None,
        title: str = 'Clustered Network'
    ):
        """
        Visualize network with cluster colors.
        
        Parameters
        ----------
        cluster_assignments : dict
            Mapping from node to cluster label
        figsize : tuple
            Figure size
        save_path : str, optional
            Path to save figure
        title : str
            Plot title
        """
        if self.network is None:
            raise ValueError("Network not provided.")
        
        plt.figure(figsize=figsize)
        pos = nx.spring_layout(self.network, seed=42, k=0.5)
        
        # Get colors for each node
        node_colors = [cluster_assignments.get(node, 0) for node in self.network.nodes()]
        
        # Draw network
        nx.draw_networkx_nodes(
            self.network,
            pos,
            node_size=500,
            node_color=node_colors,
            cmap=plt.cm.tab10,
            alpha=0.9
        )
        
        nx.draw_networkx_edges(
            self.network,
            pos,
            alpha=0.3,
            edge_color='gray'
        )
        
        nx.draw_networkx_labels(
            self.network,
            pos,
            font_size=9,
            font_weight='bold'
        )
        
        plt.title(title, fontsize=14, fontweight='bold')
        plt.axis('off')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Clustered network visualization saved to {save_path}")
        
        plt.show()


class SocialNetworkMiner:
    """
    Main class orchestrating the social network mining pipeline.
    """
    
    def __init__(self, xes_path: str, output_dir: str = 'outputs'):
        """
        Initialize SocialNetworkMiner.
        
        Parameters
        ----------
        xes_path : str
            Path to XES event log
        output_dir : str
            Directory for saving outputs
        """
        self.xes_path = xes_path
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        self.log_processor = None
        self.log = None
        self.df = None
        
    def run_full_pipeline(
        self,
        similarity_metric: str = 'cosine_distance',
        similarity_threshold: float = 0.85,
        handover_threshold: float = 0.0,
        kmeans_k: int = None,
        hierarchical_k: int = 5,
        use_elbow: bool = True
    ):
        """
        Run complete social network mining pipeline.
        
        Parameters
        ----------
        similarity_metric : str
            Distance metric for similarity network
        similarity_threshold : float
            Threshold for similarity network edges
        handover_threshold : float
            Threshold for handover network edges
        kmeans_k : int, optional
            Number of clusters for K-Means (None = use elbow)
        hierarchical_k : int
            Number of clusters for hierarchical clustering
        use_elbow : bool
            Whether to use elbow method for K-Means
        """
        print("=" * 80)
        print("SOCIAL NETWORK MINING PIPELINE")
        print("=" * 80)
        
        # 1. Load and preprocess data
        print("\n[1] Loading and preprocessing event log...")
        self.log_processor = EventLogProcessor(self.xes_path)
        self.log = self.log_processor.import_log()
        self.df = self.log_processor.convert_to_dataframe()
        
        # 2. Resource-Activity Similarity Network
        print("\n[2] Building Resource-Activity Similarity Network...")
        ra_network = ResourceActivityNetwork(self.df)
        ra_matrix = ra_network.create_resource_activity_matrix()
        
        # Save matrix
        ra_matrix.to_csv(
            os.path.join(self.output_dir, 'resource_activity_matrix.csv')
        )
        
        ra_network.compute_distance_matrix(metric=similarity_metric)
        ra_graph = ra_network.build_network(
            measure_preference='distance',
            threshold=similarity_threshold
        )
        ra_network.visualize_network(
            save_path=os.path.join(self.output_dir, 'similarity_network.png')
        )
        
        # 3. Clustering on Similarity Network
        print("\n[3] Clustering Resource-Activity Network...")
        ra_clusterer = NetworkClusterer(
            network=ra_graph,
            feature_matrix=ra_matrix
        )
        
        # K-Means
        print("\n[3a] K-Means Clustering...")
        kmeans_clusters = ra_clusterer.kmeans_clustering(
            n_clusters=kmeans_k,
            use_elbow=use_elbow,
            max_k=15
        )
        ra_clusterer.visualize_clustered_network(
            cluster_assignments=kmeans_clusters,
            title='K-Means Clustering - Similarity Network',
            save_path=os.path.join(self.output_dir, 'similarity_kmeans.png')
        )
        
        # Hierarchical
        print("\n[3b] Hierarchical Clustering...")
        hierarchical_clusters = ra_clusterer.hierarchical_clustering(
            n_clusters=hierarchical_k,
            plot_dendrogram=True
        )
        ra_clusterer.visualize_clustered_network(
            cluster_assignments=hierarchical_clusters,
            title='Hierarchical Clustering - Similarity Network',
            save_path=os.path.join(self.output_dir, 'similarity_hierarchical.png')
        )
        
        # 4. Handover Network
        print("\n[4] Building Handover-of-Work Network...")
        ho_network = HandoverNetwork(self.log)
        ho_network.discover_process_model()
        ho_network.extract_concurrent_activities()
        ho_matrix = ho_network.compute_handover_matrix()
        
        # Save matrix
        ho_matrix.to_csv(
            os.path.join(self.output_dir, 'handover_matrix.csv')
        )
        
        ho_graph = ho_network.build_network(
            threshold=handover_threshold,
            directed=True
        )
        ho_network.visualize_network(
            save_path=os.path.join(self.output_dir, 'handover_network.png')
        )
        
        # 5. Clustering on Handover Network
        print("\n[5] Clustering Handover Network...")
        
        # Prepare feature matrix (outgoing + incoming handovers)
        ho_features = pd.DataFrame(
            np.hstack([ho_matrix.values, ho_matrix.values.T]),
            index=ho_matrix.index
        )
        
        ho_clusterer = NetworkClusterer(
            network=ho_graph,
            feature_matrix=ho_features
        )
        
        # K-Means
        print("\n[5a] K-Means Clustering...")
        ho_kmeans = ho_clusterer.kmeans_clustering(
            n_clusters=kmeans_k,
            use_elbow=use_elbow,
            max_k=15
        )
        ho_clusterer.visualize_clustered_network(
            cluster_assignments=ho_kmeans,
            title='K-Means Clustering - Handover Network',
            save_path=os.path.join(self.output_dir, 'handover_kmeans.png')
        )
        
        # Hierarchical
        print("\n[5b] Hierarchical Clustering...")
        ho_hierarchical = ho_clusterer.hierarchical_clustering(
            n_clusters=hierarchical_k,
            plot_dendrogram=True
        )
        ho_clusterer.visualize_clustered_network(
            cluster_assignments=ho_hierarchical,
            title='Hierarchical Clustering - Handover Network',
            save_path=os.path.join(self.output_dir, 'handover_hierarchical.png')
        )
        
        print("\n" + "=" * 80)
        print("PIPELINE COMPLETED SUCCESSFULLY")
        print(f"All outputs saved to: {self.output_dir}")
        print("=" * 80)
        
        return {
            'similarity_network': ra_graph,
            'handover_network': ho_graph,
            'similarity_kmeans': kmeans_clusters,
            'similarity_hierarchical': hierarchical_clusters,
            'handover_kmeans': ho_kmeans,
            'handover_hierarchical': ho_hierarchical
        }