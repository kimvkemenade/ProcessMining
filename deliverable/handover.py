"""
Handover of Work module for computing handover networks from event logs and process models.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import pm4py
from pm4py.objects.process_tree.obj import Operator


class Handover:
    """Computes handover of work between resources."""
    
    def __init__(self, log):
        """
        Initialize handover analyzer.
        
        Args:
            log: PM4Py event log object
        """
        self.log = log
        self.process_tree = None
        self.parallel_pairs = None
        self.handover_matrix = None

    def discover_process_tree(self):
        """
        Discover process tree from event log using inductive miner.
        
        Returns:
            ProcessTree: Discovered process tree
        """
        self.process_tree = pm4py.discover_process_tree_inductive(self.log)
        print("Process tree discovered using inductive miner")
        return self.process_tree

    def get_leaf_activities(self, node):
        """
        Recursively extract all leaf activities from a process tree node.
        
        Args:
            node: Process tree node
            
        Returns:
            set: Set of activity labels
        """
        if node.label is not None:  # Leaf node (activity)
            return {node.label}
        
        activities = set()
        if hasattr(node, 'children') and node.children:
            for child in node.children:
                activities |= self.get_leaf_activities(child)
        
        return activities

    def extract_parallel_pairs(self, node=None):
        """
        Extract all pairs of activities that can occur in parallel according to the process model.
        
        Args:
            node: Process tree node (if None, uses self.process_tree)
            
        Returns:
            set: Set of frozensets, each containing two parallel activities
        """
        if node is None:
            if self.process_tree is None:
                self.discover_process_tree()
            node = self.process_tree
        
        pairs = set()
    
        # Safe check for parallel operator
        if hasattr(node, 'operator') and node.operator == Operator.PARALLEL:
            children_sets = [self.get_leaf_activities(c) for c in node.children]
            for i in range(len(children_sets)):
                for j in range(i+1, len(children_sets)):
                    for a in children_sets[i]:
                        for b in children_sets[j]:
                            pairs.add(frozenset([a, b]))
        
        # Safe check for children before recursing
        if hasattr(node, 'children') and node.children:
            for child in node.children:
                pairs |= self.extract_parallel_pairs(child)
        
        self.parallel_pairs = pairs
        return pairs

    def compute_handover_matrix(self, df, case_col='case_id', activity_col='activity',
                                resource_col='resource', timestamp_col='timestamp'):
        """
        Compute handover of work matrix from event log.
        
        A handover occurs when consecutive activities in a case are performed by different resources,
        excluding cases where the activities are known to be parallel in the process model.
        
        Args:
            df (pd.DataFrame): Event log DataFrame
            case_col (str): Case ID column name
            activity_col (str): Activity column name
            resource_col (str): Resource column name
            timestamp_col (str): Timestamp column name
            
        Returns:
            tuple: (handover_df, total_counts)
                - handover_df: Normalized handover matrix (proportions)
                - total_counts: Raw handover counts
        """
        if self.parallel_pairs is None:
            print("Warning: No parallel pairs extracted. All consecutive different resources will count as handovers.")
            self.parallel_pairs = set()

        resources = sorted(df[resource_col].dropna().unique().tolist())
        r_index = {r: i for i, r in enumerate(resources)}
        n = len(resources)

        # total handover counts across all cases (directed)
        total_counts = np.zeros((n, n), dtype=float)

        # number of cases (denominator)
        cases = df[case_col].unique()
        n_cases = len(cases)
        if n_cases == 0:
            raise ValueError("Empty log or no cases found.")

        # group by case
        grouped = df.groupby(case_col)

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
                if frozenset([act_i, act_j]) in self.parallel_pairs:
                    continue

                # otherwise increment handover count res_i -> res_j
                if res_i in r_index and res_j in r_index:
                    total_counts[r_index[res_i], r_index[res_j]] += 1.0
        print(total_counts)
        # mean per case
        mean_per_case = total_counts / np.array(total_counts).sum()

        handover_df = pd.DataFrame(mean_per_case, index=resources, columns=resources)
        self.handover_matrix = handover_df
        return handover_df, total_counts

    def build_network(self, threshold=0.0002, directed=True, show_unconnected=False):
        """
        Build handover network from handover matrix.
        
        Args:
            threshold (float): Minimum edge weight to include
            directed (bool): Whether to create directed graph
            show_unconnected (bool): Whether to include isolated nodes
            
        Returns:
            nx.Graph or nx.DiGraph: Handover network
        """
        if self.handover_matrix is None:
            raise ValueError("Must compute handover matrix first using compute_handover_matrix()")
        
        handover_df = self.handover_matrix
        G = nx.DiGraph() if directed else nx.Graph()

        if directed:
            for src in handover_df.index:
                for dst in handover_df.columns:
                    weight = handover_df.at[src, dst]
                    if weight >= threshold and weight > 0:
                        G.add_edge(src, dst, weight=float(weight))
            
            if show_unconnected:
                for node in handover_df.index:
                    G.add_node(node)
        else:
            visited = set()
            for a in handover_df.index:
                for b in handover_df.columns:
                    if (a, b) in visited or (b, a) in visited:
                        continue
                    weight = (handover_df.at[a, b] + handover_df.at[b, a]) / 2.0
                    if weight >= threshold and weight > 0:
                        G.add_edge(a, b, weight=float(weight))
                    visited.add((a, b))
            
            if show_unconnected:
                for node in handover_df.index:
                    G.add_node(node)

        print(f"Handover network: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
        return G

    def plot_network(self, G, output_path=None):
        """
        Visualize the handover network.
        
        Args:
            G (nx.Graph or nx.DiGraph): Network to plot
            output_path (str, optional): Path to save figure
        """
        plt.figure(figsize=(12, 10))
        pos = nx.spring_layout(G, seed=42,)
        
        # Get edge weights for visualization
        edge_weights = [G[u][v]['weight'] for u, v in G.edges()]
        
        # Draw network
        nx.draw_networkx_nodes(G, pos, node_size=500, node_color='#E45756', alpha=0.9)
        nx.draw_networkx_edges(
            G, pos, 
            width=[max(0.5, w * 100) for w in edge_weights],
            alpha=0.6,
            arrowsize=20,
            arrowstyle='->',
            connectionstyle='arc3,rad=0.1'
        )
        nx.draw_networkx_labels(G, pos, font_size=10, font_weight='bold')
        
        plt.title('Handover of Work Network\n(Based on Activity Handovers)', fontsize=14, fontweight='bold')
        plt.axis('off')
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"Network plot saved to {output_path}")
        
        plt.show()