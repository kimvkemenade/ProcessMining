"""
Main script for Social Network Mining Assignment.
This script orchestrates the entire analysis pipeline.
"""
import argparse
import yaml
import os
import sys
from pathlib import Path

from dataset import DataSet
from resource_similarity import ResourceSimilarity
from handover import Handover
from clustering import Clustering


def load_config(config_path='config.yaml'):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def ensure_output_dir(output_dir):
    """Create output directory if it doesn't exist."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)


def main(config_path='config.yaml'):
    """
    Main execution function.
    
    Args:
        config_path (str): Path to configuration YAML file
    """
    print("SOCIAL NETWORK MINING - Process Mining Assignment")

    
    # Load configuration

    config = load_config(config_path)
    print(f"Configuration loaded from {config_path}")
    
    # Setup output directory
    output_dir = config['visualization']['output_dir']
    ensure_output_dir(output_dir)
    
    # Load dataset
    xes_file = config['dataset']['xes_file']
    if not os.path.exists(xes_file):
        print(f"ERROR: XES file not found: {xes_file}")
        print("Please update the 'xes_file' path in config.yaml")
        sys.exit(1)
    
    dataset = DataSet(xes_file)
    df = dataset.convert_to_df()
    log = dataset.log
    
    # Resource Similarity Network

    
    resource_sim = ResourceSimilarity(df)
    activity_matrix = resource_sim.create_activity_matrix()
    
    # Save activity matrix
    if config['visualization']['save_plots']:
        matrix_path = os.path.join(output_dir, 'resource_activity_matrix.csv')
        activity_matrix.to_csv(matrix_path)
        print(f"Resource-activity matrix saved to {matrix_path}")
    
    # Compute distances
    metric = config['resource_similarity']['metric']
    minkowski_p = config['resource_similarity']['minkowski_p']
    distance_matrix = resource_sim.compute_distance(metric=metric, minkowski_p=minkowski_p)
    
    # Build similarity network
    threshold = config['resource_similarity']['threshold']
    G_resource = resource_sim.build_network(threshold=threshold)
    
    # Plot similarity network
    if config['visualization']['show_plots']:
        output_path = os.path.join(output_dir, 'resource_similarity_network.png') if config['visualization']['save_plots'] else None
        resource_sim.plot_network(G_resource, output_path=output_path)
    
    # Handover Network
    
    handover = Handover(log)
    handover.discover_process_tree()
    handover.extract_parallel_pairs()
    
    # Compute handover matrix
    handover_matrix, total_counts = handover.compute_handover_matrix(
        df,
        case_col='case_id',
        activity_col='activity',
        resource_col='resource',
        timestamp_col='timestamp'
    )
    
    # Save handover matrix
    if config['visualization']['save_plots']:
        matrix_path = os.path.join(output_dir, 'handover_matrix.csv')
        handover_matrix.to_csv(matrix_path)
        print(f"✓ Handover matrix saved to {matrix_path}")
    
    # Build handover network
    handover_threshold = config['handover']['threshold']
    directed = config['handover']['directed']
    G_handover = handover.build_network(threshold=handover_threshold, directed=directed)
    
    # Plot handover network
    if config['visualization']['show_plots']:
        output_path = os.path.join(output_dir, 'handover_network.png') if config['visualization']['save_plots'] else None
        handover.plot_network(G_handover, output_path=output_path)
    
    # Clustering
    
    clustering = Clustering(
        resource_matrix=distance_matrix,
        resource_graph=G_resource,
        handover_matrix=handover_matrix,
        handover_graph=G_handover
    )
    
    # Elbow method
    if config['clustering']['use_elbow_method']:
        print("\nRunning elbow method to determine optimal k...")
        data_type = config['clustering']['cluster_on']
        clustering.elbow_method(data_type=data_type, k_range=range(2, 11))
    
    # Perform clustering
    algorithm = config['clustering']['algorithm']
    n_clusters = config['clustering']['n_clusters']
    data_type = config['clustering']['cluster_on']
    
    if algorithm == 'kmeans':
        clusters = clustering.cluster_kmeans(n_clusters=n_clusters, data_type=data_type)
    elif algorithm == 'hierarchical':
        linkage = config['clustering']['linkage']
        clusters = clustering.cluster_hierarchical(
            n_clusters=n_clusters, 
            data_type=data_type, 
            linkage=linkage
        )
    else:
        print(f"ERROR: Choose either 'kmeans' or 'hierarchical")
        sys.exit(1)
    
    # Get cluster assignments
    assignments = clustering.get_cluster_assignments(clusters, data_type=data_type)
    print(assignments.to_string(index=False))
    
    # Save cluster assignments
    if config['visualization']['save_plots']:
        assignments_path = os.path.join(output_dir, f'cluster_assignments_{data_type}.csv')
        assignments.to_csv(assignments_path, index=False)
    

    if config['visualization']['show_plots']:
        output_path = os.path.join(output_dir, f'clustered_{data_type}_network.png') if config['visualization']['save_plots'] else None
        threshold_val = handover_threshold if data_type == 'handover' else threshold
        clustering.plot_clustered_network(
            clusters, 
            data_type=data_type,
            threshold=threshold_val,
            directed=directed if data_type == 'handover' else False,
            output_path=output_path
        )



if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Social Network Mining - Process Mining Assignment',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        '--config',
        type=str,
        default='config.yaml',
        help='Path to configuration YAML file (default: config.yaml)'
    )
    
    args = parser.parse_args()
    
    try:
        main(config_path=args.config)
    except Exception as e:
        print(f"\nERROR: {str(e)}")
