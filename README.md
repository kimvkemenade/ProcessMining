# Social Network Mining - Process Mining Assignment

This project implements organizational mining techniques to analyze social networks from event logs, including resource similarity networks, handover of work networks, and clustering analysis.

## Overview

The implementation provides three main organizational mining approaches:

1. **Resource Similarity Network**: Models task-based similarities between resources using activity profiles
2. **Handover of Work Network**: Models work handovers between resources based on event sequences and process models
3. **Clustering Analysis**: Applies K-means and hierarchical clustering to discover organizational structures

## Project Structure

```
social_network_mining/
├── config.yaml                 # Configuration file for all parameters
├── main.py                     # Main orchestration script
├── dataset.py                  # Event log loading and preprocessing
├── resource_similarity.py      # Resource similarity network generation
├── handover.py                 # Handover of work network generation
├── clustering.py               # Clustering algorithms
├── requirements.txt            # Python dependencies
├── README.md                   # This file
└── output/                     # Generated outputs (created automatically)
    ├── resource_activity_matrix.csv
    ├── handover_matrix.csv
    ├── cluster_assignments_*.csv
    └── *.png (network visualizations)
```

## Requirements

- Python 3.8 or higher
- See `requirements.txt` for package dependencies

## Installation

1. Clone or download this project

2. Install required packages:
```bash
pip install -r requirements.txt
```

3. Place your XES event log file in the project directory

4. Update `config.yaml` with your event log filename

## Usage

### Quick Start

Run the complete analysis pipeline with default settings:

```bash
python main.py
```

### Custom Configuration

1. Edit `config.yaml` to customize parameters:
   - Dataset path
   - Distance metrics and thresholds
   - Clustering algorithm and number of clusters
   - Visualization options

2. Run with custom config:
```bash
python main.py --config my_config.yaml
```

### Configuration Options

#### Dataset Configuration
```yaml
dataset:
  xes_file: "your_event_log.xes"  # Path to XES file
```

#### Resource Similarity Network
```yaml
resource_similarity:
  metric: "hamming"              # Distance metric: 'minkowski', 'hamming', 'pearson'
  minkowski_p: 2                 # Parameter for Minkowski distance
  threshold: 0.4                 # Threshold for filtering edges (lower = more similar)
```

**Distance Metrics:**
- **Minkowski**: Generalized distance metric (p=1: Manhattan, p=2: Euclidean)
- **Hamming**: Proportion of differing activity profiles
- **Pearson**: Based on correlation between activity profiles

**Threshold**: Lower values create edges only between very similar resources. Higher values include more edges.

#### Handover Network
```yaml
handover:
  threshold: 0.0001              # Minimum handover frequency to include
  directed: true                 # Create directed graph (true/false)
```

#### Clustering
```yaml
clustering:
  algorithm: "kmeans"            # Algorithm: 'kmeans' or 'hierarchical'
  n_clusters: 6                  # Number of clusters
  use_elbow_method: true         # Show elbow plot before clustering
  linkage: "ward"                # For hierarchical: 'ward', 'average', 'complete', 'single'
  cluster_on: "handover"         # Data to cluster: 'handover' or 'resources'
```

#### Visualization
```yaml
visualization:
  show_plots: true               # Display plots during execution
  save_plots: true               # Save plots to files
  output_dir: "output"           # Output directory
```

## Output Files

The analysis generates several output files in the `output/` directory:

1. **resource_activity_matrix.csv**: Resource-activity profile matrix showing mean activity counts per case
2. **handover_matrix.csv**: Handover frequency matrix between resources
3. **cluster_assignments_*.csv**: Resource-to-cluster assignments
4. **resource_similarity_network.png**: Visualization of resource similarity network
5. **handover_network.png**: Visualization of handover network
6. **clustered_*_network.png**: Clustered network visualization

## Methodology

### 1. Resource Similarity Network

**Process:**
1. Build resource-activity matrix: count mean occurrences of each activity per resource per case
2. Compute pairwise distances between resource profiles using selected metric
3. Create network where edges represent similarity (distance ≤ threshold)

**Interpretation:**
- Connected resources perform similar sets of activities
- Clusters may represent functional roles or teams

### 2. Handover of Work Network

**Process:**
1. Discover process model using Inductive Miner
2. Extract parallel activity pairs from process tree
3. Count handovers: consecutive activities by different resources (excluding parallel activities)
4. Normalize to get mean handovers per case
5. Create directed network where edges represent handover frequency

**Interpretation:**
- Edges show work transfer between resources
- Strong edges indicate frequent collaboration
- Network structure reveals organizational workflow patterns

### 3. Clustering Analysis

**Algorithms:**

**K-means:**
- Partitions resources into k clusters
- Minimizes within-cluster variance
- Use elbow method to determine optimal k

**Hierarchical:**
- Builds hierarchy of clusters
- Linkage methods:
  - **Ward**: Minimizes variance within clusters (default)
  - **Average**: Uses average distance between clusters
  - **Complete**: Uses maximum distance
  - **Single**: Uses minimum distance

**Interpretation:**
- Clusters reveal organizational structure
- Can identify teams, roles, or resource groups

## Example Workflows

### Example 1: Quick Analysis with Defaults
```bash
python main.py
```

### Example 2: K-means with Elbow Method
Edit `config.yaml`:
```yaml
clustering:
  algorithm: "kmeans"
  use_elbow_method: true
  cluster_on: "handover"
```
Run: `python main.py`

### Example 3: Hierarchical Clustering on Resource Similarity
Edit `config.yaml`:
```yaml
clustering:
  algorithm: "hierarchical"
  n_clusters: 5
  linkage: "ward"
  cluster_on: "resources"
```
Run: `python main.py`

### Example 4: Compare Different Distance Metrics
Create multiple config files (config_hamming.yaml, config_pearson.yaml) with different metrics, then run:
```bash
python main.py --config config_hamming.yaml
python main.py --config config_pearson.yaml
```

## Evaluation

The assignment requires evaluation using precision and recall against a gold standard. To implement this:

1. **Create gold standard**: Manually label expected clusters or relationships
2. **Compute metrics**:
   - **Precision**: Proportion of discovered relationships that are correct
   - **Recall**: Proportion of true relationships that were discovered
   - **F1-score**: Harmonic mean of precision and recall

3. **Compare**: Network edges, cluster assignments against ground truth

Example evaluation code structure:
```python
def evaluate_clusters(predicted_clusters, gold_standard):
    # Compare cluster assignments
    # Compute precision, recall, F1
    pass

def evaluate_network(predicted_edges, gold_standard_edges):
    # Compare edge sets
    # Compute precision, recall
    pass
```

## Troubleshooting

### Common Issues

**Error: XES file not found**
- Check that the file path in `config.yaml` is correct
- Ensure the XES file is in the correct location

**Empty networks**
- Threshold may be too restrictive - try increasing/decreasing threshold values
- Check that event log has required columns: case ID, activity, resource, timestamp

**Import errors**
- Ensure all dependencies are installed: `pip install -r requirements.txt`
- Check Python version (3.8+)

**Memory issues with large datasets**
- Process a subset of the data first
- Reduce visualization resolution

## Implementation Notes

### Key Design Decisions

1. **Parallel Activity Detection**: Uses process tree from Inductive Miner to identify parallel branches
2. **Handover Definition**: Follows Song & Van der Aalst (2008) - consecutive activities by different resources, excluding parallel activities
3. **Normalization**: Handover matrix normalized by total handovers for interpretability
4. **Clustering Input**: Uses distance/handover matrices directly (not graph structure) for compatibility with scikit-learn

### Limitations

- Process discovery quality affects parallel activity detection
- Threshold selection is somewhat subjective
- Large event logs may require significant processing time

## References

Song, M., & Van der Aalst, W. M. (2008). Towards comprehensive support for organizational mining. Decision support systems, 46(1), 300-317.

## License

This project is for academic purposes as part of the Process Mining (JM0211) course assignment.

## Contact

For questions about this implementation, please refer to the assignment instructions or contact your course instructor.