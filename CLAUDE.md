# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Commands

### Testing
```bash
# Run all tests
python -m pytest tests/

# Run specific test file
python -m pytest tests/test_<module_name>.py

# Run tests with verbose output
python -m pytest tests/ -v
```

### Package Installation
```bash
# Install in development mode
pip install -e .

# Install with poetry (if preferred)
poetry install
```

## Architecture Overview

This is a Python package for managing Visual Graph Datasets (VGDs) designed for graph neural networks and explainable AI research.

### Core Components

- **`visual_graph_datasets/data.py`**: Dataset loading, saving, and management utilities. Key classes include `VisualGraphDatasetReader` and `VisualGraphDatasetWriter`.

- **`visual_graph_datasets/processing/`**: Domain-specific graph processing modules:
  - `base.py`: Core processing interfaces and base classes
  - `molecules.py`: SMILES/molecular graph processing
  - `colors.py`: Color graph processing
  - `generic.py`: Generic graph processing

- **`visual_graph_datasets/visualization/`**: Graph visualization utilities:
  - `base.py`: Core visualization functions
  - `importances.py`: Importance/attribution visualization
  - `molecules.py`: Molecular visualization
  - `colors.py`: Color graph visualization

- **`visual_graph_datasets/generation/`**: Synthetic dataset generation utilities

- **`visual_graph_datasets/experiments/`**: Experiment scripts for dataset creation, especially `generate_molecule_dataset_from_csv.py` which serves as a base for creating molecular datasets from CSV files

### Dataset Format

VGDs store each graph as two files:
- A JSON file containing the full graph representation (nodes, edges, attributes, positions)
- A PNG file with the canonical visualization

Key graph structure in JSON:
- `node_indices`, `node_attributes`: Node data
- `edge_indices`, `edge_attributes`: Edge data  
- `node_positions`: Pixel coordinates in the visualization
- `node_importances_*`, `edge_importances_*`: Ground truth explanations (optional)

### CLI Interface

The package provides a CLI via `visual_graph_datasets.cli` for:
- Downloading datasets from remote providers
- Listing available datasets
- Managing configuration

### Configuration

Uses `visual_graph_datasets/config.py` with YAML configuration files stored in `$HOME/.visual_graph_datasets/config.yaml`.

## Working with Experiments

Most dataset generation is done through experiment files in `visual_graph_datasets/experiments/`. To create a new molecular dataset from CSV:

1. Create a new experiment file extending `generate_molecule_dataset_from_csv.py`
2. Set required parameters: `CSV_FILE_NAME`, `SMILES_COLUMN_NAME`, `TARGET_TYPE`, `TARGET_COLUMN_NAMES`, `DATASET_NAME`
3. Run the experiment to generate the VGD dataset

## Dependencies

Key dependencies include:
- `rdkit`: Molecular processing
- `networkx`: Graph operations
- `matplotlib`: Visualization
- `numpy`: Numerical operations
- `pycomex`: Experiment management framework