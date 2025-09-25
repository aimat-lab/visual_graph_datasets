# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Guidelines

### Docstrings

Docstrings should use the ReStructuredText (reST) format. This is important for generating documentation and for consistency across the codebase. Docstrings should always start with a one-line summary followed by a more detailed paragraph - also including usage examples, for instance. If appropriate, docstrings should not only describe a method or function but also shed some light on the design rationale.

Documentation should also be *appropriate* in length. For simple functions, a brief docstring is sufficient. For more complex functions or classes, more detailed explanations and examples should be provided.

An example docstring may look like this:

```python

def multiply(a: int, b: int) -> int:
    """
    Multiply two integers `a` and `b`.

    This function takes two integers as input and returns their product.

    Example:
    
    ... code-block:: python

        result = multiply(3, 4)
        print(result)  # Output: 12

    :param a: The first integer to multiply.
    :param b: The second integer to multiply.

    :return: The product of the two integers.
    """
    return a * b
```

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
uv pip install -e .
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