"""
Generates a synthetic visual graph dataset from a CSV source file with motif-based labels.

This experiment generates a synthetic dataset where the target values are computed based on the
presence of specific molecular motifs (functional groups) rather than using the original labels
from the CSV file. Additionally, ground truth explanation masks are generated that indicate which
atoms and bonds belong to each motif type.

**MOTIF CONTRIBUTIONS**

The target value for each molecule is computed as the sum of contributions from detected motifs:
- OH groups (alcohol/phenol): +1 contribution each
- NH2 groups (primary amine): -1 contribution each

For example, a molecule with 2 OH groups and 1 NH2 group would have target = 2*1 + 1*(-1) = 1

**GROUND TRUTH EXPLANATIONS**

The dataset includes ground truth explanation masks stored as:
- ``node_importances``: shape (num_nodes, 2) - Binary mask indicating motif membership
  - Channel 0: OH group atoms (C and O of C-OH)
  - Channel 1: NH2 group atoms (C and N of C-NH2)
- ``edge_importances``: shape (num_edges, 2) - Binary mask indicating motif membership
  - Channel 0: OH group bonds (C-O bond)
  - Channel 1: NH2 group bonds (C-N bond)

**CHANGELOG**

0.1.0 - Initial version with OH (+1) and NH2 (-1) motif contributions
"""
import os
import typing as t

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from pycomex.functional.experiment import Experiment
from pycomex.utils import folder_path, file_namespace
from rdkit import Chem

# == SOURCE PARAMETERS ==
# These parameters determine how to handle the source CSV file of the dataset.

# :param FILE_SHARE_PROVIDER:
#       The vgd file share provider from which to download the CSV file to be used as the source for the VGD
#       conversion. Not used when CSV_FILE_NAME is an absolute local path.
FILE_SHARE_PROVIDER: str = 'main'
# :param CSV_FILE_NAME:
#       The absolute path to the CSV file on the local system to be used as the source for the dataset
#       conversion. The CSV file must contain a column with SMILES strings.
CSV_FILE_NAME: str = '/media/ssd/Programming/visual_graph_datasets/visual_graph_datasets/experiments/assets/zinc_250k_oh_nh2_subset.csv'
# :param INDEX_COLUMN_NAME:
#       (Optional) this may define the string name of the CSV column which contains the integer index
#       associated with each dataset element.
INDEX_COLUMN_NAME: t.Optional[str] = None
# :param SMILES_COLUMN_NAME:
#       This has to be the string name of the CSV column which contains the SMILES string representation of
#       the molecule.
SMILES_COLUMN_NAME: str = 'smiles'
# :param TARGET_TYPE:
#       This has to be the string name of the type of dataset that the source file represents.
#       For synthetic datasets, we use regression since the target is a sum of contributions.
TARGET_TYPE: str = 'regression'
# :param TARGET_COLUMN_NAMES:
#       This has to be a list of string column names within the source CSV file. These are required by the
#       base experiment but will be ignored since we compute synthetic targets.
#       Using 'logP' as a placeholder column from zinc_250k_oh_nh2_subset.csv.
TARGET_COLUMN_NAMES: t.List[str] = ['logP']
# :param SPLIT_COLUMN_NAMES:
#       The keys of this dictionary are integers which represent the indices of various train test splits.
SPLIT_COLUMN_NAMES: t.Dict[int, str] = {}

# == SYNTHETIC DATASET PARAMETERS ==
# These parameters control the synthetic label generation based on molecular motifs.

# :param MOTIF_CONTRIBUTIONS:
#       A dictionary mapping SMARTS patterns to their contribution values. Each detected motif instance
#       adds its contribution to the final target value.
#       - Key: SMARTS pattern string for motif detection
#       - Value: Numeric contribution to add per motif instance
MOTIF_CONTRIBUTIONS: t.Dict[str, float] = {
    # Alcohol/Phenol OH group: Carbon connected to hydroxyl oxygen
    # [CX4] matches sp3 carbon, [c] matches aromatic carbon
    # [OX2H] matches oxygen with 2 connections and one hydrogen
    '[C,c][OX2H]': +1.0,
    # Primary amine NH2 group: Carbon connected to NH2
    # [NX3H2] matches nitrogen with 3 connections and 2 hydrogens
    '[C,c][NX3H2]': -1.0,
}
# :param FILTER_EMPTY_MOTIFS:
#       If True, molecules that do not contain any of the defined motifs will be filtered out
#       and not included in the final dataset. If False (default), such molecules are included
#       with a target value of 0.
FILTER_EMPTY_MOTIFS: bool = False

# == DATASET PARAMETERS ==

# :param DATASET_CHUNK_SIZE:
#       This number will determine the chunking of the dataset.
DATASET_CHUNK_SIZE: t.Optional[int] = None
# :param DATASET_NAME:
#       The name given to the visual graph dataset folder which will be created.
DATASET_NAME: str = 'synth_easy'
# :param DATASET_META:
#       This dict will be converted into the .meta.yml file which will be added to the final visual graph
#       dataset folder.
DATASET_META: t.Optional[dict] = {
    'version': '0.1.0',
    'changelog': [
        '0.1.0 - Initial version with OH (+1) and NH2 (-1) motif contributions',
    ],
    'description': (
        'Synthetic dataset based on a balanced subset of ZINC250k (~29k molecules) where target values '
        'are computed based on molecular motif contributions. OH groups (alcohol/phenol) contribute +1 '
        'each, NH2 groups (primary amine) contribute -1 each. The subset is balanced such that ~50% of '
        'molecules contain OH-only and ~50% contain NH2-only, with other functional groups distributed '
        'evenly across both classes. Includes ground truth explanation masks indicating which atoms and '
        'bonds belong to each motif.'
    ),
    'references': [
        'ZINC250k dataset: https://zinc.docking.org/',
        'Library used for the processing and visualization of molecules. https://www.rdkit.org/',
    ],
    'visualization_description': (
        'Molecular graphs generated by RDKit based on the SMILES representation of the molecule.'
    ),
    'target_descriptions': {
        0: 'Synthetic target: sum of motif contributions (OH: +1, NH2: -1)'
    },
    'importance_channels': {
        0: 'OH group atoms/bonds (alcohol/phenol C-OH)',
        1: 'NH2 group atoms/bonds (primary amine C-NH2)',
    }
}

# == EXPERIMENT PARAMETERS ==

experiment = Experiment.extend(
    'generate_molecule_dataset_from_csv.py',
    base_path=folder_path(__file__),
    namespace=file_namespace(__file__),
    glob=globals(),
)


def find_motif_atoms(mol: Chem.Mol, smarts: str) -> t.List[t.Tuple[int, ...]]:
    """
    Find all occurrences of a SMARTS pattern in a molecule and return the matching atom indices.

    :param mol: RDKit molecule object
    :param smarts: SMARTS pattern string to search for

    :returns: List of tuples, where each tuple contains the atom indices for one match
    """
    pattern = Chem.MolFromSmarts(smarts)
    if pattern is None:
        return []

    matches = mol.GetSubstructMatches(pattern)
    return list(matches)


def get_motif_atoms_and_edges(
    mol: Chem.Mol,
    smarts: str
) -> t.Tuple[t.Set[int], t.Set[t.Tuple[int, int]]]:
    """
    Find all atoms and edges belonging to a motif pattern in a molecule.

    For each match of the SMARTS pattern, this function identifies:
    - All atoms in the match
    - All bonds connecting atoms within the match

    :param mol: RDKit molecule object
    :param smarts: SMARTS pattern string to search for

    :returns: Tuple of (atom_indices_set, edge_tuples_set)
    """
    matches = find_motif_atoms(mol, smarts)

    motif_atoms: t.Set[int] = set()
    motif_edges: t.Set[t.Tuple[int, int]] = set()

    for match in matches:
        # Add all atoms in this match
        motif_atoms.update(match)

        # Find all bonds between atoms in this match
        for i, atom_idx_i in enumerate(match):
            for atom_idx_j in match[i+1:]:
                bond = mol.GetBondBetweenAtoms(atom_idx_i, atom_idx_j)
                if bond is not None:
                    # Store edges in both directions for undirected graph matching
                    motif_edges.add((atom_idx_i, atom_idx_j))
                    motif_edges.add((atom_idx_j, atom_idx_i))

    return motif_atoms, motif_edges


def compute_synthetic_target_and_importances(
    mol: Chem.Mol,
    motif_contributions: t.Dict[str, float],
) -> t.Tuple[float, t.Dict[str, t.Set[int]], t.Dict[str, t.Set[t.Tuple[int, int]]]]:
    """
    Compute the synthetic target value and identify motif atoms/edges for a molecule.

    :param mol: RDKit molecule object
    :param motif_contributions: Dictionary mapping SMARTS patterns to contribution values

    :returns: Tuple of (target_value, motif_atoms_dict, motif_edges_dict)
        - target_value: Sum of all motif contributions
        - motif_atoms_dict: Dict mapping SMARTS to set of atom indices
        - motif_edges_dict: Dict mapping SMARTS to set of edge tuples
    """
    target = 0.0
    motif_atoms_dict: t.Dict[str, t.Set[int]] = {}
    motif_edges_dict: t.Dict[str, t.Set[t.Tuple[int, int]]] = {}

    for smarts, contribution in motif_contributions.items():
        matches = find_motif_atoms(mol, smarts)
        num_matches = len(matches)
        target += num_matches * contribution

        atoms, edges = get_motif_atoms_and_edges(mol, smarts)
        motif_atoms_dict[smarts] = atoms
        motif_edges_dict[smarts] = edges

    return target, motif_atoms_dict, motif_edges_dict


def create_importance_arrays(
    mol: Chem.Mol,
    edge_indices: np.ndarray,
    motif_contributions: t.Dict[str, float],
    motif_atoms_dict: t.Dict[str, t.Set[int]],
    motif_edges_dict: t.Dict[str, t.Set[t.Tuple[int, int]]],
) -> t.Tuple[np.ndarray, np.ndarray]:
    """
    Create node and edge importance arrays with one channel per motif type.

    :param mol: RDKit molecule object
    :param edge_indices: Array of shape (num_edges, 2) with edge source/target indices
    :param motif_contributions: Dictionary mapping SMARTS patterns to contribution values
    :param motif_atoms_dict: Dict mapping SMARTS to set of atom indices in that motif
    :param motif_edges_dict: Dict mapping SMARTS to set of edge tuples in that motif

    :returns: Tuple of (node_importances, edge_importances)
        - node_importances: Array of shape (num_atoms, num_motifs)
        - edge_importances: Array of shape (num_edges, num_motifs)
    """
    num_atoms = mol.GetNumAtoms()
    num_edges = len(edge_indices)
    num_channels = len(motif_contributions)

    node_importances = np.zeros((num_atoms, num_channels), dtype=np.float32)
    edge_importances = np.zeros((num_edges, num_channels), dtype=np.float32)

    # Create importance arrays for each motif channel
    for channel_idx, smarts in enumerate(motif_contributions.keys()):
        # Node importances
        motif_atoms = motif_atoms_dict.get(smarts, set())
        for atom_idx in motif_atoms:
            node_importances[atom_idx, channel_idx] = 1.0

        # Edge importances
        motif_edges = motif_edges_dict.get(smarts, set())
        for edge_idx, (src, dst) in enumerate(edge_indices):
            if (src, dst) in motif_edges:
                edge_importances[edge_idx, channel_idx] = 1.0

    return node_importances, edge_importances


# Filter callback for molecules without any motifs
def no_motifs_filter(mol, data) -> bool:
    """
    Filter callback that returns True if the molecule has no detected motifs.

    Only active when FILTER_EMPTY_MOTIFS is True.
    """
    # Access the experiment's MOTIF_CONTRIBUTIONS through the data dict
    motif_contributions = data.get('_motif_contributions', {})

    for smarts in motif_contributions.keys():
        matches = find_motif_atoms(mol, smarts)
        if len(matches) > 0:
            return False  # Has at least one motif, don't filter

    return True  # No motifs found, filter this molecule


@experiment.hook('modify_filter_callbacks')
def add_synth_filters(e: Experiment, filter_callbacks: t.List[t.Callable]) -> t.List[t.Callable]:
    """
    Add the no_motifs filter if FILTER_EMPTY_MOTIFS is enabled.
    """
    if e.FILTER_EMPTY_MOTIFS:
        filter_callbacks.append(no_motifs_filter)

    return filter_callbacks


@experiment.hook('additional_graph_data')
def add_synthetic_data(
    e: Experiment,
    additional_graph_data: dict,
    mol: Chem.Mol,
    data: dict,
) -> dict:
    """
    Compute synthetic target and add ground truth importance masks to the graph data.

    This hook overrides the original target values from the CSV with synthetically computed
    values based on motif contributions, and adds node_importances and edge_importances arrays.
    """
    # Store motif contributions in data for the filter callback
    data['_motif_contributions'] = e.MOTIF_CONTRIBUTIONS

    # Compute synthetic target and identify motifs
    target, motif_atoms_dict, motif_edges_dict = compute_synthetic_target_and_importances(
        mol=mol,
        motif_contributions=e.MOTIF_CONTRIBUTIONS,
    )

    # Override the target with our synthetic value
    additional_graph_data['graph_labels'] = [target]

    # Get edge indices from the graph data that will be created
    # We need to construct the edge indices the same way MoleculeProcessing does
    edge_indices = []
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        edge_indices.append([i, j])
        # Add reverse edge for undirected graph (matching UNDIRECTED_EDGES_AS_TWO=True)
        edge_indices.append([j, i])

    edge_indices = np.array(edge_indices) if edge_indices else np.zeros((0, 2), dtype=np.int32)

    # Create importance arrays
    node_importances, edge_importances = create_importance_arrays(
        mol=mol,
        edge_indices=edge_indices,
        motif_contributions=e.MOTIF_CONTRIBUTIONS,
        motif_atoms_dict=motif_atoms_dict,
        motif_edges_dict=motif_edges_dict,
    )

    # Add importance arrays to graph data
    additional_graph_data['node_importances'] = node_importances.tolist()
    additional_graph_data['edge_importances'] = edge_importances.tolist()

    return additional_graph_data


@experiment.hook('dataset_info')
def plot_synthetic_target_distribution(e: Experiment, index_data_map: dict) -> None:
    """
    Create plots showing the distribution of synthetic target values and motif statistics.

    This hook overrides the default dataset_info plotting to create visualizations specific
    to the synthetic dataset, including target value distribution and motif count histograms.
    """
    pdf_path = os.path.join(e.path, 'dataset_info.pdf')
    with PdfPages(pdf_path) as pdf:
        # Extract target values
        targets = [d['metadata']['target'][0] for d in index_data_map.values()]

        # Plot 1: Target value distribution
        e.log('Plotting synthetic target value distribution...')
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.set_title('Synthetic Target Value Distribution', fontsize=14)
        ax.set_xlabel('Target Value (sum of motif contributions)', fontsize=12)
        ax.set_ylabel('Count', fontsize=12)

        # Use integer bins since target values are discrete (sum of +1/-1 contributions)
        min_target = int(min(targets))
        max_target = int(max(targets))
        bins = range(min_target, max_target + 2)  # +2 to include the last value

        _, _, _ = ax.hist(
            targets,
            bins=bins,
            color='steelblue',
            edgecolor='black',
            alpha=0.7,
            align='left',
        )

        # Add statistics text
        stats_text = (
            f'Total samples: {len(targets)}\n'
            f'Min: {min(targets):.0f}\n'
            f'Max: {max(targets):.0f}\n'
            f'Mean: {np.mean(targets):.2f}\n'
            f'Std: {np.std(targets):.2f}'
        )
        ax.text(
            0.95, 0.95, stats_text,
            transform=ax.transAxes,
            fontsize=10,
            verticalalignment='top',
            horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        )

        plt.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

        # Plot 2: Graph size distribution
        e.log('Plotting graph size distribution...')
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.set_title('Graph Size Distribution (Number of Atoms)', fontsize=14)
        ax.set_xlabel('Number of Atoms', fontsize=12)
        ax.set_ylabel('Count', fontsize=12)

        sizes = [len(d['metadata']['graph']['node_indices']) for d in index_data_map.values()]
        ax.hist(
            sizes,
            bins=30,
            color='forestgreen',
            edgecolor='black',
            alpha=0.7,
        )

        # Add statistics text
        stats_text = (
            f'Min: {min(sizes)}\n'
            f'Max: {max(sizes)}\n'
            f'Mean: {np.mean(sizes):.1f}\n'
            f'Std: {np.std(sizes):.1f}'
        )
        ax.text(
            0.95, 0.95, stats_text,
            transform=ax.transAxes,
            fontsize=10,
            verticalalignment='top',
            horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        )

        plt.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

        # Plot 3: Target value class counts (bar chart)
        e.log('Plotting target value class counts...')
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.set_title('Target Value Class Counts', fontsize=14)
        ax.set_xlabel('Target Value', fontsize=12)
        ax.set_ylabel('Count', fontsize=12)

        unique_targets, counts = np.unique(targets, return_counts=True)
        ax.bar(
            unique_targets,
            counts,
            color='coral',
            edgecolor='black',
            alpha=0.7,
        )

        # Add count labels on bars
        for target_val, count in zip(unique_targets, counts):
            ax.text(
                target_val, count + max(counts) * 0.01,
                str(count),
                ha='center',
                va='bottom',
                fontsize=8,
            )

        plt.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

    e.log(f'Dataset statistics saved to: {pdf_path}')


experiment.run_if_main()
