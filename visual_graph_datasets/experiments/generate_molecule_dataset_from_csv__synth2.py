"""
Generates a synthetic visual graph dataset (synth2) from a CSV source file with motif-based labels.

This experiment generates a synthetic dataset where the target values are computed based on the
presence of specific molecular motifs (functional groups). Ground truth explanation masks are
generated with 2 channels: one for positive-contributing motifs and one for negative-contributing
motifs.

**MOTIF CONTRIBUTIONS**

The target value for each molecule is computed as the sum of contributions from detected motifs:
- OH groups (any hydroxyl): +1 contribution each
- C=O groups (any carbonyl): +1 contribution each
- NH2 groups (any primary amine): -1 contribution each
- C≡N groups (any nitrile): -1 contribution each

Each molecule in the source CSV is pre-filtered to contain substructures from exactly one of
these four groups, so the target and explanation masks are unambiguous.

**GROUND TRUTH EXPLANATIONS**

The dataset includes ground truth explanation masks stored as:
- ``node_importances``: shape (num_nodes, 2) - Binary mask indicating motif membership
  - Channel 0: Positive motif atoms (OH and C=O groups)
  - Channel 1: Negative motif atoms (NH2 and C≡N groups)
- ``edge_importances``: shape (num_edges, 2) - Binary mask indicating motif membership
  - Channel 0: Positive motif bonds
  - Channel 1: Negative motif bonds

**CHANGELOG**

0.2.0 - Broadened motif SMARTS to match any OH / C=O / NH2 / C≡N. Added charge neutralization
and stereochemistry removal in the pre-filtering step.

0.1.0 - Initial version with OH (+1), C=O (+1), NH2 (-1), C≡N (-1) motif contributions
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
#       conversion. This should point to the filtered subset produced by filter_zinc_for_synth2.py.
CSV_FILE_NAME: str = '/media/ssd/Programming/visual_graph_datasets/visual_graph_datasets/experiments/assets/synth2_subset.csv'
# :param INDEX_COLUMN_NAME:
#       (Optional) this may define the string name of the CSV column which contains the integer index
#       associated with each dataset element.
INDEX_COLUMN_NAME: t.Optional[str] = None
# :param SMILES_COLUMN_NAME:
#       This has to be the string name of the CSV column which contains the SMILES string representation of
#       the molecule.
SMILES_COLUMN_NAME: str = 'smiles'
# :param TARGET_TYPE:
#       Binary classification: each molecule gets target +1 (positive motif) or -1 (negative motif).
TARGET_TYPE: str = 'regression'
# :param TARGET_COLUMN_NAMES:
#       Empty because the target is fully computed by the synthetic hook, not read from the CSV.
TARGET_COLUMN_NAMES: t.List[str] = []
# :param SPLIT_COLUMN_NAMES:
#       The keys of this dictionary are integers which represent the indices of various train test splits.
SPLIT_COLUMN_NAMES: t.Dict[int, str] = {}

# == SYNTHETIC DATASET PARAMETERS ==
# These parameters control the synthetic label generation based on molecular motifs.

# :param MOTIF_CONTRIBUTIONS:
#       A dictionary mapping SMARTS patterns to their contribution values. Each detected motif instance
#       adds its contribution to the final target value.
#       The C=O pattern uses a positive specification: the carbonyl carbon's other two neighbors
#       (besides =O) must be carbon or hydrogen. This excludes carboxylic acids, esters, amides,
#       acyl halides, anhydrides, etc.
MOTIF_CONTRIBUTIONS: t.Dict[str, float] = {
    # Any OH group
    '[OX2H]': +1.0,
    # Any carbonyl C=O group
    '[CX3]=[OX1]': +1.0,
    # Any primary amine NH2 group
    '[NX3H2]': -1.0,
    # Any nitrile C≡N group
    '[CX2]#[NX1]': -1.0,
}
# :param MOTIF_CHANNEL_MAP:
#       Maps each SMARTS pattern to an importance channel index. Multiple motifs can share
#       the same channel. Channel 0 groups all positive-contributing motifs, channel 1 groups
#       all negative-contributing motifs.
MOTIF_CHANNEL_MAP: t.Dict[str, int] = {
    '[OX2H]': 0,
    '[CX3]=[OX1]': 0,
    '[NX3H2]': 1,
    '[CX2]#[NX1]': 1,
}
# :param NUM_IMPORTANCE_CHANNELS:
#       The number of importance channels in the explanation masks.
NUM_IMPORTANCE_CHANNELS: int = 2
# == DATASET PARAMETERS ==

# :param DATASET_CHUNK_SIZE:
#       This number will determine the chunking of the dataset.
DATASET_CHUNK_SIZE: t.Optional[int] = None
# :param DATASET_NAME:
#       The name given to the visual graph dataset folder which will be created.
DATASET_NAME: str = 'synth2'
# :param DATASET_META:
#       This dict will be converted into the .meta.yml file which will be added to the final visual graph
#       dataset folder.
DATASET_META: t.Optional[dict] = {
    'version': '0.2.0',
    'changelog': [
        '0.2.0 - Broadened motif SMARTS to match any OH / C=O / NH2 / C≡N. '
        'Added charge neutralization and stereochemistry removal in pre-filtering.',
        '0.1.0 - Initial version with OH (+1), C=O (+1), NH2 (-1), C≡N (-1) motif contributions',
    ],
    'description': (
        'Synthetic dataset based on a filtered subset of ZINC250k where target values are computed '
        'based on molecular motif contributions. Any OH group and any C=O group each contribute +1, '
        'while any NH2 group and any C≡N group each contribute -1. The subset contains molecules '
        'with exclusively one of the four functional groups. All molecules are sanitized '
        '(charges neutralized, stereochemistry removed). Includes ground truth explanation masks '
        'with 2 channels: positive motifs (OH, C=O) and negative motifs (NH2, C≡N).'
    ),
    'references': [
        'ZINC250k dataset: https://zinc.docking.org/',
        'Library used for the processing and visualization of molecules. https://www.rdkit.org/',
    ],
    'visualization_description': (
        'Molecular graphs generated by RDKit based on the SMILES representation of the molecule.'
    ),
    'target_descriptions': {
        0: 'Synthetic binary target: +1 if molecule contains a positive motif (OH or C=O), '
           '-1 if it contains a negative motif (NH2 or C≡N)'
    },
    'importance_channels': {
        0: 'Positive motif atoms/bonds (OH alcohol/phenol, C=O ketone/aldehyde)',
        1: 'Negative motif atoms/bonds (NH2 primary amine, C≡N nitrile)',
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

    For each match of the SMARTS pattern, this function identifies all atoms in the match
    and all bonds connecting atoms within the match.

    :param mol: RDKit molecule object
    :param smarts: SMARTS pattern string to search for

    :returns: Tuple of (atom_indices_set, edge_tuples_set)
    """
    matches = find_motif_atoms(mol, smarts)

    motif_atoms: t.Set[int] = set()
    motif_edges: t.Set[t.Tuple[int, int]] = set()

    for match in matches:
        motif_atoms.update(match)

        for i, atom_idx_i in enumerate(match):
            for atom_idx_j in match[i+1:]:
                bond = mol.GetBondBetweenAtoms(atom_idx_i, atom_idx_j)
                if bond is not None:
                    motif_edges.add((atom_idx_i, atom_idx_j))
                    motif_edges.add((atom_idx_j, atom_idx_i))

    return motif_atoms, motif_edges


def compute_synthetic_target_and_importances(
    mol: Chem.Mol,
    motif_contributions: t.Dict[str, float],
) -> t.Tuple[float, t.Dict[str, t.Set[int]], t.Dict[str, t.Set[t.Tuple[int, int]]]]:
    """
    Compute the synthetic target value and identify motif atoms/edges for a molecule.

    The target is determined by the *presence* of a motif, not by how many times it occurs.
    If any match of a SMARTS pattern is found, its contribution (+1 or -1) is added exactly
    once. Since the input molecules are pre-filtered to contain exactly one motif group,
    the target will always be either +1 or -1.

    :param mol: RDKit molecule object
    :param motif_contributions: Dictionary mapping SMARTS patterns to contribution values

    :returns: Tuple of (target_value, motif_atoms_dict, motif_edges_dict)
        - target_value: +1 or -1 depending on which motif group is present
        - motif_atoms_dict: Dict mapping SMARTS to set of atom indices
        - motif_edges_dict: Dict mapping SMARTS to set of edge tuples
    """
    target = 0.0
    motif_atoms_dict: t.Dict[str, t.Set[int]] = {}
    motif_edges_dict: t.Dict[str, t.Set[t.Tuple[int, int]]] = {}

    for smarts, contribution in motif_contributions.items():
        atoms, edges = get_motif_atoms_and_edges(mol, smarts)
        motif_atoms_dict[smarts] = atoms
        motif_edges_dict[smarts] = edges

        if atoms:
            target += contribution

    return target, motif_atoms_dict, motif_edges_dict


def create_importance_arrays(
    mol: Chem.Mol,
    edge_indices: np.ndarray,
    motif_contributions: t.Dict[str, float],
    motif_channel_map: t.Dict[str, int],
    num_channels: int,
    motif_atoms_dict: t.Dict[str, t.Set[int]],
    motif_edges_dict: t.Dict[str, t.Set[t.Tuple[int, int]]],
) -> t.Tuple[np.ndarray, np.ndarray]:
    """
    Create node and edge importance arrays with channels defined by ``motif_channel_map``.

    Unlike the synth experiment where each motif gets its own channel, here multiple motifs
    can share the same channel (e.g., all positive-contributing motifs share channel 0).

    :param mol: RDKit molecule object
    :param edge_indices: Array of shape (num_edges, 2) with edge source/target indices
    :param motif_contributions: Dictionary mapping SMARTS patterns to contribution values
    :param motif_channel_map: Dictionary mapping SMARTS patterns to channel indices
    :param num_channels: Total number of importance channels
    :param motif_atoms_dict: Dict mapping SMARTS to set of atom indices in that motif
    :param motif_edges_dict: Dict mapping SMARTS to set of edge tuples in that motif

    :returns: Tuple of (node_importances, edge_importances)
        - node_importances: Array of shape (num_atoms, num_channels)
        - edge_importances: Array of shape (num_edges, num_channels)
    """
    num_atoms = mol.GetNumAtoms()
    num_edges = len(edge_indices)

    node_importances = np.zeros((num_atoms, num_channels), dtype=np.float32)
    edge_importances = np.zeros((num_edges, num_channels), dtype=np.float32)

    for smarts in motif_contributions.keys():
        channel_idx = motif_channel_map[smarts]

        motif_atoms = motif_atoms_dict.get(smarts, set())
        for atom_idx in motif_atoms:
            node_importances[atom_idx, channel_idx] = 1.0

        motif_edges = motif_edges_dict.get(smarts, set())
        for edge_idx, (src, dst) in enumerate(edge_indices):
            if (src, dst) in motif_edges:
                edge_importances[edge_idx, channel_idx] = 1.0

    return node_importances, edge_importances


@experiment.hook('modify_filter_callbacks')
def add_synth_filters(e: Experiment, filter_callbacks: t.List[t.Callable]) -> t.List[t.Callable]:
    """
    Add filters to enforce that every molecule has motifs from exactly one importance channel.

    Molecules are excluded if they either contain no motifs at all or if they contain motifs
    belonging to more than one importance channel. This guarantees that each molecule in the
    final dataset is associated with a single, unambiguous explanation channel.
    """

    def exclusive_channel_filter(mol, data) -> bool:
        matched_channels = set()
        for smarts in e.MOTIF_CONTRIBUTIONS.keys():
            if find_motif_atoms(mol, smarts):
                matched_channels.add(e.MOTIF_CHANNEL_MAP[smarts])

        # Keep only molecules that match exactly one channel
        return len(matched_channels) != 1

    filter_callbacks.append(exclusive_channel_filter)

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
    # Compute synthetic target and identify motifs
    target, motif_atoms_dict, motif_edges_dict = compute_synthetic_target_and_importances(
        mol=mol,
        motif_contributions=e.MOTIF_CONTRIBUTIONS,
    )

    # Override the target with our synthetic value
    additional_graph_data['graph_labels'] = [target]

    # Construct edge indices the same way MoleculeProcessing does
    edge_indices = []
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        edge_indices.append([i, j])
        edge_indices.append([j, i])

    edge_indices = np.array(edge_indices) if edge_indices else np.zeros((0, 2), dtype=np.int32)

    # Create importance arrays
    node_importances, edge_importances = create_importance_arrays(
        mol=mol,
        edge_indices=edge_indices,
        motif_contributions=e.MOTIF_CONTRIBUTIONS,
        motif_channel_map=e.MOTIF_CHANNEL_MAP,
        num_channels=e.NUM_IMPORTANCE_CHANNELS,
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

        min_target = int(min(targets))
        max_target = int(max(targets))
        bins = range(min_target, max_target + 2)

        ax.hist(
            targets, bins=bins,
            color='steelblue', edgecolor='black', alpha=0.7, align='left',
        )

        stats_text = (
            f'Total samples: {len(targets)}\n'
            f'Min: {min(targets):.0f}\n'
            f'Max: {max(targets):.0f}\n'
            f'Mean: {np.mean(targets):.2f}\n'
            f'Std: {np.std(targets):.2f}'
        )
        ax.text(
            0.95, 0.95, stats_text,
            transform=ax.transAxes, fontsize=10,
            verticalalignment='top', horizontalalignment='right',
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
            sizes, bins=30,
            color='forestgreen', edgecolor='black', alpha=0.7,
        )

        stats_text = (
            f'Min: {min(sizes)}\n'
            f'Max: {max(sizes)}\n'
            f'Mean: {np.mean(sizes):.1f}\n'
            f'Std: {np.std(sizes):.1f}'
        )
        ax.text(
            0.95, 0.95, stats_text,
            transform=ax.transAxes, fontsize=10,
            verticalalignment='top', horizontalalignment='right',
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
            unique_targets, counts,
            color='coral', edgecolor='black', alpha=0.7,
        )

        for target_val, count in zip(unique_targets, counts):
            ax.text(
                target_val, count + max(counts) * 0.01,
                str(count), ha='center', va='bottom', fontsize=8,
            )

        plt.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

    e.log(f'Dataset statistics saved to: {pdf_path}')


experiment.run_if_main()
