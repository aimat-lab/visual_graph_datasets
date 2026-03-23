"""
Filters a source CSV of molecules into exclusive functional group subsets for the synth2 dataset.

Each molecule in the output contains exactly one of the following functional groups
and none of the others:

- OH: ``[OX2H]`` -> +1
- C=O: ``[CX3]=[OX1]`` -> +1
- NH2: ``[NX3H2]`` -> -1
- C≡N: ``[CX2]#[NX1]`` -> -1

Before classification, each molecule is sanitized (charges neutralized, stereochemistry removed).
Molecules that match more than one group or none are discarded.

Usage::

    python filter_zinc_for_synth2.py --balance
    python filter_zinc_for_synth2.py --csv /path/to/molecules.csv --smiles-column SMILES --balance
    python filter_zinc_for_synth2.py --balance --seed 42 --max-per-group 50000
"""
import os
import csv
import argparse
import random

from tqdm import tqdm

from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem.MolStandardize import rdMolStandardize
from rdkit.rdBase import BlockLogs

block = BlockLogs()

# SMARTS patterns for the 4 functional groups (used for classification).
# These are intentionally broad to avoid visual ambiguity: any molecule that passes the
# filter should contain substructures from exactly one of these groups and nothing that
# could be visually confused with another group.
PATTERNS = {
    'OH':  '[OX2H]',              # any OH group
    'CO':  '[CX3]=[OX1]',         # any carbonyl C=O
    'NH2': '[NX3H2]',             # any primary amine NH2
    'CN':  '[CX2]#[NX1]',         # any nitrile C≡N
}


def sanitize_mol(mol: Chem.Mol) -> Chem.Mol:
    """
    Sanitize a molecule by neutralizing charges and removing stereochemistry.

    :param mol: RDKit molecule object

    :returns: Sanitized molecule, or None if sanitization fails.
    """
    try:
        uncharger = rdMolStandardize.Uncharger()
        mol = uncharger.uncharge(mol)
        Chem.RemoveStereochemistry(mol)
        Chem.SanitizeMol(mol)
        return mol
    except Exception:
        return None


def reservoir_insert(reservoir, item, count, max_size, rng):
    """
    Reservoir sampling insert. Maintains a uniformly random sample of at most
    ``max_size`` items from a stream of ``count`` items seen so far.

    :param reservoir: The list holding the current sample.
    :param item: The new item to potentially insert.
    :param count: How many items (including this one) have been seen for this reservoir.
    :param max_size: Maximum reservoir capacity.
    :param rng: Random instance for reproducibility.
    """
    if count <= max_size:
        reservoir.append(item)
    else:
        j = rng.randint(0, count - 1)
        if j < max_size:
            reservoir[j] = item


def main():
    parser = argparse.ArgumentParser(description='Filter a molecule CSV for the synth2 dataset')
    parser.add_argument(
        '--csv',
        type=str,
        default='/media/ssd/Programming/pubchem-download/output/pubchem32.csv',
        help='Path to the source CSV file',
    )
    parser.add_argument(
        '--smiles-column',
        type=str,
        default='SMILES',
        help='Name of the column containing SMILES strings (default: SMILES)',
    )
    parser.add_argument(
        '--balance',
        action='store_true',
        help='Downsample all groups to the size of the smallest group',
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducible sampling (default: 42)',
    )
    parser.add_argument(
        '--max-per-group',
        type=int,
        default=50_000,
        help='Maximum number of candidates to keep per group via reservoir sampling (default: 50000)',
    )
    parser.add_argument(
        '--target-per-group',
        type=int,
        default=None,
        help='Stop early once every group has at least this many molecules',
    )
    args = parser.parse_args()

    base_dir = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(base_dir, 'assets', 'synth2_subset.csv')

    # Compile SMARTS patterns
    compiled = {}
    for name, smarts in PATTERNS.items():
        pat = Chem.MolFromSmarts(smarts)
        if pat is None:
            raise ValueError(f'Invalid SMARTS for {name}: {smarts}')
        compiled[name] = pat

    # Reservoir-sampled groups and true counts
    rng = random.Random(args.seed)
    reservoirs = {name: [] for name in PATTERNS}
    group_counts = {name: 0 for name in PATTERNS}

    skipped_invalid = 0
    skipped_sanitize = 0
    skipped_none = 0
    skipped_multi = 0
    total_rows = 0

    print(f'Source:         {args.csv}')
    print(f'SMILES column:  {args.smiles_column}')
    print(f'Max per group:  {args.max_per_group}')

    # Count lines for progress bar (subtract 1 for header)
    print('Counting rows...')
    with open(args.csv) as f:
        num_lines = sum(1 for _ in f) - 1
    print(f'Total rows:     {num_lines:,}')
    print()

    with open(args.csv) as f:
        reader = csv.DictReader(f)
        pbar = tqdm(reader, total=num_lines, unit=' mols', mininterval=2.0)
        for row in pbar:
            total_rows += 1

            smiles = row[args.smiles_column].strip()
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                skipped_invalid += 1
                continue

            mol = sanitize_mol(mol)
            if mol is None:
                skipped_sanitize += 1
                continue

            matches = [name for name, pat in compiled.items() if mol.HasSubstructMatch(pat)]

            if len(matches) == 0:
                skipped_none += 1
            elif len(matches) > 1:
                skipped_multi += 1
            else:
                group_name = matches[0]
                group_counts[group_name] += 1

                out_row = {
                    'smiles': Chem.MolToSmiles(mol),
                    'group': group_name,
                }
                reservoir_insert(
                    reservoirs[group_name],
                    out_row,
                    group_counts[group_name],
                    args.max_per_group,
                    rng,
                )

            pbar.set_postfix(
                OH=group_counts['OH'],
                CO=group_counts['CO'],
                NH2=group_counts['NH2'],
                CN=group_counts['CN'],
            )

            if (args.target_per_group is not None
                    and all(c >= args.target_per_group for c in group_counts.values())):
                print(f'\nEarly stop: all groups reached {args.target_per_group} molecules.')
                break

    total_exclusive = sum(group_counts.values())
    print(f'\n--- Filtering Results ---')
    print(f'Total rows processed:          {total_rows:,}')
    print(f'Invalid SMILES (skipped):      {skipped_invalid:,}')
    print(f'Sanitization failed (skipped): {skipped_sanitize:,}')
    print(f'No matching groups (skipped):  {skipped_none:,}')
    print(f'Multiple groups (skipped):     {skipped_multi:,}')
    print(f'Total exclusive molecules:     {total_exclusive:,}')
    print()

    for name in PATTERNS:
        count = group_counts[name]
        kept = len(reservoirs[name])
        pct = count / total_exclusive * 100 if total_exclusive > 0 else 0
        contribution = '+1' if name in ('OH', 'CO') else '-1'
        print(f'  {name:>4} ({contribution}): {count:>10,} total  '
              f'{kept:>6,} kept in reservoir ({pct:.1f}%)')

    # Balancing: downsample all reservoirs to the size of the smallest
    if args.balance:
        min_size = min(len(r) for r in reservoirs.values())
        print(f'\nBalancing: downsampling all groups to {min_size} molecules each')
        print(f'Total after balancing: {min_size * len(PATTERNS)}')
        for name in reservoirs:
            rng.shuffle(reservoirs[name])
            reservoirs[name] = reservoirs[name][:min_size]

    # Combine and write output
    all_rows = []
    for name in PATTERNS:
        all_rows.extend(reservoirs[name])

    out_fieldnames = ['smiles', 'group']
    with open(output_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=out_fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)

    print(f'\nOutput written to: {output_path}')
    print(f'Total molecules in output: {len(all_rows)}')

    # -- Create a 5x5 grid of randomly sampled molecules for visual inspection --
    rng_vis = random.Random(args.seed)
    sample_rows = rng_vis.sample(all_rows, min(25, len(all_rows)))

    mols = []
    legends = []
    for row in sample_rows:
        mol = Chem.MolFromSmiles(row['smiles'].strip())
        if mol is not None:
            mols.append(mol)
            legends.append(row['group'])

    img = Draw.MolsToGridImage(
        mols,
        molsPerRow=5,
        subImgSize=(400, 400),
        legends=legends,
    )
    grid_path = os.path.join(base_dir, 'assets', 'synth2_subset_grid.png')
    img.save(grid_path)
    print(f'Inspection grid saved to: {grid_path}')


if __name__ == '__main__':
    main()
