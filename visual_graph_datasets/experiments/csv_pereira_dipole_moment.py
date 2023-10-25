"""

"""
import os
import csv
import pathlib
import signal
import typing as t

from rdkit import Chem
from pycomex.functional.experiment import Experiment
from pycomex.utils import folder_path, file_namespace

PATH = pathlib.Path(__file__).parent.absolute()
ASSETS_PATH = os.path.join(PATH, 'assets')
CSV_PATH = os.path.join(ASSETS_PATH, 'dipole_moments_raw.csv')
SDF_FOLDER_PATH = os.path.join(ASSETS_PATH, 'dipole_moments_sdf')

__DEBUG__ = True


@Experiment(base_path=folder_path(__file__),
            namespace=file_namespace(__file__),
            glob=globals())
def experiment(e: Experiment):
    e.log('reading csv file...')
    dataset: t.List[dict] = []
    with open(CSV_PATH) as file:
        dict_reader = csv.DictReader(file)
        for index, row in enumerate(dict_reader):
            dataset.append({
                'sdf_path': os.path.join(SDF_FOLDER_PATH, row['Molecule_ID']),
                'value': row['Dipole_Moment(D)'],
            })

    dataset_length = len(dataset)
    e.log(f'read dataset of {dataset_length} molecules')

    e.log('extracting SMILES representations from SDF files...')
    index = 0
    for data in dataset:
        try:
            sdf_path = data['sdf_path']
            supplier = Chem.SDMolSupplier(sdf_path, sanitize=False)
            mol = supplier[0]
            Chem.SanitizeMol(mol)
            Chem.RemoveStereochemistry(mol)
            mol = Chem.RemoveAllHs(mol)
            Chem.SanitizeMol(mol)
            smiles = Chem.MolToSmiles(mol)
            data['smiles'] = smiles
            data['index'] = index
            index += 1
        except Exception as exc:
            e.log(f' ! error: {exc}')

        if index % 100 == 0:
            e.log(f' * {index}/{dataset_length} done')

    e.log('writing to new CSV file...')
    result_path = os.path.join(e.path, 'dipole_moment.csv')
    with open(result_path, mode='w') as file:
        dict_writer = csv.DictWriter(file, ['index', 'smiles', 'value'])
        dict_writer.writeheader()
        for c, data in enumerate([data for data in dataset if 'smiles' in data]):
            del data['sdf_path']
            dict_writer.writerow(data)

    e.log(f'created CSV file with {c} elements')


experiment.run_if_main()
