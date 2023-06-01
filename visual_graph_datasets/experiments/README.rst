==========================
Experiment Script Overview
==========================

This folder contains the several scripts, each one acting as a pycomex Experiment module.
Each script is executable and will execute an experiment with a specific purpose. These experiments will
create artifacts, which will automatically be stored within their own archive folder within the nested
``results`` folder. For more information...

The following list aims to provide a brief overview over the purpose of each of the scripts.

* ``csv_sanchez_lengeling_dataset.py``: In their paper
  `Evaluating Attribution for Graph Neural Networks`_ Sanchez-Lengeling et al. propose several
  "real-world" chemical datasets with known ground truth attributional explanations. These datasets are
  available as part of the `graph-attribution`_ github repository. This base experiment implements the
  generic conversion of one such dataset into a single CSV file, which can then be further processed into
  a visual graph dataset.

* ``generate_molecule_dataset_from_csv.py``: Base implementation which takes a CSV file containing a
  molecule dataset of SMILES and target values into a visual graph dataset.

    * ``generate_molecule_dataset_from_csv_aqsoldb.py``: Generates the VGD for the AqSolDB dataset based
      on the CSV file from the remote repository



_`Evaluating Attribution for Graph Neural Networks`: https://papers.nips.cc/paper/2020/hash/417fbbf2e9d5a28a855a11894b2e795a-Abstract.html
_`graph-attribution`: https://github.com/google-research/graph-attribution/tree/main/data/benzene